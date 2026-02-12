# app/infrastructure/db_service.py
"""
Database Service.
"""

import gc
import hashlib
import logging
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

# --- Optional Imports for Hardware Acceleration / Dataframes ---
try:
    import lancedb
    from lancedb.table import Table

    LANCEDB_AVAILABLE = True
except ImportError:
    lancedb = None
    Table = None
    LANCEDB_AVAILABLE = False

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

from app.domain.config import ScanConfig
from app.domain.data_models import get_fingerprint_fields_schema
from app.shared.constants import (
    DB_TABLE_NAME,
    DEFAULT_SEARCH_PRECISION,
    SEARCH_PRECISION_PRESETS,
)
from app.shared.signal_bus import APP_SIGNAL_BUS

logger = logging.getLogger("PixelHand.db_service")


class DatabaseService:
    """
    Manages access to the embedded LanceDB instance.
    Ensures thread safety for write operations.
    """

    def __init__(self, storage_path: Path):
        """
        Args:
            storage_path: Directory where database files will be stored.
                          Usually 'app_data/.cache'.
        """
        self.storage_path = storage_path
        self.db = None
        self.table: Table | None = None
        self.db_path: Path | None = None
        self.is_ready = False

        # Lock to prevent concurrent writes/reads which might corrupt embedded DB
        self._access_lock = threading.RLock()

        self.current_dim = 512

    def initialize(self, config: ScanConfig) -> bool:
        """
        Connects to (or creates) the LanceDB database for the current scan context.
        """
        with self._access_lock:
            if not LANCEDB_AVAILABLE:
                msg = "LanceDB library not found. AI search unavailable."
                logger.error(msg)
                APP_SIGNAL_BUS.scan_error.emit(msg)
                return False

            try:
                self.close()

                # Generate a unique DB name based on folder and model
                # This ensures we don't mix vectors from different models or folders
                model_name = config.ai.model_name
                sanitized_model = model_name.replace("/", "_").replace("-", "_")
                folder_hash = hashlib.md5(str(config.folder_path).encode()).hexdigest()

                db_name = f"lancedb_vectors_{folder_hash}_{sanitized_model}"
                self.db_path = self.storage_path / db_name
                self.current_dim = config.ai.model_dim

                # Ensure directory exists
                self.db_path.mkdir(parents=True, exist_ok=True)

                # Connect
                self.db = lancedb.connect(str(self.db_path))

                # Define Schema
                # Fixed schema for metadata + dynamic vector dimension
                schema_fields = [
                    pa.field("id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), self.current_dim)),
                ]
                # Append standard fingerprint fields (resolution, size, etc.)
                for name, types in get_fingerprint_fields_schema().items():
                    schema_fields.append(pa.field(name, types["pyarrow"]))

                schema = pa.schema(schema_fields)

                # Reset table if it exists (fresh scan assumption for simplicity in this logic)
                if DB_TABLE_NAME in self.db.table_names():
                    self.db.drop_table(DB_TABLE_NAME)

                self.table = self.db.create_table(DB_TABLE_NAME, schema=schema)
                self.is_ready = True

                logger.info(f"Database initialized at {self.db_path}")
                return True

            except Exception as e:
                logger.error(f"DB Init failed: {e}", exc_info=True)
                APP_SIGNAL_BUS.scan_error.emit(f"Database Error: {e}")
                self.is_ready = False
                return False

    def add_batch(self, data_dicts: list[dict[str, Any]]):
        """
        Inserts a batch of records into the database.
        """
        if not self.is_ready or not data_dicts:
            return

        # Ensure IDs are unique
        prepared_data = []
        for d in data_dicts:
            # Create deterministic ID based on path + channel
            unique_str = str(d["path"]) + (d.get("channel") or "")
            uid = str(uuid.uuid5(uuid.NAMESPACE_URL, unique_str))

            item = d.copy()
            item["id"] = uid
            prepared_data.append(item)

        with self._access_lock:
            try:
                # Use explicit schema if available to prevent type mismatch errors (List vs FixedSizeList).
                # This bypasses the need for LanceDB to guess/cast the type from a generic list.
                if self.table is not None:
                    arrow_table = pa.Table.from_pylist(prepared_data, schema=self.table.schema)
                else:
                    arrow_table = (
                        pl.DataFrame(prepared_data).to_arrow()
                        if POLARS_AVAILABLE
                        else pa.Table.from_pylist(prepared_data)
                    )

                self.table.add(data=arrow_table)
            except Exception as e:
                logger.error(f"Failed to add batch to DB: {e}")

    def create_index(self):
        """
        Creates an IVF-PQ index for fast vector search.
        Should be called after all data is inserted.
        """
        with self._access_lock:
            if not self.is_ready:
                return
            try:
                num_rows = self.table.to_lance().count_rows()
                if num_rows < 1000:
                    return  # Indexing is overhead for small datasets

                # Calculate optimal partitions
                partitions = min(2048, max(128, int(num_rows**0.5)))

                # Calculate sub-vectors for PQ (Product Quantization)
                # Ideally dim / 16, ensuring it divides evenly
                dim = self.current_dim
                if dim % 96 == 0:
                    sub_vectors = 96
                elif dim % 64 == 0:
                    sub_vectors = 64
                elif dim % 32 == 0:
                    sub_vectors = 32
                elif dim % 16 == 0:
                    sub_vectors = 16
                else:
                    sub_vectors = 8

                logger.info(f"Creating index: partitions={partitions}, sub_vectors={sub_vectors}")
                self.table.create_index(
                    metric="cosine",
                    num_partitions=partitions,
                    num_sub_vectors=sub_vectors,
                    replace=True,
                )
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")

    def search_vectors(self, query_vector: np.ndarray, limit: int, probes: int, refine: int) -> list[dict]:
        """
        Performs a vector search for a single query vector.
        """
        with self._access_lock:
            if not self.is_ready:
                return []
            try:
                res = (
                    self.table.search(query_vector)
                    .metric("cosine")
                    .limit(limit)
                    .nprobes(probes)
                    .refine_factor(refine)
                    .to_polars()
                )
                return res.to_dicts()
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return []

    def find_similar_pairs(
        self,
        config: ScanConfig,
        stop_event: threading.Event,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[str, str, str, str, float]]:
        """
        Performs an exhaustive (or indexed) search to find all similar pairs in the dataset.
        Returns: List of (path_a, channel_a, path_b, channel_b, distance)
        """
        with self._access_lock:
            if not self.is_ready:
                return []

            try:
                num_rows = self.table.to_lance().count_rows()
            except Exception:
                num_rows = 0

            if num_rows == 0:
                return []

            logger.info(f"Searching {num_rows} items for similarities...")

            found_pairs_set = set()
            found_links = []

            # Retrieve threshold
            threshold_val = getattr(config, "similarity_threshold", 95)
            dist_threshold = 1.0 - (threshold_val / 100.0)

            # Get search parameters from Performance Config
            precision_key = config.perf.search_precision
            precision_config = SEARCH_PRECISION_PRESETS.get(
                precision_key, SEARCH_PRECISION_PRESETS[DEFAULT_SEARCH_PRECISION]
            )
            nprobes = precision_config["nprobes"]
            refine_factor = precision_config["refine_factor"]
            k_neighbors = 50  # Look for top 50 matches per image

            try:
                processed_count = 0
                # Stream data to avoid loading all into RAM
                stream = self.table.to_lance().to_batches(columns=["id", "vector", "path", "channel"], batch_size=1024)

                for batch in stream:
                    if stop_event.is_set():
                        return []

                    batch_rows = batch.to_pylist()

                    for row in batch_rows:
                        if stop_event.is_set():
                            return []

                        query_vec = row["vector"]
                        query_id = row["id"]
                        query_path = row["path"]
                        query_channel = row["channel"] or "RGB"

                        # Search
                        arrow_results = (
                            self.table.search(query_vec)
                            .metric("cosine")
                            .limit(k_neighbors)
                            .nprobes(nprobes)
                            .refine_factor(refine_factor)
                            .to_arrow()
                        )

                        r_dists = arrow_results["_distance"].to_numpy()
                        r_ids = arrow_results["id"].to_pylist()
                        r_paths = arrow_results["path"].to_pylist()
                        r_channels = arrow_results["channel"].to_pylist()

                        for i in range(len(arrow_results)):
                            match_dist = r_dists[i]
                            if match_dist > dist_threshold:
                                continue

                            match_id = r_ids[i]
                            if query_id == match_id:
                                continue  # Skip self-match

                            # Deduplicate pairs (A-B is same as B-A)
                            pair_sig = tuple(sorted((query_id, match_id)))
                            if pair_sig in found_pairs_set:
                                continue

                            found_pairs_set.add(pair_sig)

                            match_path = r_paths[i]
                            match_channel = r_channels[i] or "RGB"

                            found_links.append(
                                (
                                    query_path,
                                    query_channel,
                                    match_path,
                                    match_channel,
                                    match_dist,
                                )
                            )

                        processed_count += 1

                    if progress_callback:
                        progress_callback(processed_count, num_rows)

                    if processed_count % 5000 == 0:
                        gc.collect()

                return found_links

            except Exception as e:
                logger.error(f"IVF-PQ search failed: {e}", exc_info=True)
                return []

    def get_files_by_group(self, group_id: int) -> list[dict]:
        """Retrieves all files belonging to a specific group ID from the results table."""
        with self._access_lock:
            if not self.db:
                return []
            try:
                if "scan_results" not in self.db.table_names():
                    return []
                tbl = self.db.open_table("scan_results")
                return tbl.search().where(f"group_id = {group_id}").limit(None).to_list()
            except Exception as e:
                logger.error(f"Fetch group failed: {e}")
                return []

    def save_results_table(self, data: list[dict]):
        """Saves the final scan results to a separate table for UI access."""
        with self._access_lock:
            if not self.db:
                return
            try:
                tbl_name = "scan_results"
                if tbl_name in self.db.table_names():
                    self.db.drop_table(tbl_name)

                pa_table = pl.DataFrame(data).to_arrow() if POLARS_AVAILABLE else pa.Table.from_pylist(data)

                self.db.create_table(tbl_name, data=pa_table)
            except Exception as e:
                logger.error(f"Failed to save results table: {e}")

    def delete_paths(self, paths: list[str]):
        """Removes records with the specified paths from the results table."""
        with self._access_lock:
            if not self.db:
                return
            try:
                if "scan_results" in self.db.table_names():
                    tbl = self.db.open_table("scan_results")
                    # Escape paths for SQL
                    path_list_sql = ", ".join(f"'{p.replace("'", "''")}'" for p in paths)
                    if path_list_sql:
                        tbl.delete(f"path IN ({path_list_sql})")
            except Exception as e:
                logger.error(f"Delete paths failed: {e}")

    def close(self):
        """Closes connections and cleans up resources."""
        with self._access_lock:
            self.db = None
            self.table = None
            self.is_ready = False
            gc.collect()
