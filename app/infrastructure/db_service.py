# app/infrastructure/db_service.py
import gc
import hashlib
import logging
import shutil
import threading
import uuid
from collections.abc import Callable

import numpy as np
import pyarrow as pa

from app.domain.data_models import ScanConfig, get_fingerprint_fields_schema
from app.shared.constants import (
    CACHE_DIR,
    DB_TABLE_NAME,
    DEFAULT_SEARCH_PRECISION,
    LANCEDB_AVAILABLE,
    SEARCH_PRECISION_PRESETS,
)
from app.shared.signal_bus import APP_SIGNAL_BUS

if LANCEDB_AVAILABLE:
    import lancedb

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

app_logger = logging.getLogger("PixelHand.db_service")


class DatabaseService:
    """
    Singleton service for managing access to LanceDB.
    Ensures thread safety and centralized connection management.
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.db = None
        self.table = None
        self.db_path = None
        self.is_ready = False
        self._access_lock = threading.RLock()
        self._initialized = True

    def initialize(self, config: ScanConfig) -> bool:
        with self._access_lock:
            if not LANCEDB_AVAILABLE:
                APP_SIGNAL_BUS.scan_error.emit("LanceDB library not found.")
                return False

            try:
                self.close()

                sanitized_model = config.model_name.replace("/", "_").replace("-", "_")
                folder_hash = hashlib.md5(str(config.folder_path).encode()).hexdigest()
                db_name = f"lancedb_vectors_{folder_hash}_{sanitized_model}"
                self.db_path = CACHE_DIR / db_name

                if config.lancedb_in_memory:
                    APP_SIGNAL_BUS.log_message.emit("DB Mode: In-Memory (Volatile).", "info")
                    if self.db_path.exists():
                        shutil.rmtree(self.db_path)

                self.db_path.mkdir(parents=True, exist_ok=True)
                self.db = lancedb.connect(str(self.db_path))

                schema_fields = [
                    pa.field("id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), config.model_dim)),
                ]
                for name, types in get_fingerprint_fields_schema().items():
                    schema_fields.append(pa.field(name, types["pyarrow"]))

                schema = pa.schema(schema_fields)

                if DB_TABLE_NAME in self.db.table_names():
                    self.db.drop_table(DB_TABLE_NAME)

                self.table = self.db.create_table(DB_TABLE_NAME, schema=schema)
                self.is_ready = True
                return True

            except Exception as e:
                app_logger.error(f"DB Init failed: {e}", exc_info=True)
                APP_SIGNAL_BUS.scan_error.emit(f"Database Error: {e}")
                self.is_ready = False
                return False

    def add_batch(self, data_dicts: list[dict]):
        if not self.is_ready or not data_dicts:
            return

        prepared_data = []
        for d in data_dicts:
            uid = str(uuid.uuid5(uuid.NAMESPACE_URL, d["path"] + (d["channel"] or "")))
            item = d.copy()
            item["id"] = uid
            prepared_data.append(item)

        with self._access_lock:
            try:
                arrow_table = (
                    pl.DataFrame(prepared_data).to_arrow() if POLARS_AVAILABLE else pa.Table.from_pylist(prepared_data)
                )
                self.table.add(data=arrow_table)
            except Exception as e:
                app_logger.error(f"Failed to add batch to DB: {e}")

    def create_index(self):
        with self._access_lock:
            if not self.is_ready:
                return
            try:
                num_rows = self.table.to_lance().count_rows()
                if num_rows < 1000:
                    return

                partitions = min(2048, max(128, int(num_rows**0.5)))
                self.table.create_index(metric="cosine", num_partitions=partitions, num_sub_vectors=96, replace=True)
            except Exception as e:
                app_logger.warning(f"Index creation warning: {e}")

    def search_vectors(self, query_vector: np.ndarray, limit: int, probes: int, refine: int) -> list[dict]:
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
                app_logger.error(f"Vector search failed: {e}")
                return []

    def find_similar_pairs(
        self,
        config: ScanConfig,
        stop_event: threading.Event,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[str, str, str, str, float]]:
        """
        Performs an Approximate Nearest Neighbor (ANN) search for every item in the DB.
        Replaces the old LanceDBSimilarityEngine.
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

            APP_SIGNAL_BUS.log_message.emit(f"Searching {num_rows} items using LanceDB Index...", "info")

            found_pairs_set = set()
            found_links = []
            dist_threshold = 1.0 - (config.similarity_threshold / 100.0)

            precision_config = SEARCH_PRECISION_PRESETS.get(
                config.search_precision, SEARCH_PRECISION_PRESETS[DEFAULT_SEARCH_PRECISION]
            )
            nprobes = precision_config["nprobes"]
            refine_factor = precision_config["refine_factor"]
            k_neighbors = 50

            try:
                processed_count = 0
                # Using to_lance().to_batches() to stream data without loading everything into RAM
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

                        # Perform search for this vector
                        # Note: We call search on self.table directly to keep internal lock context
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
                                continue

                            # Deduplicate pairs (A-B is same as B-A)
                            pair_sig = tuple(sorted((query_id, match_id)))
                            if pair_sig in found_pairs_set:
                                continue

                            found_pairs_set.add(pair_sig)

                            match_path = r_paths[i]
                            match_channel = r_channels[i] or "RGB"
                            found_links.append((query_path, query_channel, match_path, match_channel, match_dist))

                        processed_count += 1

                    if progress_callback:
                        progress_callback(processed_count, num_rows)

                    if processed_count % 5000 == 0:
                        gc.collect()

                APP_SIGNAL_BUS.log_message.emit(
                    f"Linking complete. Found {len(found_links)} pairs via Index Search.", "success"
                )
                return found_links

            except Exception as e:
                app_logger.error(f"IVF-PQ search failed: {e}", exc_info=True)
                APP_SIGNAL_BUS.log_message.emit(f"Search failed: {e}", "error")
                return []

    def get_files_by_group(self, group_id: int) -> list[dict]:
        with self._access_lock:
            if not self.db:
                return []
            try:
                if "scan_results" not in self.db.table_names():
                    return []
                tbl = self.db.open_table("scan_results")
                return tbl.search().where(f"group_id = {group_id}").limit(None).to_list()
            except Exception as e:
                app_logger.error(f"Fetch group failed: {e}")
                return []

    def save_results_table(self, data: list[dict]):
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
                app_logger.error(f"Failed to save results table: {e}")

    def delete_paths(self, paths: list[str]):
        with self._access_lock:
            if not self.db:
                return
            try:
                if "scan_results" in self.db.table_names():
                    tbl = self.db.open_table("scan_results")
                    path_list_sql = ", ".join(f"'{p.replace("'", "''")}'" for p in paths)
                    if path_list_sql:
                        tbl.delete(f"path IN ({path_list_sql})")
            except Exception as e:
                app_logger.error(f"Delete paths failed: {e}")

    def close(self):
        with self._access_lock:
            self.db = None
            self.table = None
            self.is_ready = False
            gc.collect()


DB_SERVICE = DatabaseService()
