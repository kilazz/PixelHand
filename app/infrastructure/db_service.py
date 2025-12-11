# app/infrastructure/db_service.py
import gc
import hashlib
import logging
import shutil
import threading
import uuid

import numpy as np
import pyarrow as pa

from app.domain.data_models import ScanConfig, get_fingerprint_fields_schema
from app.shared.constants import CACHE_DIR, DB_TABLE_NAME, LANCEDB_AVAILABLE
from app.shared.signal_bus import APP_SIGNAL_BUS

if LANCEDB_AVAILABLE:
    import lancedb

# Optional Polars integration for faster data handling
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
                    # FIX UP008: Use super() without arguments
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
        # Re-entrant lock to allow recursive calls within the same thread if necessary,
        # but block access from other threads.
        self._access_lock = threading.RLock()
        self._initialized = True

    def initialize(self, config: ScanConfig) -> bool:
        """
        Initializes the database connection and creates the table schema.
        Should be called at the start of a scan.
        """
        with self._access_lock:
            if not LANCEDB_AVAILABLE:
                APP_SIGNAL_BUS.scan_error.emit("LanceDB library not found.")
                return False

            try:
                # Close any existing connection
                self.close()

                # Generate a unique DB path based on the folder being scanned and the model used.
                # This prevents mixing caches from different scan sessions.
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

                # Define the Schema
                # 1. Vector field
                schema_fields = [
                    pa.field("id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), config.model_dim)),
                ]
                # 2. Metadata fields (resolution, file size, etc.)
                for name, types in get_fingerprint_fields_schema().items():
                    schema_fields.append(pa.field(name, types["pyarrow"]))

                schema = pa.schema(schema_fields)

                # Always start fresh for a new scan to ensure consistency.
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
        """
        Thread-safe insertion of a batch of records.
        """
        if not self.is_ready or not data_dicts:
            return

        # Data Preparation: Generate deterministic IDs
        prepared_data = []
        for d in data_dicts:
            # Create a UUID based on the file path and channel to identify unique vectors
            uid = str(uuid.uuid5(uuid.NAMESPACE_URL, d["path"] + (d["channel"] or "")))
            item = d.copy()
            item["id"] = uid
            prepared_data.append(item)

        with self._access_lock:
            try:
                # Convert to Arrow Table (Optimized with Ternary Operator)
                arrow_table = (
                    pl.DataFrame(prepared_data).to_arrow() if POLARS_AVAILABLE else pa.Table.from_pylist(prepared_data)
                )

                self.table.add(data=arrow_table)
            except Exception as e:
                app_logger.error(f"Failed to add batch to DB: {e}")

    def create_index(self):
        """
        Creates an IVF-PQ index for fast similarity search.
        Automatically scales partitions based on dataset size.
        """
        with self._access_lock:
            if not self.is_ready:
                return
            try:
                num_rows = self.table.to_lance().count_rows()
                if num_rows < 1000:
                    return  # Indexing is not necessary for small datasets

                # Heuristic for partitions
                partitions = min(2048, max(128, int(num_rows**0.5)))

                self.table.create_index(metric="cosine", num_partitions=partitions, num_sub_vectors=96, replace=True)
            except Exception as e:
                app_logger.warning(f"Index creation warning: {e}")

    def search_vectors(self, query_vector: np.ndarray, limit: int, probes: int, refine: int) -> list[dict]:
        """
        Performs an Approximate Nearest Neighbor (ANN) search using the vector index.
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
                app_logger.error(f"Vector search failed: {e}")
                return []

    def get_files_by_group(self, group_id: int) -> list[dict]:
        """
        Retrieves file metadata for a specific group ID from the results table.
        Used by the UI for lazy loading of group children.
        """
        with self._access_lock:
            if not self.db:
                return []
            try:
                # 'scan_results' is the secondary table where clustered results are stored.
                if "scan_results" not in self.db.table_names():
                    return []

                tbl = self.db.open_table("scan_results")
                return tbl.search().where(f"group_id = {group_id}").limit(None).to_list()
            except Exception as e:
                app_logger.error(f"Fetch group failed: {e}")
                return []

    def save_results_table(self, data: list[dict]):
        """
        Persists the final clustering results (groups) into a separate table in LanceDB.
        This allows the UI to query results later without holding everything in RAM.
        """
        with self._access_lock:
            if not self.db:
                return
            try:
                tbl_name = "scan_results"
                if tbl_name in self.db.table_names():
                    self.db.drop_table(tbl_name)

                # FIX SIM108: Use ternary operator
                pa_table = pl.DataFrame(data).to_arrow() if POLARS_AVAILABLE else pa.Table.from_pylist(data)

                self.db.create_table(tbl_name, data=pa_table)
            except Exception as e:
                app_logger.error(f"Failed to save results table: {e}")

    def delete_paths(self, paths: list[str]):
        """
        Removes records matching the given paths from the results table.
        Used when the user deletes files or moves them to trash.
        """
        with self._access_lock:
            if not self.db:
                return
            try:
                if "scan_results" in self.db.table_names():
                    tbl = self.db.open_table("scan_results")
                    # Escape paths to prevent SQL syntax errors in the filter string
                    path_list_sql = ", ".join(f"'{p.replace("'", "''")}'" for p in paths)
                    if path_list_sql:
                        tbl.delete(f"path IN ({path_list_sql})")
            except Exception as e:
                app_logger.error(f"Delete paths failed: {e}")

    def close(self):
        """
        Closes the connection and resets the state.
        """
        with self._access_lock:
            self.db = None
            self.table = None
            self.is_ready = False
            gc.collect()


# Global Singleton Instance
DB_SERVICE = DatabaseService()
