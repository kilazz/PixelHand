# app/infrastructure/database.py
"""
Handles initialization and connection management for LanceDB.
"""

import hashlib
import logging
import shutil

import pyarrow as pa

from app.domain.data_models import ScanConfig, get_fingerprint_fields_schema
from app.shared.constants import CACHE_DIR, DB_TABLE_NAME, LANCEDB_AVAILABLE
from app.shared.signal_bus import APP_SIGNAL_BUS

if LANCEDB_AVAILABLE:
    import lancedb

app_logger = logging.getLogger("PixelHand.infrastructure.database")


class LanceDBContext:
    """Context manager and helper for LanceDB connections."""

    def __init__(self, config: ScanConfig):
        self.config = config
        self.db = None
        self.table = None
        self.db_path = None

    def initialize(self) -> bool:
        """Sets up the LanceDB connection and table schema."""
        if not LANCEDB_AVAILABLE:
            APP_SIGNAL_BUS.scan_error.emit("LanceDB library not found. Please install lancedb.")
            return False

        try:
            folder_hash = hashlib.md5(str(self.config.folder_path).encode()).hexdigest()
            sanitized_model = self.config.model_name.replace("/", "_").replace("-", "_")
            db_name = f"lancedb_vectors_{folder_hash}_{sanitized_model}"
            self.db_path = CACHE_DIR / db_name

            if self.config.lancedb_in_memory:
                APP_SIGNAL_BUS.log_message.emit("Vector storage: LanceDB in-memory (fastest, temporary).", "info")
                if self.db_path.exists():
                    shutil.rmtree(self.db_path)
            else:
                APP_SIGNAL_BUS.log_message.emit(
                    f"Vector storage: LanceDB on-disk (scalable, persistent at {self.db_path.name}).", "info"
                )

            self.db_path.mkdir(parents=True, exist_ok=True)
            self.db = lancedb.connect(str(self.db_path))

            # Define Schema
            schema_fields = [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.config.model_dim)),
            ]

            fingerprint_fields = get_fingerprint_fields_schema()
            for name, types in fingerprint_fields.items():
                schema_fields.append(pa.field(name, types["pyarrow"]))

            schema = pa.schema(schema_fields)

            if DB_TABLE_NAME in self.db.table_names():
                self.db.drop_table(DB_TABLE_NAME)  # Always start fresh for consistency

            self.table = self.db.create_table(DB_TABLE_NAME, schema=schema)
            return True

        except Exception as e:
            app_logger.error(f"Failed to initialize LanceDB: {e}", exc_info=True)
            APP_SIGNAL_BUS.scan_error.emit(f"LanceDB setup error: {e}")
            return False

    def close(self):
        """Cleanup resources."""
        self.db = None
        self.table = None
