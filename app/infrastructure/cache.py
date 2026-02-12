# app/infrastructure/cache.py
"""
Manages the Thumbnail Cache (LanceDB or Dummy).
"""

import abc
import hashlib
import logging

# --- Refactoring: Updated Imports ---
from app.domain.config import ScanConfig
from app.shared.constants import CACHE_DIR, LANCEDB_AVAILABLE

logger = logging.getLogger("PixelHand.cache")


# --- Thumbnail Cache System ---


class AbstractThumbnailCache(abc.ABC):
    @abc.abstractmethod
    def get(self, key: str) -> bytes | None:
        pass

    @abc.abstractmethod
    def put(self, key: str, data: bytes):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class LanceDBThumbnailCache(AbstractThumbnailCache):
    def __init__(self, in_memory: bool, db_name: str = "thumbnails"):
        self.db = None
        self.table = None

        if not LANCEDB_AVAILABLE:
            return

        try:
            import lancedb
            import pyarrow as pa

            db_path = CACHE_DIR / db_name

            # Note: For now, we ignore in_memory request to ensure persistence
            # and stability across large datasets.
            # if in_memory and db_path.exists():
            #     shutil.rmtree(db_path)

            db_path.mkdir(parents=True, exist_ok=True)
            self.db = lancedb.connect(str(db_path))

            if db_name in self.db.table_names():
                self.table = self.db.open_table(db_name)
            else:
                schema = pa.schema(
                    [
                        pa.field("key", pa.string()),
                        pa.field("data", pa.binary()),
                    ]
                )
                self.table = self.db.create_table(db_name, schema=schema)

        except Exception as e:
            logger.error(f"Failed to initialize thumbnail cache: {e}")

    def get(self, key: str) -> bytes | None:
        if not self.table:
            return None
        try:
            result = self.table.query().where(f"key = '{key}'").limit(1).to_list()
            return result[0]["data"] if result else None
        except Exception:
            return None

    def put(self, key: str, data: bytes):
        if not self.table:
            return
        try:
            import pyarrow as pa

            self.table.delete(f"key = '{key}'")
            record = pa.Table.from_pylist([{"key": key, "data": data}])
            self.table.add(record)
        except Exception:
            pass

    def close(self):
        self.table = None
        self.db = None


class DummyThumbnailCache(AbstractThumbnailCache):
    def get(self, key: str) -> bytes | None:
        return None

    def put(self, key: str, data: bytes):
        pass

    def close(self):
        pass


# Global thumbnail cache instance
thumbnail_cache: AbstractThumbnailCache = DummyThumbnailCache()


def get_thumbnail_cache_key(
    path_str: str,
    mtime: float,
    target_size: int | None,
    tonemap_mode: str,
    channel_to_load: str | None,
) -> str:
    # Use 0 if target_size is None to keep key valid string
    size_str = str(target_size) if target_size is not None else "full"
    key_str = f"{path_str}|{mtime}|{size_str}|{tonemap_mode}|{channel_to_load or 'full'}"
    return hashlib.sha1(key_str.encode()).hexdigest()


def setup_caches(config: ScanConfig):
    global thumbnail_cache
    thumbnail_cache.close()

    thumbnail_cache = LanceDBThumbnailCache(in_memory=False) if LANCEDB_AVAILABLE else DummyThumbnailCache()


def teardown_caches():
    global thumbnail_cache
    thumbnail_cache.close()
    thumbnail_cache = DummyThumbnailCache()
