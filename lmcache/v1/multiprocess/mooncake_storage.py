# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional
import threading

# Third Party
from lmcache.v1.memory_management import MemoryObj

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import StorageKey
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConfig,
)

logger = init_logger(__name__)


class MooncakeMPStorage:
    """Mooncake L2 storage used by the LMCache multiprocess server."""

    def __init__(
        self,
        config_path: str,
        allocator,
        replica_num: Optional[int] = None,
        nof_replica_num: Optional[int] = None,
        preferred_nof_segments: Optional[list[str]] = None,
        async_put: bool = True,
        max_workers: int = 2,
    ) -> None:
        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except ImportError as exc:
            raise ImportError(
                "Mooncake is required for LMCache MP remote storage."
            ) from exc

        config_file = Path(config_path)
        if not config_file.is_file():
            raise FileNotFoundError(
                f"Mooncake configuration file does not exist: {config_path}"
            )

        self.config = MooncakeStoreConfig.from_file(str(config_file))
        if replica_num is not None:
            self.config.replica_num = replica_num
        if nof_replica_num is not None:
            self.config.nof_replica_num = nof_replica_num
        if preferred_nof_segments is not None:
            self.config.preferred_nof_segments = preferred_nof_segments

        self.store = MooncakeDistributedStore()
        self.store.setup(
            self.config.local_hostname,
            self.config.metadata_server,
            self.config.global_segment_size,
            self.config.local_buffer_size,
            self.config.protocol,
            self.config.device_name,
            self.config.master_server_address,
        )

        buffer = self._get_allocator_buffer(allocator)
        if buffer is None or buffer.numel() == 0:
            raise RuntimeError("MP allocator has no host buffer to register")

        self.buffer = buffer
        self.registered_buffer_ptr = buffer.data_ptr()
        result = self.store.register_buffer(
            self.registered_buffer_ptr, buffer.numel()
        )
        if result != 0:
            raise RuntimeError(f"Mooncake register_buffer failed with code {result}")

        self.replica_config = ReplicateConfig()
        self.replica_config.replica_num = self.config.replica_num
        self.replica_config.nof_replica_num = self.config.nof_replica_num
        if self.config.prefer_local_alloc:
            self.replica_config.preferred_segment = self.store.get_hostname()
        if self.config.preferred_nof_segments:
            self.replica_config.preferred_nof_segments = (
                self.config.preferred_nof_segments
            )

        self.async_put = async_put
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="lmcache-mp-mooncake",
        )
        self.pending: set[Future] = set()
        self.pending_lock = threading.Lock()
        self.closed = False

        logger.info(
            "LMCache MP Mooncake L2 initialized: config=%s, replica_num=%d, "
            "nof_replica_num=%d, async_put=%s",
            config_path,
            self.replica_config.replica_num,
            self.replica_config.nof_replica_num,
            self.async_put,
        )

    @staticmethod
    def _get_allocator_buffer(allocator):
        if hasattr(allocator, "buffer"):
            return allocator.buffer
        if hasattr(allocator, "get_underlying_buffer"):
            return allocator.get_underlying_buffer()
        if hasattr(allocator, "pin_allocator"):
            pin_allocator = allocator.pin_allocator
            if hasattr(pin_allocator, "buffer"):
                return pin_allocator.buffer
        return None

    @staticmethod
    def key_to_string(key: StorageKey) -> str:
        return (
            f"{key.model_name}@{key.world_size}@{key.worker_id}@"
            f"{key.chunk_hash.hex()}"
        )

    @staticmethod
    def _object_ptr_and_size(memory_obj: MemoryObj) -> tuple[int, int]:
        tensor = memory_obj.tensor
        if tensor is None:
            raise RuntimeError("Mooncake MP storage requires a tensor MemoryObj")
        return tensor.data_ptr(), tensor.numel() * tensor.element_size()

    def contains(self, key: StorageKey) -> bool:
        return bool(self.store.is_exist(self.key_to_string(key)))

    def _put_batch(self, keys: list[StorageKey], memory_objs: list[MemoryObj]):
        key_strings = [self.key_to_string(key) for key in keys]
        ptrs: list[int] = []
        sizes: list[int] = []
        for memory_obj in memory_objs:
            ptr, size = self._object_ptr_and_size(memory_obj)
            ptrs.append(ptr)
            sizes.append(size)

        results = self.store.batch_put_from(
            key_strings,
            ptrs,
            sizes,
            self.replica_config,
        )
        if isinstance(results, int):
            results = [results]
        if results is not None and any(result < 0 for result in results):
            raise RuntimeError(f"Mooncake batch_put_from failed: {results}")
        return results

    def submit_put(
        self,
        keys: list[StorageKey],
        memory_objs: list[MemoryObj],
        on_complete: Optional[Callable[[], None]] = None,
    ) -> Optional[Future]:
        if not keys:
            return None
        if len(keys) != len(memory_objs):
            raise ValueError("Mooncake put keys and objects have different lengths")
        if self.closed:
            raise RuntimeError("Mooncake MP storage is already closed")

        for memory_obj in memory_objs:
            memory_obj.ref_count_up()

        def put_task():
            try:
                return self._put_batch(keys, memory_objs)
            finally:
                for memory_obj in memory_objs:
                    memory_obj.ref_count_down()
                if on_complete is not None:
                    on_complete()

        if not self.async_put:
            put_task()
            return None

        future = self.executor.submit(put_task)
        with self.pending_lock:
            self.pending.add(future)

        def done_callback(done: Future):
            with self.pending_lock:
                self.pending.discard(done)
            exception = done.exception()
            if exception is not None:
                logger.error(
                    "LMCache MP Mooncake put failed for %d keys: %s",
                    len(keys),
                    exception,
                )

        future.add_done_callback(done_callback)
        return future

    def get_into(self, key: StorageKey, memory_obj: MemoryObj) -> int:
        ptr, size = self._object_ptr_and_size(memory_obj)
        result = self.store.get_into(self.key_to_string(key), ptr, size)
        if result <= 0:
            raise RuntimeError(
                f"Mooncake get_into failed for {self.key_to_string(key)}: {result}"
            )
        return result

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.executor.shutdown(wait=True)
        result = self.store.unregister_buffer(self.registered_buffer_ptr)
        if result != 0:
            logger.warning("Mooncake unregister_buffer failed with code %s", result)
        self.store.close()
        logger.info("Closed LMCache MP Mooncake L2 storage")
