# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)


class FakeReplicateConfig:
    def __init__(self):
        self.replica_num = 0
        self.nof_replica_num = 0
        self.preferred_segment = ""


class FakeMooncakeDistributedStore:
    def setup(self, *args, **kwargs):
        return 0

    def get_hostname(self):
        return "fake-host"

    def register_buffer(self, ptr, size):
        return 0

    def unregister_buffer(self, ptr):
        return 0

    def close(self):
        return 0


@pytest.fixture
def fake_mooncake_store_module(monkeypatch):
    mooncake_pkg = ModuleType("mooncake")
    mooncake_store_mod = ModuleType("mooncake.store")
    mooncake_store_mod.MooncakeDistributedStore = FakeMooncakeDistributedStore
    mooncake_store_mod.ReplicateConfig = FakeReplicateConfig
    mooncake_store_mod.bind_to_numa_node = lambda numa_id: None

    monkeypatch.setitem(sys.modules, "mooncake", mooncake_pkg)
    monkeypatch.setitem(sys.modules, "mooncake.store", mooncake_store_mod)


def make_local_cpu_backend(enable_mooncake_nof_pool: bool):
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        enable_mooncake_nof_pool=enable_mooncake_nof_pool,
        extra_config={
            "local_hostname": "127.0.0.1:50052",
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "master_server_address": "127.0.0.1:50051",
            "global_segment_size": 1024 * 1024,
            "local_buffer_size": 1024 * 1024,
            "protocol": "tcp",
            "device_name": "",
            "storage_root_dir": "",
        },
    )

    buffer = SimpleNamespace(data_ptr=lambda: 0x1234, numel=lambda: 4096)
    allocator = SimpleNamespace(
        pin_allocator=SimpleNamespace(buffer=buffer),
        numa_mapping=None,
    )

    return SimpleNamespace(
        config=config,
        metadata=MagicMock(),
        memory_allocator=allocator,
        enable_mooncake_nof_pool=enable_mooncake_nof_pool,
    )


class TestMooncakeStoreConnector:
    def test_disable_mooncake_nof_pool_keeps_nof_replica_zero(
        self, fake_mooncake_store_module
    ):
        loop = asyncio.new_event_loop()
        connector = MooncakestoreConnector(
            host="",
            port=0,
            dev_name="",
            loop=loop,
            local_cpu_backend=make_local_cpu_backend(False),
            lmcache_config=None,
        )

        assert connector.replica_config.replica_num == 1
        assert connector.replica_config.nof_replica_num == 0

        loop.run_until_complete(connector.close())
        loop.close()

    def test_enable_mooncake_nof_pool_sets_nof_replica_one(
        self, fake_mooncake_store_module
    ):
        loop = asyncio.new_event_loop()
        connector = MooncakestoreConnector(
            host="",
            port=0,
            dev_name="",
            loop=loop,
            local_cpu_backend=make_local_cpu_backend(True),
            lmcache_config=None,
        )

        assert connector.replica_config.replica_num == 1
        assert connector.replica_config.nof_replica_num == 1

        loop.run_until_complete(connector.close())
        loop.close()
