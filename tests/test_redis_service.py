import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from src.services import redis_service as redis_module
from src.services.redis_service import CACHE_SCHEMA_VERSION, RedisService, normalise


# ---------------------------------------------------------------------- #
# Helpers and fixtures                                                    #
# ---------------------------------------------------------------------- #

KEYWORDS = {"keywords": [{"keyword": "data mining", "priority": 1}]}
TTL_SECONDS = 24 * 60 * 60


def make_settings(**overrides):
    values = {
        "REDIS_ENABLED": True,
        "REDIS_HOST": "localhost",
        "REDIS_PORT": 6379,
        "REDIS_DB": 0,
        "REDIS_PASSWORD": "",
        "REDIS_SSL": False,
        "REDIS_KEY_PREFIX": "test",
        "REDIS_CACHE_TTL_DAYS": 1,
        "REDIS_QUERY_COUNTER_ENABLED": True,
        "redis_cache_ttl_seconds": TTL_SECONDS,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def client():
    """Stands in for the redis.asyncio client: commands are awaited, the
    pipeline queues them synchronously and sends them on execute()."""
    pipe = MagicMock()
    pipe.execute = AsyncMock()
    fake = MagicMock()
    fake.pipeline = Mock(return_value=pipe)
    fake.hset = AsyncMock()
    fake.expire = AsyncMock()
    fake.ping = AsyncMock()
    fake.aclose = AsyncMock()
    fake.pipe = pipe
    return fake


@pytest.fixture
def service(client):
    """A service that has connected, with the fake client in place of Redis."""
    svc = RedisService()
    svc._enabled = True
    svc._client = client
    return svc


# ---------------------------------------------------------------------- #
# normalise and build_key                                                 #
# ---------------------------------------------------------------------- #

class TestNormalise:
    def test_lowercases_trims_and_collapses_whitespace(self):
        assert normalise("  Data \t Mining\n ") == "data mining"


class TestBuildKey:
    def test_key_shape(self):
        key = RedisService().build_key("data mining", False, make_settings())

        digest = hashlib.sha256(b"data mining").hexdigest()
        assert key == f"test:{CACHE_SCHEMA_VERSION}:nosyn:{digest}"

    def test_same_query_typed_differently_shares_a_key(self):
        svc, settings = RedisService(), make_settings()
        assert svc.build_key("  Data   MINING ", False, settings) == svc.build_key("data mining", False, settings)

    def test_synonyms_get_their_own_key(self):
        svc, settings = RedisService(), make_settings()
        assert svc.build_key("ai", True, settings) != svc.build_key("ai", False, settings)
        assert ":syn:" in svc.build_key("ai", True, settings)


# ---------------------------------------------------------------------- #
# Connection lifecycle                                                    #
# ---------------------------------------------------------------------- #

class TestInitCache:
    @pytest.fixture
    def redis_lib(self, monkeypatch, client):
        pool = Mock(name="pool")
        connection_pool = Mock(return_value=pool)
        redis_cls = Mock(return_value=client)
        monkeypatch.setattr(redis_module.redis, "ConnectionPool", connection_pool)
        monkeypatch.setattr(redis_module.redis, "Redis", redis_cls)
        return SimpleNamespace(ConnectionPool=connection_pool, Redis=redis_cls, pool=pool)

    async def test_disabled_never_connects(self, redis_lib):
        svc = RedisService()

        await svc.init_cache(make_settings(REDIS_ENABLED=False))

        redis_lib.ConnectionPool.assert_not_called()
        assert svc._available is False

    async def test_connects_and_pings(self, redis_lib, client):
        svc = RedisService()

        await svc.init_cache(make_settings())

        kwargs = redis_lib.ConnectionPool.call_args.kwargs
        assert (kwargs["host"], kwargs["port"], kwargs["db"]) == ("localhost", 6379, 0)
        assert kwargs["password"] is None
        assert kwargs["connection_class"] is redis_module.redis.Connection
        assert kwargs["decode_responses"] is True
        redis_lib.Redis.assert_called_once_with(connection_pool=redis_lib.pool)
        client.ping.assert_awaited_once()
        assert svc._available is True

    async def test_ssl_and_password(self, redis_lib):
        await RedisService().init_cache(make_settings(REDIS_SSL=True, REDIS_PASSWORD="secret"))

        kwargs = redis_lib.ConnectionPool.call_args.kwargs
        assert kwargs["connection_class"] is redis_module.redis.SSLConnection
        assert kwargs["password"] == "secret"

    async def test_unreachable_at_startup_keeps_the_client(self, redis_lib, client):
        client.ping.side_effect = ConnectionError("refused")
        svc = RedisService()

        await svc.init_cache(make_settings())  # must not raise

        # Kept, so that later requests reconnect once Redis is back.
        assert svc._client is client
        assert svc._available is True


class TestCloseCache:
    async def test_closes_the_pool_and_forgets_the_client(self, service, client):
        await service.close_cache()

        client.aclose.assert_awaited_once_with(close_connection_pool=True)
        assert service._client is None

    async def test_close_failure_is_swallowed(self, service, client):
        client.aclose.side_effect = ConnectionError("gone")

        await service.close_cache()  # must not raise

        assert service._client is None

    async def test_nothing_to_close(self):
        svc = RedisService()
        await svc.close_cache()
        assert svc._client is None


# ---------------------------------------------------------------------- #
# lookup                                                                  #
# ---------------------------------------------------------------------- #

class TestLookup:
    async def test_unavailable_falls_back_to_the_model(self):
        assert await RedisService().lookup("key", "q", make_settings()) == (None, None)

    async def test_hit_counts_the_request_and_returns_the_answer(self, service, client):
        client.pipe.execute.return_value = [5, 0, json.dumps(KEYWORDS)]

        result = await service.lookup("key", "  Data Mining ", make_settings())

        assert result == (KEYWORDS, 5)
        client.pipeline.assert_called_once_with(transaction=False)
        client.pipe.hincrby.assert_called_once_with("key", "count", 1)
        client.pipe.hsetnx.assert_called_once_with("key", "query", "data mining")
        client.pipe.hget.assert_called_once_with("key", "data")
        client.expire.assert_not_awaited()

    async def test_first_request_gives_the_record_its_lifespan(self, service, client):
        client.pipe.execute.return_value = [1, 1, None]

        assert await service.lookup("key", "q", make_settings()) == (None, 1)
        client.expire.assert_awaited_once_with("key", TTL_SECONDS)

    async def test_expiry_failure_is_swallowed(self, service, client):
        client.pipe.execute.return_value = [1, 1, None]
        client.expire.side_effect = ConnectionError("gone")

        assert await service.lookup("key", "q", make_settings()) == (None, 1)

    async def test_miss_on_a_known_query(self, service, client):
        client.pipe.execute.return_value = [4, 0, None]

        assert await service.lookup("key", "q", make_settings()) == (None, 4)
        client.expire.assert_not_awaited()

    async def test_without_counting_only_reads(self, service, client):
        client.pipe.execute.return_value = [json.dumps(KEYWORDS)]

        result = await service.lookup("key", "q", make_settings(REDIS_QUERY_COUNTER_ENABLED=False))

        assert result == (KEYWORDS, None)
        client.pipe.hincrby.assert_not_called()
        client.pipe.hsetnx.assert_not_called()
        client.expire.assert_not_awaited()

    async def test_redis_failure_falls_back_to_the_model(self, service, client):
        client.pipe.execute.side_effect = ConnectionError("timeout")

        assert await service.lookup("key", "q", make_settings()) == (None, None)

    async def test_unreadable_entry_is_discarded(self, service, client):
        client.pipe.execute.return_value = [3, 0, "{not json"]

        assert await service.lookup("key", "q", make_settings()) == (None, 3)


# ---------------------------------------------------------------------- #
# store                                                                   #
# ---------------------------------------------------------------------- #

class TestStore:
    async def test_unavailable_does_nothing(self, client):
        svc = RedisService()
        svc._client = client  # connected once, but caching switched off

        await svc.store("key", KEYWORDS, "q", make_settings())

        client.hset.assert_not_awaited()

    async def test_writes_the_answer_and_sets_expiry(self, service, client):
        await service.store("key", KEYWORDS, "  Data Mining ", make_settings())

        client.hset.assert_awaited_once_with(
            "key", mapping={"query": "data mining", "data": json.dumps(KEYWORDS)},
        )
        client.expire.assert_awaited_once_with("key", TTL_SECONDS)

    async def test_failed_write_skips_expiry(self, service, client):
        client.hset.side_effect = ConnectionError("gone")

        await service.store("key", KEYWORDS, "q", make_settings())  # must not raise

        client.expire.assert_not_awaited()

    async def test_stored_answer_reads_back(self, service, client):
        await service.store("key", KEYWORDS, "q", make_settings())
        stored = client.hset.await_args.kwargs["mapping"]["data"]
        client.pipe.execute.return_value = [2, 0, stored]

        assert await service.lookup("key", "q", make_settings()) == (KEYWORDS, 2)
