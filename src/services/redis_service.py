import hashlib
import json
import logging

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Part of every key. Bump when the shape of a stored record changes: new keys
# are read, records in the old shape go untouched and expire on their own.
CACHE_SCHEMA_VERSION = "v1"

# Kept short: a slow cache must never be slower than just calling the LLM.
REDIS_CONNECT_TIMEOUT = 0.5
REDIS_SOCKET_TIMEOUT = 0.5


def normalise(query: str) -> str:
    """Lowercase, trim, and collapse runs of whitespace to single spaces.

    Everything Redis keys on or stores uses this form, so that the same
    question typed differently is one entry rather than several.
    """
    return " ".join(query.split()).casefold()


class RedisService:
    """Response cache for search answers, one hash per query.

    Every Redis failure is logged and swallowed, so a cache that is down or
    unreachable degrades into a plain LLM call rather than an error.
    """

    def __init__(self) -> None:
        self._client: redis.Redis | None = None
        self._enabled = False

    # ------------------------------------------------------------------ #
    # Connection lifecycle                                                #
    # ------------------------------------------------------------------ #

    def init_cache(self, settings) -> None:
        """Build the connection pool once, at application startup.

        A failed ping here is logged but not fatal: the client is kept so that
        later requests reconnect on their own if Redis comes back.
        """
        self._enabled = settings.REDIS_ENABLED
        if not self._enabled:
            logger.info("Redis cache is disabled (REDIS_ENABLED=false), serving every query from the LLM.")
            return

        try:
            pool = redis.ConnectionPool(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD or None,
                connection_class=redis.SSLConnection if settings.REDIS_SSL else redis.Connection,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
                socket_timeout=REDIS_SOCKET_TIMEOUT,
                health_check_interval=30,
                decode_responses=True,
                # No retries: a request must not wait out the timeout twice over
                # for a cache lookup it can do without.
                retry=Retry(NoBackoff(), 0),
            )
            self._client = redis.Redis(connection_pool=pool)
            self._client.ping()
            logger.info(f"Redis cache connected :: {settings.REDIS_HOST}:{settings.REDIS_PORT} db={settings.REDIS_DB}")
        except Exception as e:
            logger.warning(f"Redis unreachable at startup, running without cache until it recovers :: {e}")

    def close_cache(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Failed to close Redis connection :: {e}")
            self._client = None

    @property
    def _available(self) -> bool:
        return self._enabled and self._client is not None

    # ------------------------------------------------------------------ #
    # Redis primitives                                                    #
    # ------------------------------------------------------------------ #

    def _hset(self, key: str, mapping: dict) -> bool:
        """Write fields into a hash. False if Redis could not be reached."""
        try:
            self._client.hset(key, mapping=mapping)
            logger.debug(f"Redis HSET {key} ({len(mapping)} fields)")
            return True
        except Exception as e:
            logger.warning(f"Redis write failed, response served but not cached :: {e}")
            return False

    def _expire(self, key: str, settings) -> None:
        try:
            self._client.expire(key, settings.redis_cache_ttl_seconds)
            logger.debug(f"Redis EXPIRE {key} ttl={settings.REDIS_CACHE_TTL_DAYS}d")
        except Exception as e:
            logger.warning(f"Could not set expiry on {key} :: {e}")

    # ------------------------------------------------------------------ #
    # Query cache                                                         #
    # ------------------------------------------------------------------ #

    def build_key(self, query: str, synonyms: bool, settings) -> str:
        """Key of the record for one query.

        One hash per query holds both the cached answer and how often it has been
        asked. The query is normalised and hashed, which keeps keys a fixed length
        whatever the user typed. Records outlive changes to the model or the
        prompts, so clear the cache by hand after changing either.
        """
        return ":".join([
            settings.REDIS_KEY_PREFIX,
            CACHE_SCHEMA_VERSION,
            "syn" if synonyms else "nosyn",
            hashlib.sha256(normalise(query).encode("utf-8")).hexdigest(),
        ])

    def lookup(self, key: str, query: str, settings):
        """Count this request and read back any cached answer, in one round trip.

        Returns (data, count), either of which may be None: data when nothing is
        cached yet, count when counting is switched off. Any Redis failure gives
        (None, None), so the caller simply falls back to the LLM.
        """
        if not self._available:
            return None, None

        count_it = settings.REDIS_QUERY_COUNTER_ENABLED

        try:
            pipe = self._client.pipeline(transaction=False)
            if count_it:
                # Redis does the increment itself, so the count stays correct with
                # several workers or pods serving requests at once.
                pipe.hincrby(key, "count", 1)
                pipe.hsetnx(key, "query", normalise(query))
            pipe.hget(key, "data")
            results = pipe.execute()
        except Exception as e:
            logger.warning(f"Redis read failed, falling back to the LLM :: {e}")
            return None, None

        count = results[0] if count_it else None
        raw = results[-1]

        # The very first request for a query creates the record, so give it its
        # lifespan now. Counting alone can create a record whose answer never
        # arrives (an unparseable model response), and that must expire too.
        if count == 1:
            self._expire(key, settings)

        if raw is None:
            return None, count

        try:
            return json.loads(raw), count
        except ValueError as e:
            logger.warning(f"Discarding unreadable cache entry {key} :: {e}")
            return None, count

    def store(self, key: str, data, query: str, settings) -> None:
        """Add a fresh answer to a record. Failures are logged and otherwise ignored."""
        if not self._available:
            return

        fields = {
            "query": normalise(query),
            "data": json.dumps(data),
        }

        if not self._hset(key, fields):
            return

        # With counting switched off nothing else sets the lifespan, and a record
        # rewritten after expiry needs a new one.
        self._expire(key, settings)
        logger.info(f"Cached response :: {key} ttl={settings.REDIS_CACHE_TTL_DAYS}d")


redis_service = RedisService()
