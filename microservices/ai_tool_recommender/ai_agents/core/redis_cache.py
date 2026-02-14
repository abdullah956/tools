"""Redis caching for ultra-fast responses."""

import asyncio
import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Redis configuration with support for Upstash Redis (TLS + Auth)
REDIS_URL = os.getenv("REDIS_URL")  # Full Redis URL (e.g., redis://user:pass@host:port)
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "default")
REDIS_TLS = os.getenv("REDIS_TLS", "false").lower() == "true"

# Build Redis configuration
if REDIS_URL:
    # Parse Redis URL for Upstash Redis
    parsed_url = urlparse(REDIS_URL)
    REDIS_CONFIG = {
        "host": parsed_url.hostname,
        "port": parsed_url.port or 6379,
        "password": parsed_url.password,
        "username": parsed_url.username,
        "decode_responses": True,
        "socket_connect_timeout": 5,
        "socket_timeout": 5,
    }

    # Add SSL/TLS if using rediss:// or if TLS flag is set
    if parsed_url.scheme == "rediss" or REDIS_TLS:
        REDIS_CONFIG["ssl"] = True
else:
    # Use individual parameters (for local Redis)
    REDIS_CONFIG = {
        "host": REDIS_HOST,
        "port": REDIS_PORT,
        "db": REDIS_DB,
        "decode_responses": True,
        "socket_connect_timeout": 5,
        "socket_timeout": 5,
    }

    # Add authentication if provided
    if REDIS_PASSWORD:
        REDIS_CONFIG["password"] = REDIS_PASSWORD
    if REDIS_USERNAME:
        REDIS_CONFIG["username"] = REDIS_USERNAME
    if REDIS_TLS:
        REDIS_CONFIG["ssl"] = True

# FastAPI Background Tasks Configuration
# No Celery needed - using FastAPI BackgroundTasks

# Cache TTL settings
CACHE_TTL = {
    "query_results": 300,  # 5 minutes
    "tool_data": 1800,  # 30 minutes
    "search_results": 600,  # 10 minutes
    "workflow": 900,  # 15 minutes
}


class RedisCache:
    """High-performance Redis caching system."""

    def __init__(self):
        """Initialize Redis cache with connection pool."""
        self.redis_client = None
        self._connection_pool = None

    async def connect(self):
        """Connect to Redis."""
        try:
            logger.info(f"🔗 Connecting to Redis with config: {REDIS_CONFIG}")
            self._connection_pool = redis.ConnectionPool(**REDIS_CONFIG)
            self.redis_client = redis.Redis(connection_pool=self._connection_pool)
            await self.redis_client.ping()
            logger.info("✅ Redis connected successfully")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            logger.error(f"❌ Redis config: {REDIS_CONFIG}")
            self.redis_client = None

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis disconnected")

    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        hash_key = hashlib.md5(data_str.encode(), usedforsecurity=False).hexdigest()
        return f"{prefix}:{hash_key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        if not self.redis_client:
            return None

        try:
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(self, key: str, data: Any, ttl: int = 300) -> bool:
        """Set data in cache."""
        if not self.redis_client:
            return False

        try:
            json_data = json.dumps(data, default=str)
            await self.redis_client.setex(key, ttl, json_data)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete data from cache."""
        if not self.redis_client:
            return False

        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def get_or_set(self, key: str, func, ttl: int = 300, *args, **kwargs) -> Any:
        """Get from cache or set using function."""
        # Try to get from cache first
        cached_data = await self.get(key)
        if cached_data is not None:
            logger.info(f"Cache hit for key: {key}")
            return cached_data

        # Cache miss - execute function
        logger.info(f"Cache miss for key: {key}")
        if asyncio.iscoroutinefunction(func):
            data = await func(*args, **kwargs)
        else:
            data = func(*args, **kwargs)

        # Store in cache
        await self.set(key, data, ttl)
        return data


class QueryCache:
    """Specialized cache for query results."""

    def __init__(self, redis_cache: RedisCache):
        """Initialize query cache with Redis cache instance."""
        self.redis = redis_cache

    async def get_query_results(
        self, query: str, max_results: int
    ) -> Optional[Dict[str, Any]]:
        """Get cached query results."""
        key = self.redis._generate_cache_key(
            "query", {"query": query, "max_results": max_results}
        )
        return await self.redis.get(key)

    async def set_query_results(
        self, query: str, max_results: int, results: Dict[str, Any]
    ) -> bool:
        """Cache query results."""
        key = self.redis._generate_cache_key(
            "query", {"query": query, "max_results": max_results}
        )
        return await self.redis.set(key, results, CACHE_TTL["query_results"])

    async def get_tool_data(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get cached tool data."""
        key = f"tool:{tool_id}"
        return await self.redis.get(key)

    async def set_tool_data(self, tool_id: str, data: Dict[str, Any]) -> bool:
        """Cache tool data."""
        key = f"tool:{tool_id}"
        return await self.redis.set(key, data, CACHE_TTL["tool_data"])

    async def get_workflow(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached workflow."""
        key = self.redis._generate_cache_key("workflow", {"query": query})
        return await self.redis.get(key)

    async def set_workflow(self, query: str, workflow: Dict[str, Any]) -> bool:
        """Cache workflow."""
        key = self.redis._generate_cache_key("workflow", {"query": query})
        return await self.redis.set(key, workflow, CACHE_TTL["workflow"])


# Background tasks are now handled by FastAPI BackgroundTasks
# See background_scheduler.py for implementation


# Global instances
redis_cache = RedisCache()
query_cache = QueryCache(redis_cache)
