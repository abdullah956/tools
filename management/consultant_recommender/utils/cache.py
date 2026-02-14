"""Redis caching for consultant search results."""

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Cache TTL settings
CONSULTANT_CACHE_TTL = 600  # 10 minutes


class ConsultantCache:
    """Simple Redis cache for consultant search results."""

    def __init__(self):
        """Initialize Redis cache configuration."""
        self.redis_config = None
        self._initialize_config()

    def _initialize_config(self):
        """Initialize Redis connection configuration."""
        try:
            redis_url = os.getenv("REDIS_URL")
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_tls = os.getenv("REDIS_TLS", "false").lower() == "true"

            if redis_url:
                # Parse Redis URL
                parsed_url = urlparse(redis_url)
                self.redis_config = {
                    "host": parsed_url.hostname,
                    "port": parsed_url.port or 6379,
                    "password": parsed_url.password,
                    "username": parsed_url.username,
                    "decode_responses": True,
                    "socket_connect_timeout": 30,  # Increased from 5 to 30 seconds for Upstash
                    "socket_timeout": 30,  # Increased from 5 to 30 seconds
                    "socket_keepalive": True,  # Keep connections alive
                    "retry_on_timeout": True,  # Retry on timeout
                    "health_check_interval": 30,  # Check connection health every 30 seconds
                }

                # Add SSL/TLS ONLY if using rediss:// scheme (not for local redis://)
                if parsed_url.scheme == "rediss":
                    self.redis_config["ssl_cert_reqs"] = None
                    self.redis_config["ssl_check_hostname"] = False
            else:
                # Use individual parameters (for local Redis)
                self.redis_config = {
                    "host": redis_host,
                    "port": redis_port,
                    "decode_responses": True,
                    "socket_connect_timeout": 30,  # Increased from 5 to 30 seconds
                    "socket_timeout": 30,  # Increased from 5 to 30 seconds
                    "socket_keepalive": True,  # Keep connections alive
                    "retry_on_timeout": True,  # Retry on timeout
                    "health_check_interval": 30,  # Check connection health every 30 seconds
                }

                # Only add SSL for external Redis with TLS (not local Redis)
                if redis_tls and redis_host != "redis":
                    self.redis_config["ssl_cert_reqs"] = None
                    self.redis_config["ssl_check_hostname"] = False

            logger.info("✅ Consultant cache Redis config initialized")
        except Exception as e:
            logger.warning(
                f"⚠️ Redis config initialization failed: {e}. Caching disabled."
            )
            self.redis_config = None

    async def _get_redis_client(self):
        """Get a new Redis client for the current event loop."""
        if not self.redis_config:
            return None
        try:
            return redis.Redis(**self.redis_config)
        except Exception as e:
            logger.error(f"Failed to create Redis client: {e}")
            return None

    def _generate_cache_key(self, query: str, user_work_description: str) -> str:
        """Generate cache key from query and work description."""
        # Create a unique key based on query + work description
        data = {
            "query": query.lower().strip(),
            "user_work_description": user_work_description.lower().strip(),
        }
        data_str = json.dumps(data, sort_keys=True)
        hash_key = hashlib.md5(data_str.encode(), usedforsecurity=False).hexdigest()
        return f"consultant_search:{hash_key}"

    async def get(
        self, query: str, user_work_description: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached consultant search results."""
        if not self.redis_config:
            return None

        client = None
        try:
            client = await self._get_redis_client()
            if not client:
                return None

            key = self._generate_cache_key(query, user_work_description)
            data = await client.get(key)
            if data:
                logger.info(f"🎯 Cache HIT for query: {query[:50]}...")
                return json.loads(data)
            logger.info(f"❌ Cache MISS for query: {query[:50]}...")
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
        finally:
            if client:
                await client.aclose()

    async def set(
        self, query: str, user_work_description: str, results: Dict[str, Any]
    ) -> bool:
        """Cache consultant search results."""
        if not self.redis_config:
            return False

        client = None
        try:
            client = await self._get_redis_client()
            if not client:
                return False

            key = self._generate_cache_key(query, user_work_description)
            json_data = json.dumps(results, default=str)
            await client.setex(key, CONSULTANT_CACHE_TTL, json_data)
            logger.info(
                f"💾 Cached results for query: {query[:50]}... (TTL: {CONSULTANT_CACHE_TTL}s)"
            )
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
        finally:
            if client:
                await client.aclose()

    async def ping(self) -> bool:
        """Check if Redis is available."""
        if not self.redis_config:
            return False

        client = None
        try:
            client = await self._get_redis_client()
            if not client:
                return False
            await client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False
        finally:
            if client:
                await client.aclose()


# Global cache instance
consultant_cache = ConsultantCache()
