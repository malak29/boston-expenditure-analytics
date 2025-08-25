import redis.asyncio as redis
import json
import pickle
from typing import Any, Optional, Union, Dict, List
from datetime import timedelta
import hashlib

from app.core.config import settings
from app.core.exceptions import ExternalServiceException
from loguru import logger

class CacheService:
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.default_ttl = settings.cache_ttl
        self.key_prefix = "boston_analytics"
    
    async def connect(self):
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            await self.redis_client.ping()
            logger.info("Connected to Redis cache")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ExternalServiceException("Redis", f"Connection failed: {str(e)}")
    
    async def disconnect(self):
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis cache")
    
    def _generate_key(self, key: str, namespace: str = "general") -> str:
        return f"{self.key_prefix}:{namespace}:{key}"
    
    def _hash_complex_key(self, key_data: Union[str, Dict, List]) -> str:
        if isinstance(key_data, (dict, list)):
            key_string = json.dumps(key_data, sort_keys=True)
        else:
            key_string = str(key_data)
        
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: str = "general",
        serialize_method: str = "json"
    ) -> bool:
        try:
            if not self.redis_client:
                await self.connect()
            
            cache_key = self._generate_key(key, namespace)
            expire_time = ttl or self.default_ttl
            
            if serialize_method == "json":
                serialized_value = json.dumps(value, default=str)
            elif serialize_method == "pickle":
                serialized_value = pickle.dumps(value)
            else:
                serialized_value = str(value)
            
            await self.redis_client.setex(cache_key, expire_time, serialized_value)
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def get(
        self,
        key: str,
        namespace: str = "general",
        serialize_method: str = "json"
    ) -> Optional[Any]:
        try:
            if not self.redis_client:
                await self.connect()
            
            cache_key = self._generate_key(key, namespace)
            cached_value = await self.redis_client.get(cache_key)
            
            if cached_value is None:
                return None
            
            if serialize_method == "json":
                return json.loads(cached_value)
            elif serialize_method == "pickle":
                return pickle.loads(cached_value)
            else:
                return cached_value.decode('utf-8')
                
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def delete(self, key: str, namespace: str = "general") -> bool:
        try:
            if not self.redis_client:
                await self.connect()
            
            cache_key = self._generate_key(key, namespace)
            deleted_count = await self.redis_client.delete(cache_key)
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str, namespace: str = "general") -> bool:
        try:
            if not self.redis_client:
                await self.connect()
            
            cache_key = self._generate_key(key, namespace)
            return await self.redis_client.exists(cache_key) > 0
            
        except Exception as e:
            logger.error(f"Cache exists check error for key {key}: {e}")
            return False
    
    async def invalidate_pattern(self, pattern: str, namespace: str = "general") -> int:
        try:
            if not self.redis_client:
                await self.connect()
            
            search_pattern = self._generate_key(pattern, namespace)
            keys = await self.redis_client.keys(search_pattern)
            
            if keys:
                deleted_count = await self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted_count} cache keys matching pattern {pattern}")
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache pattern invalidation error for {pattern}: {e}")
            return 0
    
    async def cache_analytics_result(
        self,
        analytics_type: str,
        parameters: Dict[str, Any],
        result: Any,
        ttl: Optional[int] = None
    ) -> bool:
        cache_key = f"analytics_{analytics_type}_{self._hash_complex_key(parameters)}"
        return await self.set(
            cache_key,
            result,
            ttl or self.default_ttl,
            namespace="analytics"
        )
    
    async def get_cached_analytics(
        self,
        analytics_type: str,
        parameters: Dict[str, Any]
    ) -> Optional[Any]:
        cache_key = f"analytics_{analytics_type}_{self._hash_complex_key(parameters)}"
        return await self.get(cache_key, namespace="analytics")
    
    async def cache_ml_prediction(
        self,
        model_name: str,
        input_data: Dict[str, Any],
        prediction: Any,
        ttl: int = 3600
    ) -> bool:
        cache_key = f"prediction_{model_name}_{self._hash_complex_key(input_data)}"
        return await self.set(
            cache_key,
            prediction,
            ttl,
            namespace="ml_predictions"
        )
    
    async def get_cached_prediction(
        self,
        model_name: str,
        input_data: Dict[str, Any]
    ) -> Optional[Any]:
        cache_key = f"prediction_{model_name}_{self._hash_complex_key(input_data)}"
        return await self.get(cache_key, namespace="ml_predictions")
    
    async def health_check(self) -> bool:
        try:
            if not self.redis_client:
                await self.connect()
            
            await self.redis_client.ping()
            return True
        except Exception:
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        try:
            if not self.redis_client:
                await self.connect()
            
            info = await self.redis_client.info()
            
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "total_keys": await self.redis_client.dbsize(),
                "cache_hit_ratio": info.get("keyspace_hits", 0) / max(
                    info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1
                )
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}

cache_service = CacheService()