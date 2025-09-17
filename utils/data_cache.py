"""
Data Cache Manager for API calls and processed data
"""
import time
import hashlib
import logging
from typing import Any, Optional, Dict
from functools import wraps
import threading

logger = logging.getLogger(__name__)

class DataCache:
    """Thread-safe cache for API calls and processed data"""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default TTL
        self.cache = {}
        self.default_ttl = default_ttl
        self.lock = threading.RLock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        with self.lock:
            if key in self.cache:
                value, timestamp, ttl = self.cache[key]
                if time.time() - timestamp < ttl:
                    logger.debug(f"Cache hit for key: {key}")
                    return value
                else:
                    # Expired, remove from cache
                    del self.cache[key]
                    logger.debug(f"Cache expired for key: {key}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        with self.lock:
            ttl = ttl or self.default_ttl
            self.cache[key] = (value, time.time(), ttl)
            logger.debug(f"Cached value for key: {key} with TTL: {ttl}")
    
    def clear(self) -> None:
        """Clear all cached data"""
        with self.lock:
            self.cache.clear()
            logger.info("Data cache cleared")
    
    def clear_expired(self) -> None:
        """Remove expired entries from cache"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp, ttl) in self.cache.items()
                if current_time - timestamp >= ttl
            ]
            for key in expired_keys:
                del self.cache[key]
            if expired_keys:
                logger.debug(f"Cleared {len(expired_keys)} expired cache entries")

def cached(ttl: int = 300):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = data_cache._generate_key(func.__name__, *args, **kwargs)
            cached_result = data_cache.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            result = func(*args, **kwargs)
            data_cache.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator

# Global cache instance
data_cache = DataCache()