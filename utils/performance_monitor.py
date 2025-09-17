"""
Performance monitoring utilities for the trading bot
"""
import time
import logging
from functools import wraps
from typing import Dict, Any
import threading

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor application performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, unit: str = "ms"):
        """Record a performance metric"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append({
                'value': value,
                'unit': unit,
                'timestamp': time.time()
            })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self.lock:
            return self.metrics.copy()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        if PSUTIL_AVAILABLE:
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'available_memory_gb': psutil.virtual_memory().available / (1024**3)
            }
        else:
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'disk_percent': 0,
                'available_memory_gb': 0
            }
    
    def clear_metrics(self):
        """Clear all recorded metrics"""
        with self.lock:
            self.metrics.clear()

def monitor_performance(metric_name: str = None):
    """Decorator to monitor function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                metric = metric_name or f"{func.__module__}.{func.__name__}"
                performance_monitor.record_metric(metric, execution_time)
                
                logger.debug(f"{func.__name__} executed in {execution_time:.2f}ms")
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                metric = metric_name or f"{func.__module__}.{func.__name__}_error"
                performance_monitor.record_metric(metric, execution_time)
                raise e
        return wrapper
    return decorator

def log_memory_usage(func):
    """Decorator to log memory usage before and after function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024**2)  # MB
            
            result = func(*args, **kwargs)
            
            memory_after = process.memory_info().rss / (1024**2)  # MB
            memory_diff = memory_after - memory_before
            
            logger.info(f"{func.__name__} memory usage: {memory_before:.2f}MB -> {memory_after:.2f}MB (Δ{memory_diff:+.2f}MB)")
            return result
        else:
            # If psutil not available, just run the function
            return func(*args, **kwargs)
    return wrapper

# Global performance monitor instance
performance_monitor = PerformanceMonitor()