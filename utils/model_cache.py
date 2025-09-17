"""
Model Cache Manager for performance optimization
"""
import os
import logging
from typing import Dict, Any, Optional
from functools import lru_cache
import threading

# Optional imports
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelCache:
    """Singleton class to manage model caching and avoid repeated loading"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.models = {}
            self.scalers = {}
            if TORCH_AVAILABLE:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = None
            self._initialized = True
    
    def get_model(self, model_type: str, coin: str, model_save_path: str) -> Optional[Dict[str, Any]]:
        """Get cached model or load from disk"""
        cache_key = f"{model_type}_{coin}"
        
        if cache_key in self.models:
            logger.info(f"Loading {model_type} model for {coin} from cache")
            return self.models[cache_key]
        
        model_file_path = os.path.join(model_save_path, f'{model_type}_{coin}.pth')
        
        if not os.path.exists(model_file_path):
            logger.warning(f"Model file not found: {model_file_path}")
            return None
        
        try:
            if TORCH_AVAILABLE:
                model_info = torch.load(model_file_path, map_location=self.device, weights_only=True)
                self.models[cache_key] = model_info
                logger.info(f"Loaded and cached {model_type} model for {coin}")
                return model_info
            else:
                logger.error("PyTorch not available for model loading")
                return None
        except Exception as e:
            logger.error(f"Error loading model {model_type}_{coin}: {e}")
            return None
    
    def get_scaler(self, model_type: str, coin: str, model_save_path: str) -> Optional[Any]:
        """Get cached scaler or load from disk"""
        cache_key = f"{model_type}_{coin}"
        
        if cache_key in self.scalers:
            logger.info(f"Loading scaler for {model_type}_{coin} from cache")
            return self.scalers[cache_key]
        
        scaler_file = os.path.join(model_save_path, f'scaler_{model_type}_{coin}.pkl')
        
        if not os.path.exists(scaler_file):
            logger.warning(f"Scaler file not found: {scaler_file}")
            return None
        
        try:
            if JOBLIB_AVAILABLE:
                scaler = joblib.load(scaler_file)
                self.scalers[cache_key] = scaler
                logger.info(f"Loaded and cached scaler for {model_type}_{coin}")
                return scaler
            else:
                logger.error("Joblib not available for scaler loading")
                return None
        except Exception as e:
            logger.error(f"Error loading scaler {model_type}_{coin}: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached models and scalers"""
        self.models.clear()
        self.scalers.clear()
        logger.info("Model cache cleared")
    
    def remove_model(self, model_type: str, coin: str):
        """Remove specific model from cache"""
        cache_key = f"{model_type}_{coin}"
        if cache_key in self.models:
            del self.models[cache_key]
        if cache_key in self.scalers:
            del self.scalers[cache_key]
        logger.info(f"Removed {cache_key} from cache")

# Global cache instance
model_cache = ModelCache()