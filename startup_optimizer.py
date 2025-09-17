#!/usr/bin/env python3
"""
Startup optimizer for the Binance AI Trading Bot
Preloads models and optimizes application startup
"""
import os
import sys
import logging
import time
from pathlib import Path
from config import Config
from utils.model_cache import model_cache
from utils.data_cache import data_cache
from utils.performance_monitor import performance_monitor

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preload_models():
    """Preload commonly used models for faster predictions"""
    logger.info("Starting model preloading...")
    
    model_save_path = Config.MODEL_SAVE_PATH
    if not os.path.exists(model_save_path):
        logger.warning(f"Model directory {model_save_path} does not exist")
        return
    
    # Find available models
    model_files = []
    for file in os.listdir(model_save_path):
        if file.endswith('.pth') or file.endswith('.pkl'):
            model_type, coin_with_ext = file.split('_', 1)
            coin = coin_with_ext.split('.')[0]
            model_files.append((model_type, coin))
    
    if not model_files:
        logger.info("No models found to preload")
        return
    
    # Preload models
    preloaded_count = 0
    for model_type, coin in model_files:
        try:
            start_time = time.time()
            
            # Preload model
            model = model_cache.get_model(model_type, coin, model_save_path)
            scaler = model_cache.get_scaler(model_type, coin, model_save_path)
            
            load_time = (time.time() - start_time) * 1000
            logger.info(f"Preloaded {model_type}_{coin} in {load_time:.2f}ms")
            preloaded_count += 1
            
        except Exception as e:
            logger.error(f"Failed to preload {model_type}_{coin}: {e}")
    
    logger.info(f"Successfully preloaded {preloaded_count}/{len(model_files)} models")

def optimize_system():
    """Perform system-level optimizations"""
    logger.info("Performing system optimizations...")
    
    # Clear any existing cache
    data_cache.clear()
    
    # Set up performance monitoring
    if Config.ENABLE_PERFORMANCE_LOGGING:
        system_info = performance_monitor.get_system_info()
        logger.info(f"System resources: CPU {system_info['cpu_percent']:.1f}%, "
                   f"Memory {system_info['memory_percent']:.1f}%, "
                   f"Available Memory {system_info['available_memory_gb']:.2f}GB")
    
    # Create necessary directories
    directories = [
        'static/css',
        'static/js',
        'logs',
        'saved_models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("System optimization completed")

def check_dependencies():
    """Check if all required dependencies are available"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'flask',
        'pandas',
        'numpy',
        'torch',
        'scikit-learn',
        'requests',
        'beautifulsoup4',
        'transformers'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are available")
    return True

def main():
    """Main startup optimization function"""
    logger.info("Starting Binance AI Trading Bot optimization...")
    
    start_time = time.time()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Optimize system
    optimize_system()
    
    # Preload models if enabled
    if Config.ENABLE_MODEL_PRELOADING:
        preload_models()
    
    total_time = (time.time() - start_time) * 1000
    logger.info(f"Startup optimization completed in {total_time:.2f}ms")
    
    # Log performance metrics
    if Config.ENABLE_PERFORMANCE_LOGGING:
        performance_monitor.record_metric('startup_time', total_time)
        system_info = performance_monitor.get_system_info()
        logger.info(f"Final system state: CPU {system_info['cpu_percent']:.1f}%, "
                   f"Memory {system_info['memory_percent']:.1f}%")

if __name__ == "__main__":
    main()