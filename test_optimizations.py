#!/usr/bin/env python3
"""
Test script to verify performance optimizations
"""
import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all optimized modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        print("✓ Basic data science libraries imported")
        
        # Test Flask
        from flask import Flask
        print("✓ Flask imported")
        
        # Test our custom modules
        from utils.data_cache import DataCache, cached
        print("✓ Data cache module imported")
        
        from utils.model_cache import ModelCache
        print("✓ Model cache module imported")
        
        from utils.performance_monitor import PerformanceMonitor, monitor_performance
        print("✓ Performance monitor module imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_caching():
    """Test caching functionality"""
    print("\nTesting caching...")
    
    try:
        from utils.data_cache import data_cache
        
        # Test basic cache operations
        data_cache.set('test_key', 'test_value', 60)
        value = data_cache.get('test_key')
        
        if value == 'test_value':
            print("✓ Data cache working correctly")
        else:
            print("✗ Data cache not working")
            return False
            
        # Test cache expiration
        data_cache.set('expire_key', 'expire_value', 1)  # 1 second TTL
        time.sleep(2)
        expired_value = data_cache.get('expire_key')
        
        if expired_value is None:
            print("✓ Cache expiration working")
        else:
            print("✗ Cache expiration not working")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Caching test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring"""
    print("\nTesting performance monitoring...")
    
    try:
        from utils.performance_monitor import performance_monitor, monitor_performance
        
        # Test decorator
        @monitor_performance('test_function')
        def test_function():
            time.sleep(0.1)  # Simulate work
            return "success"
        
        result = test_function()
        
        if result == "success":
            print("✓ Performance monitoring decorator working")
        else:
            print("✗ Performance monitoring decorator failed")
            return False
            
        # Test metrics recording
        performance_monitor.record_metric('test_metric', 100.0, 'ms')
        metrics = performance_monitor.get_metrics()
        
        if 'test_metric' in metrics:
            print("✓ Metrics recording working")
        else:
            print("✗ Metrics recording failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
        return False

def test_model_cache():
    """Test model cache functionality"""
    print("\nTesting model cache...")
    
    try:
        from utils.model_cache import model_cache
        
        # Test cache initialization
        if hasattr(model_cache, 'models') and hasattr(model_cache, 'scalers'):
            print("✓ Model cache initialized correctly")
        else:
            print("✗ Model cache initialization failed")
            return False
            
        # Test cache clearing
        model_cache.clear_cache()
        print("✓ Model cache clear working")
        
        return True
        
    except Exception as e:
        print(f"✗ Model cache test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Binance AI Trading Bot - Performance Optimization Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_caching,
        test_performance_monitoring,
        test_model_cache
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All optimizations are working correctly!")
        return True
    else:
        print("❌ Some optimizations need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)