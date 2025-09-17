#!/usr/bin/env python3
"""
Simple test for core optimizations without external dependencies
"""
import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_cache():
    """Test data caching functionality"""
    print("Testing data cache...")
    
    try:
        from utils.data_cache import data_cache
        
        # Test basic operations
        data_cache.set('test', 'value', 60)
        result = data_cache.get('test')
        
        if result == 'value':
            print("✓ Data cache set/get working")
        else:
            print("✗ Data cache set/get failed")
            return False
            
        # Test expiration
        data_cache.set('expire', 'temp', 1)
        time.sleep(1.1)
        expired = data_cache.get('expire')
        
        if expired is None:
            print("✓ Cache expiration working")
        else:
            print("✗ Cache expiration failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Data cache test failed: {e}")
        return False

def test_model_cache():
    """Test model cache functionality"""
    print("Testing model cache...")
    
    try:
        from utils.model_cache import model_cache
        
        # Test initialization
        if hasattr(model_cache, 'models') and hasattr(model_cache, 'scalers'):
            print("✓ Model cache initialized")
        else:
            print("✗ Model cache initialization failed")
            return False
            
        # Test cache operations
        model_cache.clear_cache()
        print("✓ Model cache clear working")
        
        return True
        
    except Exception as e:
        print(f"✗ Model cache test failed: {e}")
        return False

def test_performance_monitor():
    """Test performance monitoring without psutil"""
    print("Testing performance monitor...")
    
    try:
        from utils.performance_monitor import PerformanceMonitor, monitor_performance
        
        monitor = PerformanceMonitor()
        
        # Test metrics recording
        monitor.record_metric('test', 100.0, 'ms')
        metrics = monitor.get_metrics()
        
        if 'test' in metrics:
            print("✓ Performance monitor working")
        else:
            print("✗ Performance monitor failed")
            return False
            
        # Test decorator
        @monitor_performance('decorator_test')
        def test_func():
            time.sleep(0.01)
            return True
            
        result = test_func()
        if result:
            print("✓ Performance decorator working")
        else:
            print("✗ Performance decorator failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Performance monitor test failed: {e}")
        return False

def test_file_structure():
    """Test that all optimization files exist"""
    print("Testing file structure...")
    
    required_files = [
        'utils/data_cache.py',
        'utils/model_cache.py',
        'utils/performance_monitor.py',
        'static/css/styles.css',
        'static/js/scripts.js',
        'PERFORMANCE_OPTIMIZATIONS.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All optimization files present")
        return True

def main():
    """Run all tests"""
    print("Binance AI Trading Bot - Core Optimization Tests")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_data_cache,
        test_model_cache,
        test_performance_monitor
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Core optimizations are working!")
        return True
    else:
        print("❌ Some optimizations need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)