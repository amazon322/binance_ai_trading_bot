// Optimized JavaScript for Binance AI Trading Bot

// Performance optimizations
(function() {
    'use strict';
    
    // Debounce function for performance
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    // Throttle function for performance
    function throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
    
    // Initialize when DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
        initializeApp();
    });
    
    function initializeApp() {
        // Initialize form validation
        initializeFormValidation();
        
        // Initialize charts if present
        initializeCharts();
        
        // Initialize real-time updates
        initializeRealTimeUpdates();
        
        // Initialize performance monitoring
        initializePerformanceMonitoring();
    }
    
    function initializeFormValidation() {
        const forms = document.querySelectorAll('form');
        
        forms.forEach(form => {
            form.addEventListener('submit', function(e) {
                if (!form.checkValidity()) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                form.classList.add('was-validated');
            });
        });
    }
    
    function initializeCharts() {
        const chartContainer = document.getElementById('trading-chart');
        if (chartContainer) {
            // Placeholder for chart initialization
            // This would integrate with Chart.js or similar library
            console.log('Chart container found, ready for chart initialization');
        }
    }
    
    function initializeRealTimeUpdates() {
        // Throttled function for real-time updates
        const updateData = throttle(function() {
            // This would handle real-time data updates
            console.log('Updating real-time data...');
        }, 5000); // Update every 5 seconds
        
        // Set up periodic updates
        setInterval(updateData, 5000);
    }
    
    function initializePerformanceMonitoring() {
        // Monitor performance metrics
        if ('performance' in window) {
            window.addEventListener('load', function() {
                setTimeout(function() {
                    const perfData = performance.getEntriesByType('navigation')[0];
                    console.log('Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
                }, 0);
            });
        }
    }
    
    // Utility functions
    window.TradingBot = {
        // Show loading state
        showLoading: function(element) {
            if (element) {
                element.classList.add('loading');
            }
        },
        
        // Hide loading state
        hideLoading: function(element) {
            if (element) {
                element.classList.remove('loading');
            }
        },
        
        // Format currency
        formatCurrency: function(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(value);
        },
        
        // Format percentage
        formatPercentage: function(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'percent',
                minimumFractionDigits: 2
            }).format(value / 100);
        },
        
        // Show notification
        showNotification: function(message, type = 'info') {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
            alertDiv.innerHTML = `
                ${message}
                <button type="button" class="close" data-dismiss="alert">
                    <span>&times;</span>
                </button>
            `;
            
            const container = document.querySelector('.container');
            if (container) {
                container.insertBefore(alertDiv, container.firstChild);
                
                // Auto-remove after 5 seconds
                setTimeout(() => {
                    if (alertDiv.parentNode) {
                        alertDiv.parentNode.removeChild(alertDiv);
                    }
                }, 5000);
            }
        }
    };
    
})();