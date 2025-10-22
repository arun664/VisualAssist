// AI Navigation Assistant - Configuration
// Automatically detects backend URL based on environment

const CONFIG = {
  // Backend URL configuration
  getBackendUrl: () => {
    // Check if we're on HTTPS (GitHub Pages) and need to handle Mixed Content
    if (window.location.protocol === 'https:') {
      // For HTTPS pages, we need to use a proxy or different approach
      // For now, return HTTP but handle Mixed Content in client
      return 'http://18.222.141.234:8000';
    }
    return 'http://18.222.141.234:8000';
  },
  
  // Get HTTP URL specifically for mixed content handling
  getHttpBackendUrl: () => {
    return 'http://18.222.141.234:8000';
  },
  
  // Check if current page is HTTPS
  isHttpsPage: () => {
    return window.location.protocol === 'https:';
  },
  
  // WebSocket URL (derived from backend URL)
  getWebSocketUrl: () => {
    const backendUrl = CONFIG.getBackendUrl();
    return backendUrl.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws';
  },
  
  // Video stream URL
  getVideoStreamUrl: () => {
    return CONFIG.getBackendUrl() + '/processed_video_stream';
  },
  
  // Environment detection
  isDevelopment: () => {
    return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  },
  
  isProduction: () => {
    return window.location.hostname.includes('github.io');
  }
};

// Make config globally available
window.AI_NAV_CONFIG = CONFIG;

console.log('🤖 AI Navigation Assistant Config Loaded');
console.log('Environment:', CONFIG.isDevelopment() ? 'Development' : 'Production');
console.log('Backend URL:', CONFIG.getBackendUrl());
console.log('WebSocket URL:', CONFIG.getWebSocketUrl());
console.log('Is HTTPS Page:', CONFIG.isHttpsPage());