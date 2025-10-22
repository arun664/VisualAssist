// AI Navigation Assistant - Configuration
// Automatically detects backend URL based on environment

const CONFIG = {
  // Backend URL configuration
  getBackendUrl: () => {
    // Check if we're running on GitHub Pages (production)
    if (window.location.hostname.includes('github.io')) {
      // Production: Use Railway deployed backend
      return window.RAILWAY_BACKEND_URL || 'https://your-app-name.up.railway.app';
    }
    
    // Development: Use local backend
    return 'http://localhost:8000';
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