// AI Navigation Assistant - Configuration
// Automatically detects backend URL based on environment

const CONFIG = {
  // Backend URL configuration
  getBackendUrl: () => {
    // Detect if running locally or from file system
    const isLocalhost = window.location.hostname === 'localhost' || 
                       window.location.hostname === '127.0.0.1' ||
                       window.location.hostname.includes('localhost');
    const isFileProtocol = window.location.protocol === 'file:';
    
    if (isLocalhost || isFileProtocol) {
      // Use local backend when running on localhost or from filesystem
      return 'http://localhost:8000';
    } else if (window.location.hostname.includes('github.io')) {
      // For GitHub Pages, use the ngrok URL
      return 'https://flagless-clinographic-janita.ngrok-free.dev';
    } else {
      // Fallback for other environments
      return 'http://localhost:8000';
    }
  },
  
  // Get CORS proxy URL as temporary workaround (NOT for production)
  getCorsProxyUrl: () => {
    // TEMPORARY WORKAROUND: Use public CORS proxy
    // WARNING: Only for testing - not secure for production!
    return 'https://cors-anywhere.herokuapp.com/https://flagless-clinographic-janita.ngrok-free.dev';
  },
  
  // Get suggested HTTPS solutions
  getHttpsSolutions: () => {
    return [
      {
        name: 'Cloudflare Tunnel (Recommended)',
        description: 'Free, reliable, custom domain',
        setup: 'Install cloudflared on AWS, create tunnel, get HTTPS URL'
      },
      {
        name: 'ngrok',
        description: 'Quick setup, temporary URLs',
        setup: 'Run "ngrok http 8000" on AWS, get https://xxx.ngrok.io'
      },
      {
        name: 'localtunnel',
        description: 'Free, custom subdomain',
        setup: 'Run "lt --port 8000 --subdomain visualassist" on AWS'
      }
    ];
  },
  
  // Get direct backend URL without proxy
  getDirectBackendUrl: () => {
    // First try to use local backend if available
    const isLocalhost = window.location.hostname === 'localhost' || 
                        window.location.hostname === '127.0.0.1' ||
                        window.location.hostname.includes('localhost');
    const isFileProtocol = window.location.protocol === 'file:';
    
    if (isLocalhost || isFileProtocol) {
      return 'http://localhost:8000';
    }
    return 'https://flagless-clinographic-janita.ngrok-free.dev';
  },
  
  // Check if current page is HTTPS
  isHttpsPage: () => {
    return window.location.protocol === 'https:';
  },
  
  // Check if we have Mixed Content issue
  hasMixedContentIssue: () => {
    return CONFIG.isHttpsPage() && CONFIG.getBackendUrl().startsWith('http://');
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
    return window.location.hostname === 'localhost' || 
           window.location.hostname === '127.0.0.1' || 
           window.location.protocol === 'file:';
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
console.log('Mixed Content Issue:', CONFIG.hasMixedContentIssue());