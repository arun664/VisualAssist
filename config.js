// AI Navigation Assistant - Configuration
// Automatically detects backend URL based on environment

const CONFIG = {
  // Backend URL configuration
  getBackendUrl: () => {
    // Check if we're on HTTPS (GitHub Pages)
    if (window.location.protocol === 'https:') {
      // TODO: Replace with your HTTPS tunnel URL once set up
      // For now, we'll try the HTTP URL and handle the error gracefully
      // 
      // Options to set up HTTPS backend:
      // 1. Cloudflare Tunnel: https://your-domain.com
      // 2. ngrok: https://abc123.ngrok.io  
      // 3. localtunnel: https://your-subdomain.loca.lt
      
      // Temporary: Return HTTP URL and let error handling guide users
      return 'http://18.222.141.234:8000';
    }
    return 'http://18.222.141.234:8000';
  },
  
  // Get CORS proxy URL as temporary workaround (NOT for production)
  getCorsProxyUrl: () => {
    // TEMPORARY WORKAROUND: Use public CORS proxy
    // WARNING: Only for testing - not secure for production!
    return 'https://cors-anywhere.herokuapp.com/http://18.222.141.234:8000';
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
    return 'http://18.222.141.234:8000';
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
console.log('Mixed Content Issue:', CONFIG.hasMixedContentIssue());