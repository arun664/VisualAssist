#!/usr/bin/env python3
"""
Simple HTTP server that properly serves index.html as default
"""

import os
import sys
import http.server
import socketserver
from pathlib import Path

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves index.html for directory requests"""
    
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        # If requesting a directory, try to serve index.html
        if self.path.endswith('/') or '.' not in os.path.basename(self.path):
            index_path = os.path.join(self.directory, self.path.lstrip('/'), 'index.html')
            if os.path.exists(index_path):
                self.path = self.path.rstrip('/') + '/index.html'
        
        return super().do_GET()

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_server.py <port> [directory]")
        print("Example: python simple_server.py 3000 frontend")
        sys.exit(1)
    
    port = int(sys.argv[1])
    directory = sys.argv[2] if len(sys.argv) > 2 else '.'
    
    # Change to the specified directory
    if directory != '.':
        if not os.path.exists(directory):
            print(f"Error: Directory '{directory}' does not exist")
            sys.exit(1)
        os.chdir(directory)
    
    # Set up the server
    handler = CustomHTTPRequestHandler
    handler.directory = os.getcwd()
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving HTTP on port {port}")
        print(f"Directory: {os.getcwd()}")
        print(f"URL: http://localhost:{port}")
        
        # Check if index.html exists
        if os.path.exists('index.html'):
            print("✓ index.html found - will be served as default")
        else:
            print("⚠ index.html not found - directory listing will be shown")
        
        print("Press Ctrl+C to stop")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")

if __name__ == "__main__":
    main()