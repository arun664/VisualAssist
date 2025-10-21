#!/usr/bin/env python3
import http.server
import socketserver
import os
import sys

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

PORT = 8080
Handler = MyHTTPRequestHandler

print(f"Starting server on all interfaces at port {PORT}")
print("Access URLs:")
print(f"  Local: http://localhost:{PORT}")
print(f"  Network: http://192.168.180.185:{PORT}")
print("Press Ctrl+C to stop")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        sys.exit(0)