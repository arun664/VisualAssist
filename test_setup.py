#!/usr/bin/env python3
"""
Test script to verify AI Navigation Assistant setup
"""

import requests
import json
import sys
from pathlib import Path

def test_backend_api():
    """Test backend API endpoints"""
    print("🧪 Testing Backend API...")
    
    try:
        # Test root endpoint
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint: {data['message']}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
            
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health endpoint: {data['status']}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
            
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend API connection failed: {e}")
        return False

def test_frontend_server():
    """Test frontend server"""
    print("🧪 Testing Frontend Server...")
    
    try:
        response = requests.get("http://localhost:3000/", timeout=5)
        if response.status_code == 200 and "AI Navigation Assistant" in response.text:
            print("✅ Frontend server serving HTML correctly")
            return True
        else:
            print(f"❌ Frontend server failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend server connection failed: {e}")
        return False

def test_client_server():
    """Test client server"""
    print("🧪 Testing Client Server...")
    
    try:
        response = requests.get("http://localhost:3001/", timeout=5)
        if response.status_code == 200 and "Navigation Client" in response.text:
            print("✅ Client server serving HTML correctly")
            return True
        else:
            print(f"❌ Client server failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Client server connection failed: {e}")
        return False

def test_file_structure():
    """Test project file structure"""
    print("🧪 Testing Project Structure...")
    
    required_files = [
        "backend/main.py",
        "backend/requirements.txt",
        "frontend/index.html",
        "frontend/styles.css",
        "frontend/app.js",
        "client/index.html",
        "client/client-styles.css",
        "client/client.js",
        "config.json",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def main():
    """Run all tests"""
    print("🚀 AI Navigation Assistant Setup Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_backend_api,
        test_frontend_server,
        test_client_server
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    if all(results):
        print("🎉 All tests passed! Setup is working correctly.")
        print("\n📋 Summary:")
        print("- Backend API: http://localhost:8000")
        print("- Frontend UI: http://localhost:3000")
        print("- Client Interface: http://localhost:3001")
        print("- API Documentation: http://localhost:8000/docs")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())