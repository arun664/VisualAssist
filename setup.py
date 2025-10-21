#!/usr/bin/env python3
"""
AI Navigation Assistant Setup Script
Automates the initial setup process for development environment
"""

import os
import subprocess
import sys
import json
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {command}")
        print(f"Error: {e.stderr}")
        return False

def setup_backend():
    """Set up Python backend environment"""
    print("\n🔧 Setting up Backend...")
    
    backend_dir = Path("backend")
    
    # Create virtual environment
    if not (backend_dir / "venv").exists():
        if not run_command("python -m venv venv", cwd=backend_dir):
            return False
    
    # Activate and install dependencies
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Linux/Mac
        pip_cmd = "venv/bin/pip"
    
    return run_command(f"{pip_cmd} install -r requirements.txt", cwd=backend_dir)

def create_env_file():
    """Create .env file from example"""
    print("\n📝 Creating environment configuration...")
    
    env_example = Path("backend/.env.example")
    env_file = Path("backend/.env")
    
    if env_example.exists() and not env_file.exists():
        env_file.write_text(env_example.read_text())
        print("✓ Created backend/.env from example")
        return True
    return True

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def main():
    """Main setup function"""
    print("🚀 AI Navigation Assistant Setup")
    print("=" * 40)
    
    # Check prerequisites
    if not check_python_version():
        return 1
    
    # Setup backend
    if not setup_backend():
        print("❌ Backend setup failed")
        return 1
    
    # Create environment file
    if not create_env_file():
        print("❌ Environment setup failed")
        return 1
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Download Vosk model: https://alphacephei.com/vosk/models")
    print("2. Extract to backend/models/vosk-model-en-us-0.22/")
    print("3. Run: python backend/main.py")
    print("4. Open frontend: python -m http.server 3000 (in frontend/)")
    print("5. Open client: python -m http.server 3001 (in client/)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())