#!/usr/bin/env python3
"""
Development startup script for AI Navigation Assistant
Starts all components with development configuration and monitoring
"""

import os
import sys
import asyncio
import subprocess
import signal
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add backend to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from logging_config import setup_environment_logging
from monitoring import monitoring_manager


class DevelopmentServer:
    """Manages development server processes"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.logger = setup_environment_logging("development", "dev_server")
        self.running = False
        
    def start_backend(self):
        """Start the backend server"""
        self.logger.info("Starting backend server...")
        
        # Set environment variables
        env = os.environ.copy()
        env.update({
            'ENVIRONMENT': 'development',
            'PYTHONPATH': str(backend_path)
        })
        
        # Load development environment file
        env_file = backend_path / ".env.development"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env[key] = value
        
        # Start backend process
        cmd = [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "debug"
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=backend_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        self.processes['backend'] = process
        self.logger.info(f"Backend server started with PID {process.pid}")
        
        return process
    
    def start_frontend(self):
        """Start the frontend server"""
        self.logger.info("Starting frontend server...")
        
        frontend_path = Path(__file__).parent.parent / "frontend"
        
        # Start frontend process
        cmd = [sys.executable, "-m", "http.server", "3000"]
        
        process = subprocess.Popen(
            cmd,
            cwd=frontend_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        self.processes['frontend'] = process
        self.logger.info(f"Frontend server started with PID {process.pid}")
        
        return process
    
    def start_client(self):
        """Start the client server"""
        self.logger.info("Starting client server...")
        
        client_path = Path(__file__).parent.parent / "client"
        
        # Start client process
        cmd = [sys.executable, "-m", "http.server", "3001"]
        
        process = subprocess.Popen(
            cmd,
            cwd=client_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        self.processes['client'] = process
        self.logger.info(f"Client server started with PID {process.pid}")
        
        return process
    
    def monitor_processes(self):
        """Monitor all processes and restart if needed"""
        while self.running:
            for name, process in list(self.processes.items()):
                if process.poll() is not None:
                    self.logger.warning(f"{name} process died, restarting...")
                    
                    if name == 'backend':
                        self.start_backend()
                    elif name == 'frontend':
                        self.start_frontend()
                    elif name == 'client':
                        self.start_client()
            
            time.sleep(5)
    
    def stop_all(self):
        """Stop all processes"""
        self.logger.info("Stopping all processes...")
        self.running = False
        
        for name, process in self.processes.items():
            if process.poll() is None:
                self.logger.info(f"Stopping {name} (PID {process.pid})")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Force killing {name}")
                    process.kill()
        
        self.processes.clear()
    
    def start_all(self):
        """Start all components"""
        self.logger.info("Starting AI Navigation Assistant in development mode...")
        
        try:
            # Create necessary directories
            self.create_directories()
            
            # Note: Monitoring is handled by the backend server itself
            
            # Start all servers
            self.start_backend()
            time.sleep(2)  # Give backend time to start
            
            self.start_frontend()
            self.start_client()
            
            self.running = True
            
            # Print startup information
            self.print_startup_info()
            
            # Start monitoring processes
            self.monitor_processes()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Error during startup: {e}")
        finally:
            self.stop_all()
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            "logs",
            "backend/models",
            "backend/logs"
        ]
        
        for directory in directories:
            dir_path = Path(__file__).parent.parent / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {dir_path}")
    
    def print_startup_info(self):
        """Print startup information"""
        print("\n" + "="*60)
        print("AI Navigation Assistant - Development Mode")
        print("="*60)
        print(f"Backend:  http://localhost:8000")
        print(f"Frontend: http://localhost:3000")
        print(f"Client:   http://localhost:3001")
        print(f"API Docs: http://localhost:8000/docs")
        print(f"Health:   http://localhost:8000/health")
        print("="*60)
        print("Press Ctrl+C to stop all servers")
        print("="*60 + "\n")


def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\nReceived interrupt signal, shutting down...")
    sys.exit(0)


def main():
    """Main entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start development server
    server = DevelopmentServer()
    server.start_all()


if __name__ == "__main__":
    main()