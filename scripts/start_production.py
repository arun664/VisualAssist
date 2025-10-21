#!/usr/bin/env python3
"""
Production startup script for AI Navigation Assistant
Starts all components with production configuration, monitoring, and process management
"""

import os
import sys
import asyncio
import subprocess
import signal
import time
import json
from pathlib import Path
from typing import List, Dict, Optional

# Add backend to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from logging_config import setup_environment_logging
from monitoring import monitoring_manager


class ProductionServer:
    """Manages production server processes with enhanced monitoring and reliability"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.logger = setup_environment_logging("production", "prod_server")
        self.running = False
        self.restart_counts: Dict[str, int] = {}
        self.max_restarts = 5
        self.restart_window = 300  # 5 minutes
        self.restart_times: Dict[str, List[float]] = {}
        
    def start_backend(self):
        """Start the backend server with production configuration"""
        self.logger.info("Starting backend server in production mode...")
        
        # Set environment variables
        env = os.environ.copy()
        env.update({
            'ENVIRONMENT': 'production',
            'PYTHONPATH': str(backend_path)
        })
        
        # Load production environment file
        env_file = backend_path / ".env.production"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env[key] = value
        
        # Start backend with Gunicorn for production
        cmd = [
            "gunicorn", "main:app",
            "--bind", "0.0.0.0:8000",
            "--workers", "4",
            "--worker-class", "uvicorn.workers.UvicornWorker",
            "--max-requests", "1000",
            "--max-requests-jitter", "100",
            "--timeout", "30",
            "--keepalive", "10",
            "--log-level", "info",
            "--access-logfile", "logs/access.log",
            "--error-logfile", "logs/error.log",
            "--pid", "logs/backend.pid"
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
        """Start the frontend server with production configuration"""
        self.logger.info("Starting frontend server in production mode...")
        
        frontend_path = Path(__file__).parent.parent / "frontend"
        
        # Use a production-ready HTTP server (nginx would be better, but using Python for simplicity)
        cmd = [
            sys.executable, "-m", "http.server", "3000",
            "--bind", "0.0.0.0"
        ]
        
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
        """Start the client server with production configuration"""
        self.logger.info("Starting client server in production mode...")
        
        client_path = Path(__file__).parent.parent / "client"
        
        cmd = [
            sys.executable, "-m", "http.server", "3001",
            "--bind", "0.0.0.0"
        ]
        
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
    
    def should_restart_process(self, name: str) -> bool:
        """Check if a process should be restarted based on restart policy"""
        current_time = time.time()
        
        if name not in self.restart_times:
            self.restart_times[name] = []
        
        # Remove old restart times outside the window
        self.restart_times[name] = [
            t for t in self.restart_times[name] 
            if current_time - t < self.restart_window
        ]
        
        # Check if we've exceeded the restart limit
        if len(self.restart_times[name]) >= self.max_restarts:
            self.logger.error(
                f"Process {name} has been restarted {len(self.restart_times[name])} times "
                f"in the last {self.restart_window} seconds. Not restarting."
            )
            return False
        
        return True
    
    def record_restart(self, name: str):
        """Record a process restart"""
        if name not in self.restart_times:
            self.restart_times[name] = []
        
        self.restart_times[name].append(time.time())
        self.restart_counts[name] = self.restart_counts.get(name, 0) + 1
    
    def monitor_processes(self):
        """Monitor all processes and restart if needed with backoff"""
        while self.running:
            for name, process in list(self.processes.items()):
                if process.poll() is not None:
                    exit_code = process.returncode
                    self.logger.warning(f"{name} process died with exit code {exit_code}")
                    
                    if self.should_restart_process(name):
                        self.logger.info(f"Restarting {name} process...")
                        self.record_restart(name)
                        
                        # Exponential backoff
                        restart_count = self.restart_counts.get(name, 0)
                        backoff_time = min(2 ** restart_count, 60)  # Max 60 seconds
                        time.sleep(backoff_time)
                        
                        try:
                            if name == 'backend':
                                self.start_backend()
                            elif name == 'frontend':
                                self.start_frontend()
                            elif name == 'client':
                                self.start_client()
                        except Exception as e:
                            self.logger.error(f"Failed to restart {name}: {e}")
                    else:
                        self.logger.error(f"Not restarting {name} due to restart policy")
                        del self.processes[name]
            
            time.sleep(10)  # Check every 10 seconds in production
    
    def health_check(self) -> Dict[str, bool]:
        """Perform health checks on all services"""
        health_status = {}
        
        try:
            import requests
            
            # Check backend health
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                health_status['backend'] = response.status_code == 200
            except:
                health_status['backend'] = False
            
            # Check frontend availability
            try:
                response = requests.get("http://localhost:3000", timeout=5)
                health_status['frontend'] = response.status_code == 200
            except:
                health_status['frontend'] = False
            
            # Check client availability
            try:
                response = requests.get("http://localhost:3001", timeout=5)
                health_status['client'] = response.status_code == 200
            except:
                health_status['client'] = False
                
        except ImportError:
            self.logger.warning("requests library not available for health checks")
            # Fallback to process checks
            health_status = {
                name: process.poll() is None 
                for name, process in self.processes.items()
            }
        
        return health_status
    
    def log_system_status(self):
        """Log system status periodically"""
        while self.running:
            try:
                health_status = self.health_check()
                process_status = {
                    name: {
                        'pid': process.pid,
                        'running': process.poll() is None,
                        'restart_count': self.restart_counts.get(name, 0)
                    }
                    for name, process in self.processes.items()
                }
                
                status_report = {
                    'timestamp': time.time(),
                    'health': health_status,
                    'processes': process_status,
                    'all_healthy': all(health_status.values())
                }
                
                self.logger.info(f"System status: {json.dumps(status_report, indent=2)}")
                
                # Alert if any service is unhealthy
                if not all(health_status.values()):
                    unhealthy_services = [
                        name for name, healthy in health_status.items() 
                        if not healthy
                    ]
                    self.logger.error(f"Unhealthy services detected: {unhealthy_services}")
                
            except Exception as e:
                self.logger.error(f"Error in system status logging: {e}")
            
            time.sleep(60)  # Log status every minute
    
    def stop_all(self):
        """Stop all processes gracefully"""
        self.logger.info("Stopping all processes...")
        self.running = False
        
        for name, process in self.processes.items():
            if process.poll() is None:
                self.logger.info(f"Stopping {name} (PID {process.pid})")
                
                # Send SIGTERM for graceful shutdown
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=30)
                    self.logger.info(f"{name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Force killing {name}")
                    process.kill()
                    process.wait()
        
        self.processes.clear()
        self.logger.info("All processes stopped")
    
    def start_all(self):
        """Start all components with production configuration"""
        self.logger.info("Starting AI Navigation Assistant in production mode...")
        
        try:
            # Create necessary directories
            self.create_directories()
            
            # Note: Monitoring is handled by the backend server itself
            
            # Start all servers
            self.start_backend()
            time.sleep(5)  # Give backend more time to start in production
            
            self.start_frontend()
            self.start_client()
            
            self.running = True
            
            # Print startup information
            self.print_startup_info()
            
            # Start background monitoring tasks
            import threading
            monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
            status_thread = threading.Thread(target=self.log_system_status, daemon=True)
            
            monitor_thread.start()
            status_thread.start()
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Error during startup: {e}")
        finally:
            self.stop_all()
    
    def create_directories(self):
        """Create necessary directories for production"""
        directories = [
            "logs",
            "backend/models",
            "backend/logs",
            "data",
            "backups"
        ]
        
        for directory in directories:
            dir_path = Path(__file__).parent.parent / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            # Set appropriate permissions for production
            os.chmod(dir_path, 0o755)
            self.logger.debug(f"Created directory: {dir_path}")
    
    def print_startup_info(self):
        """Print production startup information"""
        print("\n" + "="*60)
        print("AI Navigation Assistant - Production Mode")
        print("="*60)
        print(f"Backend:  http://0.0.0.0:8000")
        print(f"Frontend: http://0.0.0.0:3000")
        print(f"Client:   http://0.0.0.0:3001")
        print(f"Health:   http://0.0.0.0:8000/health")
        print("="*60)
        print("Production server started. Check logs for details.")
        print("Use 'kill -TERM <pid>' to stop gracefully")
        print("="*60 + "\n")


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    sys.exit(0)


def main():
    """Main entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start production server
    server = ProductionServer()
    server.start_all()


if __name__ == "__main__":
    main()