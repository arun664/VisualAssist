#!/usr/bin/env python3
"""
Health check script for AI Navigation Assistant
Performs comprehensive system health checks and reports status
"""

import sys
import time
import json
import asyncio
import aiohttp
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Add backend to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from logging_config import setup_environment_logging


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: str  # "healthy", "warning", "critical", "unknown"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    response_time: Optional[float] = None


class HealthChecker:
    """Comprehensive health checker for AI Navigation Assistant"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.logger = setup_environment_logging("development", "health_checker")
        self.results: List[HealthCheckResult] = []
        
    async def check_backend_api(self) -> HealthCheckResult:
        """Check backend API health"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health", timeout=10) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        return HealthCheckResult(
                            name="backend_api",
                            status="healthy",
                            message="Backend API is responding normally",
                            details=data,
                            response_time=response_time
                        )
                    else:
                        return HealthCheckResult(
                            name="backend_api",
                            status="critical",
                            message=f"Backend API returned status {response.status}",
                            response_time=response_time
                        )
                        
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name="backend_api",
                status="critical",
                message="Backend API request timed out",
                response_time=time.time() - start_time
            )
        except Exception as e:
            return HealthCheckResult(
                name="backend_api",
                status="critical",
                message=f"Backend API check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def check_websocket_connection(self) -> HealthCheckResult:
        """Check WebSocket connection"""
        start_time = time.time()
        
        try:
            import websockets
            
            ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
            
            async with websockets.connect(ws_url, timeout=10) as websocket:
                # Send a test message
                test_message = json.dumps({"type": "ping", "timestamp": time.time()})
                await websocket.send(test_message)
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                response_time = time.time() - start_time
                
                return HealthCheckResult(
                    name="websocket_connection",
                    status="healthy",
                    message="WebSocket connection is working",
                    details={"response": response},
                    response_time=response_time
                )
                
        except ImportError:
            return HealthCheckResult(
                name="websocket_connection",
                status="warning",
                message="websockets library not available for testing"
            )
        except Exception as e:
            return HealthCheckResult(
                name="websocket_connection",
                status="critical",
                message=f"WebSocket connection failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def check_frontend_availability(self) -> HealthCheckResult:
        """Check frontend server availability"""
        start_time = time.time()
        
        try:
            frontend_url = "http://localhost:3000"
            async with aiohttp.ClientSession() as session:
                async with session.get(frontend_url, timeout=10) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        return HealthCheckResult(
                            name="frontend_server",
                            status="healthy",
                            message="Frontend server is accessible",
                            response_time=response_time
                        )
                    else:
                        return HealthCheckResult(
                            name="frontend_server",
                            status="warning",
                            message=f"Frontend server returned status {response.status}",
                            response_time=response_time
                        )
                        
        except Exception as e:
            return HealthCheckResult(
                name="frontend_server",
                status="critical",
                message=f"Frontend server check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def check_client_availability(self) -> HealthCheckResult:
        """Check client server availability"""
        start_time = time.time()
        
        try:
            client_url = "http://localhost:3001"
            async with aiohttp.ClientSession() as session:
                async with session.get(client_url, timeout=10) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        return HealthCheckResult(
                            name="client_server",
                            status="healthy",
                            message="Client server is accessible",
                            response_time=response_time
                        )
                    else:
                        return HealthCheckResult(
                            name="client_server",
                            status="warning",
                            message=f"Client server returned status {response.status}",
                            response_time=response_time
                        )
                        
        except Exception as e:
            return HealthCheckResult(
                name="client_server",
                status="critical",
                message=f"Client server check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage"""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }
            
            # Determine status based on resource usage
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 95:
                status = "critical"
                message = "System resources are critically high"
            elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 85:
                status = "warning"
                message = "System resources are elevated"
            else:
                status = "healthy"
                message = "System resources are within normal limits"
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                details=details
            )
            
        except ImportError:
            return HealthCheckResult(
                name="system_resources",
                status="warning",
                message="psutil library not available for system monitoring"
            )
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status="unknown",
                message=f"System resource check failed: {str(e)}"
            )
    
    async def check_required_files(self) -> HealthCheckResult:
        """Check for required files and directories"""
        required_files = [
            "backend/main.py",
            "backend/requirements.txt",
            "frontend/index.html",
            "client/index.html"
        ]
        
        required_dirs = [
            "backend",
            "frontend", 
            "client",
            "scripts"
        ]
        
        missing_files = []
        missing_dirs = []
        
        project_root = Path(__file__).parent.parent
        
        # Check files
        for file_path in required_files:
            if not (project_root / file_path).exists():
                missing_files.append(file_path)
        
        # Check directories
        for dir_path in required_dirs:
            if not (project_root / dir_path).is_dir():
                missing_dirs.append(dir_path)
        
        details = {
            "missing_files": missing_files,
            "missing_directories": missing_dirs,
            "checked_files": len(required_files),
            "checked_directories": len(required_dirs)
        }
        
        if missing_files or missing_dirs:
            return HealthCheckResult(
                name="required_files",
                status="critical",
                message=f"Missing {len(missing_files)} files and {len(missing_dirs)} directories",
                details=details
            )
        else:
            return HealthCheckResult(
                name="required_files",
                status="healthy",
                message="All required files and directories are present",
                details=details
            )
    
    async def check_python_dependencies(self) -> HealthCheckResult:
        """Check Python dependencies"""
        required_packages = [
            "fastapi",
            "uvicorn",
            "websockets",
            "opencv-python",
            "ultralytics",
            "vosk",
            "numpy",
            "aiortc"
        ]
        
        missing_packages = []
        installed_packages = {}
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                # Try to get version info
                try:
                    import pkg_resources
                    version = pkg_resources.get_distribution(package).version
                    installed_packages[package] = version
                except:
                    installed_packages[package] = "unknown"
            except ImportError:
                missing_packages.append(package)
        
        details = {
            "missing_packages": missing_packages,
            "installed_packages": installed_packages,
            "total_required": len(required_packages)
        }
        
        if missing_packages:
            return HealthCheckResult(
                name="python_dependencies",
                status="critical",
                message=f"Missing {len(missing_packages)} required Python packages",
                details=details
            )
        else:
            return HealthCheckResult(
                name="python_dependencies",
                status="healthy",
                message="All required Python packages are installed",
                details=details
            )
    
    async def check_ai_models(self) -> HealthCheckResult:
        """Check AI model availability"""
        model_checks = []
        
        # Check YOLO model
        yolo_model_path = Path("backend/yolov11n.pt")
        if yolo_model_path.exists():
            model_checks.append({"name": "YOLOv11", "status": "available", "path": str(yolo_model_path)})
        else:
            model_checks.append({"name": "YOLOv11", "status": "missing", "path": str(yolo_model_path)})
        
        # Check Vosk model
        vosk_model_path = Path("backend/models/vosk-model-en-us-0.22")
        if vosk_model_path.exists():
            model_checks.append({"name": "Vosk", "status": "available", "path": str(vosk_model_path)})
        else:
            model_checks.append({"name": "Vosk", "status": "missing", "path": str(vosk_model_path)})
        
        missing_models = [check for check in model_checks if check["status"] == "missing"]
        
        details = {
            "model_checks": model_checks,
            "missing_models": len(missing_models),
            "total_models": len(model_checks)
        }
        
        if missing_models:
            return HealthCheckResult(
                name="ai_models",
                status="warning",
                message=f"{len(missing_models)} AI models are missing (will be downloaded on first use)",
                details=details
            )
        else:
            return HealthCheckResult(
                name="ai_models",
                status="healthy",
                message="All AI models are available",
                details=details
            )
    
    async def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all health checks"""
        self.logger.info("Starting comprehensive health check...")
        
        checks = [
            self.check_backend_api(),
            self.check_websocket_connection(),
            self.check_frontend_availability(),
            self.check_client_availability(),
            self.check_system_resources(),
            self.check_required_files(),
            self.check_python_dependencies(),
            self.check_ai_models()
        ]
        
        self.results = await asyncio.gather(*checks, return_exceptions=True)
        
        # Handle any exceptions
        for i, result in enumerate(self.results):
            if isinstance(result, Exception):
                self.results[i] = HealthCheckResult(
                    name=f"check_{i}",
                    status="unknown",
                    message=f"Health check failed with exception: {str(result)}"
                )
        
        self.logger.info(f"Completed {len(self.results)} health checks")
        return self.results
    
    def get_overall_status(self) -> str:
        """Get overall system health status"""
        if not self.results:
            return "unknown"
        
        statuses = [result.status for result in self.results]
        
        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"
    
    def print_results(self, verbose: bool = False):
        """Print health check results"""
        overall_status = self.get_overall_status()
        
        # Status colors
        colors = {
            "healthy": "\033[92m",    # Green
            "warning": "\033[93m",    # Yellow
            "critical": "\033[91m",   # Red
            "unknown": "\033[94m"     # Blue
        }
        reset_color = "\033[0m"
        
        print("\n" + "="*60)
        print("AI Navigation Assistant - Health Check Report")
        print("="*60)
        print(f"Overall Status: {colors.get(overall_status, '')}{overall_status.upper()}{reset_color}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Checks Performed: {len(self.results)}")
        print("="*60)
        
        for result in self.results:
            color = colors.get(result.status, "")
            status_symbol = {
                "healthy": "✓",
                "warning": "⚠",
                "critical": "✗",
                "unknown": "?"
            }.get(result.status, "?")
            
            print(f"{color}{status_symbol} {result.name.replace('_', ' ').title()}: {result.message}{reset_color}")
            
            if result.response_time:
                print(f"  Response Time: {result.response_time:.3f}s")
            
            if verbose and result.details:
                print(f"  Details: {json.dumps(result.details, indent=4)}")
            
            print()
        
        # Summary
        status_counts = {}
        for result in self.results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        print("Summary:")
        for status, count in status_counts.items():
            color = colors.get(status, "")
            print(f"  {color}{status.title()}: {count}{reset_color}")
        
        print("="*60 + "\n")
    
    def save_results_to_file(self, file_path: str):
        """Save health check results to JSON file"""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": self.get_overall_status(),
            "checks": [
                {
                    "name": result.name,
                    "status": result.status,
                    "message": result.message,
                    "details": result.details,
                    "timestamp": result.timestamp.isoformat(),
                    "response_time": result.response_time
                }
                for result in self.results
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Health check results saved to {file_path}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Navigation Assistant Health Check")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend URL to check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--continuous", "-c", type=int, help="Run checks continuously every N seconds")
    
    args = parser.parse_args()
    
    checker = HealthChecker(args.url)
    
    if args.continuous:
        print(f"Running health checks continuously every {args.continuous} seconds...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                await checker.run_all_checks()
                checker.print_results(args.verbose)
                
                if args.output:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"{args.output}_{timestamp}.json"
                    checker.save_results_to_file(output_file)
                
                await asyncio.sleep(args.continuous)
                
        except KeyboardInterrupt:
            print("\nHealth check monitoring stopped.")
    else:
        # Single run
        await checker.run_all_checks()
        checker.print_results(args.verbose)
        
        if args.output:
            checker.save_results_to_file(args.output)
        
        # Exit with appropriate code
        overall_status = checker.get_overall_status()
        if overall_status == "critical":
            sys.exit(2)
        elif overall_status == "warning":
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())