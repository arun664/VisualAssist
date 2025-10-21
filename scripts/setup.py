#!/usr/bin/env python3
"""
Setup script for AI Navigation Assistant
Handles initial setup, dependency installation, and configuration
"""

import os
import sys
import subprocess
import shutil
import urllib.request
import zipfile
import argparse
from pathlib import Path
from typing import List, Optional
import logging


class SetupManager:
    """Manages the setup process for AI Navigation Assistant"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for setup process"""
        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        self.logger.info("Checking Python version...")
        
        if sys.version_info < (3, 8):
            self.logger.error(f"Python 3.8+ required, found {sys.version}")
            return False
        
        self.logger.info(f"Python version {sys.version} is compatible")
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        self.logger.info("Creating directory structure...")
        
        directories = [
            "logs",
            "backend/logs",
            "backend/models",
            "data",
            "backups"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {dir_path}")
        
        self.logger.info("Directory structure created successfully")
    
    def setup_backend_environment(self):
        """Set up Python virtual environment for backend"""
        self.logger.info("Setting up backend Python environment...")
        
        backend_path = self.project_root / "backend"
        venv_path = backend_path / "venv"
        
        # Create virtual environment
        if not venv_path.exists():
            self.logger.info("Creating Python virtual environment...")
            subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], check=True)
        else:
            self.logger.info("Virtual environment already exists")
        
        # Determine activation script path
        if os.name == 'nt':  # Windows
            activate_script = venv_path / "Scripts" / "activate.bat"
            pip_executable = venv_path / "Scripts" / "pip.exe"
        else:  # Unix-like
            activate_script = venv_path / "bin" / "activate"
            pip_executable = venv_path / "bin" / "pip"
        
        # Install requirements
        requirements_file = backend_path / "requirements.txt"
        if requirements_file.exists():
            self.logger.info("Installing Python dependencies...")
            subprocess.run([
                str(pip_executable), "install", "-r", str(requirements_file)
            ], check=True)
            
            # Install additional production dependencies
            self.logger.info("Installing production dependencies...")
            subprocess.run([
                str(pip_executable), "install", "gunicorn", "psutil"
            ], check=True)
        else:
            self.logger.warning("requirements.txt not found, skipping dependency installation")
        
        self.logger.info("Backend environment setup completed")
    
    def download_ai_models(self):
        """Download required AI models"""
        self.logger.info("Downloading AI models...")
        
        models_dir = self.project_root / "backend" / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Download Vosk model
        vosk_model_dir = models_dir / "vosk-model-en-us-0.22"
        if not vosk_model_dir.exists():
            self.logger.info("Downloading Vosk speech recognition model...")
            vosk_url = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"
            vosk_zip = models_dir / "vosk-model.zip"
            
            try:
                urllib.request.urlretrieve(vosk_url, vosk_zip)
                
                with zipfile.ZipFile(vosk_zip, 'r') as zip_ref:
                    zip_ref.extractall(models_dir)
                
                vosk_zip.unlink()  # Remove zip file
                self.logger.info("Vosk model downloaded successfully")
                
            except Exception as e:
                self.logger.warning(f"Failed to download Vosk model: {e}")
                self.logger.info("You can download it manually from: " + vosk_url)
        else:
            self.logger.info("Vosk model already exists")
        
        # YOLOv11 model will be downloaded automatically by ultralytics on first use
        self.logger.info("YOLOv11 model will be downloaded automatically on first use")
        
        self.logger.info("AI models setup completed")
    
    def setup_configuration_files(self):
        """Set up configuration files"""
        self.logger.info("Setting up configuration files...")
        
        # Copy example environment files if they don't exist
        env_files = [
            ("backend/.env.example", "backend/.env"),
            ("backend/.env.development", "backend/.env.dev"),
            ("backend/.env.production", "backend/.env.prod")
        ]
        
        for source, target in env_files:
            source_path = self.project_root / source
            target_path = self.project_root / target
            
            if source_path.exists() and not target_path.exists():
                shutil.copy2(source_path, target_path)
                self.logger.debug(f"Copied {source} to {target}")
        
        self.logger.info("Configuration files setup completed")
    
    def check_system_dependencies(self) -> List[str]:
        """Check for required system dependencies"""
        self.logger.info("Checking system dependencies...")
        
        dependencies = {
            "git": "git --version",
            "python3": "python3 --version",
            "pip3": "pip3 --version"
        }
        
        missing = []
        
        for name, command in dependencies.items():
            try:
                subprocess.run(command.split(), 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL, 
                             check=True)
                self.logger.debug(f"Found {name}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing.append(name)
                self.logger.warning(f"Missing dependency: {name}")
        
        if missing:
            self.logger.warning(f"Missing system dependencies: {', '.join(missing)}")
        else:
            self.logger.info("All system dependencies are available")
        
        return missing
    
    def setup_git_hooks(self):
        """Set up Git hooks for development"""
        self.logger.info("Setting up Git hooks...")
        
        git_dir = self.project_root / ".git"
        if not git_dir.exists():
            self.logger.warning("Not a Git repository, skipping Git hooks setup")
            return
        
        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        
        # Pre-commit hook for code quality
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/bash
# Pre-commit hook for AI Navigation Assistant

echo "Running pre-commit checks..."

# Check Python syntax
echo "Checking Python syntax..."
find backend -name "*.py" -exec python3 -m py_compile {} \\;
if [ $? -ne 0 ]; then
    echo "Python syntax errors found. Commit aborted."
    exit 1
fi

# Run health check
echo "Running health check..."
python3 scripts/health_check.py --url http://localhost:8000 > /dev/null 2>&1
if [ $? -eq 2 ]; then
    echo "Critical health check failures detected. Consider fixing before commit."
fi

echo "Pre-commit checks completed."
"""
        
        with open(pre_commit_hook, 'w') as f:
            f.write(pre_commit_content)
        
        # Make executable
        if os.name != 'nt':  # Not Windows
            os.chmod(pre_commit_hook, 0o755)
        
        self.logger.info("Git hooks setup completed")
    
    def create_startup_shortcuts(self):
        """Create startup shortcuts/scripts"""
        self.logger.info("Creating startup shortcuts...")
        
        # Make shell scripts executable (Unix-like systems)
        if os.name != 'nt':
            shell_scripts = [
                "scripts/start.sh"
            ]
            
            for script in shell_scripts:
                script_path = self.project_root / script
                if script_path.exists():
                    os.chmod(script_path, 0o755)
                    self.logger.debug(f"Made {script} executable")
        
        self.logger.info("Startup shortcuts created")
    
    def run_initial_health_check(self):
        """Run initial health check"""
        self.logger.info("Running initial health check...")
        
        try:
            health_check_script = self.project_root / "scripts" / "health_check.py"
            if health_check_script.exists():
                result = subprocess.run([
                    sys.executable, str(health_check_script), "--verbose"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.logger.info("Initial health check passed")
                else:
                    self.logger.warning("Initial health check found issues")
                    if self.verbose:
                        self.logger.info(f"Health check output:\n{result.stdout}")
            else:
                self.logger.warning("Health check script not found")
                
        except Exception as e:
            self.logger.warning(f"Failed to run initial health check: {e}")
    
    def print_setup_summary(self):
        """Print setup summary and next steps"""
        print("\n" + "="*60)
        print("AI Navigation Assistant Setup Complete!")
        print("="*60)
        print("\nNext Steps:")
        print("1. Review configuration files in backend/.env*")
        print("2. Start the development environment:")
        print("   python scripts/start_development.py")
        print("   # OR")
        print("   ./scripts/start.sh")
        print("   # OR (Windows)")
        print("   scripts\\start.bat")
        print("\n3. Access the application:")
        print("   Backend:  http://localhost:8000")
        print("   Frontend: http://localhost:3000")
        print("   Client:   http://localhost:3001")
        print("   API Docs: http://localhost:8000/docs")
        print("\n4. Run health checks:")
        print("   python scripts/health_check.py")
        print("\n5. For production deployment, see DEPLOYMENT.md")
        print("="*60 + "\n")
    
    def run_full_setup(self):
        """Run the complete setup process"""
        self.logger.info("Starting AI Navigation Assistant setup...")
        
        try:
            # Check prerequisites
            if not self.check_python_version():
                return False
            
            missing_deps = self.check_system_dependencies()
            if missing_deps:
                self.logger.error(f"Please install missing dependencies: {', '.join(missing_deps)}")
                return False
            
            # Run setup steps
            self.create_directories()
            self.setup_configuration_files()
            self.setup_backend_environment()
            self.download_ai_models()
            self.setup_git_hooks()
            self.create_startup_shortcuts()
            
            # Final checks
            self.run_initial_health_check()
            
            self.logger.info("Setup completed successfully!")
            self.print_setup_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            if self.verbose:
                import traceback
                self.logger.error(traceback.format_exc())
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Navigation Assistant Setup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-models", action="store_true", help="Skip AI model downloads")
    parser.add_argument("--skip-venv", action="store_true", help="Skip virtual environment setup")
    parser.add_argument("--development-only", action="store_true", help="Setup for development only")
    
    args = parser.parse_args()
    
    setup_manager = SetupManager(verbose=args.verbose)
    
    if args.development_only:
        setup_manager.logger.info("Running development-only setup...")
        setup_manager.create_directories()
        setup_manager.setup_configuration_files()
        if not args.skip_venv:
            setup_manager.setup_backend_environment()
        if not args.skip_models:
            setup_manager.download_ai_models()
        setup_manager.create_startup_shortcuts()
        setup_manager.print_setup_summary()
    else:
        success = setup_manager.run_full_setup()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()