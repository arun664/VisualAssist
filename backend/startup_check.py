#!/usr/bin/env python3
"""
Startup check script for AI Navigation Assistant Backend
Verifies all dependencies and components before starting the main application
"""

import sys
import importlib
import traceback
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_dependencies():
    """Check if all required Python packages are available"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'websockets',
        'cv2',  # opencv-python-headless
        'ultralytics',
        'vosk',
        'aiortc',
        'numpy',
        'PIL',  # Pillow
        'requests',
        'pydantic',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package} - OK")
        except ImportError as e:
            logger.error(f"✗ {package} - MISSING: {e}")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        return False
    
    logger.info("All required packages are available")
    return True

def check_application_modules():
    """Check if all application modules can be imported"""
    app_modules = [
        'config_manager',
        'logging_config',
        'websocket_manager',
        'webrtc_handler',
        'navigation_fsm',
        'computer_vision',
        'speech_recognition',
        'safety_monitor',
        'monitoring',
        'workflow_coordinator'
    ]
    
    missing_modules = []
    
    for module in app_modules:
        try:
            importlib.import_module(module)
            logger.info(f"✓ {module} - OK")
        except ImportError as e:
            logger.error(f"✗ {module} - MISSING: {e}")
            missing_modules.append(module)
        except Exception as e:
            logger.warning(f"⚠ {module} - WARNING: {e}")
    
    if missing_modules:
        logger.error(f"Missing application modules: {missing_modules}")
        return False
    
    logger.info("All application modules are available")
    return True

def check_yolo_model():
    """Check if YOLO model can be loaded"""
    try:
        from ultralytics import YOLO
        logger.info("Attempting to load YOLO model...")
        model = YOLO('yolo11n.pt')
        logger.info("✓ YOLO model loaded successfully")
        return True
    except Exception as e:
        logger.warning(f"⚠ YOLO model check failed: {e}")
        logger.info("The application will attempt to download the model at runtime")
        return True  # Not critical for startup

def check_environment():
    """Check environment configuration"""
    import os
    
    env_vars = {
        'PYTHONPATH': os.getenv('PYTHONPATH', 'Not set'),
        'ENVIRONMENT': os.getenv('ENVIRONMENT', 'development'),
        'SERVER_HOST': os.getenv('SERVER_HOST', '0.0.0.0'),
        'SERVER_PORT': os.getenv('SERVER_PORT', '8000')
    }
    
    logger.info("Environment configuration:")
    for key, value in env_vars.items():
        logger.info(f"  {key}: {value}")
    
    return True

def main():
    """Main startup check function"""
    logger.info("Starting AI Navigation Assistant Backend - Startup Check")
    
    checks = [
        ("Python Dependencies", check_python_dependencies),
        ("Application Modules", check_application_modules),
        ("YOLO Model", check_yolo_model),
        ("Environment", check_environment)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        logger.info(f"\n--- Checking {check_name} ---")
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            logger.error(f"Check {check_name} failed with exception: {e}")
            logger.error(traceback.format_exc())
            all_passed = False
    
    if all_passed:
        logger.info("\n✓ All startup checks passed! Starting main application...")
        return True
    else:
        logger.error("\n✗ Some startup checks failed. Please fix the issues before starting the application.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)