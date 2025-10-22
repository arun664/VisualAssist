#!/usr/bin/env python3
"""
Diagnostic script to identify issues with backend server startup
"""

import sys
import traceback
import importlib
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_ultralytics():
    """Specifically check if ultralytics package can be imported and model can be loaded"""
    try:
        logger.info("Attempting to import ultralytics...")
        import ultralytics
        logger.info(f"✓ ultralytics imported successfully - version: {ultralytics.__version__}")
        
        logger.info("Checking if YOLO model can be loaded...")
        from ultralytics import YOLO
        # Try to load model with verbose mode to see what's happening
        logger.info("Attempting to initialize YOLO model...")
        model = YOLO("yolo11n.pt", verbose=True)
        logger.info("✓ YOLO model initialized successfully")
        
        # Try a basic inference to verify model works
        import numpy as np
        logger.info("Testing model with a dummy image...")
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(test_image)
        logger.info("✓ Model inference successful")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error with ultralytics/YOLO: {e}")
        logger.error(traceback.format_exc())
        return False

def check_computer_vision_module():
    """Check if computer_vision module loads correctly"""
    try:
        logger.info("Attempting to import computer_vision module...")
        import computer_vision
        logger.info("✓ computer_vision module imported successfully")
        
        logger.info("Trying to get vision processor instance...")
        vision_proc = computer_vision.get_vision_processor()
        logger.info("✓ Vision processor instance created successfully")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error with computer_vision module: {e}")
        logger.error(traceback.format_exc())
        return False

def check_torch_cuda():
    """Check if PyTorch CUDA is available and working"""
    try:
        logger.info("Checking PyTorch and CUDA availability...")
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        return True
    except Exception as e:
        logger.error(f"✗ Error checking PyTorch/CUDA: {e}")
        logger.error(traceback.format_exc())
        return False

def check_main_app_startup():
    """Try to import and initialize key components from main app"""
    main_logger = logging.getLogger(__name__)
    try:
        main_logger.info("Attempting to import main app components...")
        
        # Import key modules without running the full app
        from config_manager import get_config
        from logging_config import setup_environment_logging
        from computer_vision import get_vision_processor
        
        main_logger.info("Getting configuration...")
        config = get_config()
        main_logger.info(f"✓ Configuration loaded successfully - environment: {config.environment.value}")
        
        main_logger.info("Setting up logging...")
        app_logger = setup_environment_logging(config.environment.value, "diagnostic")
        main_logger.info("✓ Logging setup successfully")
        
        main_logger.info("Trying to initialize vision processor...")
        vision_processor = get_vision_processor()
        main_logger.info("✓ Vision processor initialized successfully")
        
        return True
    except Exception as e:
        main_logger.error(f"✗ Error in main app startup: {e}")
        main_logger.error(traceback.format_exc())
        return False

def main():
    """Run all diagnostic checks"""
    logger.info("=== Starting Backend Server Diagnostic Checks ===")
    
    checks = [
        ("PyTorch and CUDA", check_torch_cuda),
        ("Ultralytics Package", check_ultralytics),
        ("Computer Vision Module", check_computer_vision_module),
        ("Main Application Startup", check_main_app_startup)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        logger.info(f"\n--- Running {check_name} check ---")
        try:
            result = check_func()
            results[check_name] = "PASSED" if result else "FAILED"
        except Exception as e:
            logger.error(f"Check {check_name} failed with unexpected exception: {e}")
            logger.error(traceback.format_exc())
            results[check_name] = "ERROR"
    
    # Print summary
    logger.info("\n=== Diagnostic Results Summary ===")
    all_passed = True
    for check_name, status in results.items():
        logger.info(f"{check_name}: {status}")
        if status != "PASSED":
            all_passed = False
    
    if all_passed:
        logger.info("\n✓ All diagnostic checks passed!")
        return True
    else:
        logger.error("\n✗ Some diagnostic checks failed. Please address the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)