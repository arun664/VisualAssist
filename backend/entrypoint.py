#!/usr/bin/env python3
"""
Entrypoint script for AI Navigation Assistant Backend
Provides graceful startup with error handling and fallback modes
"""

import sys
import os
import time
import traceback
import logging

# Set up basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def safe_import(module_name, required=True):
    """Safely import a module with error handling"""
    try:
        return __import__(module_name)
    except ImportError as e:
        if required:
            logger.error(f"Required module {module_name} could not be imported: {e}")
            return None
        else:
            logger.warning(f"Optional module {module_name} could not be imported: {e}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error importing {module_name}: {e}")
        return None

def create_minimal_app():
    """Create a minimal FastAPI app for health checks when full app fails"""
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        app = FastAPI(title="AI Navigation Assistant Backend (Minimal Mode)")
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
        
        @app.get("/")
        async def root():
            return {
                "message": "AI Navigation Assistant Backend (Minimal Mode)",
                "status": "running",
                "mode": "minimal",
                "version": "1.0.0"
            }
        
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "mode": "minimal",
                "message": "Running in minimal mode due to initialization issues"
            }
        
        logger.info("Created minimal FastAPI application")
        return app
        
    except Exception as e:
        logger.error(f"Failed to create minimal app: {e}")
        return None

def start_minimal_server():
    """Start the server in minimal mode"""
    try:
        import uvicorn
        
        app = create_minimal_app()
        if app is None:
            logger.error("Could not create minimal application")
            return False
        
        host = os.getenv('SERVER_HOST', '0.0.0.0')
        # Railway uses PORT environment variable, fallback to SERVER_PORT then default
        port = int(os.getenv('PORT', os.getenv('SERVER_PORT', 8000)))
        
        logger.info(f"Starting minimal server on {host}:{port}")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to start minimal server: {e}")
        return False

def start_full_application():
    """Start the full application"""
    try:
        logger.info("Attempting to start full application...")
        
        # Run startup check first
        startup_check = safe_import('startup_check', required=False)
        if startup_check:
            if not startup_check.main():
                logger.warning("Startup checks failed, but continuing...")
        
        # Import main application
        main_module = safe_import('main', required=True)
        if main_module is None:
            logger.error("Could not import main application module")
            return False
        
        # Check if we can access the main components
        if not hasattr(main_module, 'app'):
            logger.error("Main module does not have 'app' attribute")
            return False
        
        # Start the server
        import uvicorn
        
        host = os.getenv('SERVER_HOST', '0.0.0.0')
        # Railway uses PORT environment variable, fallback to SERVER_PORT then default
        port = int(os.getenv('PORT', os.getenv('SERVER_PORT', 8000)))
        environment = os.getenv('ENVIRONMENT', 'production')
        
        logger.info(f"Starting full application on {host}:{port} in {environment} mode")
        
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=False,
            log_level="info",
            workers=1
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to start full application: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entrypoint function"""
    logger.info("=== AI Navigation Assistant Backend - Starting ===")
    
    # Print environment info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    
    # Check if we're in Docker
    if os.path.exists('/.dockerenv'):
        logger.info("Running in Docker container")
    
    # Try to start full application first
    logger.info("Attempting to start full application...")
    if start_full_application():
        logger.info("Full application started successfully")
        return 0
    
    # Fallback to minimal mode
    logger.warning("Full application failed to start, falling back to minimal mode...")
    if start_minimal_server():
        logger.info("Minimal server started successfully")
        return 0
    
    # If everything fails
    logger.error("Both full and minimal modes failed to start")
    return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)