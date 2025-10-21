"""
Logging configuration for AI Navigation Assistant Backend
Provides structured logging with different levels and outputs for development and production
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': os.getpid(),
            'thread_id': record.thread
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'component'):
            log_entry['component'] = record.component
        if hasattr(record, 'client_id'):
            log_entry['client_id'] = record.client_id
        if hasattr(record, 'processing_time'):
            log_entry['processing_time'] = record.processing_time
        if hasattr(record, 'fsm_state'):
            log_entry['fsm_state'] = record.fsm_state
        
        return json.dumps(log_entry)


class ComponentFilter(logging.Filter):
    """Filter to add component information to log records"""
    
    def __init__(self, component_name: str):
        super().__init__()
        self.component_name = component_name
    
    def filter(self, record):
        record.component = self.component_name
        return True


def setup_logging(
    environment: str = "development",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json_logging: bool = False,
    component_name: str = "backend"
) -> logging.Logger:
    """
    Set up comprehensive logging configuration
    
    Args:
        environment: development or production
        log_level: logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: path to log file (optional)
        enable_json_logging: whether to use JSON formatting
        component_name: name of the component for filtering
    
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Choose formatter based on environment and preferences
    if enable_json_logging or environment == "production":
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(component)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(ComponentFilter(component_name))
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Rotating file handler to prevent log files from growing too large
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        
        if enable_json_logging:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(component)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(ComponentFilter(component_name))
        root_logger.addHandler(file_handler)
    
    # Error file handler (separate file for errors and above)
    if log_file:
        error_log_file = log_file.replace('.log', '_errors.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        
        if enable_json_logging:
            error_formatter = JSONFormatter()
        else:
            error_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(component)s - %(funcName)s:%(lineno)d - %(message)s\n%(exc_info)s'
            )
        
        error_handler.setFormatter(error_formatter)
        error_handler.addFilter(ComponentFilter(component_name))
        root_logger.addHandler(error_handler)
    
    # Create component-specific logger
    logger = logging.getLogger(component_name)
    
    # Log startup information
    logger.info(f"Logging initialized for {component_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"JSON logging: {enable_json_logging}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return logger


def get_performance_logger(component_name: str) -> logging.Logger:
    """Get a logger specifically for performance metrics"""
    perf_logger = logging.getLogger(f"{component_name}.performance")
    return perf_logger


def get_security_logger(component_name: str) -> logging.Logger:
    """Get a logger specifically for security events"""
    security_logger = logging.getLogger(f"{component_name}.security")
    return security_logger


def get_audit_logger(component_name: str) -> logging.Logger:
    """Get a logger specifically for audit events"""
    audit_logger = logging.getLogger(f"{component_name}.audit")
    return audit_logger


def log_processing_time(logger: logging.Logger, operation: str, start_time: float, end_time: float, **kwargs):
    """Log processing time for operations"""
    processing_time = end_time - start_time
    extra_data = {
        'processing_time': processing_time,
        'operation': operation,
        **kwargs
    }
    
    if processing_time > 0.1:  # Log slow operations as warnings
        logger.warning(f"Slow operation: {operation} took {processing_time:.3f}s", extra=extra_data)
    else:
        logger.debug(f"Operation: {operation} completed in {processing_time:.3f}s", extra=extra_data)


def log_fsm_transition(logger: logging.Logger, from_state: str, to_state: str, trigger: str, **kwargs):
    """Log FSM state transitions"""
    extra_data = {
        'fsm_state': to_state,
        'previous_state': from_state,
        'transition_trigger': trigger,
        **kwargs
    }
    
    logger.info(f"FSM transition: {from_state} -> {to_state} (trigger: {trigger})", extra=extra_data)


def log_safety_event(logger: logging.Logger, event_type: str, severity: str, message: str, **kwargs):
    """Log safety-related events"""
    extra_data = {
        'safety_event_type': event_type,
        'severity': severity,
        **kwargs
    }
    
    if severity in ['critical', 'high']:
        logger.error(f"SAFETY EVENT [{severity.upper()}]: {message}", extra=extra_data)
    elif severity == 'medium':
        logger.warning(f"Safety event [{severity}]: {message}", extra=extra_data)
    else:
        logger.info(f"Safety event [{severity}]: {message}", extra=extra_data)


def log_client_connection(logger: logging.Logger, client_id: str, event: str, **kwargs):
    """Log client connection events"""
    extra_data = {
        'client_id': client_id,
        'connection_event': event,
        **kwargs
    }
    
    logger.info(f"Client {client_id}: {event}", extra=extra_data)


# Environment-specific logging configurations
LOGGING_CONFIGS = {
    'development': {
        'log_level': 'DEBUG',
        'enable_json_logging': False,
        'log_file': 'logs/development.log'
    },
    'production': {
        'log_level': 'INFO',
        'enable_json_logging': True,
        'log_file': 'logs/production.log'
    },
    'testing': {
        'log_level': 'WARNING',
        'enable_json_logging': False,
        'log_file': 'logs/testing.log'
    }
}


def setup_environment_logging(environment: str = "development", component_name: str = "backend") -> logging.Logger:
    """Set up logging based on environment configuration"""
    config = LOGGING_CONFIGS.get(environment, LOGGING_CONFIGS['development'])
    
    return setup_logging(
        environment=environment,
        component_name=component_name,
        **config
    )