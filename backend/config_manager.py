"""
Configuration management for AI Navigation Assistant Backend
Handles environment-specific configuration loading and validation
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Set up logger for this module
logger = logging.getLogger(__name__)


class Environment(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class ServerConfig:
    """Server configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    log_level: str = "info"
    worker_processes: int = 1
    worker_connections: int = 1000
    keepalive_timeout: int = 5


@dataclass
class YOLOConfig:
    """YOLO model configuration settings for pretrained models"""
    model_name: str = "yolo11n.pt"  # Pretrained model from ultralytics
    confidence_threshold: float = 0.5
    device: str = "cpu"
    
    def __post_init__(self):
        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("YOLO confidence threshold must be between 0.0 and 1.0")
        
        # Validate model name
        valid_models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
        if self.model_name not in valid_models:
            logger.warning(f"Model {self.model_name} not in recommended list: {valid_models}")
    
    @property
    def model_path(self):
        """Backward compatibility property"""
        return self.model_name


@dataclass
class VoskConfig:
    """Vosk speech recognition configuration settings"""
    model_path: str = "./models/vosk-model-en-us-0.22"
    sample_rate: int = 16000
    
    def __post_init__(self):
        # Validate sample rate
        if self.sample_rate not in [8000, 16000, 44100, 48000]:
            raise ValueError("Vosk sample rate must be one of: 8000, 16000, 44100, 48000")


@dataclass
class WebRTCConfig:
    """WebRTC configuration settings"""
    ice_servers: str = "stun:stun.l.google.com:19302"
    timeout: int = 30
    
    def get_ice_servers_list(self) -> list:
        """Convert ice_servers string to list format"""
        return [{"urls": server.strip()} for server in self.ice_servers.split(",")]


@dataclass
class OpticalFlowConfig:
    """Optical flow analysis configuration settings"""
    threshold: float = 2.0
    stationary_threshold: float = 1.0
    motion_detection_frames: int = 5
    
    def __post_init__(self):
        # Validate thresholds
        if self.threshold < 0:
            raise ValueError("Optical flow threshold must be non-negative")
        if self.stationary_threshold < 0:
            raise ValueError("Stationary threshold must be non-negative")
        if self.motion_detection_frames < 1:
            raise ValueError("Motion detection frames must be at least 1")


@dataclass
class SafetyConfig:
    """Safety monitoring configuration settings"""
    max_processing_latency_ms: int = 100
    emergency_stop_enabled: bool = True
    safety_monitoring_enabled: bool = True
    safety_alert_threshold: int = 3
    
    def __post_init__(self):
        # Validate latency threshold
        if self.max_processing_latency_ms < 1:
            raise ValueError("Max processing latency must be at least 1ms")
        if self.safety_alert_threshold < 1:
            raise ValueError("Safety alert threshold must be at least 1")


@dataclass
class ComputerVisionConfig:
    """Computer vision processing configuration settings"""
    grid_rows: int = 6
    grid_cols: int = 8
    safety_margin: float = 0.2
    frame_skip: int = 0
    
    def __post_init__(self):
        # Validate grid dimensions
        if self.grid_rows < 1 or self.grid_cols < 1:
            raise ValueError("Grid dimensions must be at least 1x1")
        if not 0.0 <= self.safety_margin <= 1.0:
            raise ValueError("Safety margin must be between 0.0 and 1.0")
        if self.frame_skip < 0:
            raise ValueError("Frame skip must be non-negative")


@dataclass
class AudioConfig:
    """Audio processing configuration settings"""
    buffer_size: int = 1024
    sample_rate: int = 16000
    channels: int = 1
    
    def __post_init__(self):
        # Validate audio settings
        if self.buffer_size < 64 or self.buffer_size > 8192:
            raise ValueError("Audio buffer size must be between 64 and 8192")
        if self.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError("Audio sample rate must be one of: 8000, 16000, 22050, 44100, 48000")
        if self.channels not in [1, 2]:
            raise ValueError("Audio channels must be 1 (mono) or 2 (stereo)")


@dataclass
class WebSocketConfig:
    """WebSocket configuration settings"""
    ping_interval: int = 20
    ping_timeout: int = 10
    close_timeout: int = 10
    
    def __post_init__(self):
        # Validate timeouts
        if self.ping_interval < 1:
            raise ValueError("WebSocket ping interval must be at least 1 second")
        if self.ping_timeout < 1:
            raise ValueError("WebSocket ping timeout must be at least 1 second")
        if self.close_timeout < 1:
            raise ValueError("WebSocket close timeout must be at least 1 second")


@dataclass
class CORSConfig:
    """CORS configuration settings"""
    origins: str = "*"
    allow_credentials: bool = True
    
    def get_origins_list(self) -> list:
        """Convert origins string to list format"""
        if self.origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.origins.split(",")]


@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_size: str = "10MB"
    backup_count: int = 5
    
    def get_max_size_bytes(self) -> int:
        """Convert max_size string to bytes"""
        size_str = self.max_size.upper()
        if size_str.endswith("KB"):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)


@dataclass
class ApplicationConfig:
    """Main application configuration container"""
    environment: Environment = Environment.DEVELOPMENT
    server: ServerConfig = field(default_factory=ServerConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    vosk: VoskConfig = field(default_factory=VoskConfig)
    webrtc: WebRTCConfig = field(default_factory=WebRTCConfig)
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    computer_vision: ComputerVisionConfig = field(default_factory=ComputerVisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    cors: CORSConfig = field(default_factory=CORSConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment.value,
            "server": self.server.__dict__,
            "yolo": self.yolo.__dict__,
            "vosk": self.vosk.__dict__,
            "webrtc": self.webrtc.__dict__,
            "optical_flow": self.optical_flow.__dict__,
            "safety": self.safety.__dict__,
            "computer_vision": self.computer_vision.__dict__,
            "audio": self.audio.__dict__,
            "websocket": self.websocket.__dict__,
            "cors": self.cors.__dict__,
            "logging": self.logging.__dict__
        }


class ConfigManager:
    """Manages application configuration loading and validation"""
    
    def __init__(self, environment: Optional[str] = None):
        self.environment = Environment(environment or os.getenv("ENVIRONMENT", "development"))
        self.config: Optional[ApplicationConfig] = None
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_file: Optional[str] = None) -> ApplicationConfig:
        """Load configuration from environment variables and files"""
        
        # Start with default configuration
        config = ApplicationConfig(environment=self.environment)
        
        # Load from environment file if it exists
        env_file = self._get_env_file_path()
        if env_file.exists():
            self._load_from_env_file(env_file, config)
        
        # Load from custom config file if provided
        if config_file and Path(config_file).exists():
            self._load_from_config_file(config_file, config)
        
        # Override with environment variables
        self._load_from_environment_variables(config)
        
        # Validate configuration
        self._validate_config(config)
        
        self.config = config
        self.logger.info(f"Configuration loaded for {self.environment.value} environment")
        
        return config
    
    def _get_env_file_path(self) -> Path:
        """Get the path to the environment-specific .env file"""
        return Path(__file__).parent / f".env.{self.environment.value}"
    
    def _load_from_env_file(self, env_file: Path, config: ApplicationConfig):
        """Load configuration from .env file"""
        self.logger.debug(f"Loading configuration from {env_file}")
        
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
                    except ValueError:
                        self.logger.warning(f"Invalid line in env file: {line}")
    
    def _load_from_config_file(self, config_file: str, config: ApplicationConfig):
        """Load configuration from JSON config file"""
        self.logger.debug(f"Loading configuration from {config_file}")
        
        with open(config_file) as f:
            config_data = json.load(f)
        
        # Apply configuration data to config object
        self._apply_config_data(config_data, config)
    
    def _load_from_environment_variables(self, config: ApplicationConfig):
        """Load configuration from environment variables"""
        
        # Server configuration
        config.server.host = os.getenv("HOST", config.server.host)
        config.server.port = int(os.getenv("PORT", config.server.port))
        config.server.debug = os.getenv("DEBUG", "false").lower() == "true"
        config.server.reload = os.getenv("RELOAD", "false").lower() == "true"
        config.server.log_level = os.getenv("LOG_LEVEL", config.server.log_level)
        config.server.worker_processes = int(os.getenv("WORKER_PROCESSES", config.server.worker_processes))
        config.server.worker_connections = int(os.getenv("WORKER_CONNECTIONS", config.server.worker_connections))
        config.server.keepalive_timeout = int(os.getenv("KEEPALIVE_TIMEOUT", config.server.keepalive_timeout))
        
        # YOLO configuration
        config.yolo.model_name = os.getenv("YOLO_MODEL_PATH", config.yolo.model_name)
        config.yolo.confidence_threshold = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", config.yolo.confidence_threshold))
        config.yolo.device = os.getenv("YOLO_DEVICE", config.yolo.device)
        
        # Vosk configuration
        config.vosk.model_path = os.getenv("VOSK_MODEL_PATH", config.vosk.model_path)
        config.vosk.sample_rate = int(os.getenv("VOSK_SAMPLE_RATE", config.vosk.sample_rate))
        
        # WebRTC configuration
        config.webrtc.ice_servers = os.getenv("WEBRTC_ICE_SERVERS", config.webrtc.ice_servers)
        config.webrtc.timeout = int(os.getenv("WEBRTC_TIMEOUT", config.webrtc.timeout))
        
        # Optical flow configuration
        config.optical_flow.threshold = float(os.getenv("OPTICAL_FLOW_THRESHOLD", config.optical_flow.threshold))
        config.optical_flow.stationary_threshold = float(os.getenv("STATIONARY_THRESHOLD", config.optical_flow.stationary_threshold))
        config.optical_flow.motion_detection_frames = int(os.getenv("MOTION_DETECTION_FRAMES", config.optical_flow.motion_detection_frames))
        
        # Safety configuration
        config.safety.max_processing_latency_ms = int(os.getenv("MAX_PROCESSING_LATENCY_MS", config.safety.max_processing_latency_ms))
        config.safety.emergency_stop_enabled = os.getenv("EMERGENCY_STOP_ENABLED", "true").lower() == "true"
        config.safety.safety_monitoring_enabled = os.getenv("SAFETY_MONITORING_ENABLED", "true").lower() == "true"
        config.safety.safety_alert_threshold = int(os.getenv("SAFETY_ALERT_THRESHOLD", config.safety.safety_alert_threshold))
        
        # Computer vision configuration
        config.computer_vision.grid_rows = int(os.getenv("CV_GRID_ROWS", config.computer_vision.grid_rows))
        config.computer_vision.grid_cols = int(os.getenv("CV_GRID_COLS", config.computer_vision.grid_cols))
        config.computer_vision.safety_margin = float(os.getenv("CV_SAFETY_MARGIN", config.computer_vision.safety_margin))
        config.computer_vision.frame_skip = int(os.getenv("CV_FRAME_SKIP", config.computer_vision.frame_skip))
        
        # Audio configuration
        config.audio.buffer_size = int(os.getenv("AUDIO_BUFFER_SIZE", config.audio.buffer_size))
        config.audio.sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", config.audio.sample_rate))
        config.audio.channels = int(os.getenv("AUDIO_CHANNELS", config.audio.channels))
        
        # WebSocket configuration
        config.websocket.ping_interval = int(os.getenv("WS_PING_INTERVAL", config.websocket.ping_interval))
        config.websocket.ping_timeout = int(os.getenv("WS_PING_TIMEOUT", config.websocket.ping_timeout))
        config.websocket.close_timeout = int(os.getenv("WS_CLOSE_TIMEOUT", config.websocket.close_timeout))
        
        # CORS configuration
        config.cors.origins = os.getenv("CORS_ORIGINS", config.cors.origins)
        config.cors.allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
        
        # Logging configuration
        config.logging.format = os.getenv("LOG_FORMAT", config.logging.format)
        config.logging.file = os.getenv("LOG_FILE", config.logging.file)
        config.logging.max_size = os.getenv("LOG_MAX_SIZE", config.logging.max_size)
        config.logging.backup_count = int(os.getenv("LOG_BACKUP_COUNT", config.logging.backup_count))
    
    def _apply_config_data(self, config_data: Dict[str, Any], config: ApplicationConfig):
        """Apply configuration data from dictionary to config object"""
        
        for section_name, section_data in config_data.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section_obj = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def _validate_config(self, config: ApplicationConfig):
        """Validate configuration settings"""
        
        # Validate that required files exist
        if not Path(config.yolo.model_path).exists() and not config.yolo.model_path.startswith("yolo"):
            self.logger.warning(f"YOLO model file not found: {config.yolo.model_path}")
        
        if not Path(config.vosk.model_path).exists():
            self.logger.warning(f"Vosk model directory not found: {config.vosk.model_path}")
        
        # Validate numeric ranges
        if config.server.port < 1 or config.server.port > 65535:
            raise ValueError("Server port must be between 1 and 65535")
        
        # Log configuration summary
        self.logger.info(f"Configuration validation completed for {config.environment.value}")
    
    def get_config(self) -> ApplicationConfig:
        """Get the current configuration"""
        if self.config is None:
            self.config = self.load_config()
        return self.config
    
    def reload_config(self) -> ApplicationConfig:
        """Reload configuration from sources"""
        self.logger.info("Reloading configuration...")
        return self.load_config()
    
    def save_config_to_file(self, file_path: str):
        """Save current configuration to a JSON file"""
        if self.config is None:
            raise ValueError("No configuration loaded")
        
        with open(file_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        self.logger.info(f"Configuration saved to {file_path}")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> ApplicationConfig:
    """Get the global application configuration"""
    return config_manager.get_config()


def reload_config() -> ApplicationConfig:
    """Reload the global application configuration"""
    return config_manager.reload_config()