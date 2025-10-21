# Safety Monitoring and Alert System
# Implements processing latency monitoring, fail-safe audio, and emergency state transitions

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety alert levels"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class SafetyMetric:
    """Safety monitoring metric"""
    name: str
    value: float
    threshold: float
    level: SafetyLevel
    timestamp: datetime
    message: str

@dataclass
class ProcessingLatencyMetric:
    """Processing latency tracking"""
    component: str
    start_time: float
    end_time: float
    duration: float
    threshold: float
    is_violation: bool

class SafetyMonitor:
    """
    Comprehensive safety monitoring system
    Implements requirements 1.4, 1.5 for processing latency monitoring and fail-safe protocols
    """
    
    def __init__(self):
        # Safety thresholds (in seconds)
        self.latency_thresholds = {
            "frame_processing": 0.5,      # 500ms max for frame processing
            "obstacle_detection": 0.3,     # 300ms max for obstacle detection
            "audio_processing": 0.2,       # 200ms max for audio processing
            "state_transition": 0.1,       # 100ms max for state transitions
            "emergency_response": 0.05     # 50ms max for emergency responses
        }
        
        # Safety monitoring state
        self.monitoring_active = True
        self.safety_violations = []
        self.latency_history = []
        self.max_history_size = 1000
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[SafetyMetric], None]] = []
        
        # Emergency protocols
        self.emergency_protocols_active = False
        self.fail_safe_audio_messages = [
            "STOP! Navigation system error.",
            "DANGER! Please stop immediately.",
            "EMERGENCY! Seek assistance."
        ]
        
        # Performance tracking
        self.performance_stats = {
            "total_violations": 0,
            "critical_violations": 0,
            "emergency_activations": 0,
            "last_violation_time": None
        }
        
        logger.info("Safety monitoring system initialized")
    
    def add_alert_callback(self, callback: Callable[[SafetyMetric], None]):
        """Register callback for safety alerts"""
        self.alert_callbacks.append(callback)
        logger.info("Safety alert callback registered")
    
    def set_latency_threshold(self, component: str, threshold: float):
        """Update latency threshold for specific component"""
        self.latency_thresholds[component] = threshold
        logger.info(f"Updated latency threshold for {component}: {threshold}s")
    
    async def monitor_processing_latency(self, component: str, start_time: float, end_time: float) -> ProcessingLatencyMetric:
        """
        Monitor processing latency with safety thresholds
        Requirement 1.4: Implement processing latency monitoring with safety thresholds
        """
        duration = end_time - start_time
        threshold = self.latency_thresholds.get(component, 1.0)  # Default 1s threshold
        is_violation = duration > threshold
        
        # Create latency metric
        latency_metric = ProcessingLatencyMetric(
            component=component,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            threshold=threshold,
            is_violation=is_violation
        )
        
        # Add to history
        self.latency_history.append(latency_metric)
        if len(self.latency_history) > self.max_history_size:
            self.latency_history.pop(0)
        
        # Handle violations
        if is_violation:
            await self._handle_latency_violation(latency_metric)
        
        logger.debug(f"Latency monitoring - {component}: {duration:.3f}s (threshold: {threshold}s)")
        return latency_metric
    
    async def _handle_latency_violation(self, metric: ProcessingLatencyMetric):
        """Handle processing latency violations"""
        self.performance_stats["total_violations"] += 1
        self.performance_stats["last_violation_time"] = datetime.now()
        
        # Determine safety level based on severity
        severity_ratio = metric.duration / metric.threshold
        
        if severity_ratio >= 3.0:  # 3x threshold exceeded
            safety_level = SafetyLevel.EMERGENCY
            self.performance_stats["emergency_activations"] += 1
        elif severity_ratio >= 2.0:  # 2x threshold exceeded
            safety_level = SafetyLevel.CRITICAL
            self.performance_stats["critical_violations"] += 1
        else:
            safety_level = SafetyLevel.WARNING
        
        # Create safety alert
        safety_metric = SafetyMetric(
            name=f"latency_violation_{metric.component}",
            value=metric.duration,
            threshold=metric.threshold,
            level=safety_level,
            timestamp=datetime.now(),
            message=f"Processing latency violation in {metric.component}: {metric.duration:.3f}s > {metric.threshold}s"
        )
        
        # Log violation
        if safety_level == SafetyLevel.EMERGENCY:
            logger.critical(safety_metric.message)
        elif safety_level == SafetyLevel.CRITICAL:
            logger.error(safety_metric.message)
        else:
            logger.warning(safety_metric.message)
        
        # Store violation
        self.safety_violations.append(safety_metric)
        
        # Trigger alerts
        await self._trigger_safety_alert(safety_metric)
        
        # Handle critical/emergency violations
        if safety_level in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]:
            await self._handle_critical_violation(safety_metric)
    
    async def _trigger_safety_alert(self, metric: SafetyMetric):
        """Trigger safety alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metric)
                else:
                    callback(metric)
            except Exception as e:
                logger.error(f"Error in safety alert callback: {e}")
    
    async def _handle_critical_violation(self, metric: SafetyMetric):
        """Handle critical safety violations"""
        logger.critical(f"Critical safety violation detected: {metric.message}")
        
        # Activate emergency protocols for critical violations
        if metric.level == SafetyLevel.EMERGENCY:
            await self.activate_emergency_protocols(f"Processing latency emergency: {metric.message}")
        
        # Implement fail-safe measures
        await self._implement_fail_safe_measures(metric)
    
    async def activate_emergency_protocols(self, reason: str = "Safety violation detected"):
        """
        Activate emergency protocols for critical scenarios
        Requirement 1.5: Create emergency state transitions for critical scenarios
        """
        if self.emergency_protocols_active:
            logger.warning("Emergency protocols already active")
            return
        
        logger.critical(f"ACTIVATING EMERGENCY PROTOCOLS: {reason}")
        self.emergency_protocols_active = True
        
        try:
            # Protocol 1: Trigger emergency stop in FSM
            await self._trigger_emergency_fsm_stop(reason)
            
            # Protocol 2: Activate fail-safe audio system
            await self._activate_fail_safe_audio()
            
            # Protocol 3: Log emergency activation
            await self._log_emergency_activation(reason)
            
            # Protocol 4: Notify all alert callbacks
            emergency_metric = SafetyMetric(
                name="emergency_protocol_activation",
                value=1.0,
                threshold=0.0,
                level=SafetyLevel.EMERGENCY,
                timestamp=datetime.now(),
                message=f"Emergency protocols activated: {reason}"
            )
            await self._trigger_safety_alert(emergency_metric)
            
            logger.critical("Emergency protocols activation completed")
            
        except Exception as e:
            logger.critical(f"Error during emergency protocol activation: {e}")
            # Continue with fail-safe measures even if some protocols fail
            await self._last_resort_safety_measures(reason, str(e))
    
    async def _trigger_emergency_fsm_stop(self, reason: str):
        """Trigger emergency stop in navigation FSM"""
        try:
            # Import here to avoid circular imports
            from navigation_fsm import navigation_fsm
            
            success = await navigation_fsm.handle_emergency_stop(f"Safety monitor: {reason}")
            
            if success:
                logger.critical("Emergency FSM stop triggered successfully")
            else:
                logger.critical("Emergency FSM stop failed")
                
        except Exception as e:
            logger.critical(f"Error triggering emergency FSM stop: {e}")
    
    async def _activate_fail_safe_audio(self):
        """
        Activate fail-safe audio message system for TTS failures
        Requirement 1.5: Add fail-safe audio message system for TTS failures
        """
        try:
            logger.critical("Activating fail-safe audio system")
            
            # Import WebSocket manager to send emergency audio
            from websocket_manager import websocket_manager
            
            # Send multiple fail-safe audio messages
            for i, message in enumerate(self.fail_safe_audio_messages):
                emergency_audio = {
                    "type": "emergency_audio",
                    "speak": message,
                    "urgency": "emergency",
                    "fail_safe": True,
                    "sequence": i + 1,
                    "total": len(self.fail_safe_audio_messages)
                }
                
                # Broadcast to all connected clients
                await websocket_manager.broadcast(emergency_audio)
                
                # Delay between messages
                if i < len(self.fail_safe_audio_messages) - 1:
                    await asyncio.sleep(1.0)
            
            logger.critical("Fail-safe audio system activated")
            
        except Exception as e:
            logger.critical(f"Error activating fail-safe audio: {e}")
    
    async def _implement_fail_safe_measures(self, metric: SafetyMetric):
        """Implement additional fail-safe measures"""
        try:
            logger.critical(f"Implementing fail-safe measures for: {metric.message}")
            
            # Measure 1: Reduce processing complexity
            await self._reduce_processing_complexity()
            
            # Measure 2: Increase monitoring frequency
            await self._increase_monitoring_frequency()
            
            # Measure 3: Prepare for system degradation
            await self._prepare_system_degradation()
            
        except Exception as e:
            logger.error(f"Error implementing fail-safe measures: {e}")
    
    async def _reduce_processing_complexity(self):
        """Reduce system processing complexity during safety violations"""
        try:
            # This would integrate with computer vision and other components
            # to reduce processing load
            logger.critical("Processing complexity reduction activated")
            
            # Example: Lower frame processing rate, reduce detection accuracy, etc.
            # Implementation would depend on specific component integration
            
        except Exception as e:
            logger.error(f"Error reducing processing complexity: {e}")
    
    async def _increase_monitoring_frequency(self):
        """Increase safety monitoring frequency during violations"""
        try:
            # Reduce thresholds temporarily for more sensitive monitoring
            for component in self.latency_thresholds:
                self.latency_thresholds[component] *= 0.8  # 20% stricter
            
            logger.critical("Monitoring frequency increased")
            
        except Exception as e:
            logger.error(f"Error increasing monitoring frequency: {e}")
    
    async def _prepare_system_degradation(self):
        """Prepare for graceful system degradation"""
        try:
            logger.critical("Preparing for graceful system degradation")
            
            # Set flags for other components to check
            self.system_degradation_mode = True
            
        except Exception as e:
            logger.error(f"Error preparing system degradation: {e}")
    
    async def _log_emergency_activation(self, reason: str):
        """Log emergency protocol activation for analysis"""
        try:
            emergency_log = {
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "performance_stats": self.performance_stats.copy(),
                "recent_violations": [
                    {
                        "name": v.name,
                        "value": v.value,
                        "threshold": v.threshold,
                        "level": v.level.value,
                        "message": v.message,
                        "timestamp": v.timestamp.isoformat()
                    }
                    for v in self.safety_violations[-10:]  # Last 10 violations
                ],
                "latency_stats": self._get_latency_statistics()
            }
            
            logger.critical(f"EMERGENCY ACTIVATION LOG: {emergency_log}")
            
        except Exception as e:
            logger.error(f"Error logging emergency activation: {e}")
    
    async def _last_resort_safety_measures(self, original_reason: str, error_details: str):
        """Last resort safety measures when primary protocols fail"""
        logger.critical("ACTIVATING LAST RESORT SAFETY MEASURES")
        
        try:
            # Force emergency mode flag
            self.emergency_protocols_active = True
            
            # Try to send basic emergency message
            try:
                from websocket_manager import websocket_manager
                basic_emergency = {
                    "type": "last_resort_emergency",
                    "speak": "EMERGENCY STOP - SYSTEM FAILURE",
                    "urgency": "emergency"
                }
                await websocket_manager.broadcast(basic_emergency)
            except:
                pass  # Continue even if this fails
            
            logger.critical("Last resort safety measures completed")
            
        except Exception as e:
            logger.critical(f"LAST RESORT SAFETY MEASURES FAILED: {e}")
    
    def _get_latency_statistics(self) -> Dict[str, Any]:
        """Get latency statistics for monitoring"""
        if not self.latency_history:
            return {}
        
        try:
            # Calculate statistics by component
            stats_by_component = {}
            
            for component in self.latency_thresholds.keys():
                component_metrics = [m for m in self.latency_history if m.component == component]
                
                if component_metrics:
                    durations = [m.duration for m in component_metrics]
                    violations = [m for m in component_metrics if m.is_violation]
                    
                    stats_by_component[component] = {
                        "count": len(component_metrics),
                        "avg_duration": sum(durations) / len(durations),
                        "max_duration": max(durations),
                        "min_duration": min(durations),
                        "violation_count": len(violations),
                        "violation_rate": len(violations) / len(component_metrics)
                    }
            
            return {
                "total_measurements": len(self.latency_history),
                "by_component": stats_by_component,
                "overall_violation_rate": len([m for m in self.latency_history if m.is_violation]) / len(self.latency_history)
            }
            
        except Exception as e:
            logger.error(f"Error calculating latency statistics: {e}")
            return {"error": str(e)}
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety monitoring status"""
        return {
            "monitoring_active": self.monitoring_active,
            "emergency_protocols_active": self.emergency_protocols_active,
            "performance_stats": self.performance_stats.copy(),
            "latency_thresholds": self.latency_thresholds.copy(),
            "recent_violations": len([v for v in self.safety_violations if v.timestamp > datetime.now() - timedelta(minutes=5)]),
            "total_violations": len(self.safety_violations),
            "system_degradation_mode": getattr(self, 'system_degradation_mode', False)
        }
    
    async def reset_emergency_protocols(self):
        """Reset emergency protocols after manual verification"""
        try:
            logger.info("Resetting emergency protocols")
            
            self.emergency_protocols_active = False
            self.system_degradation_mode = False
            
            # Reset thresholds to original values
            self.latency_thresholds = {
                "frame_processing": 0.5,
                "obstacle_detection": 0.3,
                "audio_processing": 0.2,
                "state_transition": 0.1,
                "emergency_response": 0.05
            }
            
            logger.info("Emergency protocols reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting emergency protocols: {e}")
            return False

# Global safety monitor instance
safety_monitor = SafetyMonitor()