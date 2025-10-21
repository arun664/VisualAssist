"""
Monitoring and metrics collection for AI Navigation Assistant Backend
Provides comprehensive monitoring of system performance, health, and safety metrics
"""

import time
import asyncio
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import logging

from logging_config import get_performance_logger, log_safety_event


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Metric:
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class Alert:
    name: str
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        self.logger = get_performance_logger("monitoring")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        with self.lock:
            self.counters[name] += value
            metric = Metric(name, self.counters[name], MetricType.COUNTER, labels=labels or {})
            self.metrics[name].append(metric)
            self.logger.debug(f"Counter {name} incremented to {self.counters[name]}")
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value"""
        with self.lock:
            self.gauges[name] = value
            metric = Metric(name, value, MetricType.GAUGE, labels=labels or {})
            self.metrics[name].append(metric)
            self.logger.debug(f"Gauge {name} set to {value}")
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a value in a histogram"""
        with self.lock:
            self.histograms[name].append(value)
            metric = Metric(name, value, MetricType.HISTOGRAM, labels=labels or {})
            self.metrics[name].append(metric)
            self.logger.debug(f"Histogram {name} recorded value {value}")
    
    def start_timer(self, name: str) -> float:
        """Start a timer and return the start time"""
        start_time = time.time()
        self.logger.debug(f"Timer {name} started")
        return start_time
    
    def end_timer(self, name: str, start_time: float, labels: Dict[str, str] = None):
        """End a timer and record the duration"""
        duration = time.time() - start_time
        with self.lock:
            self.timers[name].append(duration)
            metric = Metric(name, duration, MetricType.TIMER, labels=labels or {}, unit="seconds")
            self.metrics[name].append(metric)
            self.logger.debug(f"Timer {name} completed in {duration:.3f}s")
        return duration
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        with self.lock:
            if name in self.counters:
                return {"type": "counter", "value": self.counters[name]}
            elif name in self.gauges:
                return {"type": "gauge", "value": self.gauges[name]}
            elif name in self.histograms:
                values = list(self.histograms[name])
                if values:
                    return {
                        "type": "histogram",
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "recent": values[-10:]  # Last 10 values
                    }
            elif name in self.timers:
                values = self.timers[name]
                if values:
                    return {
                        "type": "timer",
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "recent": values[-10:]  # Last 10 timers
                    }
        return {"type": "unknown", "value": None}
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {k: list(v) for k, v in self.histograms.items()},
                "timers": dict(self.timers),
                "timestamp": datetime.now().isoformat()
            }
    
    def clear_metrics(self):
        """Clear all metrics (useful for testing)"""
        with self.lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()
            self.logger.info("All metrics cleared")


class SystemMonitor:
    """Monitors system resources and performance"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = logging.getLogger("monitoring.system")
        self.monitoring_active = False
        self.monitoring_task = None
        self.monitoring_interval = 5.0  # seconds
    
    async def start_monitoring(self):
        """Start system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics.set_gauge("system_cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.set_gauge("system_memory_percent", memory.percent)
            self.metrics.set_gauge("system_memory_available_mb", memory.available / 1024 / 1024)
            self.metrics.set_gauge("system_memory_used_mb", memory.used / 1024 / 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics.set_gauge("system_disk_percent", disk.percent)
            self.metrics.set_gauge("system_disk_free_gb", disk.free / 1024 / 1024 / 1024)
            
            # Network metrics
            network = psutil.net_io_counters()
            self.metrics.set_gauge("system_network_bytes_sent", network.bytes_sent)
            self.metrics.set_gauge("system_network_bytes_recv", network.bytes_recv)
            
            # Process-specific metrics
            process = psutil.Process()
            self.metrics.set_gauge("process_cpu_percent", process.cpu_percent())
            self.metrics.set_gauge("process_memory_mb", process.memory_info().rss / 1024 / 1024)
            self.metrics.set_gauge("process_threads", process.num_threads())
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.logger = logging.getLogger("monitoring.alerts")
        self.lock = threading.Lock()
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)
    
    def set_threshold(self, metric_name: str, level: AlertLevel, threshold: float):
        """Set alert threshold for a metric"""
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = {}
        self.thresholds[metric_name][level.value] = threshold
        self.logger.info(f"Set {level.value} threshold for {metric_name}: {threshold}")
    
    def check_thresholds(self):
        """Check all metrics against their thresholds"""
        current_metrics = self.metrics.get_all_metrics()
        
        for metric_name, thresholds in self.thresholds.items():
            current_value = None
            
            # Get current value from appropriate metric type
            if metric_name in current_metrics["gauges"]:
                current_value = current_metrics["gauges"][metric_name]
            elif metric_name in current_metrics["counters"]:
                current_value = current_metrics["counters"][metric_name]
            
            if current_value is not None:
                self._check_metric_thresholds(metric_name, current_value, thresholds)
    
    def _check_metric_thresholds(self, metric_name: str, value: float, thresholds: Dict[str, float]):
        """Check a specific metric against its thresholds"""
        for level_str, threshold in thresholds.items():
            level = AlertLevel(level_str)
            
            # Check if threshold is exceeded
            if value > threshold:
                # Check if we already have an active alert for this
                existing_alert = self._find_active_alert(metric_name, level)
                
                if not existing_alert:
                    alert = Alert(
                        name=f"{metric_name}_threshold_exceeded",
                        level=level,
                        message=f"{metric_name} value {value} exceeds {level.value} threshold {threshold}",
                        component="monitoring"
                    )
                    self._trigger_alert(alert)
            else:
                # Check if we need to resolve an existing alert
                existing_alert = self._find_active_alert(metric_name, level)
                if existing_alert:
                    self._resolve_alert(existing_alert)
    
    def _find_active_alert(self, metric_name: str, level: AlertLevel) -> Optional[Alert]:
        """Find an active alert for a metric and level"""
        with self.lock:
            for alert in self.alerts:
                if (not alert.resolved and 
                    alert.name == f"{metric_name}_threshold_exceeded" and 
                    alert.level == level):
                    return alert
        return None
    
    def _trigger_alert(self, alert: Alert):
        """Trigger a new alert"""
        with self.lock:
            self.alerts.append(alert)
        
        self.logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
        log_safety_event(self.logger, "threshold_exceeded", alert.level.value, alert.message)
        
        # Call all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def _resolve_alert(self, alert: Alert):
        """Resolve an existing alert"""
        alert.resolved = True
        alert.resolution_time = datetime.now()
        
        self.logger.info(f"Alert resolved: {alert.name}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def get_all_alerts(self) -> List[Alert]:
        """Get all alerts"""
        with self.lock:
            return self.alerts.copy()
    
    def clear_resolved_alerts(self):
        """Clear all resolved alerts"""
        with self.lock:
            self.alerts = [alert for alert in self.alerts if not alert.resolved]
        self.logger.info("Cleared resolved alerts")


class HealthChecker:
    """Performs health checks on system components"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = logging.getLogger("monitoring.health")
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.last_check_results: Dict[str, bool] = {}
    
    def register_health_check(self, name: str, check_function: Callable[[], bool]):
        """Register a health check function"""
        self.health_checks[name] = check_function
        self.logger.info(f"Registered health check: {name}")
    
    async def run_health_checks(self) -> Dict[str, bool]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_function in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_function()
                end_time = time.time()
                
                results[name] = result
                self.last_check_results[name] = result
                
                # Record health check metrics
                self.metrics.set_gauge(f"health_check_{name}", 1.0 if result else 0.0)
                self.metrics.record_histogram(f"health_check_{name}_duration", end_time - start_time)
                
                if not result:
                    self.logger.warning(f"Health check failed: {name}")
                else:
                    self.logger.debug(f"Health check passed: {name}")
                    
            except Exception as e:
                self.logger.error(f"Error running health check {name}: {e}")
                results[name] = False
                self.last_check_results[name] = False
                self.metrics.set_gauge(f"health_check_{name}", 0.0)
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        return {
            "healthy": all(self.last_check_results.values()),
            "checks": self.last_check_results.copy(),
            "timestamp": datetime.now().isoformat()
        }


class MonitoringManager:
    """Main monitoring manager that coordinates all monitoring components"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemMonitor(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.health_checker = HealthChecker(self.metrics_collector)
        self.logger = logging.getLogger("monitoring.manager")
        
        # Set up default thresholds
        self._setup_default_thresholds()
        
        # Set up default health checks
        self._setup_default_health_checks()
    
    def _setup_default_thresholds(self):
        """Set up default alert thresholds"""
        # CPU thresholds
        self.alert_manager.set_threshold("system_cpu_percent", AlertLevel.MEDIUM, 70.0)
        self.alert_manager.set_threshold("system_cpu_percent", AlertLevel.HIGH, 85.0)
        self.alert_manager.set_threshold("system_cpu_percent", AlertLevel.CRITICAL, 95.0)
        
        # Memory thresholds
        self.alert_manager.set_threshold("system_memory_percent", AlertLevel.MEDIUM, 70.0)
        self.alert_manager.set_threshold("system_memory_percent", AlertLevel.HIGH, 85.0)
        self.alert_manager.set_threshold("system_memory_percent", AlertLevel.CRITICAL, 95.0)
        
        # Disk thresholds
        self.alert_manager.set_threshold("system_disk_percent", AlertLevel.MEDIUM, 80.0)
        self.alert_manager.set_threshold("system_disk_percent", AlertLevel.HIGH, 90.0)
        self.alert_manager.set_threshold("system_disk_percent", AlertLevel.CRITICAL, 95.0)
    
    def _setup_default_health_checks(self):
        """Set up default health checks"""
        def check_system_resources():
            """Check if system resources are within acceptable limits"""
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent
                
                return cpu_percent < 90 and memory_percent < 90 and disk_percent < 95
            except:
                return False
        
        def check_process_health():
            """Check if the current process is healthy"""
            try:
                process = psutil.Process()
                return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            except:
                return False
        
        self.health_checker.register_health_check("system_resources", check_system_resources)
        self.health_checker.register_health_check("process_health", check_process_health)
    
    async def start_monitoring(self):
        """Start all monitoring components"""
        await self.system_monitor.start_monitoring()
        self.logger.info("Monitoring manager started")
    
    async def stop_monitoring(self):
        """Stop all monitoring components"""
        await self.system_monitor.stop_monitoring()
        self.logger.info("Monitoring manager stopped")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        return {
            "metrics": self.metrics_collector.get_all_metrics(),
            "alerts": {
                "active": len(self.alert_manager.get_active_alerts()),
                "total": len(self.alert_manager.get_all_alerts()),
                "recent": [
                    {
                        "name": alert.name,
                        "level": alert.level.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in self.alert_manager.get_active_alerts()
                ]
            },
            "health": self.health_checker.get_health_status(),
            "system_monitoring": self.system_monitor.monitoring_active
        }


# Global monitoring instance
monitoring_manager = MonitoringManager()