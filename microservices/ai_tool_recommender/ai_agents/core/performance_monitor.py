"""Performance monitoring and optimization utilities."""

from contextlib import asynccontextmanager
import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        """Initialize performance monitor with empty metrics."""
        self.metrics = {
            "total_requests": 0,
            "total_time": 0.0,
            "pinecone_time": 0.0,
            "internet_time": 0.0,
            "validation_time": 0.0,
            "average_response_time": 0.0,
        }

    @asynccontextmanager
    async def time_operation(self, operation_name: str):
        """Context manager to time operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics[f"{operation_name}_time"] = duration
            logger.info(f"{operation_name} took {duration:.2f}s")

    def record_request(self, total_time: float):
        """Record a complete request."""
        self.metrics["total_requests"] += 1
        self.metrics["total_time"] += total_time
        self.metrics["average_response_time"] = (
            self.metrics["total_time"] / self.metrics["total_requests"]
        )

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        return {
            **self.metrics,
            "performance_status": self._get_performance_status(),
            "optimization_suggestions": self._get_optimization_suggestions(),
        }

    def _get_performance_status(self) -> str:
        """Get performance status."""
        avg_time = self.metrics["average_response_time"]
        if avg_time <= 2.0:
            return "excellent"
        elif avg_time <= 4.0:
            return "good"
        elif avg_time <= 6.0:
            return "acceptable"
        else:
            return "needs_optimization"

    def _get_optimization_suggestions(self) -> list:
        """Get optimization suggestions."""
        suggestions = []

        if self.metrics["pinecone_time"] > 1.5:
            suggestions.append("Consider reducing Pinecone max_results")

        if self.metrics["internet_time"] > 3.0:
            suggestions.append("Consider reducing internet search queries")

        if self.metrics["validation_time"] > 1.0:
            suggestions.append("Consider reducing validation complexity")

        return suggestions


class SpeedOptimizer:
    """Optimize system performance based on metrics."""

    def __init__(self, monitor: PerformanceMonitor):
        """Initialize speed optimizer with performance monitor."""
        self.monitor = monitor
        self.optimization_level = "balanced"  # balanced, fast, ultra_fast

    def get_optimized_config(self) -> Dict[str, Any]:
        """Get optimized configuration based on performance."""
        report = self.monitor.get_performance_report()

        if report["performance_status"] == "needs_optimization":
            return self._get_ultra_fast_config()
        elif report["performance_status"] == "acceptable":
            return self._get_fast_config()
        else:
            return self._get_balanced_config()

    def _get_ultra_fast_config(self) -> Dict[str, Any]:
        """Ultra-fast configuration."""
        return {
            "pinecone_max_results": 5,
            "internet_max_results": 4,
            "internet_queries": 1,
            "validation_parallel": True,
            "llm_timeout": 5,
        }

    def _get_fast_config(self) -> Dict[str, Any]:
        """Fast configuration."""
        return {
            "pinecone_max_results": 8,
            "internet_max_results": 6,
            "internet_queries": 2,
            "validation_parallel": True,
            "llm_timeout": 10,
        }

    def _get_balanced_config(self) -> Dict[str, Any]:
        """Balanced configuration."""
        return {
            "pinecone_max_results": 10,
            "internet_max_results": 8,
            "internet_queries": 2,
            "validation_parallel": True,
            "llm_timeout": 15,
        }


# Global instances
performance_monitor = PerformanceMonitor()
speed_optimizer = SpeedOptimizer(performance_monitor)
