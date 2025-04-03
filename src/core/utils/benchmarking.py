# src/core/utils/benchmarking.py

import time
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import mean, median, stdev

logger = logging.getLogger(__name__)


@dataclass
class Timer:
    """Context manager for timing code blocks."""

    name: str
    start_time: float = field(default=0.0)
    end_time: float = field(default=0.0)

    def __enter__(self) -> "Timer":
        """Start timing when entering context."""
        self.start_time = time.time()
        return self

    def __exit__(self, *args) -> None:
        """Stop timing when exiting context."""
        self.end_time = time.time()

    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time == 0.0:
            return time.time() - self.start_time
        return self.end_time - self.start_time


def benchmark(func: Callable) -> Callable:
    """Decorator to benchmark a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            return func(*args, **kwargs)

    return wrapper


@dataclass
class TimingStats:
    """Statistics for a timed operation."""

    name: str
    total_time: float = 0.0
    count: int = 0
    times: List[float] = field(default_factory=list)
    min_time: float = float("inf")
    max_time: float = 0.0
    bytes_processed: int = 0

    def add_timing(self, elapsed: float, size: int = 0) -> None:
        """Add a timing measurement.

        Args:
            elapsed: Time taken in seconds
            size: Number of bytes processed (optional)
        """
        self.total_time += elapsed
        self.count += 1
        self.times.append(elapsed)
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.bytes_processed += size

    @property
    def avg_time(self) -> float:
        """Calculate average time."""
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def median_time(self) -> float:
        """Calculate median time."""
        return median(self.times) if self.times else 0.0

    @property
    def std_dev(self) -> float:
        """Calculate standard deviation."""
        return stdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def throughput(self) -> float:
        """Calculate throughput in MB/s."""
        if self.total_time > 0 and self.bytes_processed > 0:
            return (self.bytes_processed / 1024 / 1024) / self.total_time
        return 0.0

    def __str__(self) -> str:
        if not self.times:
            return f"{self.name}: No timing data"

        stats = [
            f"Total: {self.total_time:.2f}s",
            f"Count: {self.count}",
            f"Avg: {self.avg_time:.3f}s",
            f"Median: {self.median_time:.3f}s",
            f"StdDev: {self.std_dev:.3f}s",
            f"Min: {self.min_time:.3f}s",
            f"Max: {self.max_time:.3f}s",
        ]

        if self.throughput > 0:
            stats.append(f"Throughput: {self.throughput:.1f} MB/s")

        return f"{self.name}: " + ", ".join(stats)


class PerformanceStats:
    """Collect and report performance statistics."""

    def __init__(self) -> None:
        self.stats: Dict[str, TimingStats] = {}

    def get_stats(self, name: str) -> TimingStats:
        """Get or create stats for an operation."""
        if name not in self.stats:
            self.stats[name] = TimingStats(name=name)
        return self.stats[name]

    def add_timing(self, name: str, elapsed: float, size: int = 0) -> None:
        """Add a timing measurement for an operation.

        Args:
            name: Name of the operation
            elapsed: Time taken in seconds
            size: Number of bytes processed (optional)
        """
        self.get_stats(name).add_timing(elapsed, size)

    def report(self) -> str:
        """Generate a performance report."""
        if not self.stats:
            return "No performance data collected"

        lines = []
        total_time = sum(s.total_time for s in self.stats.values())

        for name in sorted(self.stats.keys()):
            stats = self.stats[name]
            if stats.total_time > 0:
                pct = (stats.total_time / total_time) * 100 if total_time > 0 else 0
                lines.append(f"{stats} ({pct:.1f}%)")

        return "\n".join(lines)


@contextmanager
def timer(name: str, stats: Optional[PerformanceStats] = None, size: int = 0):
    """Context manager for timing code blocks with optional stats collection.

    Args:
        name: Name of the operation being timed
        stats: Optional PerformanceStats object to collect metrics
        size: Number of bytes being processed (optional)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if stats:
            stats.add_timing(name, elapsed, size)
