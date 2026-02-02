# ruff: noqa: I001
import functools
import os
import psutil
import time


def track_memory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())

        # Memory before
        mem_before = process.memory_info().rss / (1024 * 1024)

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        # Memory after
        mem_after = process.memory_info().rss / (1024 * 1024)

        print(
            f"[{func.__name__}] Time: {end_time - start_time:.4f}s | "
            f"Mem Delta: {mem_after - mem_before:+.2f} MB | "
            f"Current Total: {mem_after:.2f} MB"
        )
        return result

    return wrapper
