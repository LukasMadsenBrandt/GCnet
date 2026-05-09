"""Small decorators used by legacy Benito analysis helpers."""

import time
from gene_analysis_benito.config import timing_enabled  # Import the control flag

def timing_decorator(func):
    """Print function runtime when timing is enabled in the dataset config."""
    def wrapper(*args, **kwargs):
        """Execute the wrapped function and optionally print elapsed time."""
        if timing_enabled:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Total execution time of {func.__name__}: {end_time - start_time} seconds")
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper
