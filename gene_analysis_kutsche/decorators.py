"""Small decorators used by Kutsche analysis helpers."""

import time
from gene_analysis_kutsche.config import timing_enabled  # Import the control flag

def timing_decorator(func):
    '''
    Decorator to measure the execution time of a function.
    '''
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
