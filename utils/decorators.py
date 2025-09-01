import time
from functools import wraps

def timer_decorator(func):
    """
    A decorator to measure and print the execution time of an asynchronous function.

    Args:
        func (callable): The asynchronous function to be decorated.

    Returns:
        callable: The wrapped function with timing functionality.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        """
        The wrapper function that adds the timing logic.
        """
        start_time = time.time()
        
        # Call the original async function and await its result
        result = await func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\nFunction '{func.__name__}' execution time: {execution_time:.2f} seconds.")
        
        return result
    return wrapper