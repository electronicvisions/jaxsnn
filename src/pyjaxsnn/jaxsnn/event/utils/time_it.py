import time
from typing import Any, Callable, Tuple


def time_it(timed_function: Callable, *args) -> Tuple[Any, float]:
    start = time.time()
    res = timed_function(*args)
    return res, time.time() - start
