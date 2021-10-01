from enum import Enum
from typing import Callable
import time


class RequestType(Enum):
    REQUEST = 1
    RESPONSE = 2
    OUTLIER = 3
    REFERENCE_REQUEST = 4


def retry_with_backoff(fn: Callable, retries=10, backoff_in_seconds=1, post_success_delay=2):
    x = 0
    while True:
        try:
            r = fn()
            time.sleep(post_success_delay)
            return r
        except Exception as e:
            if x == retries:
                print("Time is up!")
                raise
            else:
                sleep = int(backoff_in_seconds * 1.5 ** x)
                print("  Sleep :", str(sleep) + "s: ", str(e))
                time.sleep(sleep)
                x += 1
