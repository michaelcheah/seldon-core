from enum import Enum
from typing import Callable
import time

class RequestType(Enum):
    REQUEST = 1
    RESPONSE = 2
    OUTLIER = 3


def retry_with_backoff(fn: Callable, retries=5, backoff_in_seconds=1):
  x = 0
  while True:
    try:
      return fn()
    except:
      if x == retries:
        print("Time is up!")
        raise
      else:
        sleep = (backoff_in_seconds * 2 ** x)
        print("  Sleep :", str(sleep) + "s")
        time.sleep(sleep)
        x += 1
