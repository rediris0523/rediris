import threading
import concurrent.futures
from typing import Callable, Any, Optional
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)


class ThreadPoolManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=10,
            thread_name_prefix="rediris-worker"
        )
        self._shutdown = False
    
    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        if self._shutdown:
            raise RuntimeError("ThreadPoolManager is shutdown")
        return self.executor.submit(fn, *args, **kwargs)
    
    def shutdown(self, wait: bool = True):
        if not self._shutdown:
            self._shutdown = True
            self.executor.shutdown(wait=wait)
    
    def __del__(self):
        if not self._shutdown:
            self.shutdown(wait=False)


def get_thread_pool() -> ThreadPoolManager:
    return ThreadPoolManager()

