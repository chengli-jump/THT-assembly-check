from threading import RLock
import threading
import functools
from PyQt5.QtCore import QThread

class ThreadSafeObject(threading.Thread):
    # A class providing infrastructure for creating thread-safe objects

    def __init__(self):
        super(ThreadSafeObject, self).__init__()
        self._lock = RLock()
        self._status = threading.Event()
        #设置初始状态为阻塞
        self._status.clear()

    def setReady(self):
        self._status.set()

    def setNotReady(self):
        self._status.clear()

    def close(self):
        self.join(1)

    def isReady(self):
        if self._status.isSet():
            return True
        else:
            return False

    @property
    def myLock(self):
        return self._lock
        # Retrieve the lock of this object

    def thread_safe(f):
        # A decorator for making methods thread-safe

        @functools.wraps(f)
        def magic(self, *args, **kwargs):
            with self.myLock:
                return f(self, *args, **kwargs)

        return magic

    thread_safe = staticmethod(thread_safe)
