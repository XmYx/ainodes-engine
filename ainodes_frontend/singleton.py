from PySide6.QtCore import QObject, QMutex, QMutexLocker


class Singleton:
    _instance = None
    _mutex = QMutex()

    def __init__(self):
        if not Singleton._instance:
            self._data = {}
        else:
            raise RuntimeError("Singleton instances can only be accessed through the 'instance()' method.")

    @classmethod
    def instance(cls):
        with QMutexLocker(cls._mutex):
            if not cls._instance:
                cls._instance = cls()
            return cls._instance
