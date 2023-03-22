from qtpy.QtCore import QMutex, QMutexLocker, QObject


class Singleton(QObject):
	_instance = None
	_mutex = QMutex()

	def __new__(cls, *args, **kwargs):
		with QMutexLocker(cls._mutex):
			if not cls._instance:
				cls._instance = super(Singleton, cls).__new__(cls)
				QObject.__init__(cls._instance)
		return cls._instance
