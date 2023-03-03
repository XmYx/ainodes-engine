import datetime

from Qt.QtCore import QMutex, QMutexLocker, QObject, Signal

#from backend.torch_gc import torch_gc

#hypernetworks = {}




class Callbacks(QObject):
	selected_model_changed = Signal(str)
	doInpaintTriggered = Signal()
	paintInpaintMaskTriggered = Signal()
	sceneBrushChanged = Signal(int)

class Singleton(QObject):
	_instance = None
	_mutex = QMutex()

	def __new__(cls, *args, **kwargs):
		with QMutexLocker(cls._mutex):
			if not cls._instance:
				cls._instance = super(Singleton, cls).__new__(cls)
				QObject.__init__(cls._instance)
		return cls._instance

	def __init__(self):
		self.signals = Callbacks()