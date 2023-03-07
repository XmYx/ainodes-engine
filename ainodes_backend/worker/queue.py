from qtpy.QtCore import QObject, QThreadPool, Signal, Slot

from ainodes_backend.worker.worker import Worker


class QueueSystem(QObject):
    """
    Queue system that manages a pool of worker threads
    """
    task_finished = Signal(object)

    def __init__(self):
        super().__init__()
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)  # Only one thread running at a time
        self.tasks = []

    @Slot(object)
    def add_task(self, task, *args, **kwargs):
        """
        Add a task to the queue
        """
        self.tasks.append((task, args, kwargs))
        self.start_next_task()

    def start_next_task(self):
        """
        Start the next task in the queue, if there is one
        """
        if not self.tasks:
            return
        task, args, kwargs = self.tasks.pop(0)
        worker = Worker(task, *args, **kwargs)
        worker.setAutoDelete(True)
        worker.signals.result.connect(self.task_finished)
        self.pool.start(worker)