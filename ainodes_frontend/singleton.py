import threading

"""class Singleton:
    __instance = None
    __lock = threading.Lock()

    def __new__(cls):
        if not cls.__instance:
            with cls.__lock:
                if not cls.__instance:
                    cls.__instance = super().__new__(cls)
        return cls.__instance"""

models = {}
token = ""
use_deforum_loss = None
highlight_sockets = True