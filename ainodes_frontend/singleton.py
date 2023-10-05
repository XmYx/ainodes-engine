"""class Singleton:
    __instance = None
    __lock = threading.Lock()

    def __new__(cls):
        if not cls.__instance:
            with cls.__lock:
                if not cls.__instance:
                    cls.__instance = super().__new__(cls)
        return cls.__instance"""
try:
    import torch
    torch_available = True
except:
    torch_available = False
def get_available_gpus():
    return [str(i) for i in range(torch.cuda.device_count())]
if torch_available:
    available_gpus = get_available_gpus()