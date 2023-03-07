try:
    import gc
    import torch


    def torch_gc():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
except:
    def torch_gc():
        pass
