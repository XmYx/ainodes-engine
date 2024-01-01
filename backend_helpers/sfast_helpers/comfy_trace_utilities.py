import contextlib
import copy
import torch


def hash_arg(arg):
    # micro optimization: bool obj is an instance of int
    if isinstance(arg, (str, int, float, bytes)):
        return arg
    if isinstance(arg, (tuple, list)):
        return tuple(map(hash_arg, arg))
    if isinstance(arg, dict):
        return tuple(
            sorted(
                ((hash_arg(k), hash_arg(v)) for k, v in arg.items()), key=lambda x: x[0]
            )
        )
    return type(arg)


class ModuleWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ModuleFactory:
    def __init__(self, callable, kwargs) -> None:
        self.callable = callable
        self.kwargs = kwargs
        self.converted_kwargs = self.gen_converted_kwargs()

    def gen_converted_kwargs(self):
        return self.kwargs

    def get_converted_kwargs(self):
        return self.converted_kwargs

    def gen_cache_key(self):
        return (
            self.callable.__class__.__qualname__,
            hash_arg(self.kwargs),
        )

    @contextlib.contextmanager
    def converted_module_context(self):
        yield (self.callable, self.converted_kwargs)

    def load_state_dict_to_module(self, script_module):
        with self.converted_module_context() as (m_model, m_kwargs):
            script_module.load_state_dict(
                m_model.state_dict(), strict=False, assign=True
            )
        return script_module


class TracerWithCache:
    cache_map = {}

    @staticmethod
    def get_traced_module(module_factory: ModuleFactory, device=None):
        cache_key = module_factory.gen_cache_key()

        if not cache_key in TracerWithCache.cache_map:
            with module_factory.converted_module_context() as (m_model, m_kwargs):
                if device != None:
                    m_model.to(device=device)
                script_module = torch.jit.trace(
                    m_model,
                    example_kwarg_inputs=m_kwargs,
                    strict=True,
                    check_trace=True,
                )

            meta_script_module = script_module.to_empty(device="meta")
            TracerWithCache.cache_map[cache_key] = meta_script_module

        meta_script_module = copy.deepcopy(TracerWithCache.cache_map[cache_key])

        script_module = module_factory.load_state_dict_to_module(meta_script_module)
        return script_module
