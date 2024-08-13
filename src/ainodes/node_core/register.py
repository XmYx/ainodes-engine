import inspect
from typing import Dict, Any

# Define a global registry to store node information
NODE_REGISTRY = {}

def register_node():
    def decorator(cls_or_func):


        # Determine if the target is a class or a function
        if inspect.isclass(cls_or_func):
            # For classes, inspect the __init__ and __call__ methods
            init_signature = inspect.signature(cls_or_func.__init__)
            call_signature = inspect.signature(cls_or_func.__call__)

            # Collect parameter info from __init__
            init_params = {}
            for name, param in init_signature.parameters.items():
                if name != 'self':
                    init_params[name] = {
                        'type': param.annotation,
                        'default': param.default if param.default is not inspect.Parameter.empty else None
                    }

            # Collect parameter info from __call__
            call_params = {}
            for name, param in call_signature.parameters.items():
                if name != 'self':
                    call_params[name] = {
                        'type': param.annotation,
                        'default': param.default if param.default is not inspect.Parameter.empty else None
                    }

            # Register the class with its collected information
            NODE_REGISTRY[cls_or_func.__name__] = {
                'type': 'class',
                'class': cls_or_func,
                'init_params': init_params,
                'call_params': call_params
            }

        elif inspect.isfunction(cls_or_func):
            # For functions, inspect the function's signature
            func_signature = inspect.signature(cls_or_func)

            # Collect parameter info from the function
            func_params = {}
            for name, param in func_signature.parameters.items():
                func_params[name] = {
                    'type': param.annotation,
                    'default': param.default if param.default is not inspect.Parameter.empty else None
                }

            # Register the function with its collected information
            NODE_REGISTRY[cls_or_func.__name__] = {
                'type': 'function',
                'function': cls_or_func,
                'params': func_params
            }

        return cls_or_func
    return decorator