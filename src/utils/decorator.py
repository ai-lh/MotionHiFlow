import inspect, functools

def capture_init_kwargs(init):
    sig = inspect.signature(init)
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        # allow missing defaults, then fill them in from the signature
        bound = sig.bind_partial(self, *args, **kwargs)
        config = {}
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                config[name] = tuple(bound.arguments.get(name, ()))
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                config[name] = dict(bound.arguments.get(name, {}))
            else:
                if name in bound.arguments:
                    config[name] = bound.arguments[name]
                elif param.default is not inspect._empty:
                    config[name] = param.default
                else:
                    # missing required arg -> mirror normal behavior
                    raise TypeError(f"Missing required argument: {name}")
        cls = self.__class__
        config["_target_"] = f"{cls.__module__}.{cls.__qualname__}"
        self.config = config
        return init(self, *args, **kwargs)
    return wrapper