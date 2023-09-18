import warnings

def deprecated(func):
    def _deprecated_warning_call(*args, **kwargs):
        warnings.warn(func.__name__ + " is a deprecated method.")
        func(*args, **kwargs)
    return _deprecated_warning_call