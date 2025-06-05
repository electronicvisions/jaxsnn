try:
    from hxtorch.core import parameter
except ImportError:
    parameter = None  # Fallback if the module is not available
