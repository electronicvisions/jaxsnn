try:
    from jaxsnn.event.utils.from_nir import (
        from_nir,
        ConversionConfig,
    )
    from jaxsnn.event.utils.from_nir_data import from_nir_data
    from jaxsnn.event.utils.to_nir_data import to_nir_data
except ImportError:
    # NIR is an optional dependency
    pass
