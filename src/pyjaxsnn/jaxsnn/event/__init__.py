# FIXME: duplicate code below
try:
    # Try to import pylogging
    import pylogging as logger

    jaxsnn_logger = logger.get("jaxsnn")

    # Check if the jaxsnn logger has any appenders
    if jaxsnn_logger.get_number_of_appenders() == 0:
        # Configure the jaxsnn logger if it has no appenders
        logger.default_config(level=logger.LogLevel.WARN)
        logger.set_loglevel(jaxsnn_logger, logger.LogLevel.INFO)
except ImportError:
    # Import standard logger
    import logging

    TRACE_LEVEL = 5
    logging.addLevelName(TRACE_LEVEL, "TRACE")

    def trace(self, message, *args, **kwargs):
        if self.isEnabledFor(TRACE_LEVEL):
            self.log(TRACE_LEVEL, message, args, **kwargs)

    logging.Logger.TRACE = trace

    # Set up the standard logger configuration
    jaxsnn_logger = logging.getLogger("jaxsnn")

    if not jaxsnn_logger.hasHandlers():
        # Configure the logger if it has no handlers
        logging.basicConfig(level=logging.WARN)
        jaxsnn_logger.setLevel(logging.INFO)

try:
    from jaxsnn.event.utils.from_nir import (
        from_nir,
        ConversionConfig,
    )
    from jaxsnn.event.utils.from_nir_data import from_nir_data
    from jaxsnn.event.utils.to_nir_data import to_nir_data
except ImportError:
    jaxsnn_logger.warning(
        "NIR (or non-NIRData-enabled version) is not installed."
        "Please install a proper version.",
    )
