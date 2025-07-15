from jax import config
from jaxsnn.event.from_nir import from_nir, ConversionConfig

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

    # Set up the standard logger configuration
    jaxsnn_logger = logging.getLogger("jaxsnn")

    if not jaxsnn_logger.hasHandlers():
        # Configure the logger if it has no handlers
        logging.basicConfig(level=logging.WARN)
        jaxsnn_logger.setLevel(logging.INFO)


def get_logger(name: str):
    if 'logger' in globals() and hasattr(logger, 'get'):
        # Using pylogging to return logger
        return logger.get(name)
    # Otherwise use standard logging, to return logger
    return jaxsnn_logger.getChild(name)


config.update("jax_debug_nans", True)
