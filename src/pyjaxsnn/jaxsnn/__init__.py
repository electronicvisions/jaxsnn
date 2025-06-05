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
    from hxtorch import (  # pylint: disable=unused-import
        init_hardware,
        init_hardware_minimal,
        release_hardware,
    )
except ImportError:
    jaxsnn_logger.warning(
        "hxtorch is not installed. Please install hxtorch to use the"
        " hardware features of jaxsnn.",
    )


def get_logger(name: str):
    if 'logger' in globals() and hasattr(logger, 'get'):
        # Using pylogging to return logger
        return logger.get(name)
    # Otherwise use standard logging, to return logger
    return jaxsnn_logger.getChild(name)
