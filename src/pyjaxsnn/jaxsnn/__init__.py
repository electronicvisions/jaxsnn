import logging

from jax import config

formatter = logging.Formatter(fmt="%(levelname)s - %(module)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger("root")
logger.setLevel(logging.INFO)
logger.addHandler(handler)


config.update("jax_debug_nans", True)
