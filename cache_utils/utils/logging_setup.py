import logging
from io import StringIO

LOGGING_LEVEL = logging.INFO

log_stream = StringIO()
stream_handler = logging.StreamHandler()
stream_handler.setLevel(LOGGING_LEVEL)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s'))
memory_handler = logging.StreamHandler(log_stream)
memory_handler.setLevel(LOGGING_LEVEL)
memory_handler.setFormatter(logging.Formatter('%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s'))

logger = logging.getLogger("PTI_logger")
logger.setLevel(LOGGING_LEVEL)
logger.handlers = []  # Remove any existing handlers
logger.addHandler(stream_handler)
logger.addHandler(memory_handler)
