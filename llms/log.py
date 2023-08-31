import logging


# Get root logger
def get_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger
