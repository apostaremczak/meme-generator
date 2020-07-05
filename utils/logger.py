import logging


def get_logger(logger_name: str = "BasicLogger") -> logging.Logger:
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.INFO)
    return logging.getLogger(logger_name)
