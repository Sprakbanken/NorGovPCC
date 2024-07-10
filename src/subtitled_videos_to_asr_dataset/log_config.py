"""Logging utilities taken from stack overflow (with blue added)

CC-BY-SA-4.0: Sergey Pleshakov - https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
"""
import logging
from datetime import datetime
from pathlib import Path


class CustomFormatter(logging.Formatter):
    grey: str = "\x1b[38;20m"
    blue: str = "\x1b[34;20m"
    yellow: str = "\x1b[33;20m"
    red: str = "\x1b[31;20m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(logger):
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())

    now = datetime.now().replace(microsecond=0)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(filename=log_dir / f"{now.isoformat().replace(':','')}.log", level=logging.INFO)

    logger.addHandler(ch)
