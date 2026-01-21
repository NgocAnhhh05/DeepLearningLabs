import os
import sys
import logging
from termcolor import colored

class ColorfulFormat(logging.Formatter):
    COLORS = {
        logging.DEBUG: "blue",
        logging.INFO: "green",
        logging.ERROR: "red",
        logging.WARNING: "yellow",
        logging.CRITICAL: "red",
    }
    def format(self, record):
        log_fmt = f"[{self.formatTime(record, self.datefmt)}] {record.levelname}: {record.getMessage()}"
        if record.levelno in self.COLORS:
            return colored(log_fmt, self.COLORS[record.levelno])
        return log_fmt

def setup_logger(output_file=None, name="dl_lab03"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Handle to print in terminal
    control_handler = logging.StreamHandler(sys.stdout)
    control_handler.setFormatter(ColorfulFormat(datefmt="%d:%m:%Y %H:%M:%S"))
    logger.addHandler(control_handler)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        file_handler = logging.FileHandler(output_file, encoding="utf-8")
        file_fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%d:%m:%Y %H:%M:%S")
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    return logger
