import logging
import os

def setup_logger(output_folder, file_name, loger_name, log_level=logging.INFO):
    log_file = os.path.join(output_folder, file_name)
    logger = logging.getLogger(loger_name)
    logger.setLevel(log_level)
    logger.handlers = []
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file, mode='w')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
