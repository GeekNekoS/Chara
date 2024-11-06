import logging
import os


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = logging.FileHandler(os.path.join(log_dir, 'logs.log'), mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
