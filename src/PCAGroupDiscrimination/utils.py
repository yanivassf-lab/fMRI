import logging
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


# ==========================================
# LOGGING SETUP
# ==========================================
def norm_mode_str(norm_mode):
    if norm_mode == 'none':
        return ''
    elif norm_mode == 'row':
        return '_normalized'
    elif norm_mode == 'row_col':
        return '_normalized_standardized'
    elif norm_mode == 'ml':
        return '_ml'


def setup_logger(log_dir=None):
    """
    Setup logger to write to both file and console with simplified format.
    Process-safe logger:
        - Each process writes to its own file (based on PID)
        - Avoids duplicate handlers
        - Logs to both file and console

    Args:
        log_dir (str, optional): Directory to save log files. Defaults to './logs'.

    Returns:
        logging.Logger: Configured logger instance with file and stream handlers.
    """
    if log_dir is None:
        log_dir = './logs'

    os.makedirs(log_dir, exist_ok=True)

    pid = os.getpid()

    log_file = os.path.join(
        log_dir,
        f'clustering_{datetime.now().strftime("%Y%m%d_%H%M%S")}_pid{pid}.log'
    )

    # Create a unique logger per process
    logger = logging.getLogger(f'clustering_{pid}')
    logger.setLevel(logging.INFO)

    # Prevent propagation to the root logger (avoids duplicate logs)
    logger.propagate = False

    # Add handlers only once per process
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter (includes timestamp and PID)
        formatter = logging.Formatter(
            '%(asctime)s [PID %(process)d] %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
