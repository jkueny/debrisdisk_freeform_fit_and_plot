import logging
import os
from datetime import datetime

def configure_logging(log_to_file, which_script, save_to_dir):
    handlers = []
    if log_to_file:
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_to_dir is not None:
            log_path = os.path.join(save_to_dir, f"{which_script}_{timestamp}.log")
        else:
            log_path = os.path.join(os.getcwd(), f"{which_script}_{timestamp}.log")
        handlers.append(logging.FileHandler(log_path))
    else:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
    return log_path if log_to_file else None