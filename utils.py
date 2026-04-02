"""
Utility classes and functions for the neonatal infection prediction model.
"""

import os
import gc
import logging
import torch
import numpy as np
from torchinfo import summary


class Logger:
    def __init__(self, logger_filename=None):
        logging.basicConfig(
            filename=logger_filename,
            format='%(asctime)s - %(message)s',
            level=logging.INFO,
            filemode='w',
            force=True,
        )

    def my_print(self, message, level='info'):
        """
        Manual print operation.

        Args:
            message: input string.
            level: level of logging.

        Returns:

        """
        if level == 'info':
            print_message = 'INFO: {}'.format(message)
            logging.info(print_message)
        elif level == 'exception':
            print_message = 'EXCEPTION: {}'.format(message)
            logging.exception(print_message)
        elif level == 'warning':
            print_message = 'WARNING: {}'.format(message)
            logging.warning(print_message)
        else:
            print_message = 'INFO: {}'.format(message)
            logging.info(print_message)
        print(print_message)

    def close(self):
        """
        Closes all handlers associated with the root logger.
        This is crucial for re-initializing the logger in a new directory.
        """
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)


def create_folder_if_not_exists(folder, logger=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if logger is not None:
        logger.my_print('Creating folder: {}'.format(folder))


def get_model_summary(model, input_data, filename, device, logger):
    """
    Get model summary and number of trainable parameters.

    Args:
        model:
        input_data:
        device:
        logger:

    Returns:

    """
    # Get and save summary
    txt = str(summary(model=model, input_data=input_data, device=device))

    # Print to console and write to logger file
    logger.my_print(f"{txt}")

    # Save summary to a separate file
    file = open(filename, 'a+', encoding='utf-8')
    file.write(txt)
    file.close()

    # Determine number of trainable parameters
    # Source: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    # total_params = sum(p.numel() for p in model.parameters())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params


def cleanup_objects(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_json_serializable(obj):
    """Convert non-JSON-serializable objects to serializable types."""
    if isinstance(obj, torch.device):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


def restore_from_json(obj, device='cuda'):
    """Restore objects from JSON-serialized format."""
    if isinstance(obj, str):
        # Try to restore torch.device
        if obj.startswith('cuda') or obj == 'cpu':
            return torch.device(obj)
        return obj
    elif isinstance(obj, list):
        # Check if it looks like a numpy array (nested lists with numbers)
        if obj and isinstance(obj[0], (int, float, list)):
            return np.array(obj)
        return [restore_from_json(item, device) for item in obj]
    elif isinstance(obj, dict):
        return {k: restore_from_json(v, device) for k, v in obj.items()}
    else:
        return obj
