import json
import inspect
import logging
import logging.config
import logging.handlers
import os
from pathlib import Path
import queue
import threading
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

from tqdm import tqdm
import functools
import time  # For simulation, remove in actual use

def show_file_progress(desc="Processing files", **kwargs):
    """
    A decorator to display a progress bar while processing files.

    Args:
        desc (str): Description for the progress bar.
        **kwargs: Additional arguments for tqdm customization.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(files, *args, **func_kwargs):
            total_files = len(files)
            
            with tqdm(total=total_files, desc=desc, unit="file", dynamic_ncols=True, **kwargs) as pbar:
                processed_files = []
                for file in files:
                    func(file, *args, **func_kwargs)
                    
                    # Track processed files
                    processed_files.append(file)
                    
                    # Update the progress bar with custom postfix
                    pbar.set_postfix({
                        "Remaining time": f"{pbar.format_dict.get('remaining_time', 0):0>8.2f}s",
                        "Speed": f"{(pbar.format_dict.get('elapsed', 0) / (pbar.n or 1)):.2f}s/file"
                    })
                    pbar.update(1)
                
        return wrapper
    return decorator
