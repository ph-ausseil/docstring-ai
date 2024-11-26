# Docstring generated by docstring-ai : http://github.com/ph-ausseil/docstring-ai
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
from functools import wraps
from tqdm import tqdm

from functools import wraps
from tqdm import tqdm

def show_file_progress(desc="Processing files", **kwargs):
    """
    A decorator to display a progress bar while processing files.

    This decorator uses the `tqdm` library to provide a visual representation
    of the progress made in processing a list of files. The progress bar will
    update as each file is processed.

    Args:
        desc (str): Description for the progress bar.
        **kwargs: Additional arguments for `tqdm` customization.

    Returns:
        function: The wrapped function that processes the files.

    Raises:
        Any exceptions raised by the decorated function during file processing will be propagated.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(files, *args, **func_kwargs):
            results = []
            with tqdm(total=len(files), desc=desc, unit="file", dynamic_ncols=True, **kwargs) as pbar:
                for file in files:
                    result = func(file, *args, **func_kwargs)  # Process one file at a time
                    results.append(result)  # Store results
                    pbar.update(1)
            return results  # Return results for all files
        return wrapper
    return decorator
