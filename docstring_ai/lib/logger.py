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

# Load the .env file
load_dotenv()

JSON_LOGGING = os.environ.get("JSON_LOGGING", "false").lower() == "true"

TRACE = 5
NOTICE = 15
DB_LOG = 18
MINOR_INFO = NOTICE - 1
MAJOR_INFO = logging.INFO + 1 
DEV_NOTE = MINOR_INFO - 1
CHAT = 29

logging.addLevelName(TRACE, "TRACE")
logging.addLevelName(DB_LOG, "DB_LOG")
logging.addLevelName(NOTICE, "NOTICE")
logging.addLevelName(MINOR_INFO , "MINOR_INFO")
logging.addLevelName(MAJOR_INFO, "MAJOR_INFO")
logging.addLevelName(DEV_NOTE, "DEV_NOTE")
logging.addLevelName(CHAT, "CHAT")

if os.environ.get("PYTEST_RUN", "false").lower() == "true":
    CONSOLE_LOG_LEVEL = logging.getLevelName(
        os.getenv("PYTEST_CONSOLE_LOG_LEVEL", "ERROR").upper()
    )
    FILE_LOG_LEVEL = logging.getLevelName(
        os.getenv("PYTEST_FILE_LOG_LEVEL", "ERROR").upper()
    )
else:
    CONSOLE_LOG_LEVEL = logging.getLevelName(
        os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
    )
    FILE_LOG_LEVEL = logging.getLevelName(os.getenv("FILE_LOG_LEVEL", "DEBUG").upper())

AFAAS_BASE_PATH = Path(".").resolve()
RESET_SEQ: str = "\033[0m"
COLOR_SEQ: str = "\033[1;%dm"
BOLD_SEQ: str = "\033[1m"
UNDERLINE_SEQ: str = "\033[04m"
ITALIC_SEQ = "\033[3m"

ORANGE: str = "\033[33m"
YELLOW: str = "\033[93m"
WHITE: str = "\33[37m"
BLUE: str = "\033[34m"
LIGHT_BLUE: str = "\033[94m"
RED: str = "\033[91m"
GREY: str = "\33[90m"
GREEN: str = "\033[92m"
PURPLE: str = "\033[35m"
BRIGHT_PINK: str = "\033[95m"
CYAN: str = "\033[96m"  # A vibrant cyan for MINOR_INFO
BOLD_ORANGE: str = "\033[1;33m"  # A bold orange for MAJOR_INFO



EMOJIS: dict[str, str] = {
    "TRACE": "🔍",
    "DEBUG": "🐛",
    "MINOR_INFO": "📝",
    "MAJOR_INFO": "📝",
    "INFO": "📝",
    "CHAT": "💬",
    "DEV_NOTE": "⚠️",
    "WARNING": "⚠️",
    "ERROR": "❌",
    "CRITICAL": "💥",
    "NOTICE": "🔊",
    "DB_LOG": "📁",
}

KEYWORD_COLORS: dict[str, str] = {
    "TRACE": BRIGHT_PINK,
    "DEBUG": WHITE,
    "INFO": LIGHT_BLUE,
    "CHAT": PURPLE,
    "NOTICE": GREEN,
    "WARNING": YELLOW,
    "DEV_NOTE": ORANGE,
    "ERROR": ORANGE,
    "CRITICAL": RED,
    "DB_LOG": GREY,
    "MINOR_INFO": CYAN,  # New color
    "MAJOR_INFO": BOLD_ORANGE,  # New color
}

class SafeJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts non-serializable objects to strings."""
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

class JsonFormatter(logging.Formatter):
    """Formats log records as JSON strings, handling non-serializable objects."""
    def format(self, record):
        record_dict = record.__dict__.copy()
        # Optionally, remove or transform problematic fields
        # For example, you can remove 'exc_info' or other complex objects
        # Here, we'll let SafeJsonEncoder handle it
        return json.dumps(record_dict, cls=SafeJsonEncoder)

def formatter_message(message: str, use_color: bool = True) -> str:
    """
    Replaces placeholders in the message with ANSI codes for coloring.
    """
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


def friendly_path(path, max_length=30, min_length=20):
    """
    Construct a user-friendly relative path by including parts from the end until the length
    reaches the maximum limit, ensuring no part is truncated in the middle.

    :param path: Absolute or relative file path.
    :param max_length: Maximum allowed length for the path string.
    :param min_length: Minimum length to pad shorter paths.
    :return: User-friendly relative path.
    """
    # Convert to a Path object and make it relative to AFAAS_BASE_PATH
    path_obj = Path(path).resolve()
    relative_path = path_obj.relative_to(AFAAS_BASE_PATH)

    # Convert to string for length comparison
    relative_str = str(relative_path)

    # If already short, pad to minimum length
    if len(relative_str) <= min_length:
        return relative_str.ljust(min_length, " ")

    # Get the parts of the relative path
    parts = list(relative_path.parts)

    # Start with the last part (file name)
    result = parts[-1]

    # Add preceding parts until length limit is reached
    for part in reversed(parts[:-1]):
        new_result = f"{part}/{result}"
        if len(new_result) + 5 > max_length:  # +5 accounts for "[...]/"
            break
        result = new_result

    # Add ellipsis if not all parts are included
    if len(parts) > len(result.split("/")):
        result = f"[...]/{result}"

    return result


class ConsoleFormatter(logging.Formatter):
    """
    Custom formatter that adds colors and emojis to log messages.
    """
    def __init__(
        self, fmt: str, datefmt: str = None, style: str = "%", use_color: bool = True
    ):
        super().__init__(fmt, datefmt, style)
        self.use_color = use_color


    def format(self, record: logging.LogRecord) -> str:
        """
        Formats and enhances log records with colors and emojis.
        """
        rec = record
        levelname = rec.levelname
        current_color = ""

        if self.use_color and levelname in KEYWORD_COLORS:
            current_color = KEYWORD_COLORS[levelname]
            levelname_color = current_color + levelname + RESET_SEQ
            rec.levelname = levelname_color

        rec.name = f"{GREY}{friendly_path(path = rec.pathname, max_length = 30, min_length = 15)}"
        rec.msg = current_color + EMOJIS[levelname] + " " + str(rec.msg)

        message = logging.Formatter.format(self, rec)

        # Optionally reinstate color after each reset
        if self.use_color:
            message = message.replace(RESET_SEQ, RESET_SEQ + current_color) + RESET_SEQ

        # Truncate long TRACE messages
        if rec.levelno == TRACE and len(message) > 1000:
            message = (
                message[:800] + "[...] " + os.path.abspath(AFAASLogger.LOG_FILENAME)
            )
        return message


class SingletonMeta(type):
    """
    A Singleton metaclass to ensure only one instance of a class exists.
    """
    _instances = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

class AFAASLogger(logging.Logger, metaclass=SingletonMeta):
    """
    Custom logger that adds extra logging functions and a heartbeat mechanism.
    """
    LOG_FILENAME = "debug.log"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    MAX_TASK_LENGTH = 100  # Maximum length for the current task message

    CONSOLE_FORMAT = (
        "[%(asctime)s]$BOLD%(name)-15s,%(lineno)d]$RESET[%(levelname)-8s]\t%(message)s"
    )
    FORMAT = "%(asctime)s %(name)-15s,%(lineno)d] %(levelname)-8s %(message)s"
    COLOR_FORMAT = formatter_message(CONSOLE_FORMAT, True)
    JSON_FORMAT = (
        '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
    )

    def __init__(self, name: str, log_folder: str = "./", logLevel: str = "DEBUG"):
        super().__init__(name, logLevel)
        if hasattr(self, "_initialized"):
            return
        self._initialized = True  # Prevent re-initialization

        # Initialize the time of the last log message
        self.last_log_time = datetime.utcnow()

        # Variables for the status updater thread
        self.status_thread = None
        self.status_thread_running = False
        self.current_task = "Initializing..."

        # Lock for thread safety
        self._lock = threading.Lock()

    def set_task(self, task_name: str):
        """Set the current task being processed for status updates."""
        with self._lock:
            self.current_task = task_name
            self.last_log_time = datetime.utcnow()

    def start_status_updater(self, interval=10):
        """Start the status updater thread."""
        if not self.status_thread_running:
            self.status_thread_running = True
            self.status_thread = threading.Thread(target=self._status_updater, args=(interval,), daemon=True)
            self.status_thread.start()

    def _status_updater(self, interval):
        """Log periodic status updates if no new log messages have been made."""
        while self.status_thread_running:
            with self._lock:
                time_since_last_log = datetime.utcnow() - self.last_log_time
                if time_since_last_log > timedelta(seconds=interval):
                    if self.current_task:
                        self.info(f"Still processing: {self.current_task}")
                    else:
                        self.info("Still processing... Please wait.")
                    self.last_log_time = datetime.utcnow()
            time.sleep(interval)

    def stop_status_updater(self):
        """Stop the background status thread."""
        self.status_thread_running = False
        if self.status_thread:
            self.status_thread.join()
            self.status_thread = None

    def log(self, level, msg, *args, stacklevel: int = 2, **kwargs):
        """
        Override log method to dynamically fetch caller details and update `current_task`.
        """
        try:
            # Use the original message without adding filename and lineno
            msg_with_details = msg

            # Log the message using the superclass method, passing stacklevel
            super().log(level, msg_with_details, *args, stacklevel=stacklevel, **kwargs)

            # Update the current task with a truncated version of the message
            with self._lock:
                self.last_log_time = datetime.utcnow()
                if isinstance(msg, str):
                    self.current_task = (
                        msg[: self.MAX_TASK_LENGTH]
                        + ("..." if len(msg) > self.MAX_TASK_LENGTH else "")
                    )
        except Exception as e:
            # Handle any exceptions within the logging process
            super().error(f"Error in custom log method: {e}")

    def chat(self, role: str, openai_repsonse: dict, messages=None, *args, **kws):
        """
        Parse the content, log the message and extract the usage into prometheus metrics
        """
        role_emojis = {
            "system": "🖥️",
            "user": "👤",
            "assistant": "🤖",
            "function": "⚙️",
        }
        if self.isEnabledFor(CHAT):
            if messages:
                for message in messages:
                    self.log(
                        CHAT,
                        f"{role_emojis.get(message['role'], '🔵')}: {message['content']}",
                    )
            else:
                response = openai_repsonse  # Assuming it's already a dict

                self.log(
                    CHAT,
                    f"{role_emojis.get(role, '🔵')}: {response['choices'][0]['message']['content']}",
                )

    def trace(self, msg, *args, stacklevel: int = 3, **kwargs):
        if self.isEnabledFor(TRACE):
            self.log(TRACE, msg, *args, stacklevel=stacklevel, **kwargs)

    def notice(self, msg, *args, stacklevel: int = 3, **kwargs):
        if self.isEnabledFor(NOTICE):
            self.log(NOTICE, msg, *args, stacklevel=stacklevel, **kwargs)

    def db_log(self, msg, *args, stacklevel: int = 3, **kwargs):
        if self.isEnabledFor(DB_LOG):
            self.log(DB_LOG, msg, *args, stacklevel=stacklevel, **kwargs)

    def dev(self, msg, *args, stacklevel: int = 3, **kwargs):
        if self.isEnabledFor(DEV_NOTE):
            self.log(DEV_NOTE, msg, *args, stacklevel=stacklevel, **kwargs)

    def major_info(self, msg, *args, stacklevel: int = 3, **kwargs):
        if self.isEnabledFor(MAJOR_INFO):
            self.log(MAJOR_INFO, msg, *args, stacklevel=stacklevel, **kwargs)

    def minor_info(self, msg, *args, stacklevel: int = 3, **kwargs):
        if self.isEnabledFor(MINOR_INFO):
            self.log(MINOR_INFO, msg, *args, stacklevel=stacklevel, **kwargs)

    @staticmethod
    def bold(msg: str) -> str:
        """
        Returns the message in bold
        """
        return BOLD_SEQ + msg + RESET_SEQ

    @staticmethod
    def italic(msg: str) -> str:
        """
        Returns the message in italic
        """
        return ITALIC_SEQ + msg + RESET_SEQ

class QueueLogger(logging.Logger):
    """
    Custom logger class with queue.
    """
    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)
        queue_handler = logging.handlers.QueueHandler(queue.Queue(-1))
        self.addHandler(queue_handler)


class PathFilter(logging.Filter):
    """
    Filter log records based on the file path of the log source.
    """
    def __init__(self, base_path):
        """
        Initialize with the base path to allow logs from.
        :param base_path: The base path to include logs from.
        """
        self.base_path = Path(base_path).resolve()
        super().__init__()

    def filter(self, record):
        """
        Allow logs only if the pathname starts with the base path.
        """
        record_path = Path(record.pathname).resolve()
        return str(record_path).startswith(str(self.base_path))


logging.setLoggerClass(AFAASLogger)
def setup_logger():
    logger = logging.getLogger("AFAAS")
    logger.setLevel(CONSOLE_LOG_LEVEL)

    # Console Handler
    console_handler = logging.StreamHandler()
    if JSON_LOGGING:
        console_formatter = JsonFormatter()
    else:
        console_formatter = ConsoleFormatter(AFAASLogger.COLOR_FORMAT)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(CONSOLE_LOG_LEVEL)

    console_handler.addFilter(PathFilter(AFAAS_BASE_PATH))
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.handlers.TimedRotatingFileHandler(
        AFAASLogger.LOG_FILENAME, when="midnight", interval=1, backupCount=7
    )
    if JSON_LOGGING:
        file_formatter = JsonFormatter()
    else:
        file_formatter = logging.Formatter(AFAASLogger.FORMAT)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(FILE_LOG_LEVEL)

    # File handler does not use filters (captures all logs)
    logger.addHandler(file_handler)

    # Start the status updater
    logger.start_status_updater()

    return logger


# Initialize the logger
LOG = setup_logger()

# Log initial debug messages
LOG.debug(f"Console log level is  : {logging.getLevelName(CONSOLE_LOG_LEVEL)}")
LOG.debug(f"File log level is  : {logging.getLevelName(FILE_LOG_LEVEL)}")

from tqdm import tqdm
from functools import wraps
import time  # For simulation, remove in actual use

def show_file_progress(desc="Processing files", **kwargs):
    """
    A decorator to display a progress bar while processing files.

    Args:
        desc (str): Description for the progress bar.
        **kwargs: Additional arguments for tqdm customization.
    """
    def decorator(func):
        @wraps(func)
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
                        "Remaining time": f"{pbar.format_dict['remaining_time']:0>8.2f}s",
                        "Speed": f"{pbar.format_dict['elapsed'] / (pbar.n or 1):.2f}s/file"
                    })
                    pbar.update(1)
                
                # Print all processed files after completion
                print("\nProcessed files:")
                for processed_file in processed_files:
                    print(f"- {processed_file}")
                    
        return wrapper
    return decorator

