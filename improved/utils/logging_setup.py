import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Dict, Any, Union, List
import json

class CustomFormatter(logging.Formatter):
    """
    Custom formatter with color support for console output.
    """
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'RESET': '\033[0m'       # Reset
    }
    
    def __init__(self, use_colors: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = use_colors
    
    def format(self, record):
        log_message = super().format(record)
        if self.use_colors and hasattr(record, 'levelname'):
            color_start = self.COLORS.get(record.levelname, '')
            color_end = self.COLORS['RESET'] if color_start else ''
            return f"{color_start}{log_message}{color_end}"
        return log_message

class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    """
    def __init__(self, fields: Optional[List[str]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields = fields or ['name', 'levelname', 'pathname', 'lineno', 'message', 'exc_info']
    
    def format(self, record):
        log_record = {}
        
        # Add standard fields
        log_record['timestamp'] = self.formatTime(record, self.datefmt)
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['message'] = record.getMessage()
        
        # Add location info
        if 'pathname' in self.fields:
            log_record['file'] = record.pathname
        if 'lineno' in self.fields:
            log_record['line'] = record.lineno
        if 'funcName' in self.fields:
            log_record['function'] = record.funcName
        
        # Add exception info if present
        if record.exc_info and 'exc_info' in self.fields:
            log_record['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields from record
        for key, value in record.__dict__.items():
            if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                          'funcName', 'id', 'levelname', 'levelno', 'lineno', 'module',
                          'msecs', 'message', 'msg', 'name', 'pathname', 'process',
                          'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName']:
                log_record[key] = value
        
        return json.dumps(log_record)

def setup_logging(log_dir: str, 
                  log_level: Union[int, str] = logging.INFO,
                  log_file_prefix: str = 'urbansound_classifier',
                  console_output: bool = True,
                  file_output: bool = True,
                  json_output: bool = False,
                  max_file_size_mb: int = 10,
                  backup_count: int = 5,
                  use_colors: bool = True) -> logging.Logger:
    """
    Set up logging with file and console handlers.
    
    Args:
        log_dir (str): Directory to store log files
        log_level (Union[int, str]): Logging level (e.g., logging.INFO or 'INFO')
        log_file_prefix (str): Prefix for log files
        console_output (bool): Whether to output logs to console
        file_output (bool): Whether to output logs to file
        json_output (bool): Whether to use JSON formatting for file logs
        max_file_size_mb (int): Maximum size of each log file in MB
        backup_count (int): Number of backup log files to keep
        use_colors (bool): Whether to use colors in console output
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('urbansound_classifier')
    
    # Convert string log level to numeric if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    standard_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler if enabled
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        if use_colors and sys.stdout.isatty():
            # Use colored formatter for console if terminal supports it
            console_formatter = CustomFormatter(
                use_colors=True,
                fmt='%(levelname)s: %(message)s'
            )
        else:
            console_formatter = logging.Formatter('%(levelname)s: %(message)s')
            
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Create file handlers if enabled
    if file_output:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Standard log file with rotation by size
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(log_dir, f"{log_file_prefix}_{timestamp}.log")
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        
        if json_output:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(detailed_formatter)
            
        logger.addHandler(file_handler)
        
        # Error log file (ERROR and above)
        error_log_file = os.path.join(log_dir, f"{log_file_prefix}_{timestamp}_error.log")
        error_file_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        error_file_handler.setLevel(logging.ERROR)
        
        if json_output:
            error_file_handler.setFormatter(JsonFormatter())
        else:
            error_file_handler.setFormatter(detailed_formatter)
            
        logger.addHandler(error_file_handler)
        
        # Daily rotating log file
        daily_log_file = os.path.join(log_dir, f"{log_file_prefix}_daily.log")
        daily_file_handler = TimedRotatingFileHandler(
            daily_log_file,
            when='midnight',
            interval=1,
            backupCount=30  # Keep a month of daily logs
        )
        daily_file_handler.setLevel(log_level)
        
        if json_output:
            daily_file_handler.setFormatter(JsonFormatter())
        else:
            daily_file_handler.setFormatter(standard_formatter)
            
        logger.addHandler(daily_file_handler)
    
    # Log the setup
    logger.info(f"Logging initialized at level {logging.getLevelName(log_level)}")
    if file_output:
        logger.info(f"Log files will be saved to {log_dir}")
    
    return logger

class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context information to log messages.
    """
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        # Add extra context to the message
        if self.extra:
            context_str = ' '.join(f"[{k}={v}]" for k, v in self.extra.items())
            msg = f"{msg} {context_str}"
        return msg, kwargs

def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> Union[logging.Logger, LoggerAdapter]:
    """
    Get a logger with optional context information.
    
    Args:
        name (str): Logger name
        context (Optional[Dict[str, Any]]): Context information to add to log messages
        
    Returns:
        Union[logging.Logger, LoggerAdapter]: Logger or LoggerAdapter instance
    """
    logger = logging.getLogger(f"urbansound_classifier.{name}")
    
    if context:
        return LoggerAdapter(logger, context)
    
    return logger