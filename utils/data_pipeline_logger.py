import logging
from pathlib import Path


class DataPipelineLogger:
    """Centralized logger for data pipeline operations"""
    
    _loggers = {}  # Class variable to store logger instances
    
    def __init__(self, name, log_file_path=None, log_level=logging.INFO):
        """
        Initialize logger with console and optional file output
        
        Args:
            name: Logger name (typically class name or module name)
            log_file_path: Path to log file (optional)
            log_level: Logging level (default: INFO)
        """
        self.name = name
        self.log_file_path = log_file_path
        self.log_level = log_level
        
        # Avoid duplicate loggers
        if name not in DataPipelineLogger._loggers:
            self.logger = self._setup_logger()
            DataPipelineLogger._loggers[name] = self.logger
        else:
            self.logger = DataPipelineLogger._loggers[name]
    
    def _setup_logger(self):
        """Setup logger with console and file handlers"""
        logger = logging.getLogger(self.name)
        
        # Avoid adding handlers multiple times
        if logger.handlers:
            return logger
        
        logger.setLevel(self.log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if log file path is provided)
        if self.log_file_path:
            log_file = Path(self.log_file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)
    
    @classmethod
    def get_logger(cls, name, log_file_path=None, log_level=logging.INFO):
        """
        Factory method to get or create a logger instance
        
        Args:
            name: Logger name
            log_file_path: Path to log file (optional)
            log_level: Logging level (default: INFO)
        
        Returns:
            DataPipelineLogger instance
        """
        return cls(name, log_file_path, log_level)