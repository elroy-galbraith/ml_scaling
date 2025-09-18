"""
Logging utilities for the Random Forest scaling research.

This module provides standardized logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config


def setup_logging(
    log_level: str = None,
    log_file: Optional[Path] = None,
    log_format: str = None
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, logs only to console)
        log_format: Log message format string
        
    Returns:
        Configured logger instance
    """
    if log_level is None:
        log_level = config.LOG_LEVEL
    
    if log_format is None:
        log_format = config.LOG_FORMAT
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    # Create logger
    logger = logging.getLogger('scaling_research')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    logger.info(f"Logging configured at {log_level} level")
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'scaling_research.{name}')


class ProgressLogger:
    """Utility class for logging progress in long-running operations."""
    
    def __init__(self, logger: logging.Logger, total_steps: int, log_interval: int = 10):
        """
        Initialize progress logger.
        
        Args:
            logger: Logger instance to use
            total_steps: Total number of steps in the operation
            log_interval: How often to log progress (every N steps)
        """
        self.logger = logger
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.current_step = 0
    
    def update(self, step_name: str = None) -> None:
        """
        Update progress and log if necessary.
        
        Args:
            step_name: Optional name of the current step
        """
        self.current_step += 1
        
        if (self.current_step % self.log_interval == 0 or 
            self.current_step == self.total_steps):
            
            progress_pct = (self.current_step / self.total_steps) * 100
            
            message = f"Progress: {self.current_step}/{self.total_steps} ({progress_pct:.1f}%)"
            if step_name:
                message += f" - {step_name}"
            
            self.logger.info(message)
    
    def finish(self, message: str = "Operation completed") -> None:
        """
        Log completion message.
        
        Args:
            message: Completion message
        """
        self.logger.info(f"{message} ({self.total_steps} steps)")


# Initialize default logger
default_logger = setup_logging()