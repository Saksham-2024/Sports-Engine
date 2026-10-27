import pytest
import logging
import sys
from pathlib import Path
from src.core.utils.logger import get_logger

def test_get_logger_console_only():
    """Test get_logger returns a logger with only a StreamHandler by default."""
    logger_name = "test_console_logger"
    logger = get_logger(logger_name)
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == logger_name
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].stream == sys.stdout

def test_get_logger_with_file(tmp_path):
    """Test get_logger returns a logger with FileHandler when log_file is provided."""
    logger_name = "test_file_logger"
    log_file = tmp_path / "test.log"
    logger = get_logger(logger_name, log_file=log_file)
    
    assert len(logger.handlers) == 2
    handler_types = [type(h) for h in logger.handlers]
    assert logging.StreamHandler in handler_types
    assert logging.FileHandler in handler_types
    
    # Check that log file was created
    assert log_file.exists()

def test_get_logger_no_duplicate_handlers():
    """Test get_logger doesn't add multiple handlers on repeated calls."""
    logger_name = "test_duplicate_logger"
    logger1 = get_logger(logger_name)
    assert len(logger1.handlers) == 1
    
    logger2 = get_logger(logger_name)
    assert logger1 is logger2
    assert len(logger2.handlers) == 1
