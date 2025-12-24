# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Logging configuration for tau2-bench evaluation framework.

This module provides:
- Custom log levels: PROFILE (15) and USER_JUDGE (16)
- Thread-safe task context management
- Structured logging with task_id, thread_id, and millisecond timestamps
"""

import logging
import threading
import time
from datetime import datetime
from typing import Optional, Any, Dict
from contextlib import contextmanager

# Custom log levels (between DEBUG=10 and INFO=20)
PROFILE = 15
USER_JUDGE = 16

# Register custom levels
logging.addLevelName(PROFILE, "PROFILE")
logging.addLevelName(USER_JUDGE, "USER_JUDGE")


class TaskContext(threading.local):
    """Thread-local storage for task execution context."""
    
    def __init__(self):
        super().__init__()
        self.task_id: Optional[str] = None
        self.step: Optional[int] = None
        self.domain: Optional[str] = None


# Global thread-local task context
_task_context = TaskContext()


def set_task_context(task_id: str, domain: Optional[str] = None):
    """Set the current task context for this thread."""
    _task_context.task_id = task_id
    _task_context.domain = domain
    _task_context.step = None


def set_step_context(step: int):
    """Set the current step number for this thread."""
    _task_context.step = step


def clear_task_context():
    """Clear the task context for this thread."""
    _task_context.task_id = None
    _task_context.step = None
    _task_context.domain = None


def get_task_context() -> Dict[str, Any]:
    """Get the current task context."""
    return {
        "task_id": getattr(_task_context, "task_id", None),
        "step": getattr(_task_context, "step", None),
        "domain": getattr(_task_context, "domain", None),
    }


@contextmanager
def task_context(task_id: str, domain: Optional[str] = None):
    """Context manager for task execution."""
    set_task_context(task_id, domain)
    try:
        yield
    finally:
        clear_task_context()


class Tau2Formatter(logging.Formatter):
    """Custom formatter that includes task context and millisecond precision timestamps."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add millisecond timestamp
        ct = datetime.fromtimestamp(record.created)
        record.timestamp_ms = ct.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(ct.microsecond / 1000):03d}"
        
        # Add thread ID
        record.thread_id = threading.current_thread().ident
        
        # Add task context
        ctx = get_task_context()
        record.task_id = ctx.get("task_id") or "global"
        record.step_num = ctx.get("step")
        
        return super().format(record)


class Tau2Logger(logging.Logger):
    """Extended logger with profile and user_judge methods."""
    
    def profile(self, msg: str, *args, **kwargs):
        """Log at PROFILE level for agent LLM calls, tool calls, and step timing."""
        if self.isEnabledFor(PROFILE):
            self._log(PROFILE, msg, args, **kwargs)
    
    def user_judge(self, msg: str, *args, **kwargs):
        """Log at USER_JUDGE level for user simulation and evaluator LLM calls."""
        if self.isEnabledFor(USER_JUDGE):
            self._log(USER_JUDGE, msg, args, **kwargs)


# Set Tau2Logger as the default logger class
logging.setLoggerClass(Tau2Logger)


def get_tau2_logger(name: str = "tau2") -> Tau2Logger:
    """Get a tau2 logger instance."""
    return logging.getLogger(name)


def configure_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    stream_handler: bool = True,
    file_handler: Optional[str] = None,
):
    """
    Configure the tau2 logging system.
    
    Args:
        level: Log level string (DEBUG, PROFILE, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (optional)
        stream_handler: Whether to add a stream handler (default: True)
        file_handler: Path to log file (optional)
    """
    # Map string levels to numeric values
    level_map = {
        "DEBUG": logging.DEBUG,
        "PROFILE": PROFILE,
        "USER_JUDGE": USER_JUDGE,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    numeric_level = level_map.get(level.upper(), logging.INFO)
    
    # Default format with task context
    if format_string is None:
        format_string = "[%(levelname)s] %(timestamp_ms)s task=%(task_id)s thread=%(thread_id)s %(message)s"
    
    formatter = Tau2Formatter(format_string)
    
    # Configure root tau2 logger
    logger = get_tau2_logger()
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    if stream_handler:
        sh = logging.StreamHandler()
        sh.setLevel(numeric_level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    
    if file_handler:
        fh = logging.FileHandler(file_handler)
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


class Timer:
    """Simple context manager for measuring wall clock time in milliseconds."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_ms: Optional[float] = None
    
    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds (can be called during timing)."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return (end - self.start_time) * 1000


def log_profile_event(
    event_type: str,
    **kwargs
) -> None:
    """
    Log a PROFILE level event with structured data.
    
    Args:
        event_type: Type of event (llm_call, expert_call, tool_call, step_complete)
        **kwargs: Additional event data (model, function, duration_ms, etc.)
    """
    logger = get_tau2_logger()
    ctx = get_task_context()
    
    # Build message parts
    parts = [f"type={event_type}"]
    
    # Add step if available
    if ctx.get("step") is not None:
        parts.append(f"step={ctx['step']}")
    
    # Add all additional kwargs
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, float):
                parts.append(f"{key}={value:.2f}")
            else:
                parts.append(f"{key}={value}")
    
    logger.profile(" ".join(parts))


def log_user_judge_event(
    event_type: str,
    **kwargs
) -> None:
    """
    Log a USER_JUDGE level event with structured data.
    
    Args:
        event_type: Type of event (user_sim, evaluator)
        **kwargs: Additional event data (model, duration_ms, etc.)
    """
    logger = get_tau2_logger()
    ctx = get_task_context()
    
    # Build message parts
    parts = [f"type={event_type}"]
    
    # Add step if available
    if ctx.get("step") is not None:
        parts.append(f"step={ctx['step']}")
    
    # Add all additional kwargs
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, float):
                parts.append(f"{key}={value:.2f}")
            else:
                parts.append(f"{key}={value}")
    
    logger.user_judge(" ".join(parts))


def log_debug_event(msg: str, **kwargs) -> None:
    """Log a DEBUG level event with structured data."""
    logger = get_tau2_logger()
    
    if kwargs:
        parts = [msg]
        for key, value in kwargs.items():
            if value is not None:
                parts.append(f"{key}={value}")
        logger.debug(" ".join(parts))
    else:
        logger.debug(msg)

