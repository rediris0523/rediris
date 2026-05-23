import logging
import sys
import os
from pathlib import Path
from typing import Optional

_module_prefix: Optional[str] = None
_global_log_level: Optional[str] = None

def set_module_prefix(prefix: str):

    global _module_prefix
    _module_prefix = prefix

    _update_all_logger_formats()


def get_module_prefix() -> Optional[str]:
    return _module_prefix


def set_global_log_level(level: str):

    global _global_log_level
    _global_log_level = level.upper()

    _update_all_logger_levels()


def get_global_log_level() -> Optional[str]:
    return _global_log_level


def _update_all_logger_levels():
    if _global_log_level is None:
        return

    level = getattr(logging, _global_log_level, logging.INFO)

    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith('rediris'):
            logger = logging.getLogger(name)
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)


def reinitialize_all_loggers(log_file: Optional[str] = None):

    log_level = _global_log_level or "INFO"
    level = getattr(logging, log_level.upper(), logging.INFO)

    moirai_logger_names = [name for name in logging.Logger.manager.loggerDict.keys() if name.startswith('rediris')]

    reinitialized_count = 0
    for name in moirai_logger_names:
        logger = logging.getLogger(name)
        logger.setLevel(level)

        logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(_get_log_format(include_file_info=False)))
        logger.addHandler(console_handler)

        if log_file:
            try:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(level)
                file_handler.setFormatter(logging.Formatter(_get_log_format(include_file_info=True)))
                logger.addHandler(file_handler)
            except Exception:
                pass

        logger.propagate = False
        reinitialized_count += 1

    root_logger = logging.getLogger('rediris')
    if not root_logger.handlers:
        root_logger.setLevel(level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(_get_log_format(include_file_info=False)))
        root_logger.addHandler(console_handler)
        root_logger.propagate = False

    print(f"[logging] Reinitialized {reinitialized_count} rediris loggers: {moirai_logger_names}", flush=True)


def _get_log_format(include_file_info: bool = False) -> str:
    if _module_prefix:
        if include_file_info:
            return f'%(asctime)s - [{_module_prefix}] - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        return f'%(asctime)s - [{_module_prefix}] - %(name)s - %(levelname)s - %(message)s'
    else:
        if include_file_info:
            return '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        return '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def _update_all_logger_formats():

    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith('rediris'):
            logger = logging.getLogger(name)
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setFormatter(logging.Formatter(_get_log_format(include_file_info=False)))
                elif isinstance(handler, logging.FileHandler):
                    handler.setFormatter(logging.Formatter(_get_log_format(include_file_info=True)))


def setup_logger(name: str, log_level: Optional[str] = None, log_file: Optional[str] = None) -> logging.Logger:

    if log_level is None:
        if _global_log_level is not None:
            log_level = _global_log_level
        else:
            try:
                from rediris.common.config import settings
                log_level = settings.LOG_LEVEL
            except (ImportError, AttributeError):
                log_level = "INFO"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    logger.propagate = False

    if logger.handlers:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setFormatter(logging.Formatter(_get_log_format(include_file_info=False)))
            elif isinstance(handler, logging.FileHandler):
                handler.setFormatter(logging.Formatter(_get_log_format(include_file_info=True)))
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_formatter = logging.Formatter(_get_log_format(include_file_info=False))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            file_formatter = logging.Formatter(_get_log_format(include_file_info=True))
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to setup file logging to {log_file}: {e}")

    return logger

