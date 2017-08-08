"""Miscellaneous utilities related to logging."""

import sys
import logging

# If any method that creates handlers is called twice (e.g., setup reconfigure or during tests),
# then we need to prevent another one from being created. Since we have multiple loggers now, we
# store them in a dictionary.
_loggers = {}


def _set_handler(logger, stream, level, use_format):
    """
    Set the StreamHandler for logger.

    Parameters
    ----------
    logger : object
        Logger object.
    level : int
        Logging level for this logger. Default is logging.INFO (level 20).
    use_format : bool
        Set to True to use the openmdao format "Level: message"
    stream : file-like
        an output stream to which logger output will be directed
    """
    handler = logging.StreamHandler(stream)

    # set a format which is simpler for console use
    if use_format:
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)

    handler.setLevel(level)
    logger.addHandler(handler)


def get_default_logger(logger=None, name='default_logger',
                       level=logging.INFO, use_format=False, stream=None):
    """
    Return a logger that prints to an I/O stream.

    Parameters
    ----------
    logger : object
        Logger object. If a valid one is not passed in, then create a new one.
    name : str
        Name of the logger to be returned, will be created if it doesn't exist.
    level : int
        Logging level for this logger. Default is logging.INFO (level 20).
        (applied only when creating a new logger or setting a new stream)
    use_format : bool
        Set to True to use the openmdao format "Level: message"
        (applied only when creating a new logger or setting a new stream)
    stream : 'stdout' or file-like,
        output stream to which logger output will be directed

    Returns
    -------
    <logging.Logger>
        Logger that writes to stdout and adheres to requested settings.
    """
    global _loggers
    if logger is None:
        if name in _loggers:
            # use existing logger
            logger = _loggers[name]

            # log may be getting redirected to a different stream
            if stream:
                for handler in logger.handlers:
                    logger.removeHandler(handler)
                _set_handler(logger, stream, level, use_format)
        else:
            # create new logger
            logger = logging.getLogger(name)
            _loggers[name] = logger

            _set_handler(logger, stream, level, use_format)

            logger.setLevel(level)

    return logger
