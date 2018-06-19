"""Miscellaneous utilities related to logging."""

import re
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
    stream : 'stdout', 'stderr' or file-like
        output stream to which logger output will be directed.
    level : int
        Logging level for this logger. Default is logging.INFO (level 20).
    use_format : bool
        Set to True to use the openmdao format "Level: message".
    """
    if stream == 'stdout':
        stream = sys.stdout
    elif stream == 'stderr':
        stream = sys.stderr
    handler = logging.StreamHandler(stream)

    # set a format which is simpler for console use
    if use_format:
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)

    handler.setLevel(level)
    logger.addHandler(handler)


def get_logger(name='default_logger', level=logging.INFO, use_format=False,
               out_stream='stdout', lock=None):
    """
    Return a logger that writes to an I/O stream.

    Parameters
    ----------
    name : str
        Name of the logger to be returned, will be created if it doesn't exist.
    level : int
        Logging level for this logger. Default is logging.INFO (level 20).
        (applied only when creating a new logger or setting a new stream).
    use_format : bool
        Set to True to use the openmdao format "Level: message".
        (applied only when creating a new logger or setting a new stream).
    out_stream : 'stdout', 'stderr' or file-like
        output stream to which logger output will be directed.
    lock : bool
        if True, do not allow the handler to be changed until unlocked.
        if False, unlock the handler for the logger.

    Returns
    -------
    <logging.Logger>
        Logger that writes to a stream and adheres to requested settings.
    """
    if name in _loggers:
        # use existing logger
        info = _loggers[name]
        logger = info['logger']
        stream = info['stream']
        locked = info['locked']

        unlock = lock is False

        # redirect log to new stream (if not locked)
        if out_stream != stream and (not locked or unlock):
            for handler in logger.handlers:
                logger.removeHandler(handler)
            if out_stream:
                _set_handler(logger, out_stream, level, use_format)
            info['stream'] = out_stream

        # update locked status
        info['locked'] = lock
    else:
        # create new logger
        logger = logging.getLogger(name)

        if out_stream:
            _set_handler(logger, out_stream, level, use_format)

        logger.setLevel(level)

        _loggers[name] = {
            'logger': logger,
            'stream': out_stream,
            'locked': lock
        }

    return logger


class TestLogger(object):
    """
    A logger replacement for testing that simplifies checking log output.

    Attributes
    ----------
    _msgs : dict
        Stores lists of messages under 'error', 'warning' and 'info' keys.
    """

    def __init__(self):
        """
        Initialize the message dict.
        """
        self._msgs = {'error': [], 'warning': [], 'info': []}

    def error(self, msg):
        """
        Collect an error message.

        Parameters
        ----------
        msg : str
            An error message.
        """
        self._msgs['error'].append(msg)

    def warning(self, msg):
        """
        Collect a warning message.

        Parameters
        ----------
        msg : str
            A warning message.
        """
        self._msgs['warning'].append(msg)

    def info(self, msg):
        """
        Collect an informational message.

        Parameters
        ----------
        msg : str
            An informational message.
        """
        self._msgs['info'].append(msg)

    def debug(self, msg):
        """
        Collect a debug message.

        Parameters
        ----------
        msg : str
            A debugging message.
        """
        self._msgs['debug'].append(msg)

    def get(self, typ):
        """
        Return all stored messages of a specific type.

        Parameters
        ----------
        typ : str
            Type of messages ('error', 'warning', 'info') to be returned.

        Returns
        -------
        list of str
            Any messages of that type that have been written to the logger.
        """
        return self._msgs[typ]

    def contains_line(self, typ, line):
        """
        Do any of the lines of stored messages of a specific type equal the given line.

        Parameters
        ----------
        typ : str
            Type of messages ('error', 'warning', 'info') to be returned.

        line : str
            Line used to look for matches.

        Returns
        -------
        bool
            True if any of the lines of stored messages of a specific type equal the line.
        """
        for s in self._msgs[typ]:
            if s == line:
                return True

        return False
