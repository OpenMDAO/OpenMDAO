"""Miscellaneous utilities related to logging."""

import sys
import logging

# If any method that creates handlers is called twice (e.g., setup reconfigure or during tests),
# then we need to prevent another one from being created. Since we have multiple loggers now, we
# store them in a dictionary.
_loggers = {}


def _set_handler(logger, handler, level, use_format):
    """
    Set the StreamHandler for logger.

    Parameters
    ----------
    logger : object
        Logger object.
    handler : logging handler
        handler to add to the logger
    level : int
        Logging level for this logger. Default is logging.INFO (level 20).
    use_format : bool
        Set to True to use the openmdao format "Level: message".
    """
    # set a format which is simpler for console use
    if use_format:
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)

    handler.setLevel(level)
    logger.addHandler(handler)


def get_logger(name='default_logger', level=logging.INFO, use_format=False,
               out_stream='stdout', out_file=None, lock=None):
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
    out_file : str or None
        If not None, add a FileHandler to write to this file.
    lock : bool
        if True, do not allow the handler to be changed until unlocked.
        if False, unlock the handler for the logger.

    Returns
    -------
    <logging.Logger>
        Logger that writes to a stream and adheres to requested settings.
    """
    if out_stream == 'stdout':
        out_stream = sys.stdout
    elif out_stream == 'stderr':
        out_stream = sys.stderr

    if name in _loggers:
        # use existing logger
        info = _loggers[name]
        logger = info['logger']
        stream = info['stream']
        ofile = info['file']
        locked = info['locked']

        unlock = lock is False

        # redirect log to new stream (if not locked)
        if (out_stream != stream or ofile != out_file) and (not locked or unlock):
            for handler in logger.handlers:
                logger.removeHandler(handler)
            if out_stream:
                _set_handler(logger, logging.StreamHandler(out_stream), level, use_format)
            if out_file:
                _set_handler(logger, logging.FileHandler(out_file, mode='w'), level, use_format)
            info['stream'] = out_stream
            info['file'] = out_file

        # update locked status
        info['locked'] = lock
    else:
        # create new logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if out_stream:
            _set_handler(logger, logging.StreamHandler(out_stream), level, use_format)
        if out_file:
            _set_handler(logger, logging.FileHandler(out_file, mode='w'), level, use_format)

        _loggers[name] = {
            'logger': logger,
            'stream': out_stream,
            'file': out_file,
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

    def contains(self, typ, message):
        """
        Do any of the stored messages of a specific type equal the given message.

        Parameters
        ----------
        typ : str
            Type of messages ('error', 'warning', 'info') to be returned.

        message : str
            The message to match.

        Returns
        -------
        bool
            True if any of the lines of stored messages of a specific type equal the line.
        """
        for s in self._msgs[typ]:
            if s == message:
                return True

        return False

    def find_in(self, typ, message):
        """
        Find the given message among the stored messages.

        Raises an exception if the given message isn't found.

        Parameters
        ----------
        typ : str
            Type of messages ('error', 'warning', 'info') to be searched.

        message : str
            The message to match.
        """
        if not self.contains(typ, message):
            raise RuntimeError('Message "{}" not found in {}.'.format(message,
                                                                      ',\n'.join(self._msgs[typ])))
