"""Miscellaneous utilities related to logging."""

import sys
import logging

# If any method that creates handlers is called twice (e.g., setup reconfigure or during tests),
# then we need to prevent another one from being created. Since we have multiple loggers now, we
# store them in a dictionary.
_set_logger = {}


def get_default_logger(logger, name='default_logger', level=logging.INFO, use_format=False):
    """
    Return a logger that prints to stdout.

    Parameters
    ----------
    logger : object
        Logger object. If a valid one is not passed in, then create a new one that writes to
        stdout.
    name : str
        Name of the logger to be created.
    level : int
        Logging level for this logger. Default is logging.INFO (level 20).
    use_format : bool
        Set to True to use the openmdao format "Level: message"

    Returns
    -------
    <logging.Logger>
        Logger that writes to stdout and adheres to requested settings.
    """
    global _set_logger
    if logger is None:
        if name not in _set_logger:

            logger = logging.getLogger(name)
            _set_logger[name] = logger
            console = logging.StreamHandler(sys.stdout)

            # set a format which is simpler for console use
            if use_format:
                formatter = logging.Formatter('%(levelname)s: %(message)s')

                # tell the handler to use this format
                console.setFormatter(formatter)

            console.setLevel(level)
            logger.addHandler(console)

        else:
            logger = _set_logger[name]

    return logger
