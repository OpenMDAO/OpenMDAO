"""Various utilty functions for testing."""

import numpy as np
from math import isnan


def assert_rel_error(test_case, actual, desired, tolerance):
    """Check relative error.

    Determine that the relative error between `actual` and `desired`
    is within `tolerance`. If `desired` is zero, then use absolute error.

    Args
    ----
    test_case : :class:`unittest.TestCase`
        TestCase instance used for assertions.

    actual : float
        The value from the test.

    desired : float
        The value expected.

    tolerance : float
        Maximum relative error ``(actual - desired) / desired``.
    """
    try:
        actual[0]
    except (TypeError, IndexError):
        if isnan(actual) and not isnan(desired):
            test_case.fail('actual nan, desired %s' % desired)
        if desired != 0:
            error = (actual - desired) / desired
        else:
            error = actual
        if abs(error) > tolerance:
            test_case.fail('actual %s, desired %s, rel error %s, tolerance %s'
                           % (actual, desired, error, tolerance))

    # array values
    else:
        if not np.all(np.isnan(actual) == np.isnan(desired)):
            test_case.fail('actual and desired values have non-matching nan'
                           ' values')

        if np.linalg.norm(desired) == 0:
            error = np.linalg.norm(actual)
        else:
            error = np.linalg.norm(actual - desired) / np.linalg.norm(desired)

        if abs(error) > tolerance:
            test_case.fail('arrays do not match, rel error %.3e > tol (%.3e)' %
                           (error, tolerance))

    return error


class TestLogger(object):
    """A logger replacement for testing that simplifies checking log output."""

    def __init__(self):
        """Initialize the message dict."""
        self._msgs = {'error': [], 'warning': [], 'info': []}

    def error(self, msg):
        """Collect an error message.

        Args
        ----
        msg : str
            An error message.
        """
        self._msgs['error'].append(msg)

    def warning(self, msg):
        """Collect a warning message.

        Args
        ----
        msg : str
            A warning message.
        """
        self._msgs['warning'].append(msg)

    def info(self, msg):
        """Collect an informational message.

        Args
        ----
        msg : str
            An informational message.
        """
        self._msgs['info'].append(msg)

    def debug(self, msg):
        """Collect a debug message.

        Args
        ----
        msg : str
            A debugging message.
        """
        self._msgs['debug'].append(msg)

    def get(self, typ):
        """Return all stored messages of a specific type."""
        return self._msgs[typ]
