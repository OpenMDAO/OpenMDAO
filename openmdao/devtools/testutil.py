"""Various utility functions for testing."""

import numpy as np
from math import isnan
from six import raise_from


def assert_rel_error(test_case, actual, desired, tolerance=1e-15):
    """Check relative error.

    Determine that the relative error between `actual` and `desired`
    is within `tolerance`. If `desired` is zero, then use absolute error.

    Parameters
    ----------
    test_case : :class:`unittest.TestCase`
        TestCase instance used for assertions.

    actual : float, array-like, dict
        The value from the test.

    desired : float, array-like, dict
        The value expected.

    tolerance : float
        Maximum relative error ``(actual - desired) / desired``.
    """
    if isinstance(actual, dict) and isinstance(desired, dict):

        actual_keys = set(actual.keys())
        desired_keys = set(desired.keys())

        if actual_keys.symmetric_difference(desired_keys):
            msg = 'Actual and desired keys differ. Actual extra keys: {}, Desired extra keys: {}'
            actual_extra = actual_keys.difference(desired_keys)
            desired_extra = desired_keys.difference(actual_keys)
            test_case.fail(msg.format(actual_extra, desired_extra))

        error = 0.

        for key in actual_keys:
            try:
                new_error = assert_rel_error(test_case, actual[key], desired[key], tolerance)
                error = max(error, new_error)
            except test_case.failureException as exception:
                msg = '{}: '.format(key) + str(exception)
                raise_from(test_case.failureException(msg), None)

    elif isinstance(actual, float) and isinstance(desired, float):
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
        actual = np.atleast_1d(actual)
        desired = np.atleast_1d(desired)
        if actual.shape != desired.shape:
            test_case.fail(
                'actual and desired have differing shapes.'
                ' actual {}, desired {}'.format(actual.shape, desired.shape))
        if not np.all(np.isnan(actual) == np.isnan(desired)):
            if actual.size == 1 and desired.size == 1:
                test_case.fail('actual %s, desired %s' % (actual, desired))
            else:
                test_case.fail('actual and desired values have non-matching nan'
                               ' values')

        if np.linalg.norm(desired) == 0:
            error = np.linalg.norm(actual)
        else:
            error = np.linalg.norm(actual - desired) / np.linalg.norm(desired)

        if abs(error) > tolerance:
            if actual.size < 10 and desired.size < 10:
                test_case.fail('actual %s, desired %s, rel error %s, tolerance %s'
                               % (actual, desired, error, tolerance))
            else:
                test_case.fail('arrays do not match, rel error %.3e > tol (%.3e)' %
                               (error, tolerance))

    return error


def assertEqualArrays(a1, a2):
    """
    Check that two arrays are equal.

    This is a simplified method useful when the arrays to be compared may
    not be numeric. It simply compares the shapes of the two arrays and then
    does a value by value comparison.

    Parameters
    ----------
    a1 : array
        The first array to compare.

    a2 : array
        The second array to compare.
    """
    assert a1.shape == a2.shape

    flat1 = a1.flatten()
    flat2 = a2.flatten()

    for x, y in zip(flat1, flat2):
        assert x == y


class TestLogger(object):
    """A logger replacement for testing that simplifies checking log output."""

    def __init__(self):
        """Initialize the message dict."""
        self._msgs = {'error': [], 'warning': [], 'info': []}

    def error(self, msg):
        """Collect an error message.

        Parameters
        ----------
        msg : str
            An error message.
        """
        self._msgs['error'].append(msg)

    def warning(self, msg):
        """Collect a warning message.

        Parameters
        ----------
        msg : str
            A warning message.
        """
        self._msgs['warning'].append(msg)

    def info(self, msg):
        """Collect an informational message.

        Parameters
        ----------
        msg : str
            An informational message.
        """
        self._msgs['info'].append(msg)

    def debug(self, msg):
        """Collect a debug message.

        Parameters
        ----------
        msg : str
            A debugging message.
        """
        self._msgs['debug'].append(msg)

    def get(self, typ):
        """Return all stored messages of a specific type."""
        return self._msgs[typ]
