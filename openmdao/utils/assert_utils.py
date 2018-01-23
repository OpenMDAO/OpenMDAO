"""
Functions for making assertions about OpenMDAO Systems.
"""
import numpy as np
from math import isnan
from six import raise_from

from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian

from numpy.testing import assert_allclose


def assert_check_partials(data, atol=1e-6, rtol=1e-6):
    """
    Raise assertion if any entry from the return from check_partials is above a tolerance.

    Parameters
    ----------
    data : dict of dicts of dicts
            First key:
                is the component name;
            Second key:
                is the (output, input) tuple of strings;
            Third key:
                is one of ['rel error', 'abs error', 'magnitude', 'J_fd', 'J_fwd', 'J_rev'];

            For 'rel error', 'abs error', 'magnitude' the value is: A tuple containing norms for
                forward - fd, adjoint - fd, forward - adjoint.
            For 'J_fd', 'J_fwd', 'J_rev' the value is: A numpy array representing the computed
                Jacobian for the three different methods of computation.
    atol : float
        absolute error. Default is 1e-6.
    rtol : float
        relative error. Default is 1e-6.
    """
    desired = (0.0, 0.0, 0.0)
    for comp in data:
        for (var, wrt) in data[comp]:
            for error_type, tolerance in [('abs error', atol), ('rel error', rtol), ]:
                actual = data[comp][var, wrt][error_type]
                if not np.isnan(actual).any():
                    assert_allclose(actual, desired, atol=tolerance,
                                    err_msg='{0} error in partial of'
                                            ' {1} wrt {2} in component {3}'.format(error_type,
                                                                                   var, wrt, comp),
                                    verbose=True)


def assert_no_approx_partials(system, include_self=True, recurse=True):
    """
    Raise assertion error if any component within system is using approximated partials.

    Parameters
    ----------
    system : System
        The system under which to search for approximated partials.
    include_self : bool
        If True, include this system in the iteration.
    recurse : bool
        If True, iterate over the whole tree under this system.

    Raises
    ------
    AssertionError
        If a subsystem of group is found to be using approximated partials.

    """
    approximated_partials = {}
    msg = 'The following components use approximated partials:\n'
    for s in system.system_iter(include_self=include_self, recurse=recurse):
        if isinstance(s, Component):
            if s._approximated_partials:
                approximated_partials[s.pathname] = s._approximated_partials
                msg += '    ' + s.pathname + '\n'
                for partial in s._approximated_partials:
                    msg += '        of={0:12s}    wrt={1:12s}    method={2:2s}\n'.format(*partial)
    if approximated_partials:
        raise AssertionError(msg)


def assert_no_dict_jacobians(system, include_self=True, recurse=True):
    """
    Raise an assertion error if any Group within system is found to be using dictionary jacobians.

    Parameters
    ----------
    system : System
        The system under which to search for approximated partials.
    include_self : bool
        If True, include this system in the iteration.
    recurse : bool
        If True, iterate over the whole tree under this system.

    Raises
    ------
    AssertionError
        If a subsystem of group is found to be using approximated partials.

    """
    groups_with_dict_jacobians = []
    msg = 'The following groups use dictionary jacobians:\n'
    for s in system.system_iter(include_self=include_self, recurse=recurse, typ=Group):
        if isinstance(s.jacobian, DictionaryJacobian):
            groups_with_dict_jacobians.append(s.pathname)
    msg = msg + '\n'.join(['    ' + s for s in groups_with_dict_jacobians])
    if groups_with_dict_jacobians:
        raise AssertionError(msg)


def assert_rel_error(test_case, actual, desired, tolerance=1e-15):
    """
    Check relative error.

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

    Returns
    -------
    float
        The error.
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


def assert_equal_arrays(a1, a2):
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
    for x, y in zip(a1.flat, a2.flat):
        assert x == y
