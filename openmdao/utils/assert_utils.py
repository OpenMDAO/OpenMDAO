"""
Functions for making assertions about OpenMDAO Systems.
"""
import numpy as np
from math import isnan

import warnings
import unittest

from contextlib import contextmanager
from functools import wraps

from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian
from openmdao.utils.general_utils import pad_name
from openmdao.warnings import warn_deprecation, reset_warning_registry


@contextmanager
def assert_warning(category, msg):
    """
    Context manager asserting that a warning is issued.

    Parameters
    ----------
    category : class
        The class of the expected warning.
    msg : str
        The text of the expected warning.

    Raises
    ------
    AssertionError
        If the expected warning is not raised.
    """
    with reset_warning_registry():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            yield

    for warn in w:
        if (issubclass(warn.category, category) and str(warn.message) == msg):
            break
    else:
        msg = f"Did not see expected {category.__name__}:\n{msg}"
        if w:
            ws = '\n'.join([str(ww.message) for ww in w])
            categories = '\n'.join([str(ww.category.__name__) for ww in w])
            msg += f"\nDid see warnings [{categories}]:\n{ws}"
        raise AssertionError(msg)


@contextmanager
def assert_warnings(expected_warnings):
    """
    Context manager asserting that expected warnings are issued.

    Parameters
    ----------
    expected_warnings : iterable of (class, str)
        The category and text of the expected warnings.

    Raises
    ------
    AssertionError
        If all the expected warnings are not raised.
    """
    with reset_warning_registry():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            yield

    for category, msg in expected_warnings:
        for warn in w:
            if (issubclass(warn.category, category) and str(warn.message) == msg):
                break
        else:
            raise AssertionError("Did not see expected %s: %s" % (category.__name__, msg))


@contextmanager
def assert_no_warning(category, msg=None):
    """
    Context manager asserting that a warning is not issued.

    Parameters
    ----------
    category : class
        The class of the warning.
    msg : str or None
        The text of the warning. If None then only the warning class will be checked.

    Raises
    ------
    AssertionError
        If the warning is raised.
    """
    with reset_warning_registry():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            yield

    for warn in w:
        if issubclass(warn.category, category):
            if msg is None:
                raise AssertionError(f"Found warning: {category} {str(warn.message)}")
            elif str(warn.message) == msg:
                raise AssertionError(f"Found warning: {category} {msg}")


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
    error_string = ''
    absrel_header = 'abs/rel'
    wrt_header = '< output > wrt < variable >'
    norm_value_header = 'norm value'
    len_absrel_width = len(absrel_header)
    norm_types = ['fwd-fd', 'rev-fd', 'fd-rev']
    len_norm_type_width = max(len(s) for s in norm_types)

    for comp in data:
        len_wrt_width = len(wrt_header)
        len_norm_width = len(norm_value_header)
        bad_derivs = []

        # Find all derivatives whose errors exceed tolerance.
        # Also, size the output to precompute column extents.
        for (var, wrt) in data[comp]:
            pair_data = data[comp][var, wrt]
            for error_type, tolerance in [('abs error', atol), ('rel error', rtol), ]:
                actual = pair_data[error_type]
                for error_val, mode in zip(actual, norm_types):
                    in_error = False

                    if error_val is None:
                        # Reverse derivatives only computed on matrix free comps.
                        continue

                    if not np.isnan(error_val):
                        if not np.allclose(error_val, 0.0, atol=tolerance):

                            if error_type == 'rel error' and mode == 'fwd-fd' and \
                               np.allclose(pair_data['J_fwd'], 0.0, atol=atol) and \
                               np.allclose(pair_data['J_fd'], 0.0, atol=atol):
                                # Special case: both fd and fwd are really tiny, so we want to
                                # ignore the rather large relative errors.
                                in_error = False
                            else:
                                # This is a bona-fide error.
                                in_error = True

                    elif error_type == 'abs error' and mode == 'fwd-fd':
                        # Either analytic or approximated derivatives contain a NaN.
                        in_error = True

                    if in_error:
                        wrt_string = '{0} wrt {1}'.format(var, wrt)
                        norm_string = '{}'.format(error_val)
                        bad_derivs.append((wrt_string, norm_string, error_type, mode))
                        len_wrt_width = max(len_wrt_width, len(wrt_string))
                        len_norm_width = max(len_norm_width, len(norm_string))

        if bad_derivs:
            comp_error_string = ''
            for wrt_string, norm_string, error_type, mode in bad_derivs:
                err_msg = '{0} | {1} | {2} | {3}'.format(
                    pad_name(wrt_string, len_wrt_width),
                    pad_name(error_type.split()[0], len_absrel_width),
                    pad_name(mode, len_norm_type_width),
                    pad_name(norm_string, len_norm_width)) + '\n'
                comp_error_string += err_msg

            name_header = 'Component: {}\n'.format(comp)
            len_name_header = len(name_header)
            header = len_name_header * '-' + '\n'
            header += name_header
            header += len_name_header * '-' + '\n'
            header += '{0} | {1} | {2} | {3}'.format(
                pad_name(wrt_header, len_wrt_width),
                pad_name(absrel_header, len_absrel_width),
                pad_name('norm', len_norm_type_width),
                pad_name(norm_value_header, len_norm_width),
            ) + '\n'
            header += '{0} | {1} | {2} | {3}'.format(
                len_wrt_width * '-',
                len_absrel_width * '-',
                len_norm_type_width * '-',
                len_norm_width * '-',
            ) + '\n'
            comp_error_string = header + comp_error_string
            error_string += comp_error_string

    # if error string then raise error with that string
    if error_string:
        header_line1 = 'Assert Check Partials failed for the following Components'
        header_line2 = 'with absolute tolerance = {} and relative tolerance = {}'.format(atol, rtol)
        header_width = max(len(header_line1), len(header_line2))
        header = '\n' + header_width * '=' + '\n'
        header += header_line1 + '\n'
        header += header_line2 + '\n'
        header += header_width * '=' + '\n'
        error_string = header + error_string
        raise ValueError(error_string)


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
    has_approx_partials = False
    msg = 'The following components use approximated partials:\n'
    for s in system.system_iter(include_self=include_self, recurse=recurse):
        if isinstance(s, Component):
            if s._approx_schemes:
                has_approx_partials = True
                approx_partials = [(k, v['method']) for k, v in s._declared_partials.items()
                                   if 'method' in v and v['method']]
                msg += '    ' + s.pathname + '\n'
                for key, method in approx_partials:
                    msg += '        of={0:12s}    wrt={1:12s}    method={2:2s}\n'.format(key[0],
                                                                                         key[1],
                                                                                         method)
    if has_approx_partials:
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
    parts = ['The following groups use dictionary jacobians:\n']
    for s in system.system_iter(include_self=include_self, recurse=recurse, typ=Group):
        if isinstance(s._jacobian, DictionaryJacobian):
            parts.append('    ' + s.pathname)
    if len(parts) > 1:
        raise AssertionError('\n'.join(parts))


def assert_rel_error(test_case, actual, desired, tolerance=1e-15):
    """
    Check relative error.

    Determine that the relative error between `actual` and `desired`
    is within `tolerance`. If `desired` is zero, then use absolute error.

    Parameters
    ----------
    test_case : class:`unittest.TestCase`
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
    warn_deprecation("'assert_rel_error' has been deprecated. Use "
                     "'assert_near_equal' instead.")

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
                raise test_case.failureException(msg) from None

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


def assert_near_equal(actual, desired, tolerance=1e-15):
    """
    Check relative error.

    Determine that the relative error between `actual` and `desired`
    is within `tolerance`. If `desired` is zero, then use absolute error.

    Parameters
    ----------
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
            raise KeyError(msg.format(actual_extra, desired_extra))

        error = 0.

        for key in actual_keys:
            try:
                new_error = assert_near_equal(
                    actual[key], desired[key], tolerance)
                error = max(error, new_error)
            except ValueError as exception:
                msg = '{}: '.format(key) + str(exception)
                raise ValueError(msg) from None
            except KeyError as exception:
                msg = '{}: '.format(key) + str(exception)
                raise KeyError(msg) from None

    elif isinstance(actual, float) and isinstance(desired, float):
        if isnan(actual) and not isnan(desired):
            raise ValueError('actual nan, desired %s' % desired)
        if desired != 0:
            error = (actual - desired) / desired
        else:
            error = actual
        if abs(error) > tolerance:
            raise ValueError('actual %s, desired %s, rel error %s, tolerance %s'
                             % (actual, desired, error, tolerance))

    # array values
    else:
        actual = np.atleast_1d(actual)
        desired = np.atleast_1d(desired)
        if actual.shape != desired.shape:
            raise ValueError(
                'actual and desired have differing shapes.'
                ' actual {}, desired {}'.format(actual.shape, desired.shape))
        if not np.all(np.isnan(actual) == np.isnan(desired)):
            if actual.size == 1 and desired.size == 1:
                raise ValueError('actual %s, desired %s' % (actual, desired))
            else:
                raise ValueError('actual and desired values have non-matching nan'
                                 ' values')

        if np.linalg.norm(desired) == 0:
            error = np.linalg.norm(actual)
        else:
            error = np.linalg.norm(actual - desired) / np.linalg.norm(desired)

        if abs(error) > tolerance:
            if actual.size < 10 and desired.size < 10:
                raise ValueError('actual %s, desired %s, rel error %s, tolerance %s'
                                 % (actual, desired, error, tolerance))
            else:
                raise ValueError('arrays do not match, rel error %.3e > tol (%.3e)' %
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


def skip_helper(msg):
    """
    Raise a SkipTest.

    Parameters
    ----------
    msg : str
        The skip messaage.

    Raises
    ------
    SkipTest
    """
    raise unittest.SkipTest(msg)


class SkipParameterized(object):
    """
    Replaces the parameterized class, skipping decorated test cases.
    """

    @classmethod
    def expand(cls, input, name_func=None, doc_func=None, skip_on_empty=False, **legacy):
        """
        Decorate a test so that it raises a SkipTest.

        Parameters
        ----------
        input : iterable
            Not used (part of parameterized API).
        name_func : function
            Not used (part of parameterized API).
        doc_func : function
            Not used (part of parameterized API).
        skip_on_empty : bool
            Not used (part of parameterized API).
        **legacy : dict
            Not used (part of parameterized API).

        Returns
        -------
        function
            The wrapper function.
        """
        skip_msg = "requires 'parameterized' (install openmdao[test])"

        def parameterized_expand_wrapper(f, instance=None):
            """
            Wrap a function so that it raises a SkipTest.

            f : function
                Function to be wrapped.
            instance : None
                Not used (part of parameterized API).

            Returns
            -------
            function
                The wrapped function.
            """
            return wraps(f)(lambda f: skip_helper(skip_msg))

        return parameterized_expand_wrapper
