"""
Functions for making assertions about OpenMDAO Systems.
"""

import sys
from fnmatch import fnmatch
import warnings
import unittest
from contextlib import contextmanager
from functools import wraps
from itertools import chain

import numpy as np

try:
    from jax import Array as JaxArray
except ImportError:
    try:
        from jaxlib.xla_extension import ArrayImpl as JaxArray
    except ImportError:
        JaxArray = None

from openmdao.core.component import Component
from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian
from openmdao.utils.general_utils import add_border, get_max_widths, strs2row_iter
from openmdao.utils.om_warnings import reset_warning_registry, issue_warning
from openmdao.utils.mpi import MPI
from openmdao.utils.testing_utils import snum_equal


@contextmanager
def assert_warning(category, msg, contains_msg=False, ranks=None):
    """
    Context manager asserting that a warning is issued.

    Parameters
    ----------
    category : class
        The class of the expected warning.
    msg : str
        The text of the expected warning.
    contains_msg : bool
        Set to True to check that the warning text contains msg, rather than checking equality.
    ranks : int or list of int, optional
        The global ranks on which the warning is expected.

    Yields
    ------
    None

    Raises
    ------
    AssertionError
        If the expected warning is not raised.
    """
    with reset_warning_registry():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            yield

    if ranks is not None:
        if MPI is None:
            raise RuntimeError("ranks argument has been specified but MPI is not active")
        else:
            if not isinstance(ranks, list):
                ranks = [ranks]
            if MPI.COMM_WORLD.rank not in ranks:
                return

    for warn in w:
        if contains_msg:
            warn_clause = msg in str(warn.message)
        else:
            warn_clause = str(warn.message) == msg

        if (issubclass(warn.category, category) and warn_clause):
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

    Yields
    ------
    None

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
def assert_no_warning(category, msg=None, contains=False):
    """
    Context manager asserting that a warning is not issued.

    Parameters
    ----------
    category : class
        The class of the warning.
    msg : str or None
        The text of the warning. If None then only the warning class will be checked.
    contains : bool
        If True, check that the warning text contains msg, rather than checking equality.

    Yields
    ------
    None

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
            elif contains:
                if msg in str(warn.message):
                    raise AssertionError(f"Found warning: {category} containing '{msg}'")
            elif str(warn.message) == msg:
                raise AssertionError(f"Found warning: {category} {msg}")


def _filter_np_err(msg):
    # remove extraneous lines from numpy error message
    lines = []
    for line in msg.split('\n'):
        if not line.startswith('Not equal to tolerance'):
            line = line.strip()
            if line:
                lines.append(line)
    return '\n'.join(lines)


def _parse_assert_allclose_error(msg):
    """
    Parse the error message from an assert_allclose failure.

    Parameters
    ----------
    msg : str
        The error message.

    Returns
    -------
    (float, float)
        The max absolute error and the max relative error.
    """
    for line in msg.split('\n'):
        line = line.strip()
        if not line.startswith('Max'):
            continue
        parts = line.split()

        # Note change in format of assert_allclose error message for NumPy 2.x

        # Numpy 1.x format:
        # ----------------
        # Not equal to tolerance rtol=1e-06, atol=1e-06
        #
        # Mismatched elements: 15 / 15 (100%)
        # Max absolute difference: 2.7
        # Max relative difference: 1.

        # Numpy 2.x format:
        # ----------------
        # Not equal to tolerance rtol=1e-06, atol=1e-06
        #
        # Mismatched elements: 15 / 15 (100%)
        # Max absolute difference among violations: 2.7
        # Max relative difference among violations: 1.

        if parts[1] == 'absolute':
            abs_err = float(parts[-1])
        elif parts[1] == 'relative':
            rel_err = float(parts[-1])
    return abs_err, rel_err


def assert_check_partials(data, atol=1e-6, rtol=1e-6, verbose=False, max_display_shape=(20, 20)):
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
                is one of ['tol violation', 'magnitude', 'J_fd', 'J_fwd', 'J_rev',
                           'vals_at_max_error', 'directional_fd_fwd',
                           'directional_fd_rev', 'directional_fwd_rev', 'rank_inconsistent',
                           'matrix_free', 'directional', 'steps', and 'rank_inconsistent'].

                For 'J_fd', 'J_fwd', 'J_rev' the value is a numpy array representing the computed
                Jacobian for the three different methods of computation.
                For 'tol violation' and 'vals_at_max_error' the value is a
                tuple containing values for forward - fd, reverse - fd, forward - reverse. For
                'magnitude' the value is a tuple indicating the maximum magnitude of values found in
                Jfwd, Jrev, and Jfd.
    atol : float
        Absolute error. Default is 1e-6.
    rtol : float
        Relative error. Default is 1e-6.
    verbose : bool
        When True, display more jacobian information.
    max_display_shape : tuple of int
        Maximum shape of the jacobians to display directly in the error message.
        Default is (20, 20).  Only active if verbose is True.
    """
    error_strings = []

    if isinstance(data, tuple):
        if len(data) != 2:
            raise RuntimeError(f"partials data format error (tuple of size {len(data)})")
        data = data[0]

    for comp in data:
        bad_derivs = []
        inconsistent_derivs = set()

        # Find all derivatives whose errors exceed tolerance.
        for key, pair_data in data[comp].items():
            if pair_data.get('rank_inconsistent'):
                inconsistent_derivs.add(key)

            J_fds = pair_data['J_fd']
            J_fwd = pair_data.get('J_fwd')
            J_rev = pair_data.get('J_rev')
            dir_fd_fwds = pair_data.get('directional_fd_fwd')
            dir_fd_revs = pair_data.get('directional_fd_rev')
            dir_fwd_rev = pair_data.get('directional_fwd_rev')
            directional = pair_data.get('directional')

            if not isinstance(J_fds, list):
                J_fds = [J_fds]
                dir_fd_fwds = [dir_fd_fwds]
                dir_fd_revs = [dir_fd_revs]
            else:
                if dir_fd_fwds is None:
                    dir_fd_fwds = [dir_fd_fwds]
                if dir_fd_revs is None:
                    dir_fd_revs = [dir_fd_revs]

            dirstr = ' directional' if directional else ''
            jacs = [(f'J_fwd{dirstr}', J_fwd, f'Forward{dirstr}'),
                    (f'J_rev{dirstr}', J_rev, f'Reverse{dirstr}')]

            steps = pair_data.get('steps', [None])

            nrows, ncols = J_fds[0].shape
            if isinstance(max_display_shape, int):
                maxrows = maxcols = max_display_shape
            else:
                try:
                    maxrows, maxcols = max_display_shape
                except ValueError:
                    issue_warning("max_display_shape must be an int or a tuple of two ints, but "
                                  f"got {max_display_shape}. Defaulting to (20, 20).")

            for J_fd, step, dfwd, drev in zip(J_fds, steps, dir_fd_fwds, dir_fd_revs):
                if step is not None:
                    stepstr = f" (step={step})"
                else:
                    stepstr = ""

                analytic_found = False
                for Jname, J, direction in jacs:
                    fwd = direction.startswith('Forward')
                    if J is not None:
                        analytic_found = True
                        try:
                            if fwd and dfwd is not None:
                                J1, J2 = dfwd
                                np.testing.assert_allclose(J1, J2, atol=atol, rtol=rtol,
                                                           verbose=False, equal_nan=False)
                            elif not fwd and drev is not None:
                                J1, J2 = drev
                                np.testing.assert_allclose(J1, J2, atol=atol, rtol=rtol,
                                                           verbose=False, equal_nan=False)
                            else:
                                J1, J2 = J, J_fd
                                np.testing.assert_allclose(J1, J2, atol=atol, rtol=rtol,
                                                           verbose=False, equal_nan=False)
                        except Exception as err:
                            abserr, relerr = _parse_assert_allclose_error(err.args[0])
                            if abserr < atol and not (np.any(J1) or np.any(J2)):
                                # if one array is all zeros and we don't violate the absolute
                                # tolerance, then don't flag the relative error.
                                continue

                            if verbose:
                                bad_derivs.append(f"\n{direction} derivatives of '{key[0]}' wrt "
                                                  f"'{key[1]}' do not match finite "
                                                  f"difference{stepstr}.\n")
                                bad_derivs[-1] += _filter_np_err(err.args[0])
                                if nrows <= maxrows and ncols <= maxcols:
                                    with np.printoptions(linewidth=10000):
                                        bad_derivs[-1] += f'\nJ_fd - {Jname}:\n' + \
                                            np.array2string(J_fd - J)
                            else:
                                bad_derivs.append([f"{key[0]} wrt {key[1]}", "abs",
                                                   f"fd-{Jname[2:]}", f"{abserr}"])
                                bad_derivs.append([f"{key[0]} wrt {key[1]}", "rel",
                                                   f"fd-{Jname[2:]}", f"{relerr}"])

                if not analytic_found:
                    # check if J_fd is all zeros.  If not, then we have a problem.
                    abserr = np.max(np.abs(J_fd))
                    if abserr > atol:
                        if verbose:
                            bad_derivs.append(f"\nAnalytic deriv for '{key[0]}' wrt '{key[1]}' "
                                              f"is assumed zero, but finite difference{stepstr} "
                                              "is nonzero.\n")
                            if nrows <= maxrows and ncols <= maxcols:
                                with np.printoptions(linewidth=10000):
                                    bad_derivs[-1] += '\nJ_fd - J_analytic:\n' + \
                                        np.array2string(J_fd)
                        else:
                            abserr = np.max(np.abs(J_fd))
                            bad_derivs.append([f"{key[0]} wrt {key[1]}", "abs", "fd-fwd",
                                               f"{abserr}"])

            if pair_data.get('matrix_free') is not None and J_fwd is not None and J_rev is not None:
                try:
                    if dir_fwd_rev is not None:
                        dJfwd, dJrev = dir_fwd_rev
                        either_zero = not (np.any(dJfwd) or np.any(dJrev))
                        np.testing.assert_allclose(dJfwd, dJrev, atol=atol, rtol=rtol,
                                                   verbose=False, equal_nan=False)
                    else:
                        either_zero = not (np.any(J_fwd) or np.any(J_rev))
                        np.testing.assert_allclose(J_fwd, J_rev, atol=atol, rtol=rtol,
                                                   verbose=False, equal_nan=False)
                except Exception as err:
                    abserr, relerr = _parse_assert_allclose_error(err.args[0])
                    if abserr < atol and either_zero:
                        # if one array is all zeros and we don't violate the absolute
                        # tolerance, then don't flag the relative error.
                        continue
                    if verbose:
                        bad_derivs.append(f"\nForward and Reverse derivatives of '{key[0]}' wrt "
                                          f"'{key[1]}' do not match.\n")
                        bad_derivs[-1] += _filter_np_err(err.args[0])
                        if nrows <= maxrows and ncols <= maxcols:
                            with np.printoptions(linewidth=10000):
                                bad_derivs[-1] += '\nJ_fwd - J_rev:\n' + \
                                    np.array2string(J_fwd - J_rev)
                    else:
                        bad_derivs.append([f"{key[0]} wrt {key[1]}", "abs", "fwd-rev", f"{abserr}"])
                        bad_derivs.append([f"{key[0]} wrt {key[1]}", "rel", "fwd-rev", f"{relerr}"])

        if bad_derivs or inconsistent_derivs:
            error_strings.extend(['', add_border(f'Component: {comp}', '-')])
            if bad_derivs:
                if verbose:
                    error_strings[-1] += '\n'.join(bad_derivs)
                else:
                    header = ['< output > wrt < variable >', 'max abs/rel', 'diff', 'value']
                    widths = get_max_widths(chain([header], bad_derivs))
                    header_str = list(strs2row_iter([header], widths, delim=' | '))[0]
                    error_strings.append(add_border(header_str, '-', above=False))
                    error_strings.extend(strs2row_iter(bad_derivs, widths, delim=' | '))

            if inconsistent_derivs:
                error_strings[-1] += (
                    "\nInconsistent derivs across processes for keys: "
                    f"{sorted(inconsistent_derivs)}.\nCheck that distributed outputs are properly "
                    "reduced when computing\nderivatives of serial inputs.")

    if error_strings:
        header = add_border('assert_check_partials failed for the following Components\n'
                            f'with absolute tolerance = {atol} and relative tolerance = {rtol}')
        err_string = '\n'.join(error_strings)
        raise ValueError(f"\n{header}\n{err_string}")


def assert_check_totals(totals_data, atol=1e-6, rtol=1e-6, max_display_shape=(20, 20)):
    """
    Raise assertion if any entry from the return from check_totals is above a tolerance.

    Parameters
    ----------
    totals_data : Dict of Dicts of Tuples of Floats
        First key:
            is the (output, input) tuple of strings;
        Second key:
            is one of ['tol violation', 'magnitude', 'J_fd', 'J_fwd', 'J_rev',
                       'vals_at_max_error', 'directional_fd_fwd',
                       'directional_fd_rev', 'directional_fwd_rev', 'rank_inconsistent',
                       'matrix_free', 'directional', 'steps', and 'rank_inconsistent'].

            For 'J_fd', 'J_fwd', 'J_rev' the value is a numpy array representing the computed
            Jacobian for the three different methods of computation.
            For 'tol violation' and 'vals_at_max_error' the value is a
            tuple containing values for forward - fd, reverse - fd, forward - reverse. For
            'magnitude' the value is a tuple indicating the maximum magnitude of values found in
            Jfwd, Jrev, and Jfd.
    atol : float
        Absolute error. Default is 1e-6.
    rtol : float
        Relative error. Default is 1e-6.
    max_display_shape : tuple of int
        Maximum shape of the jacobians to display directly in the error message.
        Default is (20, 20).
    """
    fails = []
    incon_keys = set()
    for key, dct in totals_data.items():
        if 'inconsistent_keys' in dct:
            incon_keys = dct['inconsistent_keys']
        J_fd = dct['J_fd']
        if isinstance(J_fd, list):
            J_fd = J_fd[0]
        try:
            nrows, ncols = J_fd.shape
        except ValueError:
            nrows = J_fd.shape
            ncols = 1
        break

    if isinstance(max_display_shape, int):
        maxrows = maxcols = max_display_shape
    else:
        try:
            maxrows, maxcols = max_display_shape
        except ValueError:
            issue_warning("max_display_shape must be an int or a tuple of two ints, but "
                          f"got {max_display_shape}. Defaulting to (20, 20).")

        jacs = [('J_fwd', dct.get('J_fwd'), 'Forward'), ('J_rev', dct.get('J_rev'), 'Reverse')]

        for Jname, J, direction in jacs:
            if J is not None:
                try:
                    np.testing.assert_allclose(J, J_fd, atol=atol, rtol=rtol, verbose=False,
                                               equal_nan=False)
                except Exception as err:
                    abserr, relerr = _parse_assert_allclose_error(err.args[0])
                    if abserr < atol and not (np.any(J) or np.any(J_fd)):
                        # if one array is all zeros and we don't violate the absolute
                        # tolerance, then don't flag the relative error.
                        continue
                    fails.append(f"\n{direction} derivatives of '{key[0]}' w.r.t '{key[1]}' "
                                 "do not match finite difference.\n")
                    fails[-1] += _filter_np_err(err.args[0])
                    if nrows <= maxrows and ncols <= maxcols:
                        with np.printoptions(linewidth=10000):
                            fails[-1] += f'\n{Jname} - J_fd:\n' + np.array2string(J - J_fd)

    if incon_keys:
        ders = [f"{sof} wrt {swrt}" for sof, swrt in sorted(incon_keys)]
        fails.append(f"During total derivative computation, the following partial derivatives "
                     f"resulted in serial inputs that were inconsistent across processes: {ders}.")

    if fails:
        raise ValueError('\n\n'.join(fails))


def assert_no_approx_partials(system, include_self=True, recurse=True, method='any', excludes=None):
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
    method : str
        Specifically look for Components with this method of approx partials. Values can be
        'cs', 'fd', or 'any'. 'any' means either 'cs' or 'fd'. The default is 'any'.
    excludes : str, iter of str, or None
        Glob patterns for pathnames to exclude from the check. Default is None, which
        excludes nothing.

    Raises
    ------
    AssertionError
        If a subsystem of group is found to be using approximated partials.
    """
    if isinstance(excludes, str):
        excludes = [excludes, ]

    has_approx_partials = False
    msg = 'The following components use approximated partials:\n'
    for s in system.system_iter(include_self=include_self, recurse=recurse):
        if isinstance(s, Component):
            if excludes is not None and any(fnmatch(s.pathname, exclude) for exclude in excludes):
                continue
            if s._approx_schemes:
                if method == 'any' or method in s._approx_schemes:
                    has_approx_partials = True
                    approx_partials = [(k, v['method'])
                                       for k, v in s._declared_partials_patterns.items()
                                       if 'method' in v and v['method']]
                    msg += '    ' + s.pathname + '\n'
                    for key, method in approx_partials:
                        key = (str(key[0]), str(key[1]))
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
    from openmdao.core.group import Group
    parts = ['The following groups use dictionary jacobians:\n']
    for s in system.system_iter(include_self=include_self, recurse=recurse, typ=Group):
        if isinstance(s._jacobian, DictionaryJacobian):
            parts.append('    ' + s.pathname)
    if len(parts) > 1:
        raise AssertionError('\n'.join(parts))


def assert_near_equal(actual, desired, tolerance=1e-15, tol_type='rel'):
    """
    Check relative error.

    Determine that the relative error between `actual` and `desired`
    is within `tolerance`. If `desired` is zero, then use absolute error.

    Can handle some data structures. Generates warnings for data types it cannot handle.

    Parameters
    ----------
    actual : float, array-like, dict
        The value from the test.
    desired : float, array-like, dict
        The value expected.
    tolerance : float
        Maximum relative or absolute error.
        For relative tolerance: ``(actual - desired) / desired``.
        For absolute  tolerance: ``(actual - desired)``.
    tol_type : {'rel', 'abs'}
        Type of error to use: 'rel' for relative error, 'abs' for absolute error.
        Default is set to 'rel'.

    Returns
    -------
    float
        The error.
    """
    # Try to make similar things of the same type so they can be compared
    # make arrays out of scalars
    if type(actual) in [int, float, np.int64, np.float64, np.int32, np.complex128]:
        actual = np.atleast_1d(actual)
    if type(desired) in [int, float, np.int64, np.float64, np.int32, np.complex128]:
        desired = np.atleast_1d(desired)

    # Handle jax arrays, if available
    if JaxArray is not None:
        if isinstance(actual, JaxArray):
            actual = np.atleast_1d(actual)
        if isinstance(desired, JaxArray):
            desired = np.atleast_1d(desired)

    # if desired is numeric list or tuple, make ndarray out of it
    if isinstance(actual, (list, tuple)):
        actual = np.asarray(actual)
    if isinstance(desired, (list, tuple)):
        desired = np.asarray(desired)

    # In case they are PromAbsDict and other dict-like objects
    if isinstance(actual, dict) and type(actual) is not dict:
        actual = dict(actual)
    if isinstance(desired, dict) and type(desired) is not dict:
        desired = dict(desired)

    if type(actual) is not type(desired):
        raise ValueError(f'actual {type(actual)}, desired {type(desired)} have different types')

    if isinstance(actual, type) and isinstance(desired, type):
        if actual != desired:
            raise ValueError(
                'actual type %s, and desired type %s are different' % (actual, desired))
        return 0

    # The code below can only handle these data types
    _supported_types = [dict, set, str, bool, np.ndarray, type(None)]
    if type(actual) not in _supported_types:
        warnings.warn(
            f"The function, assert_near_equal, does not support the actual value type: '"
            f"{type(actual)}'.")
        return 0
    if type(desired) not in _supported_types:
        warnings.warn(
            f"The function, assert_near_equal, does not support the desired value type: '"
            f"{type(actual)}'.")
        return 0

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
                new_error = assert_near_equal(actual[key], desired[key], tolerance, tol_type)
                error = max(error, new_error)
            except ValueError as exception:
                msg = '{}: '.format(key) + str(exception)
                raise ValueError(msg) from None
            except KeyError as exception:
                msg = '{}: '.format(key) + str(exception)
                raise KeyError(msg) from None

    elif isinstance(actual, set) and isinstance(desired, set):
        if actual.symmetric_difference(desired):
            actual_extra = actual.difference(desired)
            desired = desired.difference(actual)
            raise KeyError("Actual and desired sets differ. "
                           f"Actual extra values: {actual_extra}, "
                           f"Desired extra values: {desired_extra}")

        error = 0.

    elif isinstance(actual, str) and isinstance(desired, str):
        if actual != desired:
            raise ValueError(
                'actual %s, desired %s strings have different values' % (actual, desired))
        error = 0.0

    elif isinstance(actual, bool) and isinstance(desired, bool):
        if actual != desired:
            raise ValueError(
                'actual %s, desired %s booleans have different values' % (actual, desired))
        error = 0.0

    elif actual is None and desired is None:
        error = 0.0

    # array values
    elif isinstance(actual, np.ndarray) and isinstance(desired, np.ndarray):
        if actual.dtype == object or desired.dtype == object:
            if actual.dtype == object:
                warnings.warn(
                    f"The function, assert_near_equal, does not support the actual value ndarray "
                    f"type of: '"
                    f"{type(actual.dtype)}'.")
            if desired.dtype == object:
                warnings.warn(
                    f"The function, assert_near_equal, does not support the desired value ndarray "
                    f"type of: '"
                    f"{type(desired.dtype)}'.")
            error = 0.0
        else:

            actual = np.atleast_1d(actual)
            desired = np.atleast_1d(desired)
            if actual.shape != desired.shape:
                raise ValueError(
                    'actual and desired have differing shapes.'
                    ' actual {}, desired {}'.format(actual.shape, desired.shape))
            # check to see if the entire array is made of floats. If not, loop through all values

            if not np.all(np.isnan(actual) == np.isnan(desired)):
                if actual.size == 1 and desired.size == 1:
                    raise ValueError('actual %s, desired %s' % (actual, desired))
                else:
                    raise ValueError('actual and desired values have non-matching nan'
                                     ' values')
            if np.linalg.norm(desired) == 0 or tol_type == 'abs':
                error = np.linalg.norm(actual - desired)
            else:
                error = np.linalg.norm(actual - desired) / np.linalg.norm(desired)

            if abs(error) > tolerance:
                if actual.size < 10 and desired.size < 10:
                    raise ValueError('actual %s, desired %s, %s error %s, tolerance %s'
                                     % (actual, desired, tol_type, error, tolerance))
                else:
                    raise ValueError('arrays do not match, rel error %.3e > tol (%.3e)' %
                                     (error, tolerance))
    elif isinstance(actual, tuple) and isinstance(desired, tuple):
        error = 0.0
        for act, des in zip(actual, desired):
            new_error = assert_near_equal(act, des, tolerance, tol_type)
            error = max(error, new_error)
    else:
        raise ValueError(
            'actual and desired have unexpected types: %s, %s' % (type(actual), type(desired)))

    return error


def mimic(func, mimicfunc, *args, **kwargs):
    """
    Verify that mimicfunc produces the same result as func.

    This can be useful when refactoring to verify that the new version of a function has
    identical outputs to the original.

    Parameters
    ----------
    func : function
        The function to mimic.
    mimicfunc : function
        The mimic function.
    *args : tuple
        The arguments to pass to the functions.
    **kwargs : dict
        The keyword arguments to pass to the functions.

    Returns
    -------
    object, object
        The return values of func and mimicfunc.
    """
    ret1 = func(*args, **kwargs)
    ret2 = mimicfunc(*args, **kwargs)
    assert_near_equal(ret1, ret2)
    return ret1, ret2


def assert_sparsity_matches_fd(system, direction='fwd', outstream=sys.stdout):
    """
    Assert that the sparsity of the system matches the finite difference sparsity.

    Parameters
    ----------
    system : System
        The system to check.
    direction : str
        The direction to check. 'fwd' or 'rev'.
    outstream : file
        The stream to write the output to.  If None, no output is written.
    """
    assert system.sparsity_matches_fd(direction=direction, outstream=outstream), \
        f"{system.msginfo}: Sparsity does not match finite difference sparsity"


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


def assert_equal_numstrings(s1, s2, atol=1e-6, rtol=1e-6):
    """
    Check that two strings containing numbers are equal after convering numerical parts to floats.

    Parameters
    ----------
    s1 : str
        The first numeric string to compare.
    s2 : str
        The second numeric string to compare.
    atol : float
        Absolute error tolerance. Default is 1e-6.
    rtol : float
        Relative error tolerance. Default is 1e-6.
    """
    assert snum_equal(s1, s2, atol=atol, rtol=rtol)


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
