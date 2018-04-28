"""Some miscellaneous utility functions."""
from __future__ import division

import contextlib
import os
import sys
import warnings
import unittest
from mock import Mock
from fnmatch import fnmatchcase
from six import string_types
from six.moves import range, cStringIO as StringIO
from collections import Iterable
import numbers

import numpy as np


def warn_deprecation(msg):
    """
    Raise a warning and prints a deprecation message to stdout.

    Parameters
    ----------
    msg : str
        Message that will be printed to stdout.
    """
    # Deprecation warnings need to be printed regardless of debug level
    warnings.simplefilter('always', DeprecationWarning)

    # note, stack level 3 should take us back to original caller.
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    warnings.simplefilter('ignore', DeprecationWarning)


def ensure_compatible(name, value, shape=None, indices=None):
    """
    Make value compatible with the specified shape or the shape of indices.

    Parameters
    ----------
    name : str
        The name of the value.
    value : float or list or tuple or ndarray or Iterable
        The value of a variable.
    shape : int or tuple or list or None
        The expected or desired shape of the value.
    indices : int or list of ints or tuple of ints or int ndarray or None
        The indices of a source variable, used to determine shape if shape is None.
        If shape is not None, the shape of the indices must match shape.

    Returns
    -------
    ndarray
        The value in a shape compatible with the specified shape and/or indices.
    tuple
        The resulting shape of the value.

    Raises
    ------
    ValueError
        If value cannot be made to conform to shape or if shape and indices
        are incompatible.
    """
    if isinstance(value, Iterable):
        value = np.asarray(value)

    if indices is not None:
        indices = np.atleast_1d(indices)
        ind_shape = indices.shape

    # if shape is not given, infer from value (if not scalar) or indices
    if shape is not None:
        if isinstance(shape, numbers.Integral):
            shape = (shape,)
        elif isinstance(shape, list):
            shape = tuple(shape)
    elif not np.isscalar(value):
        shape = np.atleast_1d(value).shape
    elif indices is not None:
        if len(ind_shape) > 1:
            raise RuntimeError("src_indices for '%s' is not flat, so its input "
                               "shape must be provided. src_indices may contain "
                               "an extra dimension if the connected source is "
                               "not flat, making the input shape ambiguous." %
                               name)
        shape = ind_shape

    if shape is None:
        # shape is not determined, assume the shape of value was intended
        value = np.atleast_1d(value)
        shape = value.shape
    else:
        # shape is determined, if value is scalar assign it to array of shape
        # otherwise make sure value is an array of the determined shape
        if np.isscalar(value) or value.shape == (1,):
            value = np.ones(shape) * value
        else:
            value = np.atleast_1d(value)
            if value.shape != shape:
                raise ValueError("Incompatible shape for '%s': "
                                 "Expected %s but got %s." %
                                 (name, shape, value.shape))

    # finally make sure shape of indices is compatible
    if indices is not None and shape != ind_shape[:len(shape)]:
        raise ValueError("Shape of indices does not match shape for '%s': "
                         "Expected %s but got %s." %
                         (name, shape, ind_shape[:len(shape)]))

    return value, shape


def determine_adder_scaler(ref0, ref, adder, scaler):
    r"""
    Determine proper values of adder and scaler based on user arguments.

    Adder and Scaler are used internally because the transformation is
    slightly more efficient.

    Parameters
    ----------
    ref : float or ndarray, optional
        Value of response variable that scales to 1.0 in the driver.
    ref0 : float or ndarray, optional
        Value of response variable that scales to 0.0 in the driver.
    adder : float or ndarray, optional
        Value to add to the model value to get the scaled value. Adder
        is first in precedence.
    scaler : float or ndarray, optional
        Value to multiply the model value to get the scaled value. Scaler
        is second in precedence.

    Returns
    -------
    tuple
        adder and scaler, properly formatted and based on ref/ref0 if provided.

    Raises
    ------
    ValueError
        If both ref/ref0 and adder/scaler were provided.

    Notes
    -----
    The response can be scaled using ref and ref0.
    The argument :code:`ref0` represents the physical value when the scaled value is 0.
    The argument :code:`ref` represents the physical value when the scaled value is 1.
    """
    # Affine scaling cannot be used with scalers/adders
    if ref0 is not None or ref is not None:
        if scaler is not None or adder is not None:
            raise ValueError('Inputs ref/ref0 are mutually exclusive '
                             'with scaler/adder')
        if ref is None:
            ref = 1.0
        if ref0 is None:
            ref0 = 0.0

        # Convert ref/ref0 to scaler/adder so we can scale the bounds
        adder = -ref0
        scaler = 1.0 / (ref + adder)

    else:
        if scaler is None:
            scaler = 1.0
        if adder is None:
            adder = 0.0

    adder = format_as_float_or_array('adder', adder, val_if_none=0.0, flatten=True)
    scaler = format_as_float_or_array('scaler', scaler, val_if_none=1.0, flatten=True)

    return adder, scaler


def set_pyoptsparse_opt(optname, fallback=True):
    """
    For testing, sets the pyoptsparse optimizer using the given optimizer name.

    This may be modified based on the value of
    OPENMDAO_FORCE_PYOPTSPARSE_OPT. This can be used on systems that have
    SNOPT installed to force them to use SLSQP in order to mimic our test
    machines on travis and appveyor.

    Parameters
    ----------
    optname : str
        Name of pyoptsparse optimizer that is requested by the test.
    fallback : bool
        If True, fall back to SLSQP if optname can't be found

    Returns
    -------
    object
        Pyoptsparse optimizer instance.
    str
        Pyoptsparse optimizer string
    """
    OPT = None
    opt = None
    OPTIMIZER = None
    force = os.environ.get('OPENMDAO_FORCE_PYOPTSPARSE_OPT')
    if force:
        optname = force

    try:
        from pyoptsparse import OPT
        try:
            opt = OPT(optname)
            OPTIMIZER = optname
        except Exception:
            if fallback and optname != 'SLSQP':
                try:
                    opt = OPT('SLSQP')
                    OPTIMIZER = 'SLSQP'
                except Exception:
                    pass
        else:
            if fallback and isinstance(opt, Mock):
                try:
                    opt = OPT('SLSQP')
                    OPTIMIZER = 'SLSQP'
                except Exception:
                    pass
    except Exception:
        pass

    if isinstance(opt, Mock):
        OPT = OPTIMIZER = None

    if not fallback and OPTIMIZER != optname:
        raise unittest.SkipTest("pyoptsparse is not providing %s" % optname)

    return OPT, OPTIMIZER


def format_as_float_or_array(name, values, val_if_none=0.0, flatten=False):
    """
    Format array option values.

    Checks that the given array values are either None, float, or an iterable
    of numeric values. On output all interables of numeric values are
    converted to a flat np.ndarray. If values is scalar, it is converted
    to float.

    Parameters
    ----------
    name : str
        The path of the variable relative to the current system.
    values : float or numpy ndarray or Iterable
        Values of the array option to be formatted to the expected form.
    val_if_none : float or numpy ndarray
        The default value for the option if values is None.
    flatten : bool
        Set to True to flatten any ndarray return.

    Returns
    -------
    float or np.ndarray
        Values transformed to the expected form.

    Raises
    ------
    ValueError
        If values is Iterable but cannot be converted to a numpy ndarray
    TypeError
        If values is scalar, not None, and not a Number.
    """
    # Convert adder to ndarray/float as necessary
    if isinstance(values, np.ndarray):
        if flatten:
            values = values.flatten()
    elif not isinstance(values, string_types) \
            and isinstance(values, Iterable):
        values = np.asarray(values, dtype=float)
        if flatten:
            values = values.flatten()
    elif values is None:
        values = val_if_none
    elif values == float('inf'):
        values = sys.float_info.max
    elif values == -float('inf'):
        values = -sys.float_info.max
    elif isinstance(values, numbers.Number):
        values = float(values)
    else:
        raise TypeError('Expected values of {0} to be an Iterable of '
                        'numeric values, or a scalar numeric value. '
                        'Got {1} instead.'.format(name, values))
    return values


class ContainsAll(object):
    """
    A fake dictionary that always reports __contains__(name) to be True.
    """

    def __contains__(self, name):
        """
        Return if the named object is contained.

        Parameters
        ----------
        name : str
            Name of the object being looked up.

        Returns
        -------
        bool
            Always returns True.
        """
        return True


def all_ancestors(pathname, delim='.'):
    """
    Return a generator of pathnames of the starting object and all of its parents.

    Parameters
    ----------
    pathname : str
        Pathname of starting object.
    delim : str
        Delimiter used to split the name
    """
    parts = pathname.split(delim)
    yield parts[0]
    for i in range(2, len(parts) + 1):
        yield delim.join(parts[:i])


def find_matches(pattern, var_list):
    """
    Return list of variable names that match given pattern.

    Parameters
    ----------
    pattern : str
        String pattern
    var_list : list of str
        List of variable names to search for pattern.

    Returns
    -------
    list
        Variable names that match pattern.
    """
    if pattern == '*':
        return var_list
    elif pattern in var_list:
        return [pattern]
    return [name for name in var_list if fnmatchcase(name, pattern)]


def pad_name(name, pad_num=10, quotes=False):
    """
    Pad a string so that they all line up when stacked.

    Parameters
    ----------
    name : str
        The string to pad.
    pad_num : int
        The number of total spaces the string should take up.
    quotes : bool
        If name should be quoted.

    Returns
    -------
    str
        Padded string
    """
    l_name = len(name)
    quotes_len = 2 if quotes else 0
    if l_name + quotes_len < pad_num:
        pad = pad_num - (l_name + quotes_len)
        if quotes:
            pad_str = "'{name}'{sep:<{pad}}"
        else:
            pad_str = "{name}{sep:<{pad}}"
        pad_name = pad_str.format(name=name, sep='', pad=pad)
        return pad_name
    else:
        if quotes:
            return "'{0}'".format(name)
        else:
            return '{0}'.format(name)


def run_model(prob):
    """
    Call `run_model` on problem and capture output.

    Parameters
    ----------
    prob : Problem
        an instance of Problem

    Returns
    -------
    string
        output from calling `run_model` on the Problem, captured from stdout
    """
    stdout = sys.stdout
    strout = StringIO()

    sys.stdout = strout
    try:
        prob.run_model()
    finally:
        sys.stdout = stdout

    return strout.getvalue()


def run_driver(prob):
    """
    Call `run_driver` on problem and capture output.

    Parameters
    ----------
    prob : Problem
        an instance of Problem

    Returns
    -------
    boolean
        Failure flag; True if failed to converge, False is successful.
    string
        output from calling `run_driver` on the Problem, captured from stdout
    """
    stdout = sys.stdout
    strout = StringIO()

    sys.stdout = strout
    try:
        failed = prob.run_driver()
    finally:
        sys.stdout = stdout

    return failed, strout.getvalue()


@contextlib.contextmanager
def printoptions(*args, **kwds):
    """
    Context manager for setting numpy print options.

    Set print options for the scope of the `with` block, and restore the old
    options at the end. See `numpy.set_printoptions` for the full description of
    available options.

    Parameters
    ----------
    *args : list
        Variable-length argument list.
    **kwds : dict
        Arbitrary keyword arguments.

    Examples
    --------
    >>> with printoptions(precision=2):
    ...     print(np.array([2.0])) / 3
    [0.67]
    The `as`-clause of the `with`-statement gives the current print options:
    >>> with printoptions(precision=2) as opts:
    ...      assert_equal(opts, np.get_printoptions())

    See Also
    --------
    set_printoptions, get_printoptions
    """
    opts = np.get_printoptions()
    try:
        np.set_printoptions(*args, **kwds)
        yield np.get_printoptions()
    finally:
        np.set_printoptions(**opts)
