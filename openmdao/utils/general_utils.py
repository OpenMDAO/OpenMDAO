"""Some miscellaneous utility functions."""
from contextlib import contextmanager
import os
import re
import sys
import math
import warnings
import unittest
from fnmatch import fnmatchcase
from io import StringIO

# note: this is a Python 3.3 change, clean this up for OpenMDAO 3.x
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numbers
import json
import importlib

import numpy as np
import openmdao


# Certain command line tools can make use of this to allow visualization of models when errors
# are present that would normally cause setup to abort.
_ignore_errors = False


def ignore_errors(flag=None):
    """
    Disable certain errors that will prevent setup from completing.

    Parameters
    ----------
    flag : bool or None
        If not None, set the value of _ignore_errors to this value.

    Returns
    -------
    bool
        The current value of _ignore_errors
    """
    global _ignore_errors
    if flag is not None:
        _ignore_errors = True
    return _ignore_errors


def conditional_error(msg, exc=RuntimeError, category=UserWarning):
    """
    Raise an exception or issue a warning, depending on the value of _ignore_errors.

    Parameters
    ----------
    msg : str
        The error/warning message.
    exc : exception class
        This exception class is used to create the exception to be raised.
    category : warning class
        This category is the class of warning to be issued.
    """
    if ignore_errors():
        simple_warning(msg, category=category)
    else:
        raise exc(msg)


@contextmanager
def ignore_errors_context(flag=True):
    """
    Set ignore_errors to the given flag in this context.

    Parameters
    ----------
    flag : bool
        If not None, set ignore_errors to this value.
    """
    save = ignore_errors()
    ignore_errors(flag)
    yield
    ignore_errors(save)


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
    simple_warning(msg, DeprecationWarning, stacklevel=3)
    warnings.simplefilter('ignore', DeprecationWarning)


def _warn_simple_format(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)


def simple_warning(msg, category=UserWarning, stacklevel=2):
    """
    Display a simple warning message without the annoying extra line showing the warning call.

    Parameters
    ----------
    msg : str
        The warning message.
    category : class
        The warning class.
    stacklevel : int
        Number of levels up the stack to identify as the warning location.
    """
    old_format = warnings.formatwarning
    warnings.formatwarning = _warn_simple_format
    try:
        warnings.warn(msg, category, stacklevel)
    finally:
        warnings.formatwarning = old_format


class reset_warning_registry(object):
    """
    Context manager which archives & clears warning registry for duration of context.

    From https://bugs.python.org/file40031/reset_warning_registry.py

    Attributes
    ----------
    _pattern : regex pattern
        Causes manager to only reset modules whose names match this pattern. defaults to ``".*"``.
    """

    #: regexp for filtering which modules are reset
    _pattern = None

    #: dict mapping module name -> old registry contents
    _backup = None

    def __init__(self, pattern=None):
        """
        Initialize all attributes.

        Parameters
        ----------
        pattern : regex pattern
            Causes manager to only reset modules whose names match pattern. defaults to ``".*"``.
        """
        self._pattern = re.compile(pattern or ".*")

    def __enter__(self):
        """
        Enter the runtime context related to this object.

        Returns
        -------
        reset_warning_registry
            This context manager.

        """
        # archive and clear the __warningregistry__ key for all modules
        # that match the 'reset' pattern.
        pattern = self._pattern
        backup = self._backup = {}
        for name, mod in list(sys.modules.items()):
            if pattern.match(name):
                reg = getattr(mod, "__warningregistry__", None)
                if reg:
                    backup[name] = reg.copy()
                    reg.clear()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context related to this object.

        Parameters
        ----------
        exc_type : Exception class
            The type of the exception.
        exc_value : Exception instance
            The exception instance raised.
        traceback : regex pattern
            Traceback object.
        """
        # restore warning registry from backup
        modules = sys.modules
        backup = self._backup
        for name, content in backup.items():
            mod = modules.get(name)
            if mod is None:
                continue
            reg = getattr(mod, "__warningregistry__", None)
            if reg is None:
                setattr(mod, "__warningregistry__", content)
            else:
                reg.clear()
                reg.update(content)

        # clear all registry entries that we didn't archive
        pattern = self._pattern
        for name, mod in list(modules.items()):
            if pattern.match(name) and name not in backup:
                reg = getattr(mod, "__warningregistry__", None)
                if reg:
                    reg.clear()


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
        contains_slice = _is_slice(indices)
        ind_shape = indices.shape
    else:
        contains_slice = None

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
            value = np.atleast_1d(value).astype(np.float64)
            if value.shape != shape:
                raise ValueError("Incompatible shape for '%s': "
                                 "Expected %s but got %s." %
                                 (name, shape, value.shape))

    if indices is not None and shape != ind_shape[:len(shape)] and not contains_slice:
        raise ValueError("Shape of indices does not match shape for '%s': "
                         "Expected %s but got %s." %
                         (name, shape, ind_shape[:len(shape)]))

    return value, shape, indices


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

    This may be modified based on the value of OPENMDAO_FORCE_PYOPTSPARSE_OPT.
    This can be used on systems that have SNOPT installed to force them to use
    SLSQP in order to mimic our test machines on travis and appveyor.

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
        from mock import Mock
    except ImportError:
        Mock = None

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
            if fallback and Mock and isinstance(opt, Mock):
                try:
                    opt = OPT('SLSQP')
                    OPTIMIZER = 'SLSQP'
                except Exception:
                    pass
    except Exception:
        pass

    if Mock and isinstance(opt, Mock):
        OPT = OPTIMIZER = None

    if not fallback and OPTIMIZER != optname:
        raise unittest.SkipTest("pyoptsparse is not providing %s" % optname)

    return OPT, OPTIMIZER


def format_as_float_or_array(name, values, val_if_none=0.0, flatten=False):
    """
    Format array option values.

    Checks that the given array values are either None, float, or an iterable
    of numeric values. On output all iterables of numeric values are
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
    elif not isinstance(values, str) \
            and isinstance(values, Iterable):
        values = np.asarray(values, dtype=float)
        if flatten:
            values = values.flatten()
    elif values is None:
        values = val_if_none
    elif values == float('inf'):
        values = openmdao.INF_BOUND
    elif values == -float('inf'):
        values = -openmdao.INF_BOUND
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

    Pathnames are ordered from longest to shortest.

    Parameters
    ----------
    pathname : str
        Pathname of starting object.
    delim : str
        Delimiter used to split the name
    """
    parts = pathname.split(delim)
    for i in range(len(parts), 0, -1):
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


def run_model(prob, ignore_exception=False):
    """
    Call `run_model` on problem and capture output.

    Parameters
    ----------
    prob : Problem
        an instance of Problem
    ignore_exception : bool
        Set to True to ignore an exception of any kind.

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
    except Exception as err:
        if not ignore_exception:
            raise err
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


@contextmanager
def printoptions(*args, **kwds):
    """
    Context manager for setting numpy print options.

    Set print options for the scope of the `with` block, and restore the old
    options at the end. See `numpy.set_printoptions` for the full description of
    available options. If any invalid options are specified, they will be ignored.

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

    # ignore any keyword args that are not valid in this version of numpy
    # e.g. numpy <=1.13 does not have the 'floatmode' option
    kw_opts = dict((key, val) for key, val in kwds.items() if key in opts)

    try:
        np.set_printoptions(*args, **kw_opts)
        yield np.get_printoptions()
    finally:
        np.set_printoptions(**opts)


def do_nothing_context():
    """
    Do nothing.

    Useful when you have a block of code that only requires a context manager sometimes,
    and you don't want to repeat the context managed block.

    Returns
    -------
    contextmanager
        A do nothing context manager.
    """
    def nothing():
        yield None

    return contextmanager(nothing)()


def _byteify(data, ignore_dicts=False):
    """
    Convert any unicode items in a data structure to bytes (object_hook for json load/loads).

    Credit: Mirec Miskuf
    stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-from-json

    Parameters
    ----------
    data : any data item or structure
        the data to be converted
    ignore_dicts : bool
        a flag to prevent recursion on dicts that have already been byteified.
        False when object_hook passes a new dict to byteify, True at all other times.

    Returns
    -------
    data item or structure
        data item or structure with unicode converted to bytes
    """
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')

    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [_byteify(item, ignore_dicts=True) for item in data]

    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }

    # if it's anything else, return it in its original form
    return data


def json_load_byteified(file_handle):
    """
    Load data from a JSON file, converting unicode to bytes if Python version is 2.x.

    Intended for use only with Python 2.x, behaves the same as json.load() under Python 3.x.

    Parameters
    ----------
    file_handle : file
        file containing the data to be converted

    Returns
    -------
    data item or structure
        data item or structure with unicode converted to bytes
    """
    return json.load(file_handle)


def json_loads_byteified(json_str):
    """
    Load data from a JSON string, converting unicode to bytes if Python version is 2.x.

    Intended for use only with Python 2.x, behaves the same as json.loads() under Python 3.x.

    Parameters
    ----------
    json_str : str
        text string containing json encoded data

    Returns
    -------
    data item or structure
        data item or structure with unicode converted to bytes
    """
    return json.loads(json_str)


def remove_whitespace(s, right=False, left=False):
    """
    Remove white-space characters from the given string.

    If neither right nor left is specified (the default),
    then all white-space is removed.

    Parameters
    ----------
    s : str
        The string to be modified.
    right : bool
        If True, remove white-space from the end of the string.
    left : bool
        If True, remove white-space from the beginning of the string.

    Returns
    -------
    str
        The string with white-space removed.
    """
    if not left and not right:
        return re.sub(r"\s+", "", s, flags=re.UNICODE)
    elif right and left:
        return re.sub(r"^\s+|\s+$", "", s, flags=re.UNICODE)
    elif right:
        return re.sub(r"\s+$", "", s, flags=re.UNICODE)
    else:  # left
        return re.sub(r"^\s+", "", s, flags=re.UNICODE)


_badtab = r'`~@#$%^&*()[]{}-+=|\/?<>,.:;'
_transtab = str.maketrans(_badtab, '_' * len(_badtab))


def str2valid_python_name(s):
    """
    Translate a given string into a valid python variable name.

    Parameters
    ----------
    s : str
        The string to be translated.

    Returns
    -------
    str
        The valid python name string.
    """
    return s.translate(_transtab)


_container_classes = (list, tuple, set)


def make_serializable(o):
    """
    Recursively convert numpy types to native types for JSON serialization.

    Parameters
    ----------
    o : object
        the object to be converted

    Returns
    -------
    object
        The converted object.
    """
    if isinstance(o, _container_classes):
        return [make_serializable(item) for item in o]
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, np.number):
        return o.item()
    elif hasattr(o, '__dict__'):
        return o.__class__.__name__
    else:
        return o


def make_set(str_data, name=None):
    """
    Construct a set containing the specified character strings.

    Parameters
    ----------
    str_data : None, str, or list of strs
        Character string(s) to be included in the set.

    name : str, optional
        A name to be used in error messages.

    Returns
    -------
    set
        A set of character strings.
    """
    if not str_data:
        return set()
    elif isinstance(str_data, str):
        return {str_data}
    elif isinstance(str_data, set):
        return str_data
    elif isinstance(str_data, list):
        return set(str_data)
    elif name:
        raise TypeError("The {} argument should be str, set, or list: {}".format(name, str_data))
    else:
        raise TypeError("The argument should be str, set, or list: {}".format(str_data))


def match_includes_excludes(name, prom_name, includes, excludes):
    """
    Check to see if the variable names pass through the includes and excludes filter.

    Parameters
    ----------
    name : str
        Unpromoted variable name to be checked for match.
    prom_name : str
        Promoted variable name to be checked for match.
    includes : None or list_like
        List of glob patterns for name to include in the filtering.
    excludes : None or list_like
        List of glob patterns for name to exclude in the filtering.

    Returns
    -------
    bool
        Return True if the name passes through the filtering of includes and excludes.
    """
    # Process includes
    if includes is not None:
        for pattern in includes:
            if fnmatchcase(name, pattern) or fnmatchcase(prom_name, pattern):
                break
        else:  # didn't find any match
            return False

    # Process excludes
    if excludes is not None:
        match = False
        for pattern in excludes:
            if fnmatchcase(name, pattern) or fnmatchcase(prom_name, pattern):
                match = True
                break
        return not match

    return True


def env_truthy(env_var):
    """
    Return True if the given environment variable is 'truthy'.

    Parameters
    ----------
    env_var : str
        The name of the environment variable.

    Returns
    -------
    bool
        True if the specified environment variable is 'truthy'.
    """
    return os.environ.get(env_var, '0').lower() not in ('0', 'false', 'no', '')


def common_subpath(pathnames):
    """
    Return the common dotted subpath found in all of the given dotted pathnames.

    Parameters
    ----------
    pathnames : iter of str
        Dotted pathnames of systems.

    Returns
    -------
    str
        Common dotted subpath.  Returns '' if no common subpath is found.
    """
    if len(pathnames) == 1:
        return pathnames[0]

    if pathnames:
        npaths = len(pathnames)
        splits = [p.split('.') for p in pathnames]
        minlen = np.min([len(s) for s in splits])
        for common_loc in range(minlen):
            p0 = splits[0][common_loc]
            for i in range(1, npaths):
                if p0 != splits[i][common_loc]:
                    break
            else:
                continue
            break
        else:
            common_loc += 1

        return '.'.join(splits[0][:common_loc])

    return ''


def _is_slice(indices):
    """
    Check if an array of indices contains a slice object.

    Parameters
    ----------
    indices : ndarray
        Dotted pathnames of systems.

    Returns
    -------
    bool
        Returns True if indices contains a slice.
    """
    return any(isinstance(i, slice) for i in indices)


def _slice_indices(slicer, out_size, out_shape):
    """
    Check if an array of indices contains a slice object.

    Parameters
    ----------
    slicer : slice
        Slice object to slice array
    out_size : int
        Size of output array
    out_shape : tuple
        Tuple of output array shape

    Returns
    -------
    array
        Returns the sliced indices.
    """
    return np.arange(out_size, dtype=int).reshape(out_shape)[tuple(slicer)]


def prom2ivc_src_dict(prom_dict):
    """
    Convert a dictionary with promoted input names into one with ivc source names.

    Parameters
    ----------
    prom_dict : dict
        Original dict with some promoted paths.

    Returns
    -------
    dict
        New dict with ivc source pathnames.
    """
    src_dict = {}
    for name, meta in prom_dict.items():
        if meta['ivc_source'] is not None:
            src_name = meta['ivc_source']
            src_dict[src_name] = meta
        else:
            src_dict[name] = meta

    return src_dict
