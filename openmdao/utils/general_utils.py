"""Some miscellaneous utility functions."""
import os
import re
import sys
import textwrap
import json
import functools
import atexit
from types import TracebackType
import unittest
from contextlib import contextmanager
from fnmatch import fnmatchcase
from io import StringIO
from numbers import Integral
from inspect import currentframe, getouterframes, stack, isfunction, ismethod

from collections.abc import Iterable

import numpy as np

from openmdao.core.constants import INF_BOUND, _UNDEFINED
from openmdao.utils.array_utils import shape_to_len
from openmdao.utils.mpi import MPI


_float_inf = float('inf')


def ensure_compatible(name, value, shape=None, indices=None, default_shape=(1,)):
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
    indices : Indexer or None
        The indices into a source variable.
    default_shape : tuple
        The default shape to use if shape is not provided.

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

    # if shape is not given, infer from value (if not scalar) or indices
    if shape is not None:
        if isinstance(shape, Integral):
            shape = (shape,)
        elif isinstance(shape, list):
            shape = tuple(shape)
    elif not np.isscalar(value):
        shape = np.atleast_1d(value).shape

    if indices is not None:
        if not indices._flat_src and shape is None:
            raise RuntimeError(f"src_indices for '{name}' is not flat, so its input "
                               "shape must be provided.")
        try:
            indshape = indices.indexed_src_shape
        except (RuntimeError, ValueError, TypeError):
            pass  # use shape provided or shape of value and check vs. shape of indices later
        else:
            if shape is not None and shape_to_len(indshape) != shape_to_len(shape):
                raise ValueError(f"Shape of indices {indshape} does not match shape of {shape} for"
                                 f" '{name}'.")
            if shape is None:
                shape = indshape

    if shape is None:
        # shape is not determined, assume the shape of value was intended
        if np.isscalar(value):
            value = np.full(default_shape, value)
        else:
            value = np.asarray(value).reshape(default_shape)
        shape = value.shape
    else:
        # shape is determined, if value is scalar assign it to array of shape
        # otherwise make sure value is an array of the determined shape
        if np.ndim(value) == 0:
            if shape != ():
                value = np.full(shape, value)
        elif value.shape == (1,):
            value = np.full(shape, value)
        else:
            value = np.atleast_1d(value).astype(np.float64)
            if value.shape != shape:
                raise ValueError(f"Incompatible shape for '{name}': Expected {shape} but got "
                                 f"{value.shape}.")

    return value, shape


def _subjac_meta2value(meta):
    """
    Convert subjacobian metadata to value, rows, cols.

    Parameters
    ----------
    meta : dict
        Metadata dict.

    Returns
    -------
    ndarray
        Value of the subjacobian.
    ndarray or None
        Row indices of nonzero values in subjacobian.
    ndarray or None
        Column indices of nonzero values in subjacobian.
    """
    val = meta['val'] if 'val' in meta else None
    rows = meta['rows'] if 'rows' in meta else None
    cols = meta['cols'] if 'cols' in meta else None
    diagonal = meta['diagonal'] if 'diagonal' in meta else False
    shape = meta['shape'] if 'shape' in meta else None

    if diagonal:
        if shape is None:
            raise ValueError("Shape is required for diagonal subjacobian.")
        rows = np.arange(shape_to_len(shape))
        cols = rows
        if val is not None:
            val = np.full(shape_to_len(shape), val)
        else:
            val = None
    elif rows is not None:
        if val is not None and np.isscalar(val):
            val = np.full(len(rows), val)
    elif np.isscalar(val):
        shape = meta['shape'] if 'shape' in meta else None
        if shape is not None:
            val = np.full(shape, val)
        else:
            val = np.atleast_2d(val)
    elif val is not None:
        val = np.atleast_2d(val)

    return val, rows, cols


def determine_adder_scaler(ref0, ref, adder, scaler):
    r"""
    Determine proper values of adder and scaler based on user arguments.

    Adder and Scaler are used internally because the transformation is
    slightly more efficient.

    Parameters
    ----------
    ref0 : float or ndarray, optional
        Value of response variable that scales to 0.0 in the driver.
    ref : float or ndarray, optional
        Value of response variable that scales to 1.0 in the driver.
    adder : float or ndarray, optional
        Value to add to the model value to get the scaled value. Adder
        is first in precedence.
    scaler : float or ndarray, optional
        Value to multiply the model value to get the scaled value. Scaler
        is second in precedence.

    Returns
    -------
    tuple
        Adder and scaler, properly formatted and based on ref/ref0 if provided.

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
        If True, fall back to SLSQP if optname can't be found.

    Returns
    -------
    object
        Pyoptsparse optimizer instance.
    str
        Pyoptsparse optimizer string.
    """
    OPT = None
    opt = None
    OPTIMIZER = None
    force = os.environ.get('OPENMDAO_FORCE_PYOPTSPARSE_OPT')
    if force:
        optname = force

    from unittest.mock import Mock

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
    if isinstance(values, float):
        if values == _float_inf:
            values = INF_BOUND
        elif values == -_float_inf:
            values = -INF_BOUND
    elif isinstance(values, np.ndarray):
        if flatten:
            values = values.flatten()
    elif values is None:
        values = val_if_none
    elif isinstance(values, Iterable) and not isinstance(values, str):
        values = np.asarray(values, dtype=float)
        if flatten:
            values = values.flatten()
    else:
        try:
            values = float(values)
        except Exception:
            raise TypeError(f'Expected values of {name} to be an Iterable of '
                            'numeric values, or a scalar numeric value. '
                            f'Got {values} instead.')
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


_contains_all = ContainsAll()


def all_ancestors(pathname, delim='.'):
    """
    Return a generator of pathnames of the starting object and all of its parents.

    Pathnames are ordered from longest to shortest.

    Parameters
    ----------
    pathname : str
        Pathname of starting object.
    delim : str
        Delimiter used to split the name.

    Yields
    ------
    str
    """
    while pathname:
        yield pathname
        pathname, _, _ = pathname.rpartition(delim)


def find_matches(pattern, var_list):
    """
    Return list of variable names that match given pattern.

    Parameters
    ----------
    pattern : str
        Glob pattern or variable name.
    var_list : list of str
        List of variable names to search for pattern.

    Returns
    -------
    list
        Variable names that match pattern.
    """
    if pattern == '*':
        return var_list
    return [name for name in var_list if fnmatchcase(name, pattern)]


def pattern_filter(patterns, var_iter, name_index=None):
    """
    Yield variable names that match a given pattern.

    Parameters
    ----------
    patterns : iter of str
        Glob patterns or variable names.
    var_iter : iter of str or iter of tuple/list
        Iterator of variable names (or tuples containing variable names) to search for patterns.
    name_index : int or None
        If not None, the var_iter is assumed to yield tuples, and the
        name_index is the index of the variable name in the tuple.

    Yields
    ------
    str
        Variable name or corresponding tuple where the name matches a pattern.
    """
    if '*' in patterns:
        yield from var_iter
    else:
        if name_index is None:
            for vname in var_iter:
                for pattern in patterns:
                    if fnmatchcase(vname, pattern):
                        yield vname
                        break
        else:
            for tup in var_iter:
                vname = tup[name_index]
                for pattern in patterns:
                    if fnmatchcase(vname, pattern):
                        yield tup
                        break


def _find_dict_meta(dct, key):
    """
    Return True if the given key is found in any metadata values in the given dict.

    Parameters
    ----------
    dct : dict
        The metadata dictionary (a dict of dicts).
    key : str
        The metadata key being searched for.

    Returns
    -------
    bool
        True if metadata at the given key was found.
    """
    for meta in dct.values():
        if key in meta:
            return True
    return False


def pad_name(name, width=10, quotes=False):
    """
    Pad a string so that they all line up when stacked.

    Parameters
    ----------
    name : str
        The string to pad.
    width : int
        The number of total spaces the string should take up.
    quotes : bool
        If name should be quoted.

    Returns
    -------
    str
        Padded string.
    """
    name = f"'{name}'" if quotes else str(name)
    if width > len(name):
        return f"{name:<{width}}"
    else:
        return f"{name}"


def get_max_widths(rows):
    """
    Determine the maximum width of each column.

    Parameters
    ----------
    rows : list of list of str
        List of rows, where each row is a list of strings.

    Returns
    -------
    list of int
        List of maximum widths for each column.
    """
    if not rows:
        return []

    for irow, row in enumerate(rows):
        if irow == 0:
            widths = [len(val) for val in row]
        else:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(val))
    return widths


def strs2row_iter(strs, colwidths, delim=' '):
    """
    Yield rows of strings formatted into columns.

    Parameters
    ----------
    strs : list of str
        List of strings to be formatted into columns.
    colwidths : list of int
        List of column widths.
    delim : str
        Delimiter to use between columns.

    Yields
    ------
    str
        Formatted row of strings.
    """
    for row in strs:
        yield delim.join(f"{val:<{width}}" for val, width in zip(row, colwidths))


def add_border(msg, borderstr='=', vpad=0, above=True, below=True):
    """
    Add border lines before and/or after a message.

    Parameters
    ----------
    msg : str
        The message to be enclosed in a border.
    borderstr : str
        The repeating string to be used in the border.
    vpad : int
        The number of blank lines between the border(s) and the message.
    above : bool
        If True, add a border above the message.
    below : bool
        If True, add a border below the message.

    Returns
    -------
    str
        A string containing the original message and border(s) before and/or after.
    """
    width = max(len(line) for line in msg.split('\n'))
    border = width * borderstr
    # handle borderstr of more than 1 char
    border = border[:width]
    uborder = border if above else ''
    lborder = border if below else ''
    upadding = '\n' * (vpad + 1) if uborder else '\n' * vpad
    lpadding = '\n' * vpad if vpad else '\n'
    return f"{uborder}{upadding}{msg}{lpadding}{lborder}"


def run_model(prob, ignore_exception=False):
    """
    Call `run_model` on problem and capture output.

    Parameters
    ----------
    prob : Problem
        An instance of Problem.
    ignore_exception : bool
        Set to True to ignore an exception of any kind.

    Returns
    -------
    string
        Output from calling `run_model` on the Problem, captured from stdout.
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
        An instance of Problem.

    Returns
    -------
    bool
        Failure flag; True if failed to converge, False is successful.
    string
        Output from calling `run_driver` on the Problem, captured from stdout.
    """
    stdout = sys.stdout
    strout = StringIO()

    sys.stdout = strout
    try:
        failed = not prob.run_driver().success
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

    >>> with printoptions(precision=2):
    ...     print(np.array([2.0])) / 3
    [0.67]
    The `as`-clause of the `with`-statement gives the current print options:
    >>> with printoptions(precision=2) as opts:
    ...      assert_equal(opts, np.get_printoptions())

    Parameters
    ----------
    *args : list
        Variable-length argument list.
    **kwds : dict
        Arbitrary keyword arguments.

    Yields
    ------
    str or int

    See Also
    --------
        set_printoptions : Set printing options.
        get_printoptions : Get printing options.
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


@contextmanager
def indent_context(stream, indent='   '):
    """
    Context manager for indenting all std output.

    Parameters
    ----------
    stream : stream
        The stream to write indented output to.
    indent : str
        The string to use for indentation.

    Yields
    ------
    str
        The current stdout.
    """
    save_stdout = sys.stdout
    save_stderr = sys.stderr

    # buffer all of stdout so we can indent it all
    sys.stdout = StringIO()
    sys.stderr = sys.stdout

    try:
        yield sys.stdout
    except Exception:
        # not sure what happened, so just print the whole thing without indentation
        print(sys.stdout.getvalue(), file=save_stdout)
        errs = sys.stderr.getvalue()
        if errs:
            print(errs, file=save_stderr)
        raise
    else:
        # nothing went wrong so do indentation
        if stream is not None:
            stream.write(textwrap.indent(sys.stdout.getvalue(), indent))
    finally:
        sys.stderr = save_stderr
        sys.stdout = save_stdout


def _nothing():
    yield None


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
    return contextmanager(_nothing)()


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


_container_classes = (list, tuple, set)


def make_serializable(o):
    """
    Recursively convert numpy types to native types for JSON serialization.

    This function should NOT be passed into json.dump or json.dumps as the 'default' arg.

    Parameters
    ----------
    o : object
        The object to be converted.

    Returns
    -------
    object
        The converted object.
    """
    if isinstance(o, _container_classes):
        return [make_serializable(item) for item in o]
    elif isinstance(o, dict):
        return {make_serializable_key(k): make_serializable(v) for k, v in o.items()}
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, np.number):
        return o.item()
    elif isinstance(o, (str, float, int)):
        return o
    elif isinstance(o, bool) or isinstance(o, complex):
        return str(o)
    elif hasattr(o, '__dict__'):
        try:
            return o.to_json()
        except AttributeError:
            return o.__class__.__name__
    else:
        return o


def make_serializable_key(o):
    """
    Recursively convert numpy types to native types for JSON serialization.

    This function is for making serizializable dictionary keys, so no containers.
    This function should NOT be passed into json.dump or json.dumps as the 'default' arg.

    Parameters
    ----------
    o : object
        The object to be converted.

    Returns
    -------
    object
        The converted object.
    """
    if isinstance(o, str):
        return o
    elif isinstance(o, np.number):
        return o.item()
    elif hasattr(o, '__dict__'):
        return o.__class__.__name__
    else:
        return str(o)


def default_noraise(o):
    """
    Try to convert some extra types during JSON serialization.

    This is intended to be passed to json.dump or json.dumps as the 'default' arg.  It will
    attempt to convert values if possible, but if no conversion works, will return
    'unserializable object (<type>)' instead of raising a TypeError.

    Parameters
    ----------
    o : object
        The object to be converted.

    Returns
    -------
    object
        The converted object.
    """
    if isinstance(o, _container_classes):
        return [default_noraise(item) for item in o]
    elif isinstance(o, dict):
        return {make_serializable_key(k): default_noraise(v) for k, v in o.items()}
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, np.number):
        return o.item()
    elif isinstance(o, (str, float, int)):
        return o
    elif isinstance(o, bool) or isinstance(o, complex):
        return str(o)
    elif hasattr(o, '__dict__'):
        return o.__class__.__name__
    elif o is None:
        return None
    else:
        return f"unserializable object ({type(o).__name__})"


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
    elif isinstance(str_data, (set, list)):

        for item in str_data:
            if not isinstance(item, str):
                typ = type(item).__name__
                msg = f"Items in tags should be of type string, but type '{typ}' was found."
                raise TypeError(msg)

        if isinstance(str_data, set):
            return str_data
        elif isinstance(str_data, list):
            return set(str_data)

    elif name:
        raise TypeError("The {} argument should be str, set, or list: {}".format(name, str_data))
    else:
        raise TypeError("The argument should be str, set, or list: {}".format(str_data))


def match_includes_excludes(name, includes=None, excludes=None):
    """
    Check to see if the variable name passes through the includes and excludes filter.

    Parameters
    ----------
    name : str
        Name to be checked for match.
    includes : iter of str or None
        Glob patterns for name to include in the filtering.  None, the default, means
        include all.
    excludes : iter of str or None
        Glob patterns for name to exclude in the filtering.

    Returns
    -------
    bool
        Return True if the name passes through the filtering of includes and excludes.
    """
    # Process excludes
    if excludes is not None:
        for pattern in excludes:
            if fnmatchcase(name, pattern):
                return False

    # Process includes
    if includes is None:
        return True
    else:
        for pattern in includes:
            if fnmatchcase(name, pattern):
                return True

    return False


def meta2src_iter(meta_iter):
    """
    Yield the source name for each metadata dict in the given iterator.

    Parameters
    ----------
    meta_iter : iter of dict
        Iterator over metadata dicts.

    Yields
    ------
    str
        The source name for each metadata dict.
    """
    for meta in meta_iter:
        yield meta['source']


def match_prom_or_abs(name, prom_name, includes=None, excludes=None):
    """
    Check to see if the variable names pass through the includes and excludes filter.

    Parameters
    ----------
    name : str
        Unpromoted variable name to be checked for match.
    prom_name : str
        Promoted variable name to be checked for match.
    includes : iter of str or None
        Glob patterns for name to include in the filtering.  None, the default, means
        to include all.
    excludes : iter of str or None
        Glob patterns for name to exclude in the filtering.

    Returns
    -------
    bool
        Return True if the name passes through the filtering of includes and excludes.
    """
    diff = name != prom_name

    # Process excludes
    if excludes is not None:
        for pattern in excludes:
            if fnmatchcase(name, pattern) or (diff and fnmatchcase(prom_name, pattern)):
                return False

    # Process includes
    if includes is None:
        return True
    else:
        for pattern in includes:
            if fnmatchcase(name, pattern) or (diff and fnmatchcase(prom_name, pattern)):
                return True

    return False


_falsey = {'0', 'false', 'no', 'off', 'none', ''}


def is_truthy(s):
    """
    Return True if the given string is 'truthy'.

    Parameters
    ----------
    s : str
        The name string being tested.

    Returns
    -------
    bool
        True if the specified string is 'truthy'.
    """
    return s.lower() not in _falsey


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
    return is_truthy(os.environ.get(env_var, ''))


def env_none(env_var):
    """
    Return True if the given environment variable is None.

    Parameters
    ----------
    env_var : str
        The name of the environment variable.

    Returns
    -------
    bool
        True if the specified environment variable is None.
    """
    return os.environ.get(env_var) is None


def common_subpath(pathnames):
    """
    Return the common dotted subpath found in all of the given dotted pathnames.

    Parameters
    ----------
    pathnames : list or tuple of str
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
        for common_loc in range(np.min([len(s) for s in splits])):
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


def _is_slicer_op(indices):
    """
    Check if an indexer contains a slice or ellipsis operator.

    Parameters
    ----------
    indices : ndarray
        Indices to check.

    Returns
    -------
    bool
        Returns True if indices contains a colon or ellipsis operator.
    """
    if isinstance(indices, tuple):
        return any(isinstance(i, slice) or i is ... for i in indices)

    return isinstance(indices, slice)


def _src_name_iter(proms):
    """
    Yield keys from proms with promoted input names converted to source names.

    Parameters
    ----------
    proms : dict
        Original dict with some promoted paths.

    Yields
    ------
    str
        source pathname name
    """
    for meta in proms.values():
        yield meta['source']


def _src_or_alias_item_iter(proms):
    """
    Yield items from proms dict with promoted input names converted to source or alias names.

    Parameters
    ----------
    proms : dict
        Original dict with some promoted paths.

    Yields
    ------
    tuple
        src_or_alias_name, metadata
    """
    for name, meta in proms.items():
        if 'alias' in meta and meta['alias'] is not None:
            yield meta['alias'], meta
        elif meta['source'] is not None:
            yield meta['source'], meta
        else:
            yield name, meta


def convert_src_inds(parent_src_inds, my_src_inds, my_src_shape):
    """
    Compute lower level src_indices based on parent src_indices.

    Parameters
    ----------
    parent_src_inds : ndarray
        Parent src_indices.
    my_src_inds : ndarray or fancy index
        Src_indices at the current system level, before conversion.
    my_src_shape : tuple
        Expected source shape at the current system level.

    Returns
    -------
    ndarray
        Final src_indices based on those of the parent.
    """
    if parent_src_inds is None:
        return my_src_inds
    elif my_src_inds is None:
        return parent_src_inds

    if my_src_inds._flat_src:
        return parent_src_inds.shaped_array(flat=True)[my_src_inds.flat()]
    else:
        return parent_src_inds.shaped_array(flat=False).reshape(my_src_shape)[my_src_inds()]


def is_undefined(obj):
    """
    Return True if the object is _UNDEFINED.

    This function should be used instead of `{obj} is _UNDEFINED`, which
    is not reliable across processes. The use of `{obj} == _UNDEFINED` will
    fail if `obj` is an array.

    Parameters
    ----------
    obj : any
        Any python object.

    Returns
    -------
    bool
        True if the obj is not an array, and obj == _UNDEFINED.
    """
    if isinstance(obj, Iterable):
        return False
    return obj == _UNDEFINED


def shape2tuple(shape):
    """
    Return shape as a tuple.

    Parameters
    ----------
    shape : int or tuple
        The given shape.

    Returns
    -------
    tuple or None
        The shape as a tuple or None if shape is None.
    """
    if isinstance(shape, tuple):
        return shape
    elif isinstance(shape, int):
        return (shape,)
    elif shape is None:
        return shape
    else:
        try:
            return tuple(shape)
        except TypeError:
            if not isinstance(shape, Integral):
                raise TypeError(f"{type(shape).__name__} is not a valid shape type.")
            return (shape,)


def get_connection_owner(system, tgt):
    """
    Return (owner, promoted_src, promoted_tgt) for the given connected target.

    Note : this is not speedy.  It's intended for use only in error messages.

    Parameters
    ----------
    system : System
        Any System.  The search always goes from the model level down.
    tgt : str
        Absolute pathname of the target variable.

    Returns
    -------
    tuple
        (owning group, promoted source name, promoted target name).
    """
    from openmdao.core.group import Group

    model = system._problem_meta['model_ref']()
    src = model._conn_global_abs_in2out[tgt]
    resolver = model._resolver

    if resolver.is_abs(src, 'output') and resolver.is_abs(tgt, 'input'):
        if resolver.abs2prom(tgt, 'input') != resolver.abs2prom(src, 'output'):
            # connection is explicit
            for g in model.system_iter(include_self=True, recurse=True, typ=Group):
                if g._manual_connections:
                    tprom = g._resolver.abs2prom(tgt, 'input')
                    if tprom in g._manual_connections:
                        return g, g._resolver.abs2prom(src, 'output'), tprom

    return system, src, tgt


def _remove_old_configs(vscode_dir):
    launch_path = os.path.join(vscode_dir, 'launch.json')

    # Read existing launch.json if it exists
    if os.path.exists(launch_path):
        with open(launch_path, 'r') as f:
            config_top = json.load(f)
        configs = config_top.get('configurations', [])
        compounds = config_top.get('compounds', [])

        # remove any old auto-added configs from configurations and compounds
        configs = [c for c in configs
                   if not (c['name'].startswith('_rank_') and c['name'].endswith('_config'))]

        compounds = [c for c in compounds if c['name'] != 'MPI Debug']

        if configs:
            config_top['configurations'] = configs
        elif 'configurations' in config_top:
            del config_top['configurations']

        if compounds:
            config_top['compounds'] = compounds
        elif 'compounds' in config_top:
            del config_top['compounds']

        with open(os.path.join(vscode_dir, "launch.json"), "w") as f:
            json.dump(config_top, f, indent=2)


def generate_launch_json_file(vscode_dir, base_port, ranks):
    """
    Generate a launch.json file for the VSCode debugger.

    Parameters
    ----------
    vscode_dir : str
        The full path of the .vscode directory.
    base_port : int
        The base port number for the debugger.
    ranks : int
        The specific ranks to debug.
    """
    _remove_old_configs(vscode_dir)

    launch_path = os.path.join(vscode_dir, 'launch.json')

    # Read existing launch.json if it exists
    if os.path.exists(launch_path):
        with open(launch_path, 'r') as f:
            config_top = json.load(f)
    else:
        config_top = {"version": "0.2.0", "configurations": [], "compounds": []}

    configs = config_top.get('configurations', [])
    compounds = config_top.get('compounds', [])

    new_configs = []
    new_compound_configs = []

    # Add a configuration for each MPI rank
    for rank in ranks:
        config_name = f"_rank_{rank}_config"
        config = {
            "name": config_name,
            "type": "python",
            "request": "attach",
            "port": base_port + rank,
            "host": "localhost",
            "justMyCode": True,
            "presentation": {
                "order": rank + 2
            }
        }
        new_configs.append(config)
        new_compound_configs.append(config_name)

    configs.extend(new_configs)
    compounds.append(
        {
            "name": "MPI Debug",
            "configurations": new_compound_configs,
            "presentation": {
                "order": 1
            }
        }
    )

    config_top['configurations'] = configs
    config_top['compounds'] = compounds

    with open(os.path.join(vscode_dir, "launch.json"), "w") as f:
        json.dump(config_top, f, indent=2)

    if MPI is None or MPI.COMM_WORLD.rank == 0:
        atexit.register(functools.partial(_remove_old_configs, vscode_dir))


def _vscode_env_error(env_var):
    print("Invalid VSCODE_DBG environment variable. Expected ':<port>' or "
          f"'<rank1,rank2,...>:<port>' but got '{env_var}'. Debugging aborted.", flush=True)
    sys.exit(1)


def setup_dbg():
    """
    If WING_DBG or VSCODE_DBG is truthy in the environment, set up their debuggers.
    """
    # Get the base port from the VSCODE_DBG environment variable if set
    vscode_dbg = os.environ.get("VSCODE_DBG")

    use_def_ranks = True
    if vscode_dbg is not None:
        if MPI is None:
            myrank = 0
            ranks = [0]
        else:
            myrank = MPI.COMM_WORLD.rank
            ranks = range(MPI.COMM_WORLD.size)  # by default, debug all ranks

        default_base_port = 51111

        if vscode_dbg == '1':  # use default port and all ranks
            portstr = str(default_base_port)
        else:
            if ':' in vscode_dbg:
                ranks_str, _, portstr = vscode_dbg.partition(':')
            else:
                ranks_str = vscode_dbg
                portstr = str(default_base_port)

            if ',' in ranks_str:
                use_def_ranks = False
                try:
                    ranks = [int(r) for r in ranks_str.split(',') if r.strip()]
                except (ValueError, TypeError):
                    _vscode_env_error(vscode_dbg)
            elif ranks_str.strip() == '':
                use_def_ranks = True
            else:  # single rank
                use_def_ranks = False
                try:
                    ranks = [int(ranks_str)]
                except (ValueError, TypeError):
                    _vscode_env_error(vscode_dbg)

        try:
            base_port = int(portstr)
        except (ValueError, TypeError):
            _vscode_env_error(vscode_dbg)

        # verify ranks are valid
        if MPI is not None and not use_def_ranks:
            badranks = []
            for r in ranks:
                if not 0 <= r < MPI.COMM_WORLD.size:
                    badranks.append(r)
            if badranks:
                print("The following ranks are outside of the valid range of "
                      f"(0-{MPI.COMM_WORLD.size - 1}): {badranks}. Debugging aborted.", flush=True)
                sys.exit(1)

        debug_port = base_port + myrank

        if myrank == min(ranks):
            omdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            vscode_dir = os.path.join(omdir, ".vscode")
            if not os.path.exists(vscode_dir):
                os.makedirs(vscode_dir)
            generate_launch_json_file(vscode_dir, base_port, ranks)

        if MPI is not None:
            MPI.COMM_WORLD.barrier()

        if myrank in ranks:
            # disable annoying debugger warning message
            os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
            import debugpy
            print(f"Rank {myrank}: Debugger listening on port {debug_port}", flush=True)
            # only listen for connections from localhost for security reasons
            debugpy.listen(('127.0.0.1', debug_port))
            debugpy.wait_for_client()  # This will block until a debugger connects

    elif env_truthy('WING_DBG'):
        save = sys.path
        new = sys.path[:] + [os.environ['WINGHOME']]
        sys.path = new
        try:
            import wingdbstub  # noqa: F401
        finally:
            sys.path = save


class LocalRangeIterable(object):
    """
    Iterable object yielding local indices while iterating over local, distributed, or remote vars.

    The number of iterations for a distributed variable will be the full distributed size of the
    variable.

    None will be returned for any indices that are not local to the given rank.

    Parameters
    ----------
    system : System
        Containing System.
    vname : str
        Name of the variable.
    use_vec_offset : bool
        If True, return indices for the given variable within its parent vector, else just return
        indices within the variable itself, i.e. range(var_size).

    Attributes
    ----------
    _vname : str
        Name of the variable.
    _inds : ndarray
        Variable indices (unused for distributed variables).
    _var_size : int
        Full size of distributed or remote variable.
    _start : int
        Starting index of distributed variable on this rank.
    _end : int
        Last index + 1 of distributed variable on this rank.
    _offset : int
        Offset of this variable into the local vector,.
    _iter : method
        The iteration method used.
    """

    def __init__(self, system, vname, use_vec_offset=True):
        """
        Initialize the iterator.
        """
        self._vname = vname
        self._var_size = 0

        all_abs2meta = system._var_allprocs_abs2meta['output']
        if vname in all_abs2meta:
            sizes = system._var_sizes['output']
            vec = system._outputs
            abs2meta = system._var_abs2meta['output']
        else:
            all_abs2meta = system._var_allprocs_abs2meta['input']
            sizes = system._var_sizes['input']
            vec = system._inputs
            abs2meta = system._var_abs2meta['input']

        if all_abs2meta[vname]['distributed']:
            var_idx = system._var_allprocs_abs2idx[vname]
            rank = system.comm.rank
            self._offset = np.sum(sizes[rank, :var_idx]) if use_vec_offset else 0

            self._iter = self._dist_iter
            self._start = np.sum(sizes[:rank, var_idx])
            self._end = self._start + sizes[rank, var_idx]
            self._var_size = np.sum(sizes[:, var_idx])
        elif vname not in abs2meta:  # variable is remote
            self._iter = self._remote_iter
            self._var_size = all_abs2meta[vname]['global_size']
        else:
            self._iter = self._serial_iter
            start, stop = vec.get_range(vname)
            if use_vec_offset:
                self._inds = range(start, stop)
            else:
                self._inds = range(stop - start)
            self._var_size = all_abs2meta[vname]['global_size']

    def __repr__(self):
        """
        Return a string representation of the iterator.

        Returns
        -------
        str
            String representation of the iterator.
        """
        if self._iter is self._dist_iter:
            return f"LocalRangeIterable({self._vname}, dist: {self._start} to {self._end})"
        elif self._iter is self._remote_iter:
            return f"LocalRangeIterable({self._vname}, remote: size={self._var_size})"
        return f"LocalRangeIterable({self._vname}, serial: size={self._var_size})"

    def _serial_iter(self):
        """
        Iterate over a local non-distributed variable.

        Yields
        ------
        int
            Variable index.
        """
        yield from self._inds

    def _dist_iter(self):
        """
        Iterate over a distributed variable.

        Yields
        ------
        int or None
            Variable index or None if index is not local to this rank.
        """
        start = self._start
        end = self._end

        for i in range(self._var_size):
            if i >= start and i < end:
                yield i - start + self._offset
            else:
                yield None

    def _remote_iter(self):
        """
        Iterate over a remote variable.

        Yields
        ------
        None
            Always yields None.
        """
        for _ in range(self._var_size):
            yield None

    def __iter__(self):
        """
        Return an iterator.

        Returns
        -------
        iterator
            An iterator over our indices.
        """
        return self._iter()


def make_traceback():
    """
    Create a traceback for use later with an exception.

    The traceback will begin at the stack frame *above* the caller of make_traceback.

    Returns
    -------
    traceback
        The newly constructed traceback.
    """
    finfo = getouterframes(currentframe())[2]
    return TracebackType(None, finfo.frame, finfo.frame.f_lasti, finfo.frame.f_lineno)


def inconsistent_across_procs(comm, arr, tol=1e-15, return_array=True):
    """
    Check serial deriv values across ranks.

    This should only be run after _apply_linear.

    Parameters
    ----------
    comm : MPI communicator
        Communicator belonging to the component that owns the derivs array.
    arr : ndarray
        The array being checked for consistency across processes.
    tol : float
        Tolerance to determine if diff is 0.
    return_array : bool
        If True, return a boolean array on rank 0 indicating which indices are inconsistent.

    Returns
    -------
    ndarray on rank 0, boolean elsewhere, or bool everywhere if return_array is False
        On rank 0, boolean array with True in entries that are not consistent across all processes
        in the communicator.  On other ranks, True if there are inconsistent entries.
    """
    if comm.size < 2:
        return np.zeros(0, dtype=bool) if return_array and comm.rank == 0 else False

    if comm.rank == 0:
        result = np.zeros(arr.size, dtype=bool) if return_array else False
        for rank, val in enumerate(comm.gather(arr, root=0)):
            if rank == 0:
                baseval = val
            elif return_array:
                result |= (np.abs(baseval - val) > tol).flat
            else:
                result |= np.any(np.abs(baseval - val) > tol)

        if return_array:
            comm.bcast(np.any(result), root=0)
        else:
            comm.bcast(result, root=0)
        return result

    comm.gather(arr, root=0)
    return comm.bcast(None, root=0)


def get_rev_conns(conns):
    """
    Return a dict mapping each connected output to a list of its connected inputs.

    Parameters
    ----------
    conns : dict
        Dict mapping each input to its connected output.

    Returns
    -------
    dict
        Dict mapping each connected output to a list of its connected inputs.
    """
    rev = {}
    for tgt, src in conns.items():
        if src in rev:
            rev[src].append(tgt)
        else:
            rev[src] = [tgt]
    return rev


def vprint(it, end='\n', getter=None, file=None):
    """
    Iterate over the given iterator and print each item separated by end.

    Parameters
    ----------
    it : iter
        Iterator to be printed.
    end : str
        String written after each item.
    getter : function or None
        If not None, only print the part of each item returned by getter(item).
    file : file-like or None
        File to write to.  If None, use sys.stdout.
    """
    if file is None:
        file = sys.stdout

    for val in it:
        if getter is not None:
            val = getter(val)
        print(val, end=end, file=file)


def _default_predicate(name, obj):
    """
    Determine if a given method should be traced.

    Parameters
    ----------
    name : str
        Name of the method.
    obj : object
        The object being checked.

    Returns
    -------
    bool
        True if the method should be traced.
    """
    for n in ['solve', 'apply', 'compute', 'setup', 'coloring', 'linearize', 'get_outputs_dir',
              'approx', 'static', 'get_vars', 'abs_get']:
        if n in name:
            return True
    return False


_trace_predicate = _default_predicate


def set_trace_predicate(funct):
    """
    Set the function that determines which methods will be traced.

    Parameters
    ----------
    funct : function
        Set of functions to be traced.
    """
    global _trace_predicate
    _trace_predicate = funct


def _decorate_functs(attrs, predicate, decorator):
    """
    Decorate the functions in the given attribute dict.

    Parameters
    ----------
    cname : str
        The name of the class containing the functions.
    attrs : dict
        The attribute dict containing the functions to be decorated.
    predicate : function
        Function returning True if the function should be decorated.
    decorator : function
        The decorator function.
    """
    for name, obj in attrs.items():
        if (isfunction(obj) or ismethod(obj)) and predicate(name, obj):
            attrs[name] = decorator(obj)


SystemMetaclass = type
ProblemMetaclass = type
SolverMetaclass = type
DriverMetaclass = type
DebugMeta = type


def om_dump(*args, **kwargs):
    r"""
    Do nothing.

    Parameters
    ----------
    *args : list
        Positional args.
    **kwargs : dict
        Named args.
    """
    pass


om_dump_indent = om_dump


def dbg(funct):
    """
    Do nothing.

    Parameters
    ----------
    funct : function
        The function being decorated.

    Returns
    -------
    function
        The function.
    """
    return funct


def _wrap_comm(comm, scope=None):
    return comm


def _unwrap_comm(comm):
    return comm


_om_dump = env_truthy('OPENMDAO_DUMP')
# OPENMDAO_DUMP can have values like 'stdout', 'stderr', 'rank', 'pid', 'rank,pid', 'pid,rank'
# 'rank' means to include the rank in the dump file name, e.g., om_dump_0.out
# 'pid' means to include the pid in the dump file name, e.g., om_dump_12345.out
# if rank and pid are both included, the file name will be, e.g., om_dump_0_12345.out
# 'stdout' means to dump to stdout (so rank and pid are ignored)
# 'stderr' means to dump to stderr (so rank and pid are ignored)
# 'trace' means to print function entry and exit
# 'args' means to print function entry and exit with args and kwargs if trace is also included
# if OPENMDAO_DUMP is just a plain truthy value, like '1', then we dump to a file
# named om_dump.out.

if _om_dump:
    parts = [s.strip() for s in os.environ['OPENMDAO_DUMP'].split(',')]
    trace = 'trace' in parts
    use_rank = 'rank' in parts

    if 'stdout' in parts:
        _dump_stream = sys.stdout
    elif 'stderr' in parts:
        _dump_stream = sys.stderr
    else:
        dirname = None

        for p in parts:
            if p.startswith('dir='):
                dirname = p.partition('=')[2]
                break
        else:
            dirname = os.path.join(os.getcwd(), 'dump_dir')

        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except Exception:
                dirname = os.getcwd()

        for p in parts:
            if p.startswith('file='):
                fname = p.partition('=')[2]
                break
        else:
            testspec = os.environ.get('TESTFLO_SPEC')
            if testspec:
                tpath, ident = testspec.split(':')
                tfile = os.path.basename(tpath)
                fname = f'om_dump_{tfile}:{ident}'
                use_rank = True  # always use rank for testflo tests
            else:
                fname = 'om_dump'

        rankstr = pidstr = ''
        if use_rank:
            from openmdao.utils.mpi import MPI
            rankstr = f"_{MPI.COMM_WORLD.rank if MPI else 0}"

        if 'pid' in parts:
            pidstr = f"_{os.getpid()}"

        _dump_stream = open(os.path.join(dirname, f'{fname}{rankstr}{pidstr}.out'), 'w')

    _show_args = 'args' in parts

    def om_dump(*args, **kwargs):
        """
        Dump to a stream if OPENMDAO_DUMP is truthy in the environment.

        Depending on the value of OPENMDAO_DUMP, output will go to file(s), stdout, or stderr.

        Parameters
        ----------
        args : list
            Positional args.
        kwargs : dict
            Named args.
        """
        kwargs['file'] = _dump_stream
        kwargs['flush'] = True
        print(*args, **kwargs)

    def om_dump_indent(pathobj, *args, **kwargs):
        """
        Dump to a stream with indent if OPENMDAO_DUMP is truthy in the environment.

        Depending on the value of OPENMDAO_DUMP, output will go to file(s), stdout, or stderr.

        Parameters
        ----------
        pathobj : object
            The object to get the pathname from.
        args : list
            Positional args.
        kwargs : dict
            Named args.
        """
        pathname = pathobj.pathname
        if pathname:
            indent = ' ' * (len(pathname.split('.')) * 3)
            newargs = []
            for arg in args:
                if isinstance(arg, str):
                    newargs.append(arg)
                else:
                    newargs.append(str(arg))

            args = ' '.join(newargs)
            args = textwrap.indent(args, indent)
            args = (args,)

        kwargs['file'] = _dump_stream
        kwargs['flush'] = True
        print(*args, **kwargs)

    def dbg(cname):
        """
        Decorate function to print function entry and exit.

        Parameters
        ----------
        cname : str
            The name of the class containing the function.

        Returns
        -------
        function
            The decorated function.
        """
        def _dbg(funct):
            def wrapper(*args, **kwargs):
                try:
                    path = args[0].pathname + '.'
                except Exception:
                    path = ''
                indent = call_depth2indent()
                if _show_args:
                    argstr = f"(args={args}, kwargs={kwargs})"
                else:
                    argstr = ''
                om_dump(f"{indent}--> {cname}:{path}{funct.__name__}{argstr}")
                ret = funct(*args, **kwargs)
                om_dump(f"{indent}<-- {cname}:{path}{funct.__name__}")
                return ret

            return wrapper
        return _dbg

    if trace:
        class DebugMeta(type):
            """
            A metaclass to add trace output to some methods of the class.

            Parameters
            ----------
            name : str
                The name of the class.
            bases : tuple
                The base classes of the class.
            attrs : dict
                The attributes of the class.

            Returns
            -------
            class
                The class with the metaclass applied.
            """

            def __new__(metaclass, name, bases, attrs):
                """
                Add trace output to some methods of the class.

                Parameters
                ----------
                name : str
                    The name of the class.
                bases : tuple
                    The base classes of the class.
                attrs : dict
                    The attributes of the class.

                Returns
                -------
                class
                    The class with trace output added to some methods
                """
                _decorate_functs(attrs, _trace_predicate, dbg(name))
                return super().__new__(metaclass, name, bases, attrs)

        SystemMetaclass = DebugMeta
        ProblemMetaclass = DebugMeta
        SolverMetaclass = DebugMeta
        DriverMetaclass = DebugMeta

        def _comm_debug_decorator(fn, scope):  # pragma no cover
            def _wrap(*args, **kwargs):
                sc = '' if scope is None else f"{scope}."
                indent = call_depth2indent()
                if _show_args:
                    argstr = f"(args={args}, kwargs={kwargs})"
                else:
                    argstr = ''
                om_dump(f"{indent}--> {sc}{fn.__name__}{argstr}")
                ret = fn(*args, **kwargs)
                om_dump(f"{indent}<-- {sc}{fn.__name__}")
                return ret
            return _wrap

        class _DebugComm(object):  # pragma no cover
            """
            Debugging wrapper for an MPI communicator.
            """

            def __init__(self, comm, scope):
                if isinstance(comm, _DebugComm):
                    self.__dict__['_comm'] = comm._comm
                else:
                    self.__dict__['_comm'] = comm
                self.__dict__['_scope'] = scope
                for name in ['bcast', 'Bcast', 'gather', 'Gather', 'scatter', 'Scatter',
                             'allgather', 'Allgather', 'Allgatherv', 'allreduce', 'Allreduce',
                             'send', 'Send', 'recv', 'Recv', 'sendrecv', 'Sendrecv']:
                    self.__dict__[name] = _comm_debug_decorator(getattr(self._comm, name), scope)

            def __getattr__(self, name):
                return getattr(self._comm, name)

            def __setattr__(self, name, val):
                setattr(self._comm, name, val)

        def _wrap_comm(comm, scope=None):  # pragma no cover
            return _DebugComm(comm, scope)

        def _unwrap_comm(comm):  # pragma no cover
            if isinstance(comm, _DebugComm):
                return comm._comm
            return comm

# if OPENMDAO_DUMP is set to anything, even a falsey value, make om_dump and om_dump_indent
# available as builtins so we don't have to import them anywhere
if os.environ.get('OPENMDAO_DUMP') is not None:
    import builtins
    builtins.om_dump = om_dump
    builtins.om_dump_indent = om_dump_indent


def call_depth2indent(tabsize=2, offset=-1):
    """
    Return a string of spaces corresponding to the current call depth.

    Parameters
    ----------
    tabsize : int
        Number of spaces per tab.
    offset : int
        Offset to add to the call depth.

    Returns
    -------
    str
        A string of spaces.
    """
    return ' ' * ((len(stack()) + offset) * tabsize)


def print_with_line_numbers(text, **kwargs):
    """
    Print a string with each line preceded by its line number.

    Parameters
    ----------
    text : str
        The text to print with line numbers.
    **kwargs : dict
        Keyword arguments to pass to print.
    """
    for i, line in enumerate(text.splitlines(), 1):
        print(f"{i:5d} | {line}", **kwargs)
