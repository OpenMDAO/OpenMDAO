"""Some miscellaneous utility functions."""
import os
import re
import sys
import types
import unittest
from contextlib import contextmanager
from fnmatch import fnmatchcase
from io import StringIO
from numbers import Integral
from inspect import currentframe, getouterframes

from collections.abc import Iterable

import numpy as np

from openmdao.core.constants import INF_BOUND
from openmdao.utils.om_warnings import issue_warning, warn_deprecation
from openmdao.utils.array_utils import shape_to_len


_float_inf = float('inf')


def _convert_auto_ivc_to_conn_name(conns_dict, name):
    """
    Convert name of auto_ivc val to promoted input name.

    Parameters
    ----------
    conns_dict : dict
        Dictionary of global connections.
    name : str
        Name of auto_ivc to be found.

    Returns
    -------
    str
        Promoted input name.
    """
    for key, val in conns_dict.items():
        if val == name:
            return key


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
    warn_deprecation('simple_warning is deprecated. '
                     'Use openmdao.utils.om_warnings.issue_warning instead.')
    issue_warning(msg, stacklevel=stacklevel, category=category)


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
    indices : Indexer or None
        The indices into a source variable.

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
        value = np.atleast_1d(value)
        shape = value.shape
    else:
        # shape is determined, if value is scalar assign it to array of shape
        # otherwise make sure value is an array of the determined shape
        if np.ndim(value) == 0 or value.shape == (1,):
            value = np.full(shape, value)
        else:
            value = np.atleast_1d(value).astype(np.float64)
            if value.shape != shape:
                raise ValueError(f"Incompatible shape for '{name}': Expected {shape} but got "
                                 f"{value.shape}.")

    return value, shape


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
    parts = pathname.split(delim)
    for i in range(len(parts), 0, -1):
        yield delim.join(parts[:i])


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
    elif pattern in var_list:
        return [pattern]
    return [name for name in var_list if fnmatchcase(name, pattern)]


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
        True if non-None metadata at the given key was found.
    """
    for meta in dct.values():
        if key in meta and meta[key] is not None:
            return True
    return False


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
        Padded string.
    """
    name = f"'{name}'" if quotes else name
    if pad_num > len(name):
        return f"{name:<{pad_num}}"
    else:
        return f'{name}'


def add_border(msg, borderstr='=', vpad=0):
    """
    Add border lines before and after a message.

    The message is assumed not to span multiple lines.

    Parameters
    ----------
    msg : str
        The message to be enclosed in a border.
    borderstr : str
        The repeating string to be used in the border.
    vpad : int
        The number of blank lines between the border and the message (before and after).

    Returns
    -------
    str
        A string containing the original message enclosed in a border.
    """
    border = len(msg) * borderstr
    # handle borderstr of more than 1 char
    border = border[:len(msg)]
    return f"{border}\n{msg}\n{border}"


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
        s_key = [make_serializable_key(item) for item in o.keys()]
        s_val = [make_serializable(item) for item in o.values()]
        return dict(zip(s_key, s_val))
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
        s_key = [make_serializable_key(item) for item in o.keys()]
        s_val = [default_noraise(item) for item in o.values()]
        return dict(zip(s_key, s_val))
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


def filtered_name_iter(name_iter, includes=None, excludes=None):
    """
    Yield names that pass through the includes and excludes filters.

    Parameters
    ----------
    name_iter : iter of str
        Iterator over names to be checked for match.
    includes : iter of str or None
        Glob patterns for name to include in the filtering.  None, the default, means
        include all.
    excludes : iter of str or None
        Glob patterns for name to exclude in the filtering.

    Yields
    ------
    str
        Each name that passes through the filters.
    """
    for name in name_iter:
        if match_includes_excludes(name, includes, excludes):
            yield name


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


def _src_or_alias_name(meta):
    if 'alias' in meta:
        alias = meta['alias']
        if alias:
            return alias
    return meta['source']


def _src_or_alias_item_iter(proms):
    """
    Yield items from proms with promoted input names converted to source or alias names.

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


def _src_or_alias_dict(prom_dict):
    """
    Convert a dict with promoted input names into one with source or alias names.

    Parameters
    ----------
    prom_dict : dict
        Original dict with some promoted paths.

    Returns
    -------
    dict
        New dict with source pathnames or alias names.
    """
    return {name: meta for name, meta in _src_or_alias_item_iter(prom_dict)}


def convert_src_inds(parent_src_inds, parent_src_shape, my_src_inds, my_src_shape):
    """
    Compute lower level src_indices based on parent src_indices.

    Parameters
    ----------
    parent_src_inds : ndarray
        Parent src_indices.
    parent_src_shape : tuple
        Shape of source expected by parent.
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
    abs2prom = model._var_allprocs_abs2prom

    if src in abs2prom['output'] and tgt in abs2prom['input'][tgt]:
        if abs2prom['input'][tgt] != abs2prom['output'][src]:
            # connection is explicit
            for g in model.system_iter(include_self=True, recurse=True, typ=Group):
                if g._manual_connections:
                    tprom = g._var_allprocs_abs2prom['input'][tgt]
                    if tprom in g._manual_connections:
                        return g, g._var_allprocs_abs2prom['output'][src], tprom

    return system, src, tgt


def wing_dbg():
    """
    Make import of wingdbstub contingent on value of WING_DBG environment variable.

    Also will import wingdbstub from the WINGHOME directory.
    """
    if env_truthy('WING_DBG'):
        import sys
        import os
        save = sys.path
        new = sys.path[:] + [os.environ['WINGHOME']]
        sys.path = new
        try:
            import wingdbstub
        finally:
            sys.path = save


class LocalRangeIterable(object):
    """
    Iterable object yielding local indices while iterating over local or distributed vars.

    The number of iterations for a distributed variable will be the full distributed size of the
    variable but None will be returned for any indices that are not local to the given rank.

    Parameters
    ----------
    system : System
        Containing System.
    vname : str
        Name of the variable.
    use_vec_offset : bool
        If True, return indices for the given variable within its vector, else just return
        indices within the variable itself, i.e. range(var_size).

    Attributes
    ----------
    _inds : ndarray
        Variable indices (unused for distributed variables).
    _dist_size : int
        Full size of distributed variable.
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
        self._dist_size = 0

        abs2meta = system._var_allprocs_abs2meta['output']
        if vname in abs2meta:
            sizes = system._var_sizes['output']
            slices = system._outputs.get_slice_dict()
        else:
            abs2meta = system._var_allprocs_abs2meta['input']
            sizes = system._var_sizes['input']
            slices = system._inputs.get_slice_dict()

        if abs2meta[vname]['distributed']:
            var_idx = system._var_allprocs_abs2idx[vname]
            rank = system.comm.rank
            self._offset = np.sum(sizes[rank, :var_idx]) if use_vec_offset else 0

            self._iter = self._dist_iter
            self._start = np.sum(sizes[:rank, var_idx])
            self._end = self._start + sizes[rank, var_idx]
            self._dist_size = np.sum(sizes[:, var_idx])
        else:
            self._iter = self._serial_iter
            if use_vec_offset:
                self._inds = range(slices[vname].start, slices[vname].stop)
            else:
                self._inds = range(slices[vname].stop - slices[vname].start)

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

        for i in range(self._dist_size):
            if i >= start and i < end:
                yield i - start + self._offset
            else:
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
    return types.TracebackType(None, finfo.frame, finfo.frame.f_lasti, finfo.frame.f_lineno)


if env_truthy('OM_DBG'):
    def dprint(*args, **kwargs):
        """
        Print only if OM_DBG is truthy in the environment.

        Parameters
        ----------
        args : list
            Positional args.
        kwargs : dict
            Named args.
        """
        print(*args, **kwargs)
else:
    def dprint(*args, **kwargs):
        """
        Print only if OM_DBG is truthy in the environment.

        Parameters
        ----------
        args : list
            Positional args.
        kwargs : dict
            Named args.
        """
        pass


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
