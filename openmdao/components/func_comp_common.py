"""
Define functions and objects common to the ExplicitFuncComp and ImplicitFuncComp classes.
"""

import sys
import traceback
import re
from functools import partial

import numpy as np
try:
    import jax
    from jax import vmap
    import jax.numpy as jnp
    # linear_util moved to jax.extend in jax 0.4.17, previous location is deprecated
    try:
        from jax.extend import linear_util
    except ImportError:
        from jax import linear_util
    from jax.api_util import argnums_partial
    from jax._src.api import _jvp, _vjp
    jax.config.update("jax_enable_x64", True)  # jax by default uses 32 bit floats
except Exception:
    _, err, tb = sys.exc_info()
    if not isinstance(err, ImportError):
        traceback.print_tb(tb)
    jax = None

from openmdao.utils.om_warnings import issue_warning
from openmdao.vectors.vector import Vector
from openmdao.core.constants import INT_DTYPE


# regex to check for variable names.
namecheck_rgx = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')

# Names that are not allowed for input or output variables (keywords for options)
_disallowed_varnames = {
    'units', 'shape', 'shape_by_conn', 'run_root_only', 'distributed', 'assembled_jac_type'
}


def _copy_with_ignore(dct, keep, ignore=()):
    """
    Copy the entries in the given dict whose keys are in keep.

    Parameters
    ----------
    dct : dict
        The dictionary to be copied.
    keep : set-like
        Set of keys for entries we want to keep.
    ignore : set or tuple
        Don't issue a warning for these non-keeper keys.

    Returns
    -------
    dict
        A new dict containing 'keep' entries.
    """
    return {k: v for k, v in dct.items() if k in keep and k not in ignore}


def _check_var_name(comp, name):
    match = namecheck_rgx.match(name)
    if match is None or match.group() != name:
        raise NameError(f"{comp.msginfo}: '{name}' is not a valid variable name.")

    if name in _disallowed_varnames:
        raise NameError(f"{comp.msginfo}: cannot use variable name '{name}' because "
                        "it's a reserved keyword.")


def _add_options(comp):
    """
    Add function component specific options to the given component.

    Parameters
    ----------
    comp : ImplicitFuncComp or ExplicitFuncComp
        The function component having options added.
    """
    comp.options.declare('use_jax', types=bool, default=False,
                         desc='If True, use jax to compute derivatives.')
    comp.options.declare('use_jit', types=bool, default=False,
                         desc='If True, attempt to use jit on the function. This is ignored if '
                              'use_jax is False.')


def jac_forward(fun, argnums, tangents):
    """
    Similar to the jax.jacfwd function but allows specification of the tangent matrix.

    This allows us to generate a compressed jacobian based on coloring.

    Parameters
    ----------
    fun : function
        The function to be differentiated.
    argnums : tuple of int or None
        Specifies which positional args are dynamic.  None means all positional args are dynamic.
    tangents : ndarray
        Array of 1.0's and 0's that is used to compute the value of the jacobian matrix.

    Returns
    -------
    function
        If there are multiple output variables, returns a function that returns rows of the
        jacobian grouped by output variable, e.g., if there were 2 output variables of size 3 and 4,
        the function would return a list with two entries. The first entry would contain the first 3
        rows of J and the second would contain the next 4 rows of J.  If there is only 1 output
        variable, the values returned are grouped by input variable.
    """
    f = linear_util.wrap_init(fun)
    if argnums is None:
        def jacfunf(*args):
            return vmap(partial(_jvp, f, args), out_axes=(None, -1))(tangents)[1]
    else:
        def jacfunf(*args):
            f_partial, dyn_args = argnums_partial(f, argnums, args)
            return vmap(partial(_jvp, f_partial, dyn_args), out_axes=(None, -1))(tangents)[1]
    return jacfunf


def jac_reverse(fun, argnums, tangents):
    """
    Similar to the jax.jacrev function but allows specification of the tangent matrix.

    This allows us to generate a compressed jacobian based on coloring.

    Parameters
    ----------
    fun : function
        The function to be differentiated.
    argnums : tuple of int or None
        Specifies which positional args are dynamic.  None means all positional args are dynamic.
    tangents : ndarray
        Array of 1.0's and 0's that is used to compute the value of the jacobian matrix.

    Returns
    -------
    function
        A function that returns rows of the jacobian grouped by function input variable, e.g., if
        there were 3 input variables of size 5 and 7 and 9, the function
        would return a list with 3 entries. The first entry would contain the first 5 columns of J,
        the second the next 7 columns of J, and the third the next 9 columns of J.  Note that for
        implicit systems, the function inputs will contain both inputs and outputs in the context
        of OpenMDAO.
    """
    f = linear_util.wrap_init(fun)
    if argnums is None:
        def jacfunr(*args):
            return vmap(_vjp(f, *args)[1])(tangents)
    else:
        def jacfunr(*args):
            f_partial, dyn_args = argnums_partial(f, argnums, args)
            return vmap(_vjp(f_partial, *dyn_args)[1])(tangents)

    return jacfunr


def jacvec_prod(fun, argnums, invals, tangent):
    """
    Similar to the jvp function but gives back a flat column.

    Note: this is significantly slower (when producing a full jacobian) than jac_forward.

    Parameters
    ----------
    fun : function
        The function to be differentiated.
    argnums : tuple of int or None
        Specifies which positional args are dynamic.  None means all positional args are dynamic.
    invals : tuple of float or ndarray
        Dynamic function input values.
    tangent : ndarray
        Array of 1.0's and 0's that is used to compute a column of the jacobian matrix.

    Returns
    -------
    function
        A function to compute the jacobian vector product.
    """
    f = linear_util.wrap_init(fun)
    if argnums is not None:
        invals = list(argnums_partial(f, argnums, invals)[1])

    # compute shaped tangents to use later
    sizes = np.array([jnp.size(a) for a in invals])
    inds = np.cumsum(sizes[:-1])
    shaped_tangents = [a.reshape(s.shape) for a, s in zip(np.split(tangent, inds, axis=0), invals)]

    if argnums is None:
        def jvfun(inps):
            return _jvp(f, inps, shaped_tangents)[1]
    else:
        def jvfun(inps):
            f_partial, dyn_args = argnums_partial(f, argnums, inps)
            return _jvp(f_partial, list(dyn_args), shaped_tangents)[1]

    return jvfun


def _get_tangents(vals, direction, coloring=None, argnums=None, trans=None):
    """
    Return a tuple of tangents values for use with vmap.

    Parameters
    ----------
    vals : list
        List of function input values.
    direction : str
        Derivative computation direction ('fwd' or 'rev').
    coloring : Coloring or None
        If not None, the Coloring object used to compute a compressed tangent array.
    argnums : list of int or None
        Indices of dynamic (differentiable) function args.
    trans : ndarray
        Translation array from jacobian indices into function arg indices.  This is needed
        because OpenMDAO expects ordering to be outputs first, then inputs, but function args
        could be in any order.

    Returns
    -------
    tuple of ndarray or ndarray
        The tangents values to be passed to vmap.
    """
    if argnums is None:
        leaves = vals
    else:
        leaves = [vals[i] for i in argnums]
    sizes = [np.size(a) for a in leaves]
    inds = np.cumsum(sizes[:-1])
    if coloring is None:
        tangent = np.eye(np.sum(sizes))
        if trans is not None:
            tangent = tangent[:, trans]
    else:
        tangent = coloring.tangent_matrix(direction, trans=trans)

    shapes = [tangent.shape[:1] + np.shape(v) for v in leaves]
    tangents = tuple([np.reshape(a, shp) for a, shp in zip(np.split(tangent, inds, axis=1),
                                                           shapes)])
    if len(leaves) == 1:
        tangents = tangents[0]

    return tangents
