"""
Define functions and objects common to the ExplicitFuncComp and ImplicitFuncComp classes.
"""

try:
    import jax
    from jax import jvp, vjp, vmap, random, jit
    import jax.numpy as jnp
except ImportError:
    jax = None

import re
from openmdao.utils.units import valid_units
import openmdao.func_api as omf
from openmdao.utils.om_warnings import issue_warning


# regex to check for variable names.
namecheck_rgx = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')

# Names that are not allowed for input or output variables (keywords for options)
_disallowed_varnames = {'units', 'shape', 'shape_by_conn', 'run_root_only'}

_meta_keep = {'units', 'shape', 'val'}
_from_def = {'default_units': 'units', 'default_shape': 'shape'}


def _check_units_option(option, value):
    if value is not None and not valid_units(value):
        raise ValueError(f"The units '{value}' are invalid.")


def _copy_with_ignore(dct, keepers, ignore=()):
    """
    Copy the entries in the given dict whose keys are in keepers.

    Parameters
    ----------
    dct : dict
        The dictionary to be copied.
    keepers : set-like
        Set of keys for entries we want to keep.
    ignore : set or tuple
        Ignore these keys.

    Returns
    -------
    dict
        A new dict containing 'keepers' entries.
    """
    kept = {}
    warn = set()
    for k, v in dct.items():
        if k in keepers:
            kept[k] = v
        elif k not in ignore:
            warn.add(k)

    if warn:
        issue_warning(f"The following metadata entries were ignored: {sorted(warn)}.")

    return kept


def setup_func_comp_io(comp):
    """
    Define out inputs and outputs.

    Parameters
    ----------
    comp : ExplicitFuncComp or ImplicitFuncComp
        The component we're adding inputs and outputs to.
    """
    optignore = {'is_option'}

    for name, meta in comp._func.get_input_meta():
        _check_var_name(comp, name)
        if 'is_option' in meta and meta['is_option']:
            kwargs = _copy_with_ignore(meta, omf._allowed_declare_options_args,
                                        ignore=optignore)
            comp.options.declare(name, **kwargs)
        else:
            kwargs = _copy_with_ignore(meta, omf._allowed_add_input_args)
            comp.add_input(name, **kwargs)

    for i, (name, meta) in enumerate(comp._func.get_output_meta()):
        if name is None:
            raise RuntimeError(f"{comp.msginfo}: Can't add output corresponding to return "
                                f"value in position {i} because it has no name.  Specify the "
                                "name by returning a variable, for example 'return myvar', or "
                                "include the name in the function's metadata.")
        _check_var_name(comp, name)
        kwargs = _copy_with_ignore(meta, omf._allowed_add_output_args)
        comp.add_output(name, **kwargs)


def _check_var_name(comp, name):
    match = namecheck_rgx.match(name)
    if match is None or match.group() != name:
        raise NameError(f"{comp.msginfo}: '{name}' is not a valid variable name.")

    if name in _disallowed_varnames:
        raise NameError(f"{comp.msginfo}: cannot use variable name '{name}' because "
                        "it's a reserved keyword.")


def fill_vector(vec, vals):
    """
    Fill the given vector using variable values.

    Parameters
    ----------
    vec : Vector
        The Vector to be filled.
    vals : ndarray or float or tuple of ndarrays and floats
        Values to be inserted into the Vector.
    """
    arr = vec.asarray()

    if isinstance(vals, tuple):
        start = end = 0
        for o in vals:
            a = asarray(o) if isscalar(o) else o
            end += a.size
            arr[start:end] = a.flat
            start = end
    else:
        if isscalar(vals):
            arr[:] = vals
        else:
            arr[:] = vals.flat
