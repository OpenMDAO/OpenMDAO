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
import numpy as np
from openmdao.utils.units import valid_units
from openmdao.func_api import _shape2tuple
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
        if k in keepers and k not in ignore:
            kept[k] = v
        elif k not in ignore:
            warn.add(k)

    if warn:
        issue_warning(f"The following metadata entries were ignored: {sorted(warn)}.")

    return kept


def _check_var_name(comp, name):
    match = namecheck_rgx.match(name)
    if match is None or match.group() != name:
        raise NameError(f"{comp.msginfo}: '{name}' is not a valid variable name.")

    if name in _disallowed_varnames:
        raise NameError(f"{comp.msginfo}: cannot use variable name '{name}' because "
                        "it's a reserved keyword.")
