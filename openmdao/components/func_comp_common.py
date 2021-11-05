"""
Define functions and objects common to the ExplicitFuncComp and ImplicitFuncComp classes.
"""

import re
from openmdao.utils.om_warnings import issue_warning


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
