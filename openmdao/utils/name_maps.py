"""Maps between promoted/relative/absolute names and name pairs."""


def rel_name2abs_name(system, rel_name):
    """
    Map relative variable name to absolute variable name.

    Parameters
    ----------
    system : <System>
        System to which the given name is relative.
    rel_name : str
        Given relative variable name.

    Returns
    -------
    str
        Absolute variable name.
    """
    return system.pathname + '.' + rel_name if system.pathname else rel_name


def abs_name2rel_name(system, abs_name):
    """
    Map relative variable name to absolute variable name.

    Parameters
    ----------
    system : <System>
        System to which the given name is relative.
    abs_name : str
        Given absolute variable name.

    Returns
    -------
    str
        Relative variable name.
    """
    return abs_name[len(system.pathname) + 1:] if system.pathname else abs_name


def rel_key2abs_key(system, rel_key):
    """
    Map relative variable name pair to absolute variable name pair.

    Parameters
    ----------
    system : <System>
        System to which the given key is relative.
    rel_key : (str, str)
        Given relative variable name pair.

    Returns
    -------
    (str, str)
        Absolute variable name pair.
    """
    if system.pathname:
        of, wrt = rel_key
        return (system.pathname + '.' + of, system.pathname + '.' + wrt)
    return rel_key


def abs_key2rel_key(system, abs_key):
    """
    Map relative variable name pair to absolute variable name pair.

    Parameters
    ----------
    system : <System>
        System to which the given key is relative.
    abs_key : (str, str)
        Given absolute variable name pair.

    Returns
    -------
    (str, str)
        Relative variable name pair.
    """
    if system.pathname:
        of, wrt = abs_key
        plen = len(system.pathname) + 1
        return (of[plen:], wrt[plen:])
    return abs_key


def prom_name2abs_name(system, prom_name, iotype):
    """
    Map the given promoted name to the absolute name.

    This is only valid when the name is unique; otherwise, a KeyError is thrown.

    Parameters
    ----------
    system : <System>
        System to which prom_name is relative.
    prom_name : str
        Promoted variable name in the owning system's namespace.
    iotype : str
        Either 'input' or 'output'.

    Returns
    -------
    str or None
        Absolute variable name or None if prom_name is invalid.
    """
    prom2abs_lists = system._var_allprocs_prom2abs_list[iotype]

    if prom_name in prom2abs_lists:
        abs_list = prom2abs_lists[prom_name]
        if len(abs_list) == 1:
            return abs_list[0]

        # looks like an aliased input, which must be set via the connected output
        model = system._problem_meta['model_ref']()
        src_name = model._var_abs2prom['output'][model._conn_global_abs_in2out[abs_list[0]]]
        raise RuntimeError(f"The promoted name {prom_name} is invalid because it refers to "
                           f"multiple inputs: [{' ,'.join(abs_list)}]. Access the value from the "
                           f"connected output variable {src_name} instead.")


def name2abs_name(system, name):
    """
    Map the given promoted or relative name to the absolute name.

    This is only valid when the name is unique; otherwise, a KeyError is thrown.

    Parameters
    ----------
    system : <System>
        System to which name is relative.
    name : str
        Promoted or relative variable name in the owning system's namespace.

    Returns
    -------
    str or None
        Absolute variable name if unique abs_name found or None otherwise.
    str or None
        The type ('input' or 'output') of the corresponding variable.
    """
    if name in system._var_allprocs_abs2prom['output']:
        return name
    if name in system._var_allprocs_abs2prom['input']:
        return name

    if name in system._var_allprocs_prom2abs_list['output']:
        return system._var_allprocs_prom2abs_list['output'][name][0]

    # This may raise an exception if name is not unique
    abs_name = prom_name2abs_name(system, name, 'input')
    if abs_name is not None:
        return abs_name

    abs_name = rel_name2abs_name(system, name)
    if abs_name in system._var_allprocs_abs2prom['output']:
        return abs_name
    elif abs_name in system._var_allprocs_abs2prom['input']:
        return abs_name


def name2abs_names(system, name):
    """
    Map the given promoted, relative, or absolute name to any matching absolute names.

    This will also match any buried promotes.

    Parameters
    ----------
    system : <System>
        System to which name is relative.
    name : str
        Promoted or relative variable name in the owning system's namespace.

    Returns
    -------
    tuple or list of str
        Tuple or list of absolute variable names found.
    """
    # first check relative promoted names
    if name in system._var_allprocs_prom2abs_list['output']:
        return system._var_allprocs_prom2abs_list['output'][name]

    if name in system._var_allprocs_prom2abs_list['input']:
        return system._var_allprocs_prom2abs_list['input'][name]

    # then check absolute names
    if name in system._var_allprocs_abs2prom['output']:
        return (name,)
    if name in system._var_allprocs_abs2prom['input']:
        return (name,)

    # then check global promoted names, including buried promotes
    if name in system._problem_meta['prom2abs']['output']:
        absnames = system._problem_meta['prom2abs']['output'][name]
        # reduce scope to this system
        if absnames[0] in system._var_allprocs_abs2prom['output']:
            return absnames

    if name in system._problem_meta['prom2abs']['input']:
        absnames = system._problem_meta['prom2abs']['input'][name]
        # reduce scope to this system
        if absnames[0] in system._var_allprocs_abs2prom['input']:
            return absnames

    return ()


def prom_key2abs_key(system, prom_key):
    """
    Map the given promoted name pair to the absolute name pair.

    The first name is a continuous output, and the second name can be an output or an input.
    If the second name is non-unique, a KeyError is thrown.

    Parameters
    ----------
    system : <System>
        System to which prom_key is relative.
    prom_key : (str, str)
        Promoted name pair of sub-Jacobian.

    Returns
    -------
    (str, str) or None
        Absolute name pair of sub-Jacobian or None is prom_key is invalid.
    """
    of, wrt = prom_key
    abs_wrt = prom_name2abs_name(system, wrt, 'input')
    if abs_wrt is None:
        abs_wrt = prom_name2abs_name(system, wrt, 'output')
        if abs_wrt is None:
            return None

    abs_of = prom_name2abs_name(system, of, 'output')
    if abs_of is not None:
        return (abs_of, abs_wrt)


def key2abs_key(system, key):
    """
    Map the given absolute, promoted or relative name pair to the absolute name pair.

    The first name is an output, and the second name can be an output or an input.
    If the second name is non-unique, a KeyError is thrown.

    Parameters
    ----------
    system : <System>
        System to which prom_key is relative.
    key : (str, str)
        Promoted or relative name pair of sub-Jacobian.

    Returns
    -------
    (str, str) or None
        Absolute name pair of sub-Jacobian if unique abs_key found or None otherwise.
    """
    abs_key = rel_key2abs_key(system, key)
    of, wrt = abs_key
    if of in system._var_allprocs_abs2idx and wrt in system._var_allprocs_abs2idx:
        return abs_key

    abs_key = prom_key2abs_key(system, key)
    if abs_key is not None:
        return abs_key

    if key in system._subjacs_info:
        return key


def abs_key_iter(system, rel_ofs, rel_wrts):
    """
    Return absolute jacobian keys given relative 'of' and 'wrt' names.

    Parameters
    ----------
    system : System
        The scoping system.
    rel_ofs : iter of str
        Names of the relative 'of' variables.
    rel_wrts : iter of str
        Names of the relative 'wrt' variables.

    Yields
    ------
    abs_of
        Absolute 'of' name.
    abs_wrt
        Absolute 'wrt' name.
    """
    pname = system.pathname + '.' if system.pathname else ''
    if pname:
        abs_wrts = [pname + r for r in rel_wrts]
        for rel_of in rel_ofs:
            abs_of = pname + rel_of
            for abs_wrt in abs_wrts:
                yield abs_of, abs_wrt
    else:
        for abs_of in rel_ofs:
            for abs_wrt in rel_wrts:
                yield abs_of, abs_wrt
