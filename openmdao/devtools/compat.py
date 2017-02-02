"""Some conversion/compatibility functions for OpenMDAO v1 to OpenMDAO v2."""
import sys

from openmdao.core.component import Component

def convert_file():
    """A crude converter for OpenMDAO v1 files to OpenMDAO v2."""
    cvt_map = {
        '.add(': '.add_subsystem(',
        '.add_param(': '.add_input(',
        '.params': '._inputs',
        '.unknowns': '._outputs',
        '.resids': '._residuals',
        '.add_option(': '.declare(',
        ' Newton(': ' NewtonSolver(',
        'openmdao.test.util': 'openmdao.devtools.testutil',
        'def solve_nonlinear(self, params, unknowns, resids)':
        'def compute(self, params, unknowns)',
    }

    with open(sys.argv[1], 'r') as f:
        contents = f.read()

        if 'add_state' in contents:
            pass
        else:
            cvt_map['Component'] = 'ExplicitComponent'

        for old, new in cvt_map.items():
            contents = contents.replace(old, new)

    sys.stdout.write(contents)

def abs_varname_iter(system, typ, local=True):
    """An iter of variable absolute pathnames for the given system.

    Args
    ----
    system : System
        The System where the iteration starts.

    typ : str
        Specifies either 'input' or 'output' vars.

    local : bool(True)
        If True, iterate over only System on the current process.

    """
    for s in system.system_iter(local=local, include_self=True, recurse=True):
        # the only place where we can calculate the absolute variable
        # path is at the Component level since the rest of the framework
        # deals only with promoted names.
        if isinstance(s, Component):
            for varname in s._var_allprocs_names[typ]:
                yield '.'.join((s.pathname, varname))

def abs_meta_iter(system, typ):
    """An iter of (abs_var_name, metadata) for all local vars.

    Args
    ----
    system : System
        The System where the iteration starts.

    typ : str
        Specifies either 'input' or 'output' vars.

    """
    meta = system._var_myproc_metadata[typ]
    for i, vname in enumerate(abs_varname_iter(system, typ)):
        yield vname, meta[i]

def abs_conn_iter(system):
    """An iter of (abs_tgt_name, abs_src_name) for all connections."""
    tgt_names = list(abs_varname_iter(system, 'input', local=False))
    src_names = list(abs_varname_iter(system, 'output', local=False))
    for tgt_idx, src_idx in system._var_connections_indices:
        yield tgt_names[tgt_idx], src_names[src_idx]

def get_abs_proms(system, typ, local=True):
    """An iter of (absname, promname) for all vars.

    Args
    ----
    system : System
        The System where the iteration starts.

    typ : str
        Specifies either 'input' or 'output' vars.

    local : bool(True)
        If True, iterate over only vars from this process.

    """
    if local:
        prom_names = system._var_myproc_names[typ]
    else:
        prom_names = system._var_allprocs_names[typ]

    for i, absname in enumerate(abs_varname_iter(system, typ, local=local)):
        yield absname, prom_names[i]

def prom2abs_map(system, typ, local=True):
    """Return a dict mapping promname to absname(s) for all vars.

    Args
    ----
    system : System
        The System where the iteration starts.

    typ : str
        Specifies either 'input' or 'output' vars.

    local : bool(True)
        If True, iterate over only vars from this process.

    """
    ret = {}
    for absname, promname in get_abs_proms(system, typ, local):
        if promname in ret:
            ret[promname].append(absname)
        else:
            ret[promname] = [absname]
    return ret

def abs2prom_map(system, typ, local=True):
    """Return a dict mapping absname to promname for all vars.

    Args
    ----
    system : System
        The System where the iteration starts.

    typ : str
        Specifies either 'input' or 'output' vars.

    local : bool(True)
        If True, iterate over only vars from this process.

    """
    return dict(get_abs_proms(system, typ, local))



if __name__ == '__main__':
    convert_file()
