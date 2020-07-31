"""A module containing various configuration checks for an OpenMDAO Problem."""

from collections import defaultdict
from distutils.version import LooseVersion

import numpy as np

from openmdao.core.group import Group
from openmdao.core.component import Component
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.solvers.linear.direct import DirectSolver
from openmdao.utils.graph_utils import get_sccs_topo
from openmdao.utils.logger_utils import get_logger
from openmdao.utils.class_util import overrides_method
from openmdao.utils.mpi import MPI
from openmdao.utils.hooks import _register_hook
from openmdao.utils.general_utils import printoptions, simple_warning, ignore_errors
from openmdao.utils.units import convert_units, _has_val_mismatch
from openmdao.utils.file_utils import _load_and_exec


_UNSET = object()

# numpy default print options changed in 1.14
if LooseVersion(np.__version__) >= LooseVersion("1.14"):
    _npy_print_opts = {'legacy': '1.13'}
else:
    _npy_print_opts = {}


def _check_cycles(group, infos=None):
    """
    Report any cycles to the logger.

    Parameters
    ----------
    group : <Group>
        The Group being checked for dataflow issues
    infos : list
        List to collect informational messages.

    Returns
    -------
    list
        List of cycles, with subsystem names sorted in execution order.
    """
    graph = group.compute_sys_graph(comps_only=False)
    sccs = get_sccs_topo(graph)
    sub2i = {sub.name: i for i, sub in enumerate(group._subsystems_allprocs)}
    cycles = [sorted(s, key=lambda n: sub2i[n]) for s in sccs if len(s) > 1]

    if cycles and infos is not None:
        infos.append("   Group '%s' has the following cycles: %s\n" % (group.pathname, cycles))

    return cycles


def _check_ubcs(group, warnings):
    """
    Report any 'used before calculated' Systems to the logger.

    Parameters
    ----------
    group : <Group>
        The Group being checked for dataflow issues
    warnings : list
        List to collect warning messages.
    """
    cycles = _check_cycles(group)

    cycle_idxs = {}

    for i, cycle in enumerate(cycles):
        # keep track of cycles so we can detect when a system in
        # one cycle is out of order with a system in a different cycle.
        for s in cycle:
            cycle_idxs[s] = i

    ubcs = _get_used_before_calc_subs(group, group._conn_global_abs_in2out)

    for tgt_system, src_systems in sorted(ubcs.items()):
        keep_srcs = []

        for src_system in src_systems:
            if (src_system not in cycle_idxs or
                    tgt_system not in cycle_idxs or
                    cycle_idxs[tgt_system] != cycle_idxs[src_system]):
                keep_srcs.append(src_system)

        if keep_srcs:
            if group.pathname:
                tgt_system = '.'.join((group.pathname, tgt_system))
                keep_srcs = ['.'.join((group.pathname, n)) for n in keep_srcs]
            warnings.append("   System '%s' executes out-of-order with "
                            "respect to its source systems %s\n" %
                            (tgt_system, sorted(keep_srcs)))


def _check_cycles_prob(prob, logger):
    """
    Report any cycles.

    Parameters
    ----------
    prob : <Problem>
        The Problem being checked for cycles.
    logger : object
        The object that manages logging output.

    """
    infos = ["The following groups contain cycles:\n"]
    for group in prob.model.system_iter(include_self=True, recurse=True, typ=Group):
        _check_cycles(group, infos)

    if len(infos) > 1:
        logger.info(''.join(infos[:1] + sorted(infos[1:])))


def _check_ubcs_prob(prob, logger):
    """
    Report any out of order Systems.

    Parameters
    ----------
    prob : <Problem>
        The Problem being checked for dataflow issues.
    logger : object
        The object that manages logging output.

    """
    warnings = ["The following systems are executed out-of-order:\n"]
    for group in prob.model.system_iter(include_self=True, recurse=True, typ=Group):
        _check_ubcs(group, warnings)

    if len(warnings) > 1:
        logger.warning(''.join(warnings[:1] + sorted(warnings[1:])))


def _get_used_before_calc_subs(group, input_srcs):
    """
    Return Systems that are executed out of dataflow order.

    Parameters
    ----------
    group : <Group>
        The Group where we're checking subsystem order.
    input_srcs : {}
        dict containing variable abs names for sources of the inputs.
        This describes all variable connections, either explicit or implicit,
        in the entire model.

    Returns
    -------
    dict
        A dict mapping names of target Systems to a set of names of their
        source Systems that execute after them.
    """
    sub2i = {}
    parallel_solver = {}
    for i, sub in enumerate(group._subsystems_allprocs):
        if hasattr(sub, '_mpi_proc_allocator') and sub._mpi_proc_allocator.parallel:
            parallel_solver[sub.name] = sub.nonlinear_solver.SOLVER

        sub2i[sub.name] = i

    glen = len(group.pathname.split('.')) if group.pathname else 0

    ubcs = defaultdict(set)
    for tgt_abs, src_abs in input_srcs.items():
        if src_abs is not None:
            iparts = tgt_abs.split('.')
            oparts = src_abs.split('.')
            src_sys = oparts[glen]
            tgt_sys = iparts[glen]
            hierarchy_check = True if oparts[glen + 1] == iparts[glen + 1] else False

            if (src_sys in parallel_solver and tgt_sys in parallel_solver and
                    (parallel_solver[src_sys] not in ["NL: NLBJ", "NL: Newton", "BROYDEN"]) and
                    src_sys == tgt_sys and
                    not hierarchy_check):
                simple_warning("Need to attach NonlinearBlockJac, NewtonSolver, or BroydenSolver "
                               "to '%s' when connecting components inside parallel "
                               "groups" % (src_sys))
                ubcs[tgt_abs.rsplit('.', 1)[0]].add(src_abs.rsplit('.', 1)[0])
            if (src_sys in sub2i and tgt_sys in sub2i and
                    (sub2i[src_sys] > sub2i[tgt_sys])):
                ubcs[tgt_sys].add(src_sys)

    return ubcs


def _check_dup_comp_inputs(problem, logger):
    """
    Issue a logger warning if any components have multiple inputs that share the same source.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.
    logger : object
        The object that manages logging output.
    """
    if isinstance(problem.model, Component):
        return

    input_srcs = problem.model._conn_global_abs_in2out
    src2inps = defaultdict(list)
    for inp, src in input_srcs.items():
        src2inps[src].append(inp)

    msgs = []
    for src, inps in src2inps.items():
        comps = defaultdict(list)
        for inp in inps:
            comp, vname = inp.rsplit('.', 1)
            comps[comp].append(vname)

        dups = sorted([(c, v) for c, v in comps.items() if len(v) > 1], key=lambda x: x[0])
        if dups:
            for comp, vnames in dups:
                msgs.append("   %s has inputs %s connected to %s\n" % (comp, sorted(vnames), src))

    if msgs:
        msg = ["The following components have multiple inputs connected to the same source, ",
               "which can introduce unnecessary data transfer overhead:\n"]
        msg += sorted(msgs)
        logger.warning(''.join(msg))


def _trim_str(obj, size):
    """
    Truncate given string if it's longer than the given size.

    For arrays, use the norm if the size is exceeded.

    Parameters
    ----------
    obj : object
        Object to be stringified and trimmed.
    size : int
        Max allowable size of the returned string.

    Returns
    -------
    str
        The trimmed string.
    """
    if isinstance(obj, np.ndarray):
        with printoptions(**_npy_print_opts):
            s = str(obj)
    else:
        s = str(obj)

    if len(s) > size:
        if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.floating):
            s = 'shape={}, norm={:<.3}'.format(obj.shape, np.linalg.norm(obj))
        else:
            s = s[:size - 4] + ' ...'

    return s


def _list_has_val_mismatch(discretes, names, units, vals):
    """
    Return True if any of the given values don't match, subject to unit conversion.

    Parameters
    ----------
    discretes : set-like
        Set of discrete variable names.
    names : list
        List of variable names.
    units : list
        List of units corresponding to names.
    vals : list
        List of values corresponding to names.

    Returns
    -------
    bool
        True if a mismatch was found, otherwise False.
    """
    if len(names) < 2:
        return False

    uset = set(units)
    if '' in uset and len(uset) > 1:
        # at least one case has no units and at least one does, so there must be a mismatch
        return True

    u0 = v0 = _UNSET
    for n, u, v in zip(names, units, vals):
        if n in discretes:
            continue
        if u0 is _UNSET:
            u0 = u
            v0 = v
        elif _has_val_mismatch(u0, v0, u, v):
            return True

    return False


def _check_hanging_inputs(problem, logger):
    """
    Issue a logger warning if any inputs are not connected.

    Promoted inputs are shown alongside their corresponding absolute names.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.
    logger : object
        The object that manages logging output.
    """
    if isinstance(problem.model, Component):
        input_srcs = {}
    else:
        input_srcs = problem.model._conn_global_abs_in2out

    prom_ins = problem.model._var_allprocs_prom2abs_list['input']
    abs2meta = problem.model._var_allprocs_abs2meta
    unconns = []
    nwid = uwid = 0

    for prom, abslist in prom_ins.items():
        unconn = [a for a in abslist if a not in input_srcs or len(input_srcs[a]) == 0]
        if unconn:
            w = max([len(u) for u in unconn])
            if w > nwid:
                nwid = w
            units = [abs2meta[a]['units'] if a in abs2meta else '' for a in unconn]
            units = [u if u is not None else '' for u in units]
            lens = [len(u) for u in units]
            if lens:
                u = max(lens)
                if u > uwid:
                    uwid = u
            unconns.append((prom, unconn, units))

    if unconns:
        template_abs = "   {:<{nwid}} {:<{uwid}} {}\n"
        template_prom = "      {:<{nwid}} {:<{uwid}} {}\n"
        msg = ["The following inputs are not connected:\n"]
        for prom, absnames, units in sorted(unconns, key=lambda x: x[0]):
            if len(absnames) == 1 and prom == absnames[0]:  # not really promoted
                a = absnames[0]
                valstr = _trim_str(problem.get_val(a, get_remote=True), 25)
                msg.append(template_abs.format(a, units[0], valstr, nwid=nwid + 3, uwid=uwid))
            else:  # promoted
                vals = [problem.get_val(a, get_remote=True) for a in absnames]
                mismatch = _list_has_val_mismatch(problem.model._var_allprocs_discrete['input'],
                                                  absnames, units, vals)
                if mismatch:
                    msg.append("\n   ----- WARNING: connected input values don't match when "
                               "converted to consistent units. -----\n")
                msg.append("   {}  (p):\n".format(prom))
                for a, u, v in zip(absnames, units, vals):
                    valstr = _trim_str(problem.get_val(a, get_remote=True), 25)
                    msg.append(template_prom.format(a, u, valstr, nwid=nwid, uwid=uwid))
                if mismatch:
                    msg.append("   --------------------------------------------------------------"
                               "-----------------------------\n\n")

        logger.warning(''.join(msg))


def _check_comp_has_no_outputs(problem, logger):
    """
    Issue a logger warning if any components do not have any outputs.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.
    logger : object
        The object that manages logging output.
    """
    msg = []

    for comp in problem.model.system_iter(include_self=True, recurse=True, typ=Component):
        if len(comp._var_allprocs_abs_names['output']) == 0:
            msg.append("   %s\n" % comp.pathname)

    if msg:
        logger.warning(''.join(["The following Components do not have any outputs:\n"] + msg))


def _check_system_configs(problem, logger):
    """
    Perform any system specific configuration checks.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.
    logger : object
        The object that manages logging output.
    """
    for system in problem.model.system_iter(include_self=True, recurse=True):
        system.check_config(logger)


def _check_solvers(problem, logger):
    """
    Search over all solvers and raise an error for unsupported configurations.

    Report any implicit component that does not implement solve_nonlinear and
    solve_linear or have an iterative nonlinear and linear solver upstream of it.

    Report any cycles that do not have an iterative nonlinear solver and either
    an iterative linear solver or a DirectSolver upstream of it.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.
    logger : object
        The object that manages logging output.
    """
    iter_nl_depth = iter_ln_depth = np.inf

    for sys in problem.model.system_iter(include_self=True, recurse=True):
        path = sys.pathname
        depth = 0 if path == '' else len(path.split('.'))

        # if this system is below both a nonlinear and linear solver, then skip checks
        if (depth > iter_nl_depth) and (depth > iter_ln_depth):
            continue

        # determine if this system is a group and has cycles
        if isinstance(sys, Group):
            graph = sys.compute_sys_graph(comps_only=False)
            sccs = get_sccs_topo(graph)
            sub2i = {sub.name: i for i, sub in enumerate(sys._subsystems_allprocs)}
            has_cycles = [sorted(s, key=lambda n: sub2i[n]) for s in sccs if len(s) > 1]
        else:
            has_cycles = []

        # determine if this system has states (is an implicit component)
        has_states = isinstance(sys, ImplicitComponent)

        # determine if this system has iterative solvers or implements the solve methods
        # for handling cycles and implicit components
        if depth > iter_nl_depth:
            is_iter_nl = True
        else:
            is_iter_nl = (
                (sys.nonlinear_solver and 'maxiter' in sys.nonlinear_solver.options) or
                (has_states and overrides_method('solve_nonlinear', sys, ImplicitComponent))
            )
            iter_nl_depth = depth if is_iter_nl else np.inf

        if depth > iter_ln_depth:
            is_iter_ln = True
        else:
            is_iter_ln = (
                (sys.linear_solver and
                 ('maxiter' in sys.linear_solver.options or
                  isinstance(sys.linear_solver, DirectSolver))) or
                (has_states and overrides_method('solve_linear', sys, ImplicitComponent))
            )
            iter_ln_depth = depth if is_iter_ln else np.inf

        # if there are cycles, then check for iterative nonlinear and linear solvers
        if has_cycles:
            if not is_iter_nl:
                msg = ("Group '%s' contains cycles %s, but does not have an iterative "
                       "nonlinear solver." % (path, has_cycles))
                logger.warning(msg)
            if not is_iter_ln:
                msg = ("Group '%s' contains cycles %s, but does not have an iterative "
                       "linear solver." % (path, has_cycles))
                logger.warning(msg)

        # if there are implicit components, check for iterative solvers or the appropriate
        # solve methods
        if has_states:
            if not is_iter_nl:
                msg = ("%s '%s' contains implicit variables, but does not have an "
                       "iterative nonlinear solver and does not implement 'solve_nonlinear'." %
                       (sys.__class__.__name__, path))
                logger.warning(msg)
            if not is_iter_ln:
                msg = ("%s '%s' contains implicit variables, but does not have an "
                       "iterative linear solver and does not implement 'solve_linear'." %
                       (sys.__class__.__name__, path))
                logger.warning(msg)


def _check_missing_recorders(problem, logger):
    """
    Check to see if there are any recorders of any type on the Problem.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.
    logger : object
        The object that manages logging output.
    """
    # Look for Driver recorder
    if problem.driver._rec_mgr._recorders:
        return

    # Look for System and Solver recorders
    for system in problem.model.system_iter(include_self=True, recurse=True):
        if system._rec_mgr._recorders:
            return
        if system.nonlinear_solver and system.nonlinear_solver._rec_mgr._recorders:
            return
        if system.linear_solver and system.linear_solver._rec_mgr._recorders:
            return

    msg = "The Problem has no recorder of any kind attached"
    logger.warning(msg)


def _get_promoted_connected_ins(g):
    """
    Find all inputs that are promoted above the level where they are explicitly connected.

    Parameters
    ----------
    g : Group
        Starting Group.

    Returns
    -------
    defaultdict
        Absolute input name keyed to [promoting_groups, manually_connecting_groups]
    """
    prom2abs_list = g._var_allprocs_prom2abs_list['input']
    abs2prom_in = g._var_abs2prom['input']
    prom_conn_ins = defaultdict(lambda: ([], []))
    for prom_in in g._manual_connections:
        for abs_in in prom2abs_list[prom_in]:
            prom_conn_ins[abs_in][1].append((prom_in, g.pathname))

    for subsys in g._subgroups_myproc:
        sub_prom_conn_ins = _get_promoted_connected_ins(subsys)
        for n, tup in sub_prom_conn_ins.items():
            proms, mans = tup
            mytup = prom_conn_ins[n]
            mytup[0].extend(proms)
            mytup[1].extend(mans)

        sub_abs2prom_in = subsys._var_abs2prom['input']

        for inp, sub_prom_inp in sub_abs2prom_in.items():
            if abs2prom_in[inp] == sub_prom_inp:  # inp is promoted up from sub
                if inp in sub_prom_conn_ins and len(sub_prom_conn_ins[inp][1]) > 0:
                    prom_conn_ins[inp][0].append(subsys.pathname)

    return prom_conn_ins


def _check_explicitly_connected_promoted_inputs(problem, logger):
    """
    Check for any inputs that are explicitly connected AND promoted above their connection group.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.
    logger : object
        The object that manages logging output.
    """
    prom_conn_ins = _get_promoted_connected_ins(problem.model)

    for inp, lst in prom_conn_ins.items():
        proms, mans = lst
        if proms:
            # there can only be one manual connection (else an exception would've been raised)
            man_prom, man_group = mans[0]
            if len(proms) > 1:
                lst = [p for p in proms if p == man_group or man_group.startswith(p + '.')]
                s = "groups %s" % sorted(lst)
            else:
                s = "group '%s'" % proms[0]
            logger.warning("Input '%s' was explicitly connected in group '%s' as '%s', but was "
                           "promoted up from %s." % (inp, man_group, man_prom, s))


# Dict of all checks by name, mapped to the corresponding function that performs the check
# Each function must be of the form  f(problem, logger).
_default_checks = {
    'out_of_order': _check_ubcs_prob,
    'system': _check_system_configs,
    'solvers': _check_solvers,
    'dup_inputs': _check_dup_comp_inputs,
    'missing_recorders': _check_missing_recorders,
    'comp_has_no_outputs': _check_comp_has_no_outputs,
}

_all_checks = _default_checks.copy()
_all_checks.update({
    'cycles': _check_cycles_prob,
    'unconnected_inputs': _check_hanging_inputs,
    'promotions': _check_explicitly_connected_promoted_inputs,
})


#
# Command line interface functions
#

def _check_config_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao check' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model')
    parser.add_argument('-o', action='store', dest='outfile', help='output file')
    parser.add_argument('-p', '--problem', action='store', dest='problem', help='Problem name')
    parser.add_argument('-c', action='append', dest='checks', default=[],
                        help='Only perform specific check(s). Default checks are: %s. '
                        'Other available checks are: %s' %
                        (sorted(_default_checks), sorted(set(_all_checks) - set(_default_checks))))


def _check_config_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao check'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.

    Returns
    -------
    function
        The post-setup hook function.
    """
    def _check_config(prob):
        if not MPI or MPI.COMM_WORLD.rank == 0:
            if options.outfile is None:
                logger = get_logger('check_config', out_stream='stdout',
                                    out_file=None, use_format=True)
            else:
                logger = get_logger('check_config', out_file=options.outfile, use_format=True)

            if not options.checks:
                options.checks = sorted(_default_checks)
            elif 'all' in options.checks:
                options.checks = sorted(_all_checks)

            prob.check_config(logger, options.checks)

        exit()

    # register the hook
    _register_hook('final_setup', class_name='Problem', inst_id=options.problem, post=_check_config)

    ignore_errors(True)
    _load_and_exec(options.file[0], user_args)


def check_allocate_complex_ln(model, under_cs):
    """
    Return True if linear vector should be complex.

    This happens when a solver needs derivatives under complex step.

    Parameters
    ----------
    model : <Group>
        Model to be checked, usually the root model.
    under_cs : bool
        Flag indicates if complex vectors were allocated in a containing Group or were force
        allocated in setup.

    Returns
    -------
    bool
        True if linear vector should be complex.
    """
    under_cs |= 'cs' in model._approx_schemes

    if under_cs and model.nonlinear_solver is not None and \
       model.nonlinear_solver.supports['gradients']:
        return True

    for sub in model._subsystems_allprocs:
        chk = check_allocate_complex_ln(sub, under_cs)

        if chk:
            return True

    return False
