"""A module containing various configuration checks for an OpenMDAO Problem."""
from __future__ import print_function

from collections import defaultdict
from six import iteritems

import numpy as np

from openmdao.core.group import Group
from openmdao.core.component import Component
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.solvers.linear.direct import DirectSolver
from openmdao.utils.graph_utils import get_sccs_topo
from openmdao.utils.logger_utils import get_logger
from openmdao.utils.class_util import overrides_method
from openmdao.utils.mpi import MPI


def _check_dataflow(group, infos, warnings):
    """
    Report any cycles and out of order Systems to the logger.

    Parameters
    ----------
    group : <Group>
        The Group being checked for dataflow issues
    infos : list
        List to collect informational messages.
    warnings : list
        List to collect warning messages.
    """
    graph = group.compute_sys_graph(comps_only=False)
    sccs = get_sccs_topo(graph)
    sub2i = {sub.name: i for i, sub in enumerate(group._subsystems_allprocs)}
    cycles = [sorted(s, key=lambda n: sub2i[n]) for s in sccs if len(s) > 1]
    cycle_idxs = {}

    if cycles:
        infos.append("   Group '%s' has the following cycles: %s\n" % (group.pathname, cycles))
        for i, cycle in enumerate(cycles):
            # keep track of cycles so we can detect when a system in
            # one cycle is out of order with a system in a different cycle.
            for s in cycle:
                if group.pathname:
                    s = '.'.join((group.pathname, s))
                cycle_idxs[s] = i

    ubcs = _get_out_of_order_subs(group, group._conn_global_abs_in2out)

    for tgt_system, src_systems in sorted(ubcs.items()):
        keep_srcs = []

        for src_system in src_systems:
            if not (src_system in cycle_idxs and
                    tgt_system in cycle_idxs and
                    cycle_idxs[tgt_system] == cycle_idxs[src_system]):
                keep_srcs.append(src_system)

        if keep_srcs:
            warnings.append("   System '%s' executes out-of-order with "
                            "respect to its source systems %s\n" %
                            (tgt_system, sorted(keep_srcs)))


def _check_dataflow_prob(prob, logger):
    """
    Report any cycles and out of order Systems.

    Parameters
    ----------
    prob : <Problem>
        The Problem being checked for dataflow issues.
    logger : object
        The object that manages logging output.

    """
    infos = ["The following groups contain cycles:\n"]
    warnings = ["The following systems are executed out-of-order:\n"]
    for group in prob.model.system_iter(include_self=True, recurse=True, typ=Group):
        _check_dataflow(group, infos, warnings)

    if len(infos) > 1:
        logger.info(''.join(infos[:1] + sorted(infos[1:])))

    if len(warnings) > 1:
        logger.warning(''.join(warnings[:1] + sorted(warnings[1:])))


def _get_out_of_order_subs(group, input_srcs):
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
    subsystems = group._subsystems_allprocs
    sub2i = {sub.name: i for i, sub in enumerate(subsystems)}
    glen = len(group.pathname.split('.')) if group.pathname else 0

    ubcs = defaultdict(set)
    for in_abs, src_abs in iteritems(input_srcs):
        if src_abs is not None:
            iparts = in_abs.split('.')
            oparts = src_abs.split('.')
            src_sys = oparts[glen]
            tgt_sys = iparts[glen]
            if (src_sys in sub2i and tgt_sys in sub2i and
                    (sub2i[src_sys] > sub2i[tgt_sys])):
                ubcs['.'.join(iparts[:glen + 1])].add('.'.join(oparts[:glen + 1]))

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
    for inp, src in iteritems(input_srcs):
        src2inps[src].append(inp)

    msgs = []
    for src, inps in iteritems(src2inps):
        comps = defaultdict(list)
        for inp in inps:
            comp, vname = inp.rsplit('.', 1)
            comps[comp].append(vname)

        dups = sorted([(c, v) for c, v in iteritems(comps) if len(v) > 1], key=lambda x: x[0])
        if dups:
            for comp, vnames in dups:
                msgs.append("   %s has inputs %s connected to %s\n" % (comp, sorted(vnames), src))

    if msgs:
        msg = ["The following components have multiple inputs connected to the same source, ",
               "which can introduce unnecessary data transfer overhead:\n"]
        msg += sorted(msgs)
        logger.warning(''.join(msg))


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
    unconns = []
    for prom, abslist in iteritems(prom_ins):
        unconn = [a for a in abslist if a not in input_srcs or len(input_srcs[a]) == 0]
        if unconn:
            unconns.append(prom)

    if unconns:
        msg = ["The following inputs are not connected:\n"]
        for prom in sorted(unconns):
            absnames = prom_ins[prom]
            if len(absnames) == 1 and prom == absnames[0]:  # not really promoted
                msg.append("   %s\n" % prom)
            else:  # promoted
                msg.append("   %s: %s\n" % (prom, prom_ins[prom]))
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


def _explicit_conns_iter(group):
    """
    Iterator over explicit connections owned by the given group.

    Yields
    ------
    (str, str)
        Connection tuple (src, target).
    """
    abs2prom_in = group._var_abs2prom['input']
    abs2prom_out = group._var_abs2prom['output']
    for tup in iteritems(group._conn_abs_in2out):
        inp, out = tup
        # if promoted names differ, must be an explicit connection
        if abs2prom_in[inp] != abs2prom_out[out]:
            yield tup


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
    # get set of all explicitly conected inputs below the top level group
    exp_conn_ins = set()
    for g in problem.model.system_iter(recurse=True, typ=Group):
        exp_conn_ins.update(t[0] for t in _explicit_conns_iter(g))

    inps = []
    for g in problem.model.system_iter(include_self=True, recurse=True, typ=Group):
        abs2prom_in = g._var_abs2prom['input']
        for subsys in g._subgroups_myproc:
            sub_abs2prom_in = subsys._var_abs2prom['input']
            sub_abs2prom_out = subsys._var_abs2prom['output']
            for inp, sub_prom_inp in iteritems(sub_abs2prom_in):
                if abs2prom_in[inp] == sub_prom_inp:
                    # input has been promoted from subsys, so check
                    # if it's part of an explicit connection at this level or below
                    if inp in subsys._conn_global_abs_in2out and inp in exp_conn_ins:
                        inps.append((inp, subsys.pathname))

    return inps


# Dict of all checks by name, mapped to the corresponding function that performs the check
# Each function must be of the form  f(problem, logger).
_checks = {
    'hanging_inputs': _check_hanging_inputs,
    'cycles': _check_dataflow_prob,
    'system': _check_system_configs,
    'solvers': _check_solvers,
    'dup_inputs': _check_dup_comp_inputs,
    'missing_recorders': _check_missing_recorders,
    'comp_has_no_outputs': _check_comp_has_no_outputs,
    'promoted_connected': _check_explicitly_connected_promoted_inputs,
}


def check_config(problem, logger=None):
    """
    Perform optional error checks on a Problem.

    Parameters
    ----------
    problem : Problem
        The Problem being checked.
    logger : object
        Logging object.
    """
    logger = logger if logger else get_logger('check_config', use_format=True)

    for c in sorted(_checks.keys()):
        _checks[c](problem, logger)


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
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', action='store', dest='outfile', help='output file.')
    parser.add_argument('-c', action='append', dest='checks', default=[],
                        help='Only perform specific check(s). Available checks are: %s. '
                        'By default, will perform all checks.' % sorted(_checks.keys()))


def _check_config_cmd(options):
    """
    Return the post_setup hook function for 'openmdao check'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    def _check_config(prob):
        if not MPI or MPI.COMM_WORLD.rank == 0:
            if options.outfile is None:
                logger = get_logger('check_config', use_format=True)
            else:
                outfile = open(options.outfile, 'w')
                logger = get_logger('check_config', out_stream=outfile, use_format=True)

            if not options.checks:
                options.checks = sorted(_checks.keys())

            for c in options.checks:
                _checks[c](prob, logger)

        exit()

    return _check_config


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
