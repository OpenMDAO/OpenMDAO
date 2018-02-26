"""A module containing various configuration checks for an OpenMDAO Problem."""
from __future__ import print_function

import sys

from collections import defaultdict
from six import iteritems

import networkx as nx
import numpy as np

from openmdao.core.group import Group
from openmdao.core.component import Component
from openmdao.utils.graph_utils import get_sccs_topo
from openmdao.utils.logger_utils import get_logger


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
        The object that managers logging output.
    """
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
        The object that managers logging output.
    """
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


def _check_system_configs(problem, logger):
    """
    Perform any system specific configuration checks.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.
    logger : object
        The object that managers logging output.
    """
    for system in problem.model.system_iter(include_self=True, recurse=True):
        system.check_config(logger)


def _check_solvers(problem):
    """
    Search over all solvers and raise an error for unsupported configurations.

    Report any implicit component that does not have an appropriate nonlinear
    and linear solver (i.e. not the default solvers) upstream of it. Note that
    a linear solver is only required when doing a gradient-based optimization
    with analytic derivatives, so we need to determine if the derivatives are
    being approximated.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.
    logger : object
        The object that managers logging output.
    """
    # TODO: This will be an warning in check_setup (not a hard error)

    # if you don't have any cycles or implicit comps, but driver requests derivs,
    # then you are fine with LRO everywhere

    # TODO: if you have a (nonlinear) Newton solver in a group, you either need a
    #       linear_solver in that group, or slotted in the Newton solver.

    # nonlinear: cycle
    # linear:    (cycle or implicit) and derivs

    # all states that have some maxiter>1 linear solver above them in the tree
    # iterated_states = set()
    # group_states = []

    uses_deriv = {}
    has_cycles = {}
    has_states = {}

    # put entry for '' into has_iter_solver just in case we're a subproblem
    has_iter_ln = {'': False}
    has_iter_nl = {'': False}

    for group in self.model.system_iter(include_self=True, recurse=True, typ=Group):
        path = group.pathname

        # determine if this group requires derivatives
        derivs_needed = uses_deriv[path] = ('fd' not in group._approx_schemes and
                                            'cs' not in group._approx_schemes)

        # determine if this group has cycles
        graph = group.compute_sys_graph(comps_only=False)
        sccs = get_sccs_topo(graph)
        sub2i = {sub.name: i for i, sub in enumerate(group._subsystems_allprocs)}
        has_cycles[path] = [sorted(s, key=lambda n: sub2i[n]) for s in sccs if len(s) > 1]

        # determine if this group has states (implicit components)
        has_states[path] = [
            comp.pathname for comp in group.system_iter(recurse=True, typ=ImplicitComponent)
        ]

        # determine if the current group has appropriate solvers for
        # handling cycles, derivatives and implicit components
        is_iter_nl = has_iter_nl[path] = group.nonlinear_solver.options['maxiter'] > 1
        is_iter_ln = has_iter_ln[path] = (group.linear_solver.options['maxiter'] > 1 or
                                          isinstance(group.linear_solver, DirectSolver))

        # check upstream groups for iterative solvers and derivative requirements
        parts = path.split('.')
        for i in range(len(parts)):
            gname = '.'.join(parts[:i])
            is_iter_nl = is_iter_nl or has_iter_nl[gname]
            is_iter_ln = is_iter_ln or has_iter_ln[gname]
            derivs_needed = derivs_needed and uses_deriv[gname]

        # if you have a cycle, then you need a nonlinear solver with maxiter > 1
        # if you also are asking for derivatives up above, you need a linear solver too
        if has_cycles[path]:
            if not is_iter_nl:
                msg = ("Group '%s' contains cycles %s, but does not have an iterative "
                       "nonlinear solver." % (group.pathname, has_cycles[path]))
                self._setup_errors.append(msg)
            if derivs_needed and not is_iter_ln:
                msg = ("Group '%s' contains cycles %s and uses derivatives, but does "
                       "not have an iterative linear solver."
                       % (group.pathname, has_cycles[path]))
                self._setup_errors.append(msg)

        # if you have implicit components and you use derivatives, then you
        # need a better linear solver than LinearRunOnce
        if derivs_needed and has_states[path] and not is_iter_ln:
            msg = ("Group '%s' contains implicit components %s and uses "
                   "derivatives, but does not have an iterative linear solver."
                   % (group.pathname, has_states[path]))
            self._setup_errors.append(msg)

        # look for nonlinear solvers that require derivs under complex step.
        if 'cs' in group._approx_schemes:
            for sub in group.system_iter(include_self=True, recurse=True, typ=Group):
                if hasattr(sub.nonlinear_solver, 'linear_solver'):
                    msg = ("The solver in '%s' requires derivatives. We "
                           "currently do not support complex step around it."
                           % sub.name)
                    self._setup_errors.append(msg)


# Dict of all checks by name, mapped to the corresponding function that performs the check
# Each function must be of the form  f(problem, logger).
_checks = {
    'hanging_inputs': _check_hanging_inputs,
    'cycles': _check_dataflow_prob,
    'system': _check_system_configs,
    # 'solvers': _check_solvers,
    'dup_inputs': _check_dup_comp_inputs,
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
