"""A module containing various configuration checks for an OpenMDAO Problem."""

from collections import defaultdict
from packaging.version import Version
import pathlib
from io import StringIO

import numpy as np
import pickle

from openmdao.core.group import Group
from openmdao.core.component import Component
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.utils.array_utils import array_hash
from openmdao.utils.graph_utils import get_sccs_topo
from openmdao.utils.logger_utils import get_logger, TestLogger
from openmdao.utils.mpi import MPI
from openmdao.utils.hooks import _register_hook
from openmdao.utils.general_utils import printoptions
from openmdao.utils.file_utils import _load_and_exec, text2html
from openmdao.utils.om_warnings import issue_warning, SetupWarning
from openmdao.utils.reports_system import register_report


_UNSET = object()

# numpy default print options changed in 1.14
if Version(np.__version__) >= Version("1.14"):
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
    cycles = [sorted(s, key=lambda n: group._subsystems_allprocs[n].index)
              for s in sccs if len(s) > 1]

    if cycles and infos is not None:
        infos.append(f"   Group '{group.pathname}' has the following cycles:")
        for cycle in cycles:
            infos.append(f"      {cycle}")
        infos.append('')

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
    out_of_order = group._check_order(reorder=False, recurse=False)
    for syspath, conns in out_of_order.items():
        prefix = f"   In System '{syspath}', subsystem " if syspath else "   System "
        for tgt, srcs in conns.items():
            warnings.append(f"{prefix}'{tgt}' executes out-of-order "
                            f"with respect to its source systems {srcs}\n")

    parallel_solvers = {}
    allsubs = group._subsystems_allprocs
    for sub, _ in allsubs.values():
        if hasattr(sub, '_mpi_proc_allocator') and sub._mpi_proc_allocator.parallel:
            parallel_solvers[sub.name] = sub.nonlinear_solver.SOLVER

    if parallel_solvers:
        _check_parallel_solvers(group, parallel_solvers)


def _check_parallel_solvers(group, parallel_solvers):
    """
    Report any parallel groups that don't have the proper solver.

    Parameters
    ----------
    group : <Group>
        The Group being checked.
    parallel_solvers : dict
        Dictionary of parallel solvers keyed by subsystem names.
    """
    glen = len(group.pathname.split('.')) if group.pathname else 0

    for tgt_abs, src_abs in group._conn_global_abs_in2out.items():
        iparts = tgt_abs.split('.')
        oparts = src_abs.split('.')
        src_sys = oparts[glen]
        tgt_sys = iparts[glen]
        hierarchy_check = oparts[glen + 1] == iparts[glen + 1]

        if (src_sys in parallel_solvers and tgt_sys in parallel_solvers and
                (parallel_solvers[src_sys] not in ["NL: NLBJ", "NL: Newton", "NL: BROYDEN"]) and
                src_sys == tgt_sys and
                not hierarchy_check):
            issue_warning("Need to attach NonlinearBlockJac, NewtonSolver, or BroydenSolver to "
                          f"'{src_sys}' when connecting components inside parallel groups",
                          category=SetupWarning)


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
    infos = ["The following groups contain cycles:"]
    for group in prob.model.system_iter(include_self=True, recurse=True, typ=Group):
        _check_cycles(group, infos)

    if len(infos) > 1:
        logger.info(infos[0])
        for i in range(1, len(infos)):
            logger.info(infos[i])


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


def _check_hanging_inputs(problem, logger):
    """
    Issue a logger warning if any model inputs are not connected.

    If an input is declared as a design variable, it is considered to be connected. Promoted
    inputs are shown alongside their corresponding absolute names.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.
    logger : object
        The object that manages logging output.
    """
    model = problem.model
    if isinstance(model, Component):
        return

    abs2prom = model._resolver.abs2prom
    desvars = problem.driver._designvars
    unconns = []
    for tgts in model._auto_ivc.auto2tgt.values():
        prom_tgt = abs2prom(tgts[0], 'input')
        # Ignore inputs that are declared as design vars.
        if desvars and prom_tgt in desvars:
            continue

        for tgt in tgts:
            unconns.append((prom_tgt, tgt))

    if unconns:
        msg = ["The following inputs are connected to Auto IVC output variables:\n"]
        for prom_tgt, abs_tgt in sorted(unconns):
            msg.append(f'  {prom_tgt} ({abs_tgt})\n')
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
        if len(list(comp._resolver.abs_iter('output'))) == 0:
            msg.append("   %s\n" % comp.pathname)

    if msg:
        logger.warning(''.join(["The following Components do not have any outputs:\n"] + msg))


def _check_auto_ivc_warnings(problem, logger):
    """
    Issue a logger warning if any components have conflicting attributes.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.
    logger : object
        The object that manages logging output.
    """
    if hasattr(problem.model, "_auto_ivc_warnings"):
        for i in problem.model._auto_ivc_warnings:
            logger.warning(i)


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


def _has_ancestor_solver(path, solvers):
    """
    Return True if the given path has an ancestor with a solver.

    Parameters
    ----------
    path : str
        The path to the system being checked.
    solvers : dict
        Dictionary of solvers keyed by system pathname.

    Returns
    -------
    bool
        True if the given path has an ancestor with a solver.
    """
    while path:
        path, _, _ = path.rpartition('.')
        if path in solvers:
            return True
    return False


def _check_solvers(problem, logger):
    """
    Search over all solvers and warn about unsupported configurations.

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
    from openmdao.core.group import Group, iter_solver_info
    from openmdao.core.implicitcomponent import ImplicitComponent

    has_nl_solver = {}
    has_lin_solver = {}
    group = problem.model
    lst = []

    for tup in group._sys_tree_visitor(iter_solver_info,
                                       predicate=lambda s: isinstance(s,
                                                                      (Group, ImplicitComponent))):
        path, pathclass, sccs, lnslv, nlslv, lnmaxiter, nlmaxiter, missing, isgrp, \
            nl_cansolve, lin_cansolve = tup

        if isgrp:
            lst.append(tup)

        if not isgrp or sccs:  # a group with cycles or an implicit component
            missing = []
            if not nl_cansolve and not _has_ancestor_solver(path, has_nl_solver):
                missing.append('nonlinear')
            if not lin_cansolve and not _has_ancestor_solver(path, has_lin_solver):
                missing.append('linear')

            if missing:
                missing = ' or '.join(missing)

                if isgrp:
                    sccs = [tuple(sorted(s)) for s in sccs]
                    msg = (f"Group '{path}' contains cycles {sccs}, but does not have an iterative "
                           f"{missing} solver.")
                else:
                    fncs = []
                    if 'nonlinear' in missing and nlslv != 'solve_nonlinear':
                        fncs.append('solve_nonlinear')
                    if 'linear' in missing and lnslv != 'solve_linear':
                        fncs.append('solve_linear')
                    fncs = ' or '.join(fncs)
                    msg = (f"{pathclass} '{path}' contains implicit variables but does "
                           f"not implement {fncs} or have an iterative {missing} solver.")
                logger.warning(msg)

        if isgrp and lin_cansolve:
            has_lin_solver[path] = (lnslv, lnmaxiter)
        if isgrp and nl_cansolve:
            has_nl_solver[path] = (nlslv, nlmaxiter)

    seen = set()
    lines = []
    for tup in lst:
        path, pathclass, sccs, lnslv, nlslv, lnmaxiter, nlmaxiter, missing, _, \
            _, _ = tup
        if sccs:
            if pathclass in seen:
                continue
            if missing == 0 and len(sccs) == 1:
                continue  # don't show groups without sub-cycles

            seen.add(pathclass)
            lines.append(f"'{path}' ({pathclass})  NL: {nlslv} (maxiter={nlmaxiter}), LN: "
                         f"{lnslv} (maxiter={lnmaxiter}):")

            for i, scc in enumerate(sccs):
                lines.append(f"   Cycle {i}: {sorted(scc)}")
            if missing:
                lines.append(f"   Number of non-cycle subsystems: {missing}")
            lines.append('')

    if lines:
        final = []
        final.append("The following groups contain sub-cycles. Performance and/or convergence "
                     "may improve")
        final.append("if these sub-cycles are solved separately in their own group.")
        final.append('')
        final.extend(lines)
        logger.warning('\n'.join(final))

    # Report any additional solver warnings.
    for system in problem.model.system_iter(include_self=True, recurse=True):
        if system._nonlinear_solver is not None:
            system._nonlinear_solver.check_config(logger)
        if system._linear_solver is not None:
            system._linear_solver.check_config(logger)


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
    # Look for a Problem recorder
    if problem._rec_mgr._recorders:
        return

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


def _check_unserializable_options(problem, logger, check_recordable=True):
    """
    Check if there are any options that are not serializable, and therefore won't be recorded.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.
    logger : object
        The object that manages logging output.
    check_recordable : bool
        If False, warn about all unserializable options.
        If True, warn only about unserializable options that do not have 'recordable' set to False.
    """
    from openmdao.recorders.case_recorder import PICKLE_VER

    def _check_opts(obj, name=None):
        if obj:
            for key, val in obj.options.items():
                try:
                    pickle.dumps(val, PICKLE_VER)
                except Exception:
                    name_str = name + " " if name else ""
                    if obj.options._dict[key]['recordable']:
                        msg = f"{obj.msginfo}: {name_str}option '{key}' is not serializable " \
                              "(cannot be pickled) but 'recordable=False' has not been set. " \
                              f"No options will be recorded for this {obj.__class__.__name__} " \
                              "unless 'recordable' is set to False for this option."
                        logger.warning(msg)
                    elif not check_recordable:
                        msg = f"{obj.msginfo}: {name_str}option '{key}' is not serializable " \
                              "(cannot be pickled) and will not be recorded."
                        logger.warning(msg)

    # check options for all for Systems and Solvers
    for system in problem.model.system_iter(include_self=True, recurse=True):
        _check_opts(system)
        _check_opts(system.linear_solver, 'linear_solver')

        nl = system.nonlinear_solver
        if nl:
            _check_opts(nl, 'nonlinear_solver')
            for name in ('linear_solver', 'linesearch'):
                _check_opts(getattr(nl, name, None), name)


def _check_all_unserializable_options(problem, logger):
    """
    Check if there are any options that are not serializable, and therefore won't be recorded.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.
    logger : object
        The object that manages logging output.
    """
    _check_unserializable_options(problem, logger, False)


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
    resolver = g._resolver
    prom_conn_ins = defaultdict(lambda: ([], []))
    for prom_in in g._manual_connections:
        for abs_in in resolver.absnames(prom_in, 'input'):
            prom_conn_ins[abs_in][1].append((prom_in, g.pathname))

    for subsys in g._subgroups_myproc:
        sub_prom_conn_ins = _get_promoted_connected_ins(subsys)
        for n, tup in sub_prom_conn_ins.items():
            proms, mans = tup
            mytup = prom_conn_ins[n]
            mytup[0].extend(proms)
            mytup[1].extend(mans)

        for inp, sub_prom_inp in subsys._resolver.abs2prom_iter('input', local=True):
            if resolver.abs2prom(inp, 'input') == sub_prom_inp:  # inp is promoted up from sub
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


def _check_bad_sparsity(problem, logger):
    """
    Check for any declared sparsity patterns that don't match the computed sparsity pattern.

    Parameters
    ----------
    problem : <Problem>
        The problem being checked.
    logger : object
        The object that manages logging output.
    """
    seen = set()
    for comp in problem.model.system_iter(include_self=True, recurse=True, typ=Component):
        plen = len(comp.pathname) + 1
        for of, wrt, computed_rows, computed_cols, rows, cols, _, _, wrn in \
                comp.check_sparsity(out_stream=None):
            # don't repeat same class over if diffs are the same
            chk = (type(comp).__name__, of[plen:], wrt[plen:],
                   array_hash(np.asarray(rows)), array_hash(np.asarray(cols)),
                   array_hash(np.asarray(computed_rows)), array_hash(np.asarray(computed_cols)))
            if chk not in seen:
                seen.add(chk)
                logger.warning(wrn)


# Dict of all checks by name, mapped to the corresponding function that performs the check
# Each function must be of the form  f(problem, logger).
_default_checks = {
    'out_of_order': _check_ubcs_prob,
    'system': _check_system_configs,
    'solvers': _check_solvers,
    'dup_inputs': _check_dup_comp_inputs,
    'missing_recorders': _check_missing_recorders,
    'unserializable_options': _check_unserializable_options,
    'comp_has_no_outputs': _check_comp_has_no_outputs,
    'auto_ivc_warnings': _check_auto_ivc_warnings,
}

_all_checks = _default_checks.copy()
_all_checks.update({
    'cycles': _check_cycles_prob,
    'unconnected_inputs': _check_hanging_inputs,
    'promotions': _check_explicitly_connected_promoted_inputs,
    'all_unserializable_options': _check_all_unserializable_options,
    'sparsity': _check_bad_sparsity,
})

_all_non_redundant_checks = _all_checks.copy()
del _all_non_redundant_checks['unserializable_options']


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


def _get_checks(checks):
    if checks is True:
        checks = sorted(_default_checks)
    elif not checks:
        checks = ()
    elif 'all' in checks:
        checks = sorted(_all_non_redundant_checks)
    return checks


class _Log2File(object):
    def __init__(self, f):
        self.f = f

    def info(self, msg):
        self.f.write(msg)
        self.f.write('\n')

    error = info
    warning = info
    debug = info


def _run_check_report(prob):
    s = StringIO()
    chk = prob._check if prob._check is not None else True
    for c in _get_checks(chk):
        if c not in _all_checks:
            print(f"WARNING: '{c}' is not a recognized check.  Available checks are: "
                  f"{sorted(_all_checks)}")
            continue

        print('-' * 30, f'Checking {c}', '-' * 30, file=s)
        _all_checks[c](prob, _Log2File(s))

    output = s.getvalue()
    if output:
        path = pathlib.Path(prob.get_reports_dir() / 'checks.html')
        with open(path, 'w') as f:
            f.write(text2html(output))


# entry point for check report
def _check_report_register():
    register_report('checks', _run_check_report, 'Config checks', 'Problem',
                    'final_setup', 'post')


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
        if not options.checks:
            options.checks = sorted(_default_checks)
        elif 'all' in options.checks:
            options.checks = sorted(_all_non_redundant_checks)

        if not MPI or prob.comm.rank == 0:
            if options.outfile is None:
                logger = get_logger('check_config', out_stream='stdout',
                                    out_file=None, use_format=True)
            else:
                logger = get_logger('check_config', out_file=options.outfile, use_format=True)
        else:
            # if not rank 0, don't display anything, but still do the config check to prevent
            # any MPI hangs due to collective calls
            logger = TestLogger()

        prob.check_config(logger, options.checks)

    # register the hook
    _register_hook('final_setup', class_name='Problem', inst_id=options.problem, post=_check_config,
                   exit=True)

    _load_and_exec(options.file[0], user_args)


def check_allocate_complex_ln(group, under_cs):
    """
    Return True if linear vector should be complex.

    This happens when a solver needs derivatives under complex step.

    Parameters
    ----------
    group : <Group>
        Group to be checked.
    under_cs : bool
        Flag indicates if complex vectors were allocated in a containing Group or were force
        allocated in setup.

    Returns
    -------
    bool
        True if linear vector should be complex.
    """
    under_cs |= 'cs' in group._approx_schemes

    if under_cs and group.nonlinear_solver is not None and \
       group.nonlinear_solver.supports['gradients']:
        return True

    for sub, _ in group._subsystems_allprocs.values():
        if isinstance(sub, Group) and check_allocate_complex_ln(sub, under_cs):
            return True

        elif isinstance(sub, ImplicitComponent):
            if sub.nonlinear_solver is not None and sub.nonlinear_solver.supports['gradients']:
                # Special case, gradient-supporting solver in an ImplicitComponent.
                return True

    return False
