"""Various debugging functions."""

from __future__ import print_function

import sys
import os
from itertools import product

import numpy as np
import cProfile
from contextlib import contextmanager
from six import iteritems, iterkeys, itervalues

from six.moves import zip_longest
from openmdao.core.problem import Problem
from openmdao.core.group import Group, System
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.utils.mpi import MPI
from openmdao.approximation_schemes.finite_difference import FiniteDifference, DEFAULT_FD_OPTIONS
from openmdao.approximation_schemes.complex_step import ComplexStep, DEFAULT_CS_OPTIONS
from openmdao.utils.name_maps import abs_key2rel_key, rel_key2abs_key

# an object used to detect when a named value isn't found
_notfound = object()

def dump_dist_idxs(problem, vec_name='nonlinear', stream=sys.stdout):  # pragma: no cover
    """Print out the distributed idxs for each variable in input and output vecs.

    Output looks like this:

    C3.y     24
    C2.y     21
    sub.C3.y 18
    C1.y     17     18 C3.x
    P.x      14     15 C2.x
    C3.y     12     12 sub.C3.x
    C3.y     11     11 C1.x
    C2.y      8      8 C3.x
    sub.C2.y  5      5 C2.x
    C1.y      3      2 sub.C2.x
    P.x       0      0 C1.x

    Parameters
    ----------
    problem : <Problem>
        The problem object that contains the model.
    vec_name : str
        Name of vector to dump (when there are multiple vectors due to parallel derivs)
    stream : File-like
        Where dump output will go.
    """
    def _get_data(g, type_):

        sizes = g._var_sizes[vec_name]
        vnames = g._var_allprocs_abs_names
        abs2meta = g._var_allprocs_abs2meta

        idx = 0
        data = []
        nwid = 0
        iwid = 0
        total = 0
        for rank in range(g.comm.size):
            for ivar, vname in enumerate(vnames[type_]):
                sz = sizes[type_][rank, ivar]
                if sz > 0:
                    data.append((vname, str(total)))
                nwid = max(nwid, len(vname))
                iwid = max(iwid, len(data[-1][1]))
                total += sz

        return data, nwid, iwid

    def _dump(g, stream):

        pdata, pnwid, piwid = _get_data(g, 'input')
        udata, unwid, uiwid = _get_data(g, 'output')

        data = []
        for u, p in zip_longest(udata, pdata, fillvalue=('', '')):
            data.append((u[0], u[1], p[1], p[0]))

        template = "{0:<{wid0}} {1:>{wid1}}     {2:>{wid2}} {3:<{wid3}}\n"
        for d in data[::-1]:
            stream.write(template.format(d[0], d[1], d[2], d[3],
                                         wid0=unwid, wid1=uiwid,
                                         wid2=piwid, wid3=pnwid))
        stream.write("\n\n")

    if not MPI or MPI.COMM_WORLD.rank == 0:
        _dump(problem.model, stream)


class _NoColor(object):
    """
    A class to replace Fore, Back, and Style when colorama isn't istalled.
    """
    def __getattr__(self, name):
        return ''


def _get_color_printer(stream=sys.stdout, colors=True, rank=0):
    """
    Return a print function tied to a particular stream, along with coloring info.
    """
    try:
        from colorama import init, Fore, Back, Style
        init(autoreset=True)
    except ImportError:
        Fore = Back = Style = _NoColor()

    if not colors:
        Fore = Back = Style = _NoColor()

    if MPI and MPI.COMM_WORLD.rank != rank:
        if rank >= MPI.COMM_WORLD.size:
            if MPI.COMM_WORLD.rank == 0:
                print("Specified rank (%d) is outside of the valid range (0-%d)." %
                      (rank, MPI.COMM_WORLD.size - 1))
            sys.exit()
        def color_print(s, **kwargs):
            pass
    else:
        def color_print(s, fore='', color='', end=''):
            """
            """
            print(color + s, file=stream, end='')
            print(Style.RESET_ALL, file=stream, end='')
            print(end=end)

    return color_print, Fore, Back, Style


def tree(top, show_solvers=True, show_jacs=True, show_colors=True,
         filter=None, max_depth=0, rank=0, stream=sys.stdout):
    """
    Dump the model tree structure to the given stream.

    If you install colorama, the tree will be displayed in color if the stream is a terminal
    that supports color display.

    Parameters
    ----------
    top : System or Problem
        The top object in the tree.
    show_solvers : bool
        If True, include solver types in the tree.
    show_jacs : bool
        If True, include jacobian types in the tree.
    show_colors : bool
        If True and stream is a terminal that supports it, display in color.
    filter : function(System)
        A function taking a System arg and returning None or an iter of (name, value) tuples.
        If None is returned, that system will not be displayed.  Otherwise, the system will
        be displayed along with any name, value pairs returned from the filter, if any.
    max_depth : int
        Maximum depth for display.
    rank : int
        If MPI is active, the tree will only be displayed on this rank.  Only objects local
        to the given rank will be displayed.
    stream : File-like
        Where dump output will go.
    """
    cprint, Fore, Back, Style = _get_color_printer(stream, show_colors, rank=rank)

    tab = 0
    if isinstance(top, Problem):
        if filter is None:
            cprint('Driver: ', color=Fore.CYAN + Style.BRIGHT)
            cprint(type(top.driver).__name__, color=Fore.MAGENTA, end='\n')
            tab += 1
        top = top.model

    seenJacs = set()
    for s in top.system_iter(include_self=True, recurse=True):
        if filter is None:
            ret = ()
        else:
            ret = filter(s)
            if ret is None:
                continue

        depth = len(s.pathname.split('.')) if s.pathname else 0
        if max_depth != 0 and depth > max_depth:
            continue

        indent = '    ' * (depth + tab)
        cprint(indent, end='')

        info = ''
        if isinstance(s, Group):
            cprint("%s " % type(s).__name__, color=Fore.GREEN + Style.BRIGHT)
            cprint("%s" % s.name)
        else:
            cprint("%s " % type(s).__name__, color=Fore.CYAN + Style.BRIGHT)
            cprint("%s" % s.name)

        if show_solvers:
            lnsolver = type(s.linear_solver).__name__
            nlsolver = type(s.nonlinear_solver).__name__

            if s.linear_solver is not None and lnsolver != "LinearRunOnce":
                cprint("  LN: ")
                cprint(lnsolver, color=Fore.MAGENTA + Style.BRIGHT)
            if s.nonlinear_solver is not None and nlsolver != "NonlinearRunOnce":
                cprint("  NL: ")
                cprint(nlsolver, color=Fore.MAGENTA + Style.BRIGHT)

        if show_jacs:
            if s._assembled_jac is not None:
                cprint(" LN_jac: ")
                cprint(type(s._assembled_jac).__name__, color=Fore.MAGENTA + Style.BRIGHT)
            if s.nonlinear_solver is not None:
                jacsolvers = list(s.nonlinear_solver._assembled_jac_solver_iter())
                if jacsolvers:
                    cprint(" NL_jac: ")
                    cprint(type(jacsolvers[0]._assembled_jac).__name__,
                           color=Fore.MAGENTA + Style.BRIGHT)

        cprint('', end='\n')

        vindent = indent + '  '
        for name, val in ret:
            cprint("%s%s: %s\n" % (vindent, name, val))


def _get_printer(comm, stream):
    if comm.rank == 0:
        def p(*args, **kwargs):
            print(*args, file=stream, **kwargs)
    else:
        def p(*args, **kwargs):
            pass

    return p


def config_summary(problem, stream=sys.stdout):
    """
    Prints various high level statistics about the model structure.

    Parameters
    ----------
    problem : Problem
        The Problem to be summarized.
    stream : File-like
        Where the output will be written.
    """
    model = problem.model
    meta = model._var_allprocs_abs2meta
    locsystems = list(model.system_iter(recurse=True, include_self=True))
    locgroups = [s for s in locsystems if isinstance(s, Group)]

    grpnames = [s.pathname for s in locgroups]
    sysnames = [s.pathname for s in locsystems]
    ln_solvers = set(type(s.linear_solver).__name__ for s in locgroups)
    nl_solvers = set(type(s.nonlinear_solver).__name__ for s in locgroups)

    max_depth = max([len(name.split('.')) for name in sysnames])
    setup_done = problem._setup_status == 2

    if problem.comm.size > 1:
        local_max = np.array([max_depth])
        global_max_depth = np.zeros(1, dtype=int)
        problem.comm.Allreduce(local_max, global_max_depth, op=MPI.MAX)

        proc_names = problem.comm.gather((sysnames, grpnames, ln_solvers, nl_solvers), root=0)
        grpnames = set()
        sysnames = set()
        ln_solvers = set()
        nl_solvers = set()
        if proc_names is not None:
            for rank in range(problem.comm.size):
                systems, grps, lnsols, nlsols = proc_names[rank]
                sysnames.update(systems)
                grpnames.update(grps)
                ln_solvers.update(lnsols)
                nl_solvers.update(nlsols)
    else:
        global_max_depth = max_depth

    printer = _get_printer(problem.comm, stream)

    printer("============== Problem Summary ============")
    printer("Groups:           %5d" % len(grpnames))
    printer("Components:       %5d" % (len(sysnames) - len(grpnames)))
    printer("Max tree depth:   %5d" % global_max_depth)
    printer()

    if setup_done:
        desvars = model.get_design_vars()
        printer("Design variables: %5d   Total size: %8d" %
              (len(desvars), sum(d['size'] for d in desvars.values())))

        # TODO: give separate info for linear, nonlinear constraints, equality, inequality
        constraints = model.get_constraints()
        printer("Constraints:      %5d   Total size: %8d" %
              (len(constraints), sum(d['size'] for d in constraints.values())))

        objs = model.get_objectives()
        printer("Objectives:       %5d   Total size: %8d" %
              (len(objs), sum(d['size'] for d in objs.values())))

    printer()

    input_names = model._var_allprocs_abs_names['input']
    ninputs = len(input_names)
    if setup_done:
        printer("Input variables:  %5d   Total size: %8d" %
              (ninputs, sum(meta[n]['size'] for n in input_names)))
    else:
        printer("Input variables: %5d" % ninputs)

    output_names = model._var_allprocs_abs_names['output']
    noutputs = len(output_names)
    if setup_done:
        printer("Output variables: %5d   Total size: %8d" %
              (noutputs, sum(meta[n]['global_size'] for n in output_names)))
    else:
        printer("Output variables: %5d" % noutputs)

    if setup_done and isinstance(model, Group):
        printer()
        conns = model._conn_global_abs_in2out
        printer("Total connections: %d   Total transfer data size: %d" %
              (len(conns), sum(meta[n]['size'] for n in conns)))

    printer()
    printer("Driver type: %s" % problem.driver.__class__.__name__)
    printer("Linear Solvers: %s" % sorted(ln_solvers))
    printer("Nonlinear Solvers: %s" % sorted(nl_solvers))


@contextmanager
def profiling(outname='prof.out'):
    """
    Context manager that runs cProfile on the wrapped code and dumps stats to the given filename.

    Parameters
    ----------
    outname : str
        Name of the output file containing profiling stats.
    """
    prof = cProfile.Profile()
    prof.enable()

    yield prof

    prof.disable()
    prof.dump_stats(outname)


def compare_jacs(Jref, J, rel_trigger=1.0):
    results = []

    for key in set(J).union(Jref):
        if key in J:
            subJ = J[key]
        else:
            subJ = np.zeros(Jref[key].shape)

        if key in Jref:
            subJref = Jref[key]
        else:
            subJref = np.zeros(J[key].shape)

        diff = np.abs(subJ - subJref)
        absref = np.abs(subJref)
        rel_idxs = np.nonzero(absref > rel_trigger)
        diff[rel_idxs] /= absref[rel_idxs]

        max_diff_idx = np.argmax(diff)
        max_diff = diff.flatten()[max_diff_idx]

        # now determine if max diff is abs or rel
        diff[:] = 0.0
        diff[rel_idxs] = 1.0
        if diff.flatten()[max_diff_idx] > 0.0:
            results.append((key, max_diff, 'rel'))
        else:
            results.append((key, max_diff, 'abs'))

    return results


