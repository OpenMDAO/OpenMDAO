"""Various debugging functions."""

from __future__ import print_function

import sys
import os

from resource import getrusage, RUSAGE_SELF, RUSAGE_CHILDREN

from six.moves import zip_longest
from openmdao.core.problem import Problem
from openmdao.core.group import Group, System

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

        set_IDs = g._var_set2iset
        sizes = g._var_sizes_byset[vec_name]
        vnames = g._var_allprocs_abs_names
        set_idxs = g._var_allprocs_abs2idx_byset[vec_name]
        abs2meta = g._var_allprocs_abs2meta

        idx = 0
        data = []
        nwid = 0
        iwid = 0
        for sname in set_IDs[type_]:
            set_total = 0
            for rank in range(g.comm.size):
                for ivar, vname in enumerate(vnames[type_]):
                    vset = abs2meta[type_][vname]['var_set']
                    if vset == sname:
                        sz = sizes[type_][vset][rank, set_idxs[type_][vname]]
                        if sz > 0:
                            data.append((vname, str(set_total)))
                        nwid = max(nwid, len(vname))
                        iwid = max(iwid, len(data[-1][1]))
                        set_total += sz

            # insert a blank line to visually sparate sets
            data.append(('', '', '', ''))
        return data, nwid, iwid

    def _dump(g, stream):

        pdata, pnwid, piwid = _get_data(g, 'input')
        udata, unwid, uiwid = _get_data(g, 'output')

        data = []
        for u, p in zip_longest(udata, pdata, fillvalue=('', '')):
            data.append((u[0], u[1], p[1], p[0]))

        for d in data[::-1]:
            template = "{0:<{wid0}} {1:>{wid1}}     {2:>{wid2}} {3:<{wid3}}\n"
            stream.write(template.format(d[0], d[1], d[2], d[3],
                                         wid0=unwid, wid1=uiwid,
                                         wid2=piwid, wid3=pnwid))
        stream.write("\n\n")

    _dump(problem.model, stream)


class _NoColor(object):
    """
    A class to replace Fore, Back, and Style when colorama isn't istalled.
    """
    def __getattr__(self, name):
        return ''


def _get_color_printer(stream=sys.stdout, colors=True):
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

    def color_print(s, fore='', color='', end=''):
        """
        """
        print(color + s, file=stream, end='')
        print(Style.RESET_ALL, file=stream, end='')
        print(end=end)

    return color_print, Fore, Back, Style


def tree(top, show_solvers=True, show_colors=True, filter=None, max_depth=0, stream=sys.stdout):
    """
    Dump the model tree structure to the given stream.

    If you install colorama, the tree will be displayed in color if the stream is a terminal
    that supports color display.

    Parameters
    ----------
    top : System or Problem
        The top object in the tree.
    show_solvers : bool
        If True, include solvers in the tree.
    show_colors : bool
        If True and stream is a terminal that supports it, display in color.
    filter : function(System)
        A function taking a System arg and returning None or an iter of (name, value) tuples.
        If None is returned, that system will not be displayed.  Otherwise, the system will
        be displayed along with any name, value pairs returned from the filter, if any.
    max_depth : int
        Maximum depth for display.
    stream : File-like
        Where dump output will go.
    """
    cprint, Fore, Back, Style = _get_color_printer(stream, show_colors)

    tab = 0
    if isinstance(top, Problem):
        if filter is None:
            cprint('Driver: ', color=Fore.CYAN + Style.BRIGHT)
            cprint(type(top.driver).__name__, color=Fore.MAGENTA, end='\n')
            tab += 1
        top = top.model

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
        print(indent, file=stream, end='')

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
        print(file=stream)

        vindent = indent + '  '
        for name, val in ret:
            print("%s%s: %s" % (vindent, name, val), file=stream)


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
    allsystems = list(model.system_iter(recurse=True, include_self=True))
    allgroups = [s for s in allsystems if isinstance(s, Group)]
    sysnames = [s.pathname for s in allsystems]
    maxdepth = max([len(name.split('.')) for name in sysnames])
    setup_done = problem._setup_status == 2
    meta_in = model._var_allprocs_abs2meta['input']

    print("============== Problem Summary ============", file=stream)
    print("Groups:           %5d" % len(allgroups), file=stream)
    print("Components:       %5d" % (len(allsystems) - len(allgroups)), file=stream)
    print("Max tree depth:   %5d" % maxdepth, file=stream)
    print(file=stream)

    if setup_done:
        desvars = model.get_design_vars()
        print("Design variables: %5d   Total size: %8d" %
              (len(desvars), sum(d['size'] for d in desvars.values())), file=stream)

        # TODO: give separate info for linear, nonlinear constraints, equality, inequality
        constraints = model.get_constraints()
        print("Constraints:      %5d   Total size: %8d" %
              (len(constraints), sum(d['size'] for d in constraints.values())), file=stream)

        objs = model.get_objectives()
        print("Objectives:       %5d   Total size: %8d" %
              (len(objs), sum(d['size'] for d in objs.values())), file=stream)

    print(file=stream)

    ninputs = len(model._var_allprocs_abs_names['input'])
    if setup_done:
        print("Input variables:  %5d   Total size: %8d" %
              (ninputs, sum(d.size for d in model._inputs._data.values())), file=stream)
    else:
        print("Input variables: %5d" % ninputs, file=stream)

    noutputs = len(model._var_allprocs_abs_names['output'])
    if setup_done:
        print("Output variables: %5d   Total size: %8d" %
              (noutputs, sum(d.size for d in model._outputs._data.values())), file=stream)
    else:
        print("Output variables: %5d" % noutputs, file=stream)

    if setup_done:
        print(file=stream)
        conns = model._conn_global_abs_in2out
        print("Total connections: %d   Total transfer data size: %d" %
              (len(conns), sum(meta_in[n]['size'] for n in conns)), file=stream)

    print(file=stream)
    print("Driver type: %s" % problem.driver.__class__.__name__, file=stream)
    print("Linear Solvers: %s" % sorted(set(type(s.linear_solver).__name__ for s in allgroups)),
          file=stream)
    print("Nonlinear Solvers: %s" % sorted(set(type(s.nonlinear_solver).__name__ for s in
                                                         allgroups)),
          file=stream)


def max_mem_usage():
    """
    Returns
    -------
    The max memory used by this process and its children, in MB.
    """
    denom = 1024.
    if sys.platform == 'darwin':
        denom *= denom
    total = getrusage(RUSAGE_SELF).ru_maxrss / denom
    total += getrusage(RUSAGE_CHILDREN).ru_maxrss / denom
    return total

try:
    import psutil

    def mem_usage(msg='', out=sys.stdout):
        """
        Returns
        -------
        The current memory used by this process (and it's children?), in MB.
        """
        denom = 1024. * 1024.
        p = psutil.Process(os.getpid())
        mem = p.memory_info().rss / denom
        if msg:
            print(msg,"%6.3f MB" % mem, file=out)
        return mem

    def diff_mem(fn):
        """
        This gives the difference in memory before and after the
        decorated function is called. Requires psutil to be installed.
        """
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            startmem = mem_usage()
            ret = fn(*args, **kwargs)
            maxmem = mem_usage()
            diff = maxmem - startmem
            if diff > 0.0:
                if args and hasattr(args[0], 'pathname'):
                    name = args[0].pathname
                else:
                    name = ''
                print(name,"%s added %5.3f MB (total: %6.3f)" % (fn.__name__, diff, maxmem))
            return ret
        return wrapper
except ImportError:
    pass
