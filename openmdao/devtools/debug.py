"""Various debugging functions."""

from __future__ import print_function

import sys

from resource import getrusage, RUSAGE_SELF, RUSAGE_CHILDREN

from six.moves import zip_longest
from openmdao.core.group import Group

def dump_dist_idxs(problem, stream=sys.stdout):  # pragma: no cover
    """Print out the distributed idxs for each variable in input and output vecs.

    Output looks like this:

    C3.y     26
    C3.y     25
    C3.y     24
    C2.y     23
    C2.y     22
    C2.y     21
    sub.C3.y 20     20 C3.x
    sub.C3.y 19     19 C3.x
    sub.C3.y 18     18 C3.x
    C1.y     17     17 C2.x
    P.x      16     16 C2.x
    P.x      15     15 C2.x
    P.x      14     14 sub.C3.x
    C3.y     13     13 sub.C3.x
    C3.y     12     12 sub.C3.x
    C3.y     11     11 C1.x
    C2.y     10     10 C3.x
    C2.y      9      9 C3.x
    C2.y      8      8 C3.x
    sub.C2.y  7      7 C2.x
    sub.C2.y  6      6 C2.x
    sub.C2.y  5      5 C2.x
    C1.y      4      4 sub.C2.x
    C1.y      3      3 sub.C2.x
    P.x       2      2 sub.C2.x
    P.x       1      1 C1.x
    P.x       0      0 C1.x

    Parameters
    ----------
    problem : <Problem>
        The problem object that contains the model.
    stream : File-like
        Where dump output will go.
    """
    def _get_data(g, type_):

        set_IDs = g._var_set2iset
        sizes = g._var_sizes_byset
        vnames = g._var_allprocs_abs_names
        set_idxs = g._var_allprocs_abs2idx_byset
        abs2meta = g._var_allprocs_abs2meta

        idx = 0
        data = []
        nwid = 0
        iwid = 0
        for sname, iset in set_IDs[type_].items():
            set_total = 0
            for rank in range(g.comm.size):
                for ivar, vname in enumerate(vnames[type_]):
                    vset = abs2meta[type_][vname]['var_set']
                    if vset == sname:
                        sz = sizes[type_][vset][rank, set_idxs[type_][vname]]
                        if sz > 0.:
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

def tree(system, include_solvers=True, stream=sys.stdout):
    """
    Dump the model tree structure to the given stream.

    Parameters
    ----------
    include_solvers : bool
        If True, include solvers in the tree.
    stream : File-like
        Where dump output will go.
    """
    for s in system.system_iter(include_self=True, recurse=True):
        if s.pathname:
            depth = len(s.pathname.split('.'))
        else:
            depth = 0
        indent = '   ' * depth
        stream.write(indent)
        stream.write("%s %s\n" % (type(s).__name__, s.name))
        if include_solvers:
            if s.nl_solver is not None:
                stream.write("%s %s nl_solver\n" % (indent, type(s.nl_solver).__name__))
            if s.ln_solver is not None:
                stream.write("%s %s ln_solver\n" % (indent, type(s.ln_solver).__name__))

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
