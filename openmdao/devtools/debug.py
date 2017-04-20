"""Various debugging functions."""

import sys

from six.moves import zip_longest
from openmdao.core.group import Group

def dump_dist_idxs(problem, stream=sys.stdout, recurse=True):  # pragma: no cover
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
    stream : File-like
        Where dump output will go.

    recurse : bool
        If True, dump info for all systems contained in this system.
    """

    def _dump(g, stream=sys.stdout):
        idx = 0
        pdata = []
        pnwid = 0
        piwid = 0
        set_IDs = g._assembler._var_set_IDs
        sizes = g._assembler._var_sizes_by_set
        vnames = g._assembler._var_allprocs_abs_names
        offsets = g._assembler._var_offsets_by_set
        vidxs = g._assembler._var_allprocs_abs2idx_io
        set_idxs = g._assembler._var_set_indices

        for _, iset in set_IDs['input'].items():
            set_total = 0
            for rank in range(g.comm.size):
                for ivar, (vset, setidx) in enumerate(set_idxs['input']):
                    if vset == iset and sizes['input'][vset][rank, setidx] > 0:
                        name = vnames['input'][ivar]
                        pdata.append((name, str(set_total)))
                        pnwid = max(pnwid, len(name))
                        piwid = max(piwid, len(pdata[-1][1]))
                        set_total += sizes['input'][vset][rank, setidx]

                # insert a blank line to visually sparate sets
                pdata.append(('', '', '', ''))

        idx = 0
        udata = []
        unwid = 0
        uiwid = 0
        for _, iset in set_IDs['output'].items():
            set_total = 0
            for rank in range(g.comm.size):
                for ivar, (vset, setidx) in enumerate(set_idxs['output']):
                    if vset == iset and sizes['output'][vset][rank, setidx] > 0:
                        name = vnames['output'][ivar]
                        udata.append((name, str(set_total)))
                        unwid = max(unwid, len(name))
                        uiwid = max(uiwid, len(udata[-1][1]))
                        set_total += sizes['output'][vset][rank, setidx]

                # insert a blank line to visually sparate sets
                udata.append(('', '', '', ''))

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
            parts = s.pathname.split('.')
            depth = len(parts)
        else:
            depth = 0
        indent = '   ' * depth
        stream.write(indent)
        stream.write("%s %s\n" % (type(s).__name__, s.pathname))
        if include_solvers:
            if s.nl_solver is not None:
                stream.write("%s %s nl_solver\n" % (indent, type(s.nl_solver).__name__))
            if s.ln_solver is not None:
                stream.write("%s %s ln_solver\n" % (indent, type(s.ln_solver).__name__))
