"""Various debugging functions."""


import sys
import pathlib
from io import StringIO

import numpy as np
from contextlib import contextmanager
from collections import Counter

from openmdao.core.constants import _SetupStatus, _DEFAULT_OUT_STREAM
from openmdao.utils.mpi import MPI
from openmdao.utils.om_warnings import issue_warning, MPIWarning
from openmdao.utils.reports_system import register_report
from openmdao.utils.file_utils import text2html, _load_and_exec
from openmdao.utils.rangemapper import RangeMapper
from openmdao.visualization.tables.table_builder import generate_table


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
        def color_print(s, color='', end=''):
            """
            """
            print(color + s, file=stream, end='')
            print(Style.RESET_ALL, file=stream, end='')
            print(end=end)

    return color_print, Fore, Back, Style


def tree(top, show_solvers=True, show_jacs=True, show_colors=True, show_approx=True,
         filter=None, show_sizes=False, max_depth=0, rank=0, stream=sys.stdout):
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
    show_approx : bool
        If True, mark systems that are approximating their derivatives.
    filter : function(System)
        A function taking a System arg and returning None or an iter of (name, value) tuples.
        If None is returned, that system will not be displayed.  Otherwise, the system will
        be displayed along with any name, value pairs returned from the filter.
    show_sizes : bool
        If True, show input and output sizes for each System.
    max_depth : int
        Maximum depth for display.
    rank : int
        If MPI is active, the tree will only be displayed on this rank.  Only objects local
        to the given rank will be displayed.
    stream : File-like
        Where dump output will go.
    """
    from openmdao.core.problem import Problem
    from openmdao.core.group import Group
    from openmdao.core.implicitcomponent import ImplicitComponent

    cprint, Fore, Back, Style = _get_color_printer(stream, show_colors, rank=rank)

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
        cprint(indent, end='')

        info = ''
        if isinstance(s, Group):
            cprint("%s " % type(s).__name__, color=Fore.GREEN + Style.BRIGHT)
            cprint("%s" % s.name)
        else:
            if isinstance(s, ImplicitComponent):
                colr = Back.CYAN + Fore.BLACK + Style.BRIGHT
            else:
                colr = Fore.CYAN + Style.BRIGHT
            cprint("%s " % type(s).__name__, color=colr)
            cprint("%s" % s.name)
            if s._has_distrib_vars:
                cprint(" (distributed)", color=Fore.MAGENTA)

        # FIXME: these sizes could be wrong under MPI
        if show_sizes:
            cprint(" (%d / %d)" % (s._inputs._data.size, s._outputs._data.size),
                color=Fore.RED + Style.BRIGHT)

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
            jacs = []
            lnjac = nljac = None
            if s._assembled_jac is not None:
                lnjac = s._assembled_jac
                jacs.append(lnjac)
            if s.nonlinear_solver is not None:
                jacsolvers = list(s.nonlinear_solver._assembled_jac_solver_iter())
                if jacsolvers:
                    nljac = jacsolvers[0]._assembled_jac
                    if nljac is not lnjac:
                        jacs.append(nljac)

            if len(jacs) == 2:
                jnames = [' LN Jac: ', ' NL Jac: ']
            elif lnjac is not None:
                if lnjac is nljac:
                    jnames = [' Jac: ']
                else:
                    jnames = [' LN Jac: ']
            elif nljac is not None:
                jnames = [' NL Jac: ']
            else:
                jnames = []

            for jname, jac in zip(jnames, jacs):
                cprint(jname)
                cprint(type(jac).__name__, color=Fore.MAGENTA + Style.BRIGHT)

        if show_approx and s._approx_schemes:
            approx_keys = set()
            keys = set()
            for k, sjac in s._subjacs_info.items():
                if 'method' in sjac and sjac['method']:
                    approx_keys.add(k)
                else:
                    keys.add(k)
            diff = approx_keys - keys
            cprint("  APPROX: ", color=Fore.MAGENTA + Style.BRIGHT)
            cprint("%s (%d of %d)" % (list(s._approx_schemes), len(diff), len(s._subjacs_info)))

        cprint('', end='\n')

        vindent = indent + '  '
        for name, val in ret:
            cprint("%s%s: %s\n" % (vindent, name, val))


def _get_printer(comm, stream, rank=0):
    if comm.rank == rank:
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
    from openmdao.core.group import Group

    model = problem.model
    meta = model._var_allprocs_abs2meta
    locsystems = list(model.system_iter(recurse=True, include_self=True))
    locgroups = [s for s in locsystems if isinstance(s, Group)]

    grpnames = [s.pathname for s in locgroups]
    sysnames = [s.pathname for s in locsystems]
    ln_solvers = [(s.pathname, type(s.linear_solver).__name__) for s in locsystems
                              if s.linear_solver is not None]
    nl_solvers = [(s.pathname, type(s.nonlinear_solver).__name__) for s in locsystems
                         if s.nonlinear_solver is not None]

    max_depth = max([len(name.split('.')) for name in sysnames])
    setup_done = model._problem_meta['setup_status'] >= _SetupStatus.POST_FINAL_SETUP

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
            for systems, grps, lnsols, nlsols in proc_names:
                sysnames.update(systems)
                grpnames.update(grps)
                ln_solvers.update(lnsols)
                nl_solvers.update(nlsols)
    else:
        global_max_depth = max_depth
        ln_solvers = set(ln_solvers)
        nl_solvers = set(nl_solvers)

    ln_solvers = Counter([sname for _, sname in ln_solvers])
    nl_solvers = Counter([sname for _, sname in nl_solvers])

    # this gives us a printer that only prints on rank 0
    printer = _get_printer(problem.comm, stream)

    printer("============== Problem Summary ============")
    printer("Groups:           %5d" % len(grpnames))
    printer("Components:       %5d" % (len(sysnames) - len(grpnames)))
    printer("Max tree depth:   %5d" % global_max_depth)
    printer()

    if setup_done:
        desvars = model.get_design_vars()
        printer("Design variables:        %5d   Total size: %8d" %
                (len(desvars), sum(d['size'] for d in desvars.values())))

        con_nonlin_eq = {}
        con_nonlin_ineq = {}
        con_linear_eq = {}
        con_linear_ineq = {}
        for con, vals in model.get_constraints().items():
            if vals['linear']:
                if vals['equals'] is not None:
                    con_linear_eq[con] = vals
                else:
                    con_linear_ineq[con] = vals
            else:
                if vals['equals'] is not None:
                    con_nonlin_eq[con]= vals
                else:
                    con_nonlin_ineq[con]= vals

        con_nonlin = con_nonlin_eq.copy()
        con_nonlin.update(con_nonlin_ineq)
        con_linear = con_linear_eq.copy()
        con_linear.update(con_linear_ineq)

        printer("\nNonlinear Constraints:   %5d   Total size: %8d" %
                (len(con_nonlin), sum(d['size'] for d in con_nonlin.values())))
        printer("    equality:            %5d               %8d" %
                (len(con_nonlin_eq), sum(d['size'] for d in con_nonlin_eq.values())))
        printer("    inequality:          %5d               %8d" %
                (len(con_nonlin_ineq), sum(d['size'] for d in con_nonlin_ineq.values())))
        printer("\nLinear Constraints:      %5d   Total size: %8d" %
                (len(con_linear), sum(d['size'] for d in con_linear.values())))
        printer("    equality:            %5d               %8d" %
                (len(con_linear_eq), sum(d['size'] for d in con_linear_eq.values())))
        printer("    inequality:          %5d               %8d" %
                (len(con_linear_ineq), sum(d['size'] for d in con_linear_ineq.values())))

        objs = model.get_objectives()
        printer("\nObjectives:              %5d   Total size: %8d" %
                (len(objs), sum(d['size'] for d in objs.values())))

    printer()

    input_names = model._var_allprocs_abs2meta['input']
    ninputs = len(input_names)
    if setup_done:
        printer("Input variables:         %5d   Total size: %8d" %
                (ninputs, sum(meta['input'][n]['size'] for n in input_names)))
    else:
        printer("Input variables:         %5d" % ninputs)

    output_names = model._var_allprocs_abs2meta['output']
    noutputs = len(output_names)
    if setup_done:
        printer("Output variables:        %5d   Total size: %8d" %
                (noutputs, sum(meta['output'][n]['global_size'] for n in output_names)))
    else:
        printer("Output variables:        %5d" % noutputs)

    if setup_done and isinstance(model, Group):
        printer()
        conns = model._conn_global_abs_in2out
        printer("Total connections: %d   Total transfer data size: %d" %
                (len(conns), sum(meta['input'][n]['size'] for n in conns)))

    printer()
    printer("Driver type: %s" % problem.driver.__class__.__name__)
    linstr = []
    for slvname, num in ln_solvers.most_common():
        if num > 1:
            linstr.append('{} x {}'.format(slvname, num))
        else:
            linstr.append(slvname)
    printer("Linear Solvers: [{}]".format(', '.join(linstr)))


    nlstr = []
    for slvname, num in nl_solvers.most_common():
        if num > 1:
            nlstr.append('{} x {}'.format(slvname, num))
        else:
            nlstr.append(slvname)
    printer("Nonlinear Solvers: [{}]".format(', '.join(nlstr)))


def _summary_report(prob):
    path = str(pathlib.Path(prob.get_reports_dir()).joinpath('summary.html'))
    s = StringIO()
    config_summary(prob, s)
    with open(path, 'w') as f:
        f.write(text2html(s.getvalue()))


def _summary_report_register():
    register_report('summary', _summary_report, 'Model summary', 'Problem', 'final_setup', 'post')


@contextmanager
def profiling(outname='prof.out'):
    """
    Context manager that runs cProfile on the wrapped code and dumps stats to the given filename.

    Parameters
    ----------
    outname : str
        Name of the output file containing profiling stats.
    """
    import cProfile
    prof = cProfile.Profile()
    prof.enable()

    try:
        yield prof
    finally:
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


def trace_dump(fname='trace_dump', skip=(), flush=True):
    """
    Dump traces to the specified filename<.rank> showing openmdao and c calls.

    Under MPI it will write a separate file for each rank.

    Parameters
    ----------
    fname : str
        Name of the trace file(s).  <.rank> will be appended to the name on each rank.
    skip : set-like
        Collection of function names to skip.
    flush : bool
        If True, flush print buffer after every print call.
    """
    if sys.getprofile() is not None:
        raise RuntimeError("another profile function is already active.")

    suffix = '.0' if MPI is None else '.' + str(MPI.COMM_WORLD.rank)
    my_fname = fname + suffix

    outfile = open(my_fname, 'w')

    stack = []

    _c_map = {
        'c_call': '(c) -->',
        'c_return': '(c) <--',
        'c_exception': '(c_exception)',
    }


    def _print_c_func(frame, arg, typestr):
        s = str(arg)
        if 'mpi4py' in s or 'petsc4py' in s:
            c = arg.__self__.__class__
            if stack:
                pname = f"(scope: {stack[-1][0]})"
            else:
                pname = ''
            print('   ' * len(stack), typestr, f"{c.__module__}.{c.__name__}.{arg.__name__}",
                  f"{frame.f_code.co_filename}:{frame.f_code.co_firstlineno} {pname}",
                  file=outfile, flush=True)


    def _mpi_trace_callback(frame, event, arg):
        pname = None
        commsize = ''
        if event == 'call':
            if 'openmdao' in frame.f_code.co_filename:
                if frame.f_code.co_name in skip:
                    return
                if 'self' in frame.f_locals:
                    try:
                        pname = frame.f_locals['self'].msginfo
                    except:
                        pass
                    try:
                        commsize = frame.f_locals['self'].comm.size
                    except:
                        pass
                if pname is not None:
                    if not stack or pname != stack[-1][0]:
                        stack.append([pname, 1])
                        print('   ' * len(stack), commsize, f"(scope: {pname})", file=outfile, flush=flush)
                    else:
                        stack[-1][1] += 1
                print('   ' * len(stack), '-->', frame.f_code.co_name, "%s:%d" %
                      (frame.f_code.co_filename, frame.f_code.co_firstlineno),
                      file=outfile, flush=flush)
        elif event == 'return':
            if 'openmdao' in frame.f_code.co_filename:
                if frame.f_code.co_name in skip:
                    return
                if 'self' in frame.f_locals:
                    try:
                        pname = frame.f_locals['self'].msginfo
                    except:
                        pass
                    try:
                        commsize = frame.f_locals['self'].comm.size
                    except:
                        pass
                print('   ' * len(stack), '<--', frame.f_code.co_name, "%s:%d" %
                      (frame.f_code.co_filename, frame.f_code.co_firstlineno),
                      file=outfile, flush=flush)
                if pname is not None and stack and pname == stack[-1][0]:
                    stack[-1][1] -= 1
                    if stack[-1][1] < 1:
                        stack.pop()
                        if stack:
                            print('   ' * len(stack), commsize, stack[-1][0], file=outfile,
                                  flush=flush)
        else:
            _print_c_func(frame, arg, _c_map[event])

    sys.setprofile(_mpi_trace_callback)


def prom_info_dump(system, tgt):
    """
    Dump the promotion src_indices/src_shape data for the given absolute target name.

    The data actually lives in the Problem metadata, but is more convenient to access during
    debugging by using a System instance to access that metadata.

    Promotion src_indices/src_shape data is displayed for all inputs, including tgt, that
    are connected to the same source.

    Parameters
    ----------
    system : System
        Any System instance.
    tgt : str
        Absolute name of an input variable.
    """
    probmeta = system._problem_meta
    model = probmeta['model_ref']()
    src = model._conn_global_abs_in2out[tgt]
    abs_in2prom_info = probmeta['abs_in2prom_info']
    print('For tgt', tgt, 'and src', src, 'connected tgts and prom info are:')
    for t, s in model._conn_global_abs_in2out.items():
        if s == src:
            print('    ', t)
            if t in abs_in2prom_info:
                for p in abs_in2prom_info[t]:
                    print('        ', p)
    print(flush=True)


def comm_info(system, outfile=None, verbose=False, table_format='box_grid'):
    """
    Display MPI communicator information for Systems in the model.

    Parameters
    ----------
    system : System
        A System in the model.
    outfile : str or None
        Name of file where output will be written. If None, output is written to stdout.
    verbose : bool
        If True, display comm size and rank range for all Systems. Otherwise, display only Systems
        having a comm size different from their parent system.
    table_format : str
        Table format.  All formats compatible with the generate_table function are available.
    """
    if MPI and MPI.COMM_WORLD.size > 1:
        dct = {}
        for path, csize, rank, wrank in system.comm_info_iter():
            if path not in dct:
                dct[path] = [csize, wrank, wrank]
            else:
                csize, minwrnk, maxwrnk = dct[path]
                # do min/max here *and* after the gather so we don't have to
                # gather all the data to get the min/max
                if wrank < minwrnk:
                    minwrnk = wrank
                if wrank > maxwrnk:
                    maxwrnk = wrank
                dct[path] = [csize, minwrnk, maxwrnk]

        # collect dct from all procs so we can get the full min/max rank range
        alldcts = system.comm.gather(dct, root=0)

        if MPI.COMM_WORLD.rank == 0:
            alldata = {}
            sizes = {}
            for dct in alldcts:
                for path, v in dct.items():
                    csize, newmin, newmax = v
                    sizes[path] = csize
                    if path in alldata:
                        _, minwrnk, maxwrnk = alldata[path]
                        if newmin < minwrnk:
                            minwrnk = newmin
                        if newmax > maxwrnk:
                            maxwrnk = newmax
                        alldata[path] = [csize, minwrnk, maxwrnk]
                    else:
                        alldata[path] = [csize, newmin, newmax]

            table_data = []
            headers = ['Comm Size', 'COMM_WORLD Range(s)', 'System Pathname']
            for path, lst in sorted(alldata.items(), key=lambda x: (-x[1][0], x[0])):
                csize, minwrnk, maxwrnk = lst
                if verbose or path == '' or sizes[path.rpartition('.')[0]] != csize:
                    if minwrnk == maxwrnk:
                        rng = str(minwrnk)
                    else:
                        rng = f"{minwrnk} - {maxwrnk}"
                    table_data.append([csize, rng, path])

            col_meta = [
                {'align': 'center', 'header_align': 'center'},
                {'align': 'center', 'header_align': 'center'},
                {}
            ]
            if table_format != 'tabulator':
                col_meta[0]['max_width'] = 5
                col_meta[1]['max_width'] = 15

            probpath = system._problem_meta['pathname']
            if outfile is None:
                print(f"Printing comm info table for Problem '{probpath}'")

            outf = generate_table(table_data, headers=headers, column_meta=col_meta,
                                  tablefmt=table_format).display(outfile=outfile)

            if outf is not None and MPI.COMM_WORLD.rank == 0:
                print(f"comm info table for Problem '{probpath}' written to {outf}")
    else:
        if outfile is None:
            print("No MPI process info available.")
        else:
            with open(outfile, 'w') as f:
                print("No MPI process info available.", file=f)


def is_full_slice(range, inds):
    size = range[1] - range[0]
    inds = np.asarray(inds)
    if len(inds) > 1 and inds[0] == 0 and inds[-1] == size - 1:
        step = inds[1] - inds[0]
        diffs = inds[1:] - inds[:-1]
        return np.all(diffs == step)

    return len(inds) == 1 and inds[0] == 0


def show_dist_var_conns(group, rev=False, out_stream=_DEFAULT_OUT_STREAM):
    """
    Show all distributed variable connections in the given group and below.

    The ranks displayed will be relative to the communicator of the given top group.

    Parameters
    ----------
    group : Group
        The top level group to be searched.  Connections in all subgroups will also be displayed.
    rev : bool
        If True show reverse transfers instead of forward transfers.
    out_stream : file-like
        Where the output will go.

    Returns
    -------
    dict or None
        Dictionary containing the data for the connections. None is returned on all ranks except
        rank 0.
    """
    from openmdao.core.group import Group

    if out_stream is _DEFAULT_OUT_STREAM:
        out_stream = sys.stdout

    if out_stream is None:
        printer = lambda *args, **kwargs: None
    else:
        printer = print

    direction = 'rev' if rev else 'fwd'
    arrowdir = {'fwd': '->', 'rev': '<-'}
    arrow = arrowdir[direction]

    gdict = {}

    for g in group.system_iter(typ=Group, include_self=True):
        if g._transfers[direction]:
            in_ranges = list(g.dist_size_iter('input', group.comm))
            out_ranges = list(g.dist_size_iter('output', group.comm))

            inmapper = RangeMapper.create(in_ranges)
            outmapper = RangeMapper.create(out_ranges)

            gprint = False

            gprefix = g.pathname + '.' if g.pathname else ''
            skip = len(gprefix)

            for sub, transfer in g._transfers[direction].items():
                if sub is not None and (not isinstance(sub, tuple) or sub[0] is not None):
                    if not gprint:
                        gdict[g.pathname] = {}
                        gprint = True

                    conns = {}
                    for iidx, oidx in zip(transfer._in_inds, transfer._out_inds):
                        idata, irind = inmapper.index2key_rel(iidx)
                        ivar, irank = idata
                        odata, orind = outmapper.index2key_rel(oidx)
                        ovar, orank = odata

                        if odata not in conns:
                            conns[odata] = {}
                        if idata not in conns[odata]:
                            conns[odata][idata] = []
                        conns[odata][idata].append((orind, irind))

                    strs = {}  # use to avoid duplicate printouts for different ranks

                    for odata, odict in conns.items():
                        ovar, orank = odata
                        ovar = ovar[skip:]

                        for idata, dlist in odict.items():
                            ivar, irank = idata
                            ivar = ivar[skip:]
                            ranktup = (orank, irank)

                            oinds = [d[0] for d in dlist]
                            iinds = [d[1] for d in dlist]

                            orange = outmapper.key2range(odata)
                            irange = inmapper.key2range(idata)

                            if is_full_slice(orange, oinds) and is_full_slice(irange, iinds):
                                s = f"{ovar} {arrow} {ivar}"
                                if s not in strs:
                                    strs[s] = set()
                                strs[s].add(ranktup)
                                continue

                            for oidx, iidx in zip(oinds, iinds):
                                s = f"{ovar}[{oidx}] {arrow} {ivar}[{iidx}]"
                                if s not in strs:
                                    strs[s] = set()
                                strs[s].add(ranktup)

                    gdict[g.pathname][str(sub)] = strs

    do_ranks = False

    if group.comm.size > 1:
        do_ranks = True
        final = {}
        gatherlist = group.comm.gather(gdict, root=0)
        if group.comm.rank == 0:
            for dct in gatherlist:
                for gpath, subdct in dct.items():
                    if gpath not in final:
                        final[gpath] = subdct
                    else:
                        fgpath = final[gpath]
                        for sub, strs in subdct.items():
                            if sub not in fgpath:
                                fgpath[sub] = strs
                            else:
                                fgpathsub = fgpath[sub]
                                for s, ranks in strs.items():
                                    if s not in fgpathsub:
                                        fgpathsub[s] = ranks
                                    else:
                                        fgpathsub[s] |= ranks

        gdict = final

    if group.comm.rank == 0:
        fwd = direction == 'fwd'
        for gpath, subdct in sorted(gdict.items(), key=lambda x: x[0]):
            indent = 0 if gpath == '' else gpath.count('.') + 1
            pad = '   ' * indent
            printer(f"{pad}In Group '{gpath}'", file=out_stream)
            for sub, strs in sorted(subdct.items(), key=lambda x: x[0]):
                if fwd:
                    printer(f"{pad}   {arrow} {sub}", file=out_stream)
                else:
                    printer(f"{pad}   {sub} {arrow}", file=out_stream)
                for s, ranks in strs.items():
                    if do_ranks:
                        oranks = np.empty(len(ranks), dtype=int)
                        iranks = np.empty(len(ranks), dtype=int)
                        for i, (ornk, irnk) in enumerate(sorted(ranks)):
                            oranks[i] = ornk
                            iranks[i] = irnk

                        if np.all(oranks == oranks[0]):
                            orstr = str(oranks[0])
                        else:
                            sorted_ranks = sorted(oranks)
                            orstr = str(sorted_ranks)
                            if len(sorted_ranks) > 3:
                                for j, r in enumerate(sorted_ranks):
                                    if j == 0 or r - val == 1:
                                        val = r
                                    else:
                                        break
                                else:
                                    orstr = f"[{sorted_ranks[0]} to {sorted_ranks[-1]}]"

                        if np.all(iranks == iranks[0]):
                            irstr = str(iranks[0])
                        else:
                            sorted_ranks = sorted(iranks)
                            irstr = str(sorted(iranks))
                            if len(sorted_ranks) > 3:
                                for j, r in enumerate(sorted_ranks):
                                    if j == 0 or r - val == 1:
                                        val = r
                                    else:
                                        break
                                else:
                                    irstr = f"[{sorted_ranks[0]} to {sorted_ranks[-1]}]"

                        if orstr == irstr and '[' not in orstr:
                            printer(f"{pad}      {s}    rank {orstr}", file=out_stream)
                        else:
                            printer(f"{pad}      {s}    ranks {orstr} {arrow} {irstr}", file=out_stream)
                    else:
                        printer(f"{pad}      {s}", file=out_stream)

        return gdict


def _dist_conns_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao dist_conns' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', default=None, action='store', dest='outfile',
                        help='Name of output file.  By default, output goes to stdout.')
    parser.add_argument('-r', '--rev', action='store_true', dest='rev',
                        help='If set, use "rev" transfer direction instead of "fwd".')
    parser.add_argument('-p', '--problem', action='store', dest='problem', help='Problem name')


def _dist_conns_cmd(options, user_args):
    """
    Run the `openmdao dist_conns` command.

    The command shows connections, with relative indexing information, between all
    variables in the model across all MPI processes.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    import openmdao.utils.hooks as hooks

    def _dist_conns(prob):
        model = prob.model
        if options.problem:
            if model._problem_meta['name'] != options.problem and \
                    model._problem_meta['pathname'] != options.problem:
                return
        elif '/' in model._problem_meta['pathname']:
            # by default, only display comm info for a top level problem
            return

        if options.outfile is None:
            out_stream = sys.stdout
        else:
            out_stream = open(options.outfile, 'w')

        try:
            show_dist_var_conns(model, rev=options.rev, out_stream=out_stream)
        finally:
            if out_stream is not sys.stdout:
                out_stream.close()

        exit()

    # register the hook to be called right after final_setup on the problem
    hooks._register_hook('final_setup', 'Problem', post=_dist_conns)

    _load_and_exec(options.file[0], user_args)
