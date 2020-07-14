
import os
import sys
from timeit import default_timer as etime
import argparse
import json
import atexit
from collections import defaultdict
from itertools import chain

from openmdao.utils.mpi import MPI

from openmdao.utils.webview import webview
from openmdao.devtools.iprof_utils import func_group, find_qualified_name, _collect_methods, \
     _setup_func_group, _get_methods, _Options


def _prof_node(fpath, parts):
    pathparts = fpath.split('|')
    obj, etime, count = parts

    return {
        'id': fpath,
        'time': etime,
        'count': count,
        'tot_time': 0.,
        'tot_count': 0,
        'obj': obj,
        'depth': len(pathparts) - 1,
    }

_profile_prefix = None
_profile_out = None
_profile_start = None
_profile_setup = False
_profile_total = 0.0
_matches = {}
_call_stack = []
_inst_data = {}


def _setup(options, finalize=True):

    global _profile_prefix, _matches
    global _profile_setup, _profile_total, _profile_out

    if _profile_setup:
        raise RuntimeError("profiling is already set up.")

    _profile_prefix = os.path.join(os.getcwd(), 'iprof')
    _profile_setup = True

    methods = _get_methods(options, default='openmdao')

    rank = MPI.COMM_WORLD.rank if MPI else 0
    _profile_out = open("%s.%s" % (_profile_prefix, rank), 'wb')

    if finalize:
        atexit.register(_finalize_profile)

    _matches = _collect_methods(methods)


def setup(methods=None, finalize=True):
    """
    Instruments certain important openmdao methods for profiling.

    Parameters
    ----------

    methods : list, optional
        A list of tuples of profiled methods to override the default set.  The first
        entry is the method name or glob pattern and the second is a tuple of class
        objects used for isinstance checking.  The default set of methods is:

        .. code-block:: python

            [
                "*": (System, Jacobian, Matrix, Solver, Driver, Problem),
            ]

    finalize : bool
        If True, register a function to finalize the profile before exit.

    """
    if not func_group:
        _setup_func_group()
    _setup(_Options(methods=methods), finalize=finalize)


def start():
    """
    Turn on profiling.
    """
    global _profile_start, _profile_setup, _call_stack, _inst_data
    if _profile_start is not None:
        print("profiling is already active.")
        return

    if not _profile_setup:
        setup()  # just do a default setup

    _profile_start = etime()
    _call_stack.append(('$total', _profile_start, None))
    if '$total' not in _inst_data:
        _inst_data['$total'] = [None, 0., 0]

    if sys.getprofile() is not None:
        raise RuntimeError("another profile function is already active.")
    sys.setprofile(_instance_profile_callback)


def stop():
    """
    Turn off profiling.
    """
    global _profile_total, _profile_start, _call_stack, _inst_data
    if _profile_start is None:
        return

    sys.setprofile(None)

    _call_stack.pop()

    _profile_total += (etime() - _profile_start)
    _inst_data['$total'][1] = _profile_total
    _inst_data['$total'][2] += 1
    _profile_start = None


def _instance_profile_callback(frame, event, arg):
    """
    Collects profile data for functions that match _matches and pass the isinstance check.

    Elapsed time and number of calls are collected.
    """
    global _call_stack, _inst_data, _matches

    if event == 'call':
        if 'self' in frame.f_locals and frame.f_code.co_name in _matches and \
                isinstance(frame.f_locals['self'], _matches[frame.f_code.co_name]):
            _call_stack.append(("%s#%d#%d" % (frame.f_code.co_filename,
                                              frame.f_code.co_firstlineno,
                                              id(frame.f_locals['self'])),
                                etime(), frame))
    elif event == 'return' and _call_stack:
        _, start, oldframe = _call_stack[-1]
        if oldframe is frame:
            final = etime()
            path = '|'.join(s[0] for s in _call_stack)
            if path not in _inst_data:
                _inst_data[path] = pdata = [frame.f_locals['self'], 0., 0]
            else:
                pdata = _inst_data[path]
            pdata[1] += final - start
            pdata[2] += 1

            _call_stack.pop()


def _finalize_profile():
    """
    Called at exit to write out the profiling data.
    """
    global _profile_prefix, _profile_total, _inst_data

    stop()

    # fix names in _inst_data
    _obj_map = {}
    cache = {}
    idents = defaultdict(dict)  # map idents to a smaller number
    for funcpath, data in _inst_data.items():
        _inst_data[funcpath] = data = _prof_node(funcpath, data)
        parts = funcpath.rsplit('|', 1)
        fname = parts[-1]
        if fname == '$total':
            continue
        filename, line, ident = fname.split('#')
        qfile, qclass, qname = find_qualified_name(filename, int(line), cache, full=False)

        idict = idents[(qfile, qclass)]
        if ident not in idict:
            idict[ident] = len(idict)
        ident = idict[ident] + 1  # so we'll agree with ident scheme in other tracing/profiling functions

        try:
            name = data['obj'].pathname
        except AttributeError:
            if qfile is None:
                _obj_map[fname] = "<%s#%d.%s>" % (qclass, ident, qname)
            else:
                _obj_map[fname] = "<%s:%d.%s>" % (qfile, line, qname)
        else:
            if name is None:
                name = '%s#%d' % (qclass, ident)
            _obj_map[fname] = '.'.join((name, "<%s.%s>" % (qclass, qname)))

    _obj_map['$total'] = '$total'

    rank = MPI.COMM_WORLD.rank if MPI else 0

    fname = os.path.basename(_profile_prefix)
    with open("%s.%d" % (fname, rank), 'w') as f:
        for name, data in _inst_data.items():
            new_name = '|'.join([_obj_map[s] for s in name.split('|')])
            f.write("%s %d %f\n" % (new_name, data['count'], data['time']))


def _iter_raw_prof_file(rawname):
    """
    Returns an iterator of (funcpath, count, elapsed_time)
    from a raw profile data file.
    """
    with open(rawname, 'r') as f:
        for line in f:
            path, count, elapsed = line.split()
            yield path, int(count), float(elapsed)


def _process_1_profile(fname):
    """
    Take the generated raw profile data, potentially from multiple files,
    and combine it to get execution counts and timing data.

    Parameters
    ----------

    flist : list of str
        Names of raw profiling data files.

    """

    totals = {}
    tree_nodes = {}
    tree_parts = []

    for funcpath, count, t in _iter_raw_prof_file(fname):
        parts = funcpath.split('|')

        tree_nodes[funcpath] = node = _prof_node(funcpath, [None, t, count])

        funcname = parts[-1]

        if funcname not in totals:
            totals[funcname] = [0., 0]

        totals[funcname][0] += t
        totals[funcname][1] += count

        tree_parts.append((parts, node))

    for parts, node in tree_parts:
        short = parts[-1]
        node['tot_time'] = totals[short][0]
        node['tot_count'] = totals[short][1]
        del node['obj']

    tree_nodes['$total']['tot_time'] = tree_nodes['$total']['time']

    return tree_nodes, totals


def _process_profile(flist):
    """
    Take the generated raw profile data, potentially from multiple files,
    and combine it to get execution counts and timing data.

    Parameters
    ----------

    flist : list of str
        Names of raw profiling data files.

    """

    nfiles = len(flist)
    top_nodes = []
    top_totals = []

    if nfiles == 1:
        return _process_1_profile(flist[0])

    for fname in sorted(flist):
        ext = os.path.splitext(fname)[1]
        try:
            int(ext.lstrip('.'))
            dec = ext
            tot_names.append('$total' + dec)
        except:
            dec = None

        nodes, tots = _process_1_profile(fname)
        top_nodes.append(nodes)
        top_totals.append(tots)

    tree_nodes = {}
    grand_total = _prof_node('$total', [None, 0., 1])
    grand_total['tot_count'] = 1

    for i, nodes in enumerate(top_nodes):
        grand_total['tot_time'] += nodes['$total']['tot_time']
        grand_total['time'] += nodes['$total']['time']

        for name, node in nodes.items():
            newname = _fix_name(name, i)
            node['id'] = newname
            node['depth'] += 1
            tree_nodes[newname] = node

    tree_nodes['$total'] = grand_total

    totals = {}
    tot_names = []
    for i, tot in enumerate(top_totals):
        tot_names.append('$total.%d' % i)
        for name, tots in tot.items():
            if name == '$total':
                totals[tot_names[-1]] = tots
            else:
                totals[name] = tots

    totals['$total'] = [0., 0]
    for tname in tot_names:
        totals['$total'][0] += totals[tname][0]
        totals['$total'][1] += totals[tname][1]

    return tree_nodes, totals


def _fix_name(name, i):
    parts = name.split('|')
    parts[0] = '$total.%d' % i
    return '|'.join(['$total'] + parts)


def _iprof_totals_setup_parser(parser):
    parser.add_argument('-o', '--outfile', action='store', dest='outfile',
                        metavar='OUTFILE', default='sys.stdout',
                        help='Name of file containing function total counts and elapsed times.')
    parser.add_argument('-g', '--group', action='store', dest='methods',
                        default='openmdao',
                        help='Determines which group of methods will be tracked.')
    parser.add_argument('-m', '--maxcalls', action='store', dest='maxcalls', type=int,
                        default=999999,
                        help='Max number of results to display.')
    parser.add_argument('file', metavar='file', nargs='*',
                        help='Raw profile data files or a python file.')


def _iprof_totals_exec(options, user_args):
    """
    Called from the command line (openmdao prof_totals command) to create a file containing total
    elapsed times and number of calls for all profiled functions.
    """

    if not options.file:
        print("No files to process.")
        sys.exit(0)

    if options.outfile == 'sys.stdout':
        out_stream = sys.stdout
    else:
        out_stream = open(options.outfile, 'w')

    if options.file[0].endswith('.py'):
        if len(options.file) > 1:
            print("iprofview can only process a single python file.", file=sys.stderr)
            sys.exit(-1)
        _iprof_py_file(options, user_args)
        if MPI:
            options.file = ['iprof.%d' % i for i in range(MPI.COMM_WORLD.size)]
        else:
            options.file = ['iprof.0']

    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    call_data, totals = _process_profile(options.file)

    total_time = call_data['$total']['tot_time']

    try:

        out_stream.write("\nTotal     Total           Function\n")
        out_stream.write("Calls     Time (s)    %   Name\n")

        calls = sorted(totals.items(), key=lambda x : x[1][0])
        for func, data in calls[-options.maxcalls:]:
            out_stream.write("%6d %11f %6.2f %s\n" %
                               (data[1], data[0],
                               (data[0]/total_time*100.), func))
    finally:
        if out_stream is not sys.stdout:
            out_stream.close()


def _iprof_py_file(options, user_args):
    """
    Run instance-based profiling on the given python script.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Command line options after '--' (if any).  Passed to user script.
    """
    if not func_group:
        _setup_func_group()

    progname = options.file[0]
    sys.path.insert(0, os.path.dirname(progname))

    # update sys.argv in case python script takes cmd line args
    sys.argv[:] = [progname] + user_args

    with open(progname, 'rb') as fp:
        code = compile(fp.read(), progname, 'exec')

    globals_dict = {
        '__file__': progname,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }

    _setup(options, finalize=False)
    start()
    exec (code, globals_dict)
    _finalize_profile()
