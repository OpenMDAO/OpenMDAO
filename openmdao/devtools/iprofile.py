from __future__ import print_function

import os
import sys
from timeit import default_timer as etime
import argparse
import json
import atexit
from collections import defaultdict
from itertools import chain

from six import iteritems

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from openmdao.devtools.webview import webview
from openmdao.devtools.iprof_utils import func_group, find_qualified_name, _collect_methods


def _prof_node(fpath, parts):
    pathparts = fpath.split('-')
    return {
        'id': fpath,
        'time': parts[1],
        'count': parts[2],
        'tot_time': 0.,
        'tot_count': 0,
        'obj': parts[0],
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


def setup(prefix='iprof', methods=None, prof_dir=None, finalize=True):
    """
    Instruments certain important openmdao methods for profiling.

    Parameters
    ----------

    prefix : str ('iprof')
        Prefix used for the raw profile data. Process rank will be appended
        to it to get the actual filename.  When not using MPI, rank=0.

    methods : list, optional
        A list of tuples of profiled methods to override the default set.  The first
        entry is the method name or glob pattern and the second is a tuple of class
        objects used for isinstance checking.  The default set of methods is:

        ::

            [
                "*": (System, Jacobian, Matrix, Solver, Driver, Problem),
            ]

    prof_dir : str
        Directory where the profile files will be written. Defaults to the
        current directory.

    finallize : bool
        If True, register a function to finalize the profile before exit.

    """

    global _profile_prefix, _matches
    global _profile_setup, _profile_total, _profile_out

    if _profile_setup:
        raise RuntimeError("profiling is already set up.")

    if prof_dir is None:
        _profile_prefix = os.path.join(os.getcwd(), prefix)
    else:
        _profile_prefix = os.path.join(os.path.abspath(prof_dir), prefix)

    _profile_setup = True

    if methods is None:
        methods = func_group['openmdao']

    rank = MPI.COMM_WORLD.rank if MPI else 0
    _profile_out = open("%s.%s" % (_profile_prefix, rank), 'wb')

    if finalize:
        atexit.register(_finalize_profile)

    _matches = _collect_methods(methods)


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
            path = '-'.join(s[0] for s in _call_stack)
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
    for funcpath, data in iteritems(_inst_data):
        _inst_data[funcpath] = data = _prof_node(funcpath, data)
        parts = funcpath.rsplit('-', 1)
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
            _obj_map[fname] = '.'.join((name, "<%s.%s>" % (qclass, qname)))

    _obj_map['$total'] = '$total'

    rank = MPI.COMM_WORLD.rank if MPI else 0

    fname = os.path.basename(_profile_prefix)
    with open("%s.%d" % (fname, rank), 'w') as f:
        for name, data in iteritems(_inst_data):
            new_name = '-'.join([_obj_map[s] for s in name.split('-')])
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
    totals = {}

    tree_nodes = {}
    tree_parts = []

    for fname in flist:
        ext = os.path.splitext(fname)[1]
        try:
            int(ext.lstrip('.'))
            dec = ext
        except:
            dec = False

        for funcpath, count, t in _iter_raw_prof_file(fname):

            parts = funcpath.split('-')

            # for multi-file MPI profiles, decorate names with the rank
            if nfiles > 1 and dec:
                parts = ["%s%s" % (p,dec) for p in parts]
                funcpath = '-'.join(parts)

            tree_nodes[funcpath] = node = _prof_node(funcpath, [None, t, count])

            funcname = parts[-1]

            if funcname in totals:
                tnode = totals[funcname]
            else:
                totals[funcname] = tnode = _prof_node(funcpath, [None, 0., 0])

            tnode['tot_time'] += t
            tnode['tot_count'] += count

            tree_parts.append((parts, node))

    for parts, node in tree_parts:
        short = parts[-1]
        node['tot_time'] = totals[short]['tot_time']
        node['tot_count'] = totals[short]['tot_count']
        del node['obj']

    tree_nodes['$total']['tot_time'] = tree_nodes['$total']['time']

    return tree_nodes, totals


def _prof_totals():
    """
    Called from the command line to create a file containing total elapsed
    times and number of calls for all profiled functions.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', action='store', dest='outfile',
                        metavar='OUTFILE', default='sys.stdout',
                        help='Name of file containing function total counts and elapsed times.')
    parser.add_argument('-g', '--group', action='store', dest='group',
                        default='openmdao',
                        help='Determines which group of methods will be tracked.')
    parser.add_argument('-m', '--maxcalls', action='store', dest='maxcalls', type=int,
                        default=999999,
                        help='Max number of results to display.')
    parser.add_argument('files', metavar='file', nargs='*',
                        help='Raw profile data files or a python file.')

    options = parser.parse_args()

    if not options.files:
        print("No files to process.")
        sys.exit(0)

    if options.outfile == 'sys.stdout':
        out_stream = sys.stdout
    else:
        out_stream = open(options.outfile, 'w')

    if options.files[0].endswith('.py'):
        if len(options.files) > 1:
            print("iprofview can only process a single python file.", file=sys.stderr)
            sys.exit(-1)
        _profile_py_file(options.files[0], methods=func_group[options.group])
        options.files = ['iprof.0']

    call_data, totals = _process_profile(options.files)

    total_time = totals['$total']['tot_time']

    try:

        out_stream.write("\nTotal     Total           Function\n")
        out_stream.write("Calls     Time (s)    %   Name\n")

        calls = sorted(totals.items(), key=lambda x : x[1]['tot_time'])
        for func, data in calls[-options.maxcalls:]:
            out_stream.write("%6d %11f %6.2f %s\n" %
                               (data['tot_count'], data['tot_time'], (data['tot_time']/total_time*100.), func))
    finally:
        if out_stream is not sys.stdout:
            out_stream.close()


def _profile_py_file(fname=None, methods=None):
    """
    Run instance-based profiling on the given python script.

    Parameters
    ----------
    fname : str
        Name of the python script.
    methods : list of (glob, (classes...)) tuples or None
        List indicating which methods to track.
    """
    if fname is None:
        args = sys.argv[1:]
        if not args:
            print("No files to process.", file=sys.stderr)
            sys.exit(2)
        fname = args[0]
    sys.path.insert(0, os.path.dirname(fname))

    with open(fname, 'rb') as fp:
        code = compile(fp.read(), fname, 'exec')

    globals_dict = {
        '__file__': fname,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }

    setup(methods=methods, finalize=False)
    start()
    exec (code, globals_dict)
    _finalize_profile()
