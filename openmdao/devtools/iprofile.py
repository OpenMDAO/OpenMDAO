from __future__ import print_function

import os
import sys
from time import time as etime
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


def _prof_node(parts):
    return {
        'name': parts[0],
        'short_name': parts[0].rsplit('@', 1)[-1],
        'time': parts[2],
        'count': parts[3],
        'tot_time': 0.,
        'tot_count': 0,
        'pct_total': 0.,
        'tot_pct_total': 0.,
        'pct_parent': 0.,
        'child_time': 0.,
        'obj': parts[1],
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

    Args
    ----

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
    _profile_out = open("%s.%d" % (_profile_prefix, rank), 'wb')

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
        _inst_data['$total'] = ['$total', None, 0., 0]

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
    _inst_data['$total'][2] = _profile_total
    _inst_data['$total'][3] += 1
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
            path = '@'.join(s[0] for s in _call_stack)
            if path not in _inst_data:
                _inst_data[path] = [path, frame.f_locals['self'], 0., 0]

            _call_stack.pop()

            pdata = _inst_data[path]
            pdata[2] += final - start
            pdata[3] += 1


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
        _inst_data[funcpath] = _prof_node(data)
        data = _inst_data[funcpath]
        parts = funcpath.rsplit('@', 1)
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
    _obj_map['$parent'] = '$parent'

    # compute child times
    for funcpath, data in iteritems(_inst_data):
        parts = funcpath.rsplit('@', 1)
        if len(parts) > 1:
            _inst_data[parts[0]]['child_time'] += data['time']

    # in order to make the D3 partition layout give accurate proportions, we can only put values
    # into leaf nodes because the parent node values get overridden by the sum of the children. To
    # get around this, we create a child for each non-leaf node with the name '$parent' and put the
    # time exclusive to the parent into that child, so that when all of the children are summed, they'll
    # add up to the correct time for the parent and the visual proportions of the parent will be correct.

    # compute child timings
    parnodes = []
    for funcpath, node in iteritems(_inst_data):
        if node['child_time'] > 0.:
            parts = funcpath.split('@')
            pparts = parts + ['$parent']
            chname = '@'.join(pparts)
            ex_child_node = _prof_node([chname, None, node['time'] - node['child_time'], 1])
            parnodes.append((chname, ex_child_node))

    rank = MPI.COMM_WORLD.rank if MPI else 0

    fname = os.path.basename(_profile_prefix)
    with open("%s.%d" % (fname, rank), 'w') as f:
        for name, data in chain(iteritems(_inst_data), parnodes):
            new_name = '@'.join([_obj_map[s] for s in name.split('@')])
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


def process_profile(flist):
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

    for fname in flist:
        ext = os.path.splitext(fname)[1]
        try:
            int(ext.lstrip('.'))
            dec = ext
        except:
            dec = False

        for funcpath, count, t in _iter_raw_prof_file(fname):

            parts = funcpath.split('@')

            # for multi-file MPI profiles, decorate names with the rank
            if nfiles > 1 and dec:
                parts = ["%s%s" % (p,dec) for p in parts]
                funcpath = '@'.join(parts)

            tree_nodes[funcpath] = node = _prof_node([funcpath, None, t, count])

            funcname = parts[-1]
            if funcname == '$parent':
                continue

            if funcname in totals:
                tnode = totals[funcname]
            else:
                totals[funcname] = tnode = _prof_node([funcpath, None, 0., 0])

            tnode['tot_time'] += t
            tnode['tot_count'] += count

    for funcpath, node in iteritems(tree_nodes):
        parts = funcpath.rsplit('@', 1)
        if parts[-1] != '$parent':
            node['tot_time'] = totals[parts[-1]]['tot_time']
            node['tot_count'] = totals[parts[-1]]['tot_count']
            node['pct_parent'] = node['time'] / tree_nodes[parts[0]]['time']
            node['pct_total'] = node['time'] / tree_nodes['$total']['time']
            node['tot_pct_total'] = totals[parts[-1]]['tot_time'] / tree_nodes['$total']['time']
        del node['obj']
        del node['child_time']

    tree_nodes['$total']['tot_time'] = tree_nodes['$total']['time']

    for funcpath, node in iteritems(tree_nodes):
        parts = funcpath.rsplit('@', 1)
        # D3 sums up all children to get parent value, so we need to
        # zero out the parent value else we get double the value we want
        # once we add in all of the times from descendants.
        if parts[-1] == '$parent':
            tree_nodes[parts[0]]['time'] = 0.

    return list(tree_nodes.values()), totals


def prof_totals():
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
    parser.add_argument('files', metavar='file', nargs='*',
                        help='Raw profile data files or a python file.')

    #TODO: add arg to set max number of results (starting at largest)

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
        profile_py_file(options.files[0], methods=func_group[options.group])
        options.files = ['iprof.0']

    _, totals = process_profile(options.files)

    total_time = totals['$total']['tot_time']

    try:

        out_stream.write("\nTotal     Total           Function\n")
        out_stream.write("Calls     Time (s)    %   Name\n")

        for func, data in sorted([(k,v) for k,v in iteritems(totals)],
                                    key=lambda x:x[1]['tot_time']):
            out_stream.write("%6d %11f %6.2f %s\n" %
                               (data['tot_count'], data['tot_time'], (data['tot_time']/total_time*100.), func))
    finally:
        if out_stream is not sys.stdout:
            out_stream.close()


def prof_view():
    """
    Called from a command line to generate an html viewer for profile data.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--noshow', action='store_true', dest='noshow',
                        help="Don't pop up a browser to view the data.")
    parser.add_argument('-t', '--title', action='store', dest='title',
                        default='Profile of Method Calls by Instance',
                        help='Title to be displayed above profiling view.')
    parser.add_argument('-g', '--group', action='store', dest='group',
                        default='openmdao',
                        help='Determines which group of methods will be tracked. Current '
                             'options are: %s and "openmdao" is the default' %
                              sorted(func_group.keys()))
    parser.add_argument('files', metavar='file', nargs='+',
                        help='Raw profile data files or a python file.')

    options = parser.parse_args()

    if options.files[0].endswith('.py'):
        if len(options.files) > 1:
            print("iprofview can only process a single python file.", file=sys.stderr)
            sys.exit(-1)
        profile_py_file(options.files[0], methods=func_group[options.group])
        options.files = ['iprof.0']

    call_graph, _ = process_profile(options.files)

    viewer = "icicle.html"
    code_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(code_dir, viewer), "r") as f:
        template = f.read()

    graphjson = json.dumps(call_graph)

    outfile = 'profile_' + viewer
    with open(outfile, 'w') as f:
        template = template.replace('$call_graph_data', graphjson)
        f.write(template.replace('$title', options.title))

    if not options.noshow:
        webview(outfile)


def profile_py_file(fname=None, methods=None):
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
