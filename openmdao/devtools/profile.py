from __future__ import print_function

import os
import sys
from time import time as etime
from inspect import getmembers, getmro
from fnmatch import fnmatchcase
import argparse
import json
import atexit
import types
from string import Template
from collections import OrderedDict
from functools import wraps
from struct import Struct
from ctypes import Structure, c_uint, c_float
from types import MethodType

from six import iteritems

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from openmdao.devtools.webview import webview

def get_method_class(meth):
    """Return the class that actually defined the given method."""
    for cls in getmro(meth.__self__.__class__):
        if meth.__name__ in cls.__dict__:
            return cls


class _ProfData(Structure):
    _fields_ = [ ('t',c_float), ('ovr',c_float), ('tstamp',c_float), ('id',c_uint) ]

_profile_methods = None
_profile_prefix = None
_profile_out = None
_profile_start = None
_profile_setup = False
_profile_total = 0.0
_profile_struct = _ProfData()
_profile_funcs_dict = OrderedDict()
_profile_matches = {}
_call_stack = []
_timing_stack = []


def setup(prefix='prof_raw', methods=None, prof_dir=None):
    """
    Instruments certain important openmdao methods for profiling.

    Args
    ----

    prefix : str ('prof_raw')
        Prefix used for the raw profile data. Process rank will be appended
        to it to get the actual filename.  When not using MPI, rank=0.

    methods : dict, optional
        A dict of profiled methods to override the default set.  The key
        is the method name or glob pattern and the value is a tuple of class
        objects used for isinstance checking.  The default set of methods is:

        ::

            {
                "*": (System, Jacobian, Matrix, Solver, Driver, Problem),
            }

    prof_dir : str
        Directory where the profile files will be written.

    """

    from openmdao.core.problem import Problem
    from openmdao.core.system import System
    from openmdao.core.driver import Driver
    from openmdao.solvers.solver import Solver
    from openmdao.jacobians.jacobian import Jacobian
    from openmdao.matrices.matrix import Matrix

    global _profile_prefix, _profile_methods, _profile_matches
    global _profile_setup, _profile_total, _profile_out

    if _profile_setup:
        raise RuntimeError("profiling is already set up.")

    if prof_dir is None:
        _profile_prefix = os.path.join(os.getcwd(), prefix)
    else:
        _profile_prefix = os.path.join(os.path.abspath(prof_dir), prefix)

    _profile_setup = True

    if methods is None:
        _profile_methods = {
            "*": (System, Jacobian, Matrix, Solver, Driver, Problem),
        }
    else:
        _profile_methods = methods

    rank = MPI.COMM_WORLD.rank if MPI else 0
    _profile_out = open("%s.%d" % (_profile_prefix, rank), 'wb')

    atexit.register(_finalize_profile)

    _profile_matches = _collect_methods(_profile_methods)


def _collect_methods(method_dict):
    """
    Iterate over a dict of method name patterns mapped to classes.  Search
    through the classes for anything that matches and return a dict of
    exact name matches and their correspoding classes.

    Parameters
    ----------
    method_dict : {pattern1: classes1, ... pattern_n: classes_n}
        Dict of glob patterns mapped to lists of classes used for isinstance checks

    Returns
    -------
    dict
        Dict of method names and tuples of all classes that matched for that method.
    """
    matches = {}
    for pattern, classes in iteritems(method_dict):
        for class_ in classes:
            for name, obj in getmembers(class_):
                if callable(obj) and (pattern == '*' or fnmatchcase(name, pattern)):
                    if name in matches:
                        matches[name].append(class_)
                    else:
                        matches[name] = [class_]

    # convert values to tuples so we can use in isinstance call
    for name in matches:
        matches[name] = tuple(matches[name])

    return matches

def _instance_profile(frame, event, arg):
    """
    Collects profile data for functions that match _profile_matches.
    The data collected will include time elapsed, number of calls, ...
    """
    global _call_stack, _profile_out, _profile_struct, \
           _profile_funcs_dict, _profile_start, _profile_matches

    if event == 'call':
        func_name = frame.f_code.co_name
        if func_name in _profile_matches:
            loc = frame.f_locals
            if 'self' in loc:
                self = loc['self']
                if isinstance(self, _profile_matches[func_name]):
                    if 'pathname' in self.__dict__:
                        name = '.'.join([self.pathname, func_name])
                    else:
                        name = "<%s_%d>.%s" % (type(self).__name__, id(self), func_name)
                    _call_stack.append(name)
                    _timing_stack.append(etime())

    elif event == 'return':
        func_name = frame.f_code.co_name
        if func_name in _profile_matches:
            loc = frame.f_locals
            if 'self' in loc:
                self = loc['self']
                if isinstance(self, _profile_matches[func_name]):
                    path = ','.join(_call_stack)
                    if path not in _profile_funcs_dict:
                        # save the id for this path
                        _profile_funcs_dict[path] = len(_profile_funcs_dict)

                    _call_stack.pop()
                    start = _timing_stack.pop()

                    _profile_struct.t = etime() - start
                    _profile_struct.ovr = 0. #start - ovr # keep track of overhead for later subtraction
                    _profile_struct.tstamp = start
                    _profile_struct.id = _profile_funcs_dict[path]
                    _profile_out.write(_profile_struct)


def start():
    """Turn on profiling.
    """
    global _profile_start, _profile_setup
    if _profile_start is not None:
        print("profiling is already active.")
        return

    if not _profile_setup:
        setup()  # just do a default setup

    _profile_start = etime()

    sys.setprofile(_instance_profile)

def stop():
    """Turn off profiling.
    """
    global _profile_total, _profile_start
    if _profile_start is None:
        return

    sys.setprofile(None)

    _profile_total += (etime() - _profile_start)
    _profile_start = None

def _iter_raw_prof_file(rawname, fdict=None):
    """Returns an iterator of (elapsed_time, timestamp, funcpath)
    from a raw profile data file.
    """
    global _profile_struct

    if fdict is None:
        fdict = {}

    fn, ext = os.path.splitext(rawname)
    dname = os.path.dirname(rawname)
    fname = os.path.basename(fn)
    funcs_fname = os.path.join(dname, "funcs_" + fname + ext)

    with open(funcs_fname, 'r') as f:
        for line in f:
            line = line.strip()
            path, ident = line.split(' ')
            fdict[ident] = path

    with open(rawname, 'rb') as f:
        while f.readinto(_profile_struct):
            path = fdict[str(_profile_struct.id)]
            yield _profile_struct.t, _profile_struct.ovr, _profile_struct.tstamp, path

def _finalize_profile():
    """called at exit to write out the file mapping function call paths
    to identifiers.
    """
    global _profile_prefix, _profile_funcs_dict, _profile_total

    stop()

    rank = MPI.COMM_WORLD.rank if MPI else 0

    dname = os.path.dirname(_profile_prefix)
    fname = os.path.basename(_profile_prefix)
    funcs_fname = os.path.join(dname, "funcs_" + fname)

    with open("%s.%d" % (funcs_fname, rank), 'w') as f:
        for name, ident in iteritems(_profile_funcs_dict):
            f.write("%s %s\n" % (name, ident))
        # also write out the total time so that we can report how much of
        # the runtime is invisible to our profile.
        f.write("%s %s\n" % (_profile_total, "@total"))


def _update_counts(dct, name, elapsed, overhead):
    try:
        d = dct[name]
    except KeyError:
        dct[name] = d = {
                'count': 0,
                'time': 0.,
                'ovr': 0.,
            }

    d['count'] += 1
    d['time'] += elapsed
    d['ovr'] += overhead

def process_profile(flist):
    """Take the generated raw profile data, potentially from multiple files,
    and combine it to get hierarchy structure and total execution counts and
    timing data.

    Args
    ----

    flist : list of str
        Names of raw profiling data files.

    """

    nfiles = len(flist)
    proc_trees = []
    funcs = {}
    totals = {}
    total_under_profile = 0.0
    tops = set()

    for fname in flist:
        fdict = {}

        ext = os.path.splitext(fname)[1]
        try:
            extval = int(ext.lstrip('.'))
            dec = ext
        except:
            dec = False

        for t, ovr, tstamp, funcpath in _iter_raw_prof_file(fname, fdict):
            parts = funcpath.split(',')

            # for multi-file MPI profiles, decorate names with the rank
            if nfiles > 1 and dec:
                parts = ["%s%s" % (p,dec) for p in parts]
                funcpath = ','.join(parts)

            elapsed = float(t)
            overhead = float(ovr)

            _update_counts(funcs, funcpath, elapsed, overhead)

            stack = parts[:-1]
            if not stack:
                tops.add(funcpath)

        total_under_profile += float(fdict['@total'])

    tree = {
        'name': '.', # this name has to be '.' and not '', else we have issues
                     # when combining multiple files due to sort order
        'time': 0.,
        'ovr': 0.,
        # keep track of total time under profiling, so that we
        # can see if there is some time that isn't accounted for by the
        # functions we've chosen to profile.
        'tot_time': total_under_profile,
        'count': 1,
        'tot_count': 1,
        'children': [],
    }

    tmp = {} # just for temporary lookup of objects

    for path, fdict in sorted(iteritems(funcs)):
        parts = path.split(',')

        dct = {
            'name': parts[-1],
            'children': [],
            'time': fdict['time'],
            'ovr': fdict['ovr'],
            'tot_time': 0.,
            'count': fdict['count'],
            'tot_count': 0,
        }

        tmp[path] = dct

        if path in tops:
            tree['children'].append(dct)
            tree['time'] += dct['time']
            tree['ovr'] += dct['ovr']
        else:
            caller = ','.join(parts[:-1])
            tmp[caller]['children'].append(dct)

    # sum up the child overheads and subtract from the elapsed time for each
    # function
    def remove_child_ovr(node):
        kids = node['children']
        total = 0.
        if kids:
            for k in kids:
                total += remove_child_ovr(k)
            node['time'] -= total
        return total + node['ovr']

    remove_child_ovr(tree)

    # now that func times are corrected, calculate totals
    for path, dct in iteritems(tmp):
        name = path.split(',')[-1]
        if name in totals:
            totals[name]['time'] += dct['time']
            totals[name]['count'] += dct['count']
        else:
            totals[name] = { 'time': dct['time'], 'count': dct['count'] }

    for path, dct in iteritems(tmp):
        name = path.split(',')[-1]
        dct['tot_time'] = totals[name]['time']
        dct['tot_count'] = totals[name]['count']

    return tree, totals

def prof_dump(fname=None, include_tstamp=True):
    """Print the contents of the given raw profile data file to stdout.

    Args
    ----

    fname : str
        Name of raw profile data file.

    include_tstamp : bool (True)
        If True, include the timestamp in the dump.
    """

    if fname is None:
        fname = sys.argv[1]

    if include_tstamp:
        for t, ovr, tstamp, funcpath in _iter_raw_prof_file(fname):
            print(funcpath, t, ovr, tstamp)
    else:
        for t, ovr, _, funcpath in _iter_raw_prof_file(fname):
            print(funcpath, t, ovr)

def prof_totals():
    """Called from the command line to create a file containing total elapsed
    times and number of calls for all profiled functions.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', action='store', dest='outfile',
                        metavar='OUTFILE', default='sys.stdout',
                        help='Name of file containing function total counts and elapsed times.')
    parser.add_argument('rawfiles', metavar='rawfile', nargs='*',
                        help='File(s) containing raw profile data to be processed. Wildcards are allowed.')

    options = parser.parse_args()

    if not options.rawfiles:
        print("No files to process.")
        sys.exit(0)

    if options.outfile == 'sys.stdout':
        out_stream = sys.stdout
    else:
        out_stream = open(options.outfile, 'w')

    _, totals = process_profile(options.rawfiles)

    try:

        out_stream.write("\nTotals\n-------------\n")
        out_stream.write("Function Name, Total Time, Calls\n")
        grands = {}
        for func, data in sorted(((k,v) for k,v in iteritems(totals)),
                                    key=lambda x:x[1]['time'],
                                    reverse=True):
            out_stream.write("%s, %s, %s\n" %
                               (func, data['time'], data['count']))

            func_name = func.split('.')[-1]
            if func_name not in grands:
                grands[func_name] = {}
                grands[func_name]['count'] = 0
                grands[func_name]['time'] = 0
            grands[func_name]['count'] += int(data['count'])
            grands[func_name]['time'] += float(data['time'])

        out_stream.write("\nGrand Totals\n-------------\n")
        out_stream.write("Function Name, Total Time, Calls\n")
        for func, data in iteritems(grands):
            out_stream.write("%s, %s, %s\n" %
                             (func, data['time'], data['count']))

    finally:
        if out_stream is not sys.stdout:
            out_stream.close()

def prof_view():
    """Called from a command line to generate an html viewer for profile data."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--noshow', action='store_true', dest='noshow',
                        help="Don't pop up a browser to view the data.")
    parser.add_argument('-t', '--title', action='store', dest='title',
                        default='Profile of Method Calls by Instance',
                        help='Title to be displayed above profiling view.')
    parser.add_argument('rawfiles', metavar='rawfile', nargs='*',
                        help='File(s) containing raw profile data to be processed. Wildcards are allowed.')

    options = parser.parse_args()

    if not options.rawfiles:
        print("No files to process.")
        sys.exit(0)

    call_graph, totals = process_profile(options.rawfiles)

    viewer = "icicle.html"
    code_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(code_dir, viewer), "r") as f:
        template = f.read()

    graphjson = json.dumps(call_graph)

    outfile = 'profile_' + viewer
    with open(outfile, 'w') as f:
        f.write(Template(template).substitute(call_graph_data=graphjson,
                                              title=options.title))

    if not options.noshow:
        webview(outfile)

if __name__ == '__main__':
    prof_dump(sys.argv[1])
