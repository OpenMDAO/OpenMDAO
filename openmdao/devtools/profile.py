from __future__ import print_function

import os
import sys
from time import time as etime
import inspect
import fnmatch
import argparse
import json
import atexit
import types
from string import Template
from collections import OrderedDict
from functools import wraps
from struct import Struct
from ctypes import Structure, c_uint, c_float

from six import iteritems

from openmdao.core.mpi_wrap import MPI
from openmdao.core.problem import Problem
from openmdao.core.system import System
from openmdao.core.group import Group
from openmdao.core.component import Component
from openmdao.core.driver import Driver
from openmdao.solvers.solver_base import SolverBase
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.devtools.webview import webview

def get_method_class(meth):
    """Return the class that actually defined the given method."""
    for cls in inspect.getmro(meth.__self__.__class__):
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

def _obj_iter(top):
    """Iterator over objects to be checked for functions to wrap for profiling.
    The top object must be a Problem or a System or an exception will be raised.
    """

    if not isinstance(top, (Problem, System)):
        raise TypeError("Error in profile object iterator.  "
                        "Top object must be a Problem or System.")

    if isinstance(top, Problem):
        yield top
        yield top.driver
        if top.driver.recorders._recorders:
            yield top.driver.recorders
        root = top.root
    else:
        root = top

    for s in root.system_iter(recurse=True, include_self=True):
        yield s
        if isinstance(s, Group):
            yield s.ln_solver
            yield s.nl_solver
            if s.ln_solver.recorders._recorders:
                yield s.ln_solver.recorders
            if s.nl_solver.recorders._recorders:
                yield s.nl_solver.recorders

def setup(top, prefix='prof_raw', methods=None,
          obj_iter=_obj_iter, prof_dir=None):
    """
    Instruments certain important openmdao methods for profiling.

    Args
    ----

    top : object
        The top object to be profiled. The top object must be an instance
        of a class that is compatible with the object iterator function.
        The default object iterator function expects the top object to
        be a Problem or a System.

    prefix : str ('prof_raw')
        Prefix used for the raw profile data. Process rank will be appended
        to it to get the actual filename.  When not using MPI, rank=0.

    methods : dict, optional
        A dict of profiled methods to override the default set.  The key
        is the method name and the value is a tuple of class objects used
        for isinstance checking.  The default set of methods is:

        ::

            {
                "setup": (Problem,),
                "run": (Problem,),
                "calc_gradient": (Problem,),
                "solve_nonlinear": (System,),
                "apply_nonlinear": (System,),
                "solve_linear": (System,),
                "apply_linear": (System,),
                "solve": (SolverBase,),
                "fd_jacobian": (System,),
                "linearize": (System,),
                "complex_step_jacobian": (Component,),
                "record_iteration": (RecordingManager,),
                "record_derivatives": (RecordingManager,),
                "_transfer_data": (Group,),
            }

    obj_iter : function, optional
        An iterator that provides objects to be checked for matching profile
        methods.  The default object iterator iterates over a Problem or System.

    prof_dir : str
        Directory where the profile files will be written.

    """

    global _profile_prefix, _profile_methods
    global _profile_setup, _profile_total, _profile_out

    if _profile_setup:
        raise RuntimeError("profiling is already set up.")

    if prof_dir is None:
        _profile_prefix = os.path.join(os.getcwd(), prefix)
    else:
        _profile_prefix = os.path.join(os.path.abspath(prof_dir), prefix)

    _profile_setup = True

    if methods:
        _profile_methods = methods
    else:
        _profile_methods = {
            "setup": (Problem,),
            "run": (Problem,),
            "calc_gradient": (Problem,),
            "solve_nonlinear": (System,),
            "apply_nonlinear": (System,),
            "solve_linear": (System,),
            "apply_linear": (System,),
            "solve": (SolverBase,),
            "fd_jacobian": (System,),
            "linearize": (System,),
            "complex_step_jacobian": (Component,),
            "record_iteration": (RecordingManager,),
            "record_derivatives": (RecordingManager,),
            "_transfer_data": (Group,),
        }

    rank = MPI.COMM_WORLD.rank if MPI else 0
    _profile_out = open("%s.%d" % (_profile_prefix, rank), 'wb')

    atexit.register(_finalize_profile)

    wrap_methods(obj_iter(top), _profile_methods, _profile_dec)

def wrap_methods(obj_iter, methods, dec_factory):
    """
    Iterate over a collection of objects and wrap any of their methods that
    match the given set of method names with a decorator created using the
    given dectorator factory.
    """
    for obj in obj_iter:
        for meth, classes in iteritems(methods):
            if isinstance(obj, classes):
                match = getattr(obj, meth, None)
                if match is not None:
                    setattr(obj, meth,
                            dec_factory()(match).__get__(obj, obj.__class__))

def start():
    """Turn on profiling.
    """
    global _profile_start
    if _profile_start is not None:
        print("profiling is already active.")
        return

    _profile_start = etime()

def stop():
    """Turn off profiling.
    """
    global _profile_total, _profile_start
    if _profile_start is None:
        return

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

class _profile_dec(object):
    """ Use as a decorator on functions that should be profiled.
    The data collected will include time elapsed, number of calls, ...
    """
    _call_stack = []

    def __init__(self):
        self.name = None

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            global _profile_out, _profile_struct, \
                   _profile_funcs_dict, _profile_start

            ovr = etime()

            if _profile_start is not None:
                if self.name is None:
                    try:
                        name = fn.__self__.pathname
                    except AttributeError:
                        name = "<%s>" % args[0].__class__.__name__

                    name = '.'.join((name, fn.__name__))
                    self.name = name
                else:
                    name = self.name

                stack = _profile_dec._call_stack

                if stack:
                    caller = stack[-1]
                else:
                    caller = ''

                stack.append(name)

                path = ','.join(stack)

                if path not in _profile_funcs_dict:
                    # save the id for this path
                    _profile_funcs_dict[path] = len(_profile_funcs_dict)

                start = etime()
                ret = fn(*args[1:], **kwargs)
                end = etime()

                stack.pop()

                _profile_struct.t = end - start
                _profile_struct.ovr = start - ovr # keep track of overhead for later subtraction
                _profile_struct.tstamp = start
                _profile_struct.id = _profile_funcs_dict[path]
                _profile_out.write(_profile_struct)

                return ret
            else:
                return fn(*args[1:], **kwargs)

        return wrapper

def _update_counts(dct, name, elapsed, overhead):
    try:
        d = dct[name]
    except KeyError:
        dct[name] = d = {
                'count': 1,
                'time': elapsed,
                'ovr': overhead,
            }
        return

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
