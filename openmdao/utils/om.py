"""
A console script wrapper for multiple post setup hook functions.
"""

import sys
import os
import argparse

_func_map = {
    'view_model': 'openmdao.devtools.problem_viewer.problem_viewer:view_model',
    'view_connections': 'openmdao.devtools.viewconns:view_connections',
    'summary': 'openmdao.devtools.debug:config_summary',
    'tree': 'openmdao.devtools.debug:tree',
    'dump_idxs': 'openmdao.devtools.debug:dump_dist_idxs',
}


def om_cmd():
    """
    Wrap a number of Problem viewing/debugging command line functions.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('func', choices=list(_func_map.keys()), help='Function to run.')
    parser.add_argument('file', nargs=1, help='Python file to profile.')

    options = parser.parse_args()

    progname = options.file[0]

    sys.path.insert(0, os.path.dirname(progname))

    with open(progname, 'rb') as fp:
        code = compile(fp.read(), progname, 'exec')

    globals_dict = {
        '__file__': progname,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }

    func_spec = _func_map[options.func]
    os.environ['OPENMDAO_POST_SETUP'] = func_spec

    exec (code, globals_dict)
