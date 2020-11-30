
"""Define a function to view driver scaling."""
import os
import sys
import json
from itertools import chain
from collections import defaultdict

import numpy as np

from openmdao.core.problem import Problem
from openmdao.utils.units import convert_units
from openmdao.utils.mpi import MPI
from openmdao.utils.webview import webview
from openmdao.utils.general_utils import printoptions


def _val2str(val):
    if isinstance(val, np.ndarray):
        if val.size > 5:
            return 'array %s' % str(val.shape)
        else:
            return np.array2string(val)

    return str(val)


def _unscale(val, scaler, adder):
    if scaler is not None:
        val = val * (1.0 / scaler)
    if adder is not None:
        val = val - adder
    return val


def view_driver_scaling(problem, outfile='driver_scaling.html', show_browser=True,
                        show_values=True, precision=6, title=None):
    """
    Generate a self-contained html file containing a detailed connection viewer.

    Optionally pops up a web browser to view the file.

    Parameters
    ----------
    problem : Problem
        The Problem containing the driver.

    outfile : str, optional
        The name of the output html file.  Defaults to 'connections.html'.

    show_browser : bool, optional
        If True, pop up a browser to view the generated html file.
        Defaults to True.

    precision : int, optional
        Sets the precision for displaying array values.

    title : str, optional
        Sets the title of the web page.
    """
    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    driver = problem.driver

    dv_table = []
    con_table = []
    obj_table = []

    dv_vals = driver.get_design_var_values(get_remote=True)
    con_vals = driver.get_constraint_values(get_remote=True)
    obj_vals = driver.get_objective_values(get_remote=True)

    idx = 1  # unique ID for use by Tabulator
    for name, meta in driver._designvars.items():
        val = dv_vals[name]
        scaler = meta['total_scaler']
        adder = meta['total_adder']
        dv_table.append({
            'id': idx,
            'name': name,
            'size': meta['size'],
            'unscaled val': _unscaled(val, scaler, adder),
            'scaled val': val,
            'ref': meta['ref'],
            'ref0': meta['ref0'],
            'scaler': scaler,
            'adder': adder,
            'unscaled lower': _unscaled(meta['lower'], scaler, adder),
            'scaled lower': meta['lower'],
            'unscaled uppper': _unscaled(meta['upper'], scaler, adder),
            'scaled upper': meta['upper'],
        })
        idx += 1

    for name, meta in driver._cons.items():
        val = con_vals[name]
        scaler = meta['total_scaler']
        adder = meta['total_adder']
        dv_table.append({
            'id': idx,
            'name': name,
            'size': meta['size'],
            'unscaled val': _unscaled(val, scaler, adder),
            'scaled val': val,
            'ref': meta['ref'],
            'ref0': meta['ref0'],
            'scaler': scaler,
            'adder': adder,
            'unscaled lower': _unscaled(meta['lower'], scaler, adder),
            'scaled lower': meta['lower'],
            'unscaled uppper': _unscaled(meta['upper'], scaler, adder),
            'scaled upper': meta['upper'],
            'unscaled equals': _unscaled(meta['equals'], scaler, adder),
            'scaled equals': meta['equals'],
            'linear': meta['linear'],
        })
        idx += 1

    if title is None:
        title = ''

    data = {
        'title': title,
        'dv_table': dv_table,
        'con_table': con_table,
        'obj_table': obj_table,
    }

    viewer = 'scaling_table.html'

    code_dir = os.path.dirname(os.path.abspath(__file__))
    libs_dir = os.path.join(code_dir, 'libs')
    style_dir = os.path.join(code_dir, 'style')

    with open(os.path.join(code_dir, viewer), "r") as f:
        template = f.read()

    with open(os.path.join(libs_dir, 'tabulator.min.js'), "r") as f:
        tabulator_src = f.read()

    with open(os.path.join(style_dir, 'tabulator.min.css'), "r") as f:
        tabulator_style = f.read()

    jsontxt = json.dumps(data)

    with open(outfile, 'w') as f:
        s = template.replace("<scaling_data>", jsontxt)
        s = s.replace("<tabulator_src>", tabulator_src)
        s = s.replace("<tabulator_style>", tabulator_style)
        f.write(s)

    if show_browser:
        webview(outfile)

def _driver_scaling_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao driver_scaling' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', default='connections.html', action='store', dest='outfile',
                        help='html output file.')
    parser.add_argument('-t', '--title', action='store', dest='title',
                        help='title of web page.')
    parser.add_argument('--no_browser', action='store_true', dest='no_browser',
                        help="don't display in a browser.")
    parser.add_argument('-p', '--problem', action='store', dest='problem', help='Problem name')


def _driver_scaling_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao driver_scaling'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    def _scaling(prob):
        if options.title:
            title = options.title
        else:
            title = "Driver scaling for %s" % os.path.basename(options.file[0])
        view_driver_scaling(prob, outfile=options.outfile, show_browser=not options.no_browser,
                            title=title)
        exit()

    # register the hook
    hooks._register_hook('final_setup', class_name='Problem', inst_id=options.problem,
                         post=_scaling)

    ignore_errors(True)
    _load_and_exec(options.file[0], user_args)
