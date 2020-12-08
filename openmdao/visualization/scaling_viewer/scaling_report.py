
"""Define a function to view driver scaling."""
import os
import sys
import json
from itertools import chain
from collections import defaultdict

import numpy as np

import openmdao.utils.coloring as coloring_mod
import openmdao.utils.hooks as hooks
from openmdao.core.problem import Problem
from openmdao.utils.units import convert_units
from openmdao.utils.mpi import MPI
from openmdao.utils.webview import webview
from openmdao.utils.general_utils import printoptions, ignore_errors, default_noraise
from openmdao.utils.file_utils import _load_and_exec, _to_filename


def _val2str(val):
    if isinstance(val, np.ndarray):
        if val.size > 5:
            return 'array %s' % str(val.shape)
        else:
            return np.array2string(val)

    return str(val)


def _unscale(val, scaler, adder, default=''):
    if val is None:
        return default
    if scaler is not None:
        val = val * (1.0 / scaler)
    if adder is not None:
        val = val - adder
    return val


def _scale(val, scaler, adder, default=''):
    if val is None:
        return default
    if adder is not None:
        val = val + adder
    if scaler is not None:
        val = val * scaler
    return val


def _getdef(val, default):
    if val is None:
        return default
    return val


def view_driver_scaling(driver, outfile='driver_scaling_report.html', show_browser=True,
                        precision=6, title=None, jac=True):
    """
    Generate a self-contained html file containing a detailed connection viewer.

    Optionally pops up a web browser to view the file.

    Parameters
    ----------
    driver : Driver
        The driver used for the scaling report.

    outfile : str, optional
        The name of the output html file.  Defaults to 'connections.html'.

    show_browser : bool, optional
        If True, pop up a browser to view the generated html file.
        Defaults to True.

    precision : int, optional
        Sets the precision for displaying array values.

    title : str, optional
        Sets the title of the web page.

    jac : bool
        If True, show jacobian information.
    """
    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    dv_table = []
    con_table = []
    obj_table = []

    dv_vals = driver.get_design_var_values(get_remote=True)
    obj_vals = driver.get_objective_values(driver_scaling=True)
    con_vals = driver.get_constraint_values(driver_scaling=True)

    mod_meta = driver._problem().model._var_allprocs_abs2meta['output']

    default = ''

    idx = 1  # unique ID for use by Tabulator
    for name, meta in driver._designvars.items():
        src_name = meta['ivc_source']
        val = dv_vals[name]  # dv_vals are unscaled
        scaler = meta['total_scaler']
        adder = meta['total_adder']

        # TODO: convert scaler/adder to ref/ref0 if needed...

        dv_table.append({
            'id': idx,
            'name': name,
            'size': meta['size'],
            'driver_val': _scale(val, scaler, adder, default),
            'driver_units': _getdef(meta['units'], default),
            'model_val': val,
            'model_units': _getdef(mod_meta[meta['ivc_source']]['units'], default),
            'ref': _getdef(meta['ref'], default),
            'ref0': _getdef(meta['ref0'], default),
            'scaler': _getdef(scaler, default),
            'adder': _getdef(adder, default),
            'lower': _getdef(meta['lower'], default),  # scaled
            'upper': _getdef(meta['upper'], default),  # scaled
        })
        idx += 1

    for name, meta in driver._cons.items():
        val = con_vals[name]
        scaler = meta['total_scaler']
        adder = meta['total_adder']
        dct = {
            'id': idx,
            'name': name,
            'size': meta['size'],
            'driver_val': _scale(val, scaler, adder, default),
            'driver_units': _getdef(meta['units'], default),
            'model_val': val,
            'model_units': _getdef(mod_meta[meta['ivc_source']]['units'], default),
            'ref': _getdef(meta['ref'], default),
            'ref0': _getdef(meta['ref0'], default),
            'scaler': _getdef(scaler, default),
            'adder': _getdef(adder, default),
            'lower': _getdef(meta['lower'], default),  # scaled
            'upper': _getdef(meta['upper'], default),  # scaled
            'equals': _getdef(meta['equals'], default), # scaled
            'linear': meta['linear'],
        }
        if dct['upper'] != default and dct['lower'] != default:
            # add separate rows for upper and lower
            # TODO: must also copy child rows if it's an array
            d = dct.copy()
            d['lower'] = default
            con_table.append(d)
            d = dct.copy()
            d['upper'] = default
            con_table.append(d)
        else:
            con_table.append(dct)
        idx += 1

    for name, meta in driver._objs.items():
        val = obj_vals[name]
        scaler = meta['total_scaler']
        adder = meta['total_adder']
        obj_table.append({
            'id': idx,
            'name': name,
            'size': meta['size'],
            'driver_val': val,
            'driver_units': _getdef(meta['units'], default),
            'model_val': _unscale(val, scaler, adder, default),
            'model_units': _getdef(mod_meta[meta['ivc_source']]['units'], default),
            'ref': _getdef(meta['ref'], default),
            'ref0': _getdef(meta['ref0'], default),
            'scaler': _getdef(scaler, default),
            'adder': _getdef(adder, default),
        })
        idx += 1

    data = {
        'title': _getdef(title, ''),
        'dv_table': dv_table,
        'con_table': con_table,
        'obj_table': obj_table,
    }

    if jac:
        coloring = driver._get_static_coloring()
        if coloring_mod._use_total_sparsity and jac:
            if coloring is None and driver._coloring_info['dynamic']:
                coloring = coloring_mod.dynamic_total_coloring(driver)

        # assemble data for jacobian visualization
        data['oflabels'] = list(chain(obj_vals, con_vals))
        data['wrtlabels'] = list(dv_vals)

        totals = driver._compute_totals(of=data['oflabels'], wrt=data['wrtlabels'],
                                        return_format='array')

        rownames = [None] * totals.shape[0]
        colnames = [None] * totals.shape[1]

        start = end = 0
        data['ofslices'] = slices = []
        for n, v in chain(obj_vals.items(), con_vals.items()):
            end += v.size
            slices.append([start, end])
            rownames[start:end] = [n] * (end - start)
            start = end

        start = end = 0
        data['wrtslices'] = slices = []
        for n, v in dv_vals.items():
            end += v.size
            slices.append([start, end])
            colnames[start:end] = [n] * (end - start)
            start = end

        norm_mat = np.zeros((len(data['ofslices']), len(data['wrtslices'])))

        def mat_magnitude(mat):
            mag = np.log10(np.abs(mat))
            finite = mag[np.isfinite(mag)]
            max_mag = np.max(finite)
            min_mag = np.min(finite)
            cap = np.abs(min_mag)
            if max_mag > cap:
                cap = max_mag
            mag[np.isinf(mag)] = -cap
            return mag

        for i, of in enumerate(chain(obj_vals, con_vals)):
            ofstart, ofend = data['ofslices'][i]
            for j, wrt in enumerate(dv_vals):
                wrtstart, wrtend = data['wrtslices'][j]
                norm_mat[i, j] = np.linalg.norm(totals[ofstart:ofend, wrtstart:wrtend])

        var_matrix = mat_magnitude(norm_mat)
        matrix = mat_magnitude(totals)

        if coloring is not None: # factor in the sparsity
            mask = np.ones(totals.shape, dtype=bool)
            mask[coloring._nzrows, coloring._nzcols] = 0
            matrix[mask] = np.inf  # we know matrix cannot contain infs by this point

        # create matrix data that includes sparsity
        nonempty_submats = set()  # submats with any nonzero values (inf now indicates zero entries)
        linear_cons = [n for n in driver._cons if driver._cons[n]['linear']]
        # coloring._nzrows/cols don't contain linear constraints, so add them to nonempty_submats
        for con in linear_cons:
            for dv in data['wrtlabels']:
                nonempty_submats.add((con, dv))

        matlist = [None] * matrix.size
        idx = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isinf(val):
                    val = None
                else:
                    nonempty_submats.add((rownames[i], colnames[j]))
                matlist[idx] = [i, j, val]
                idx += 1

        data['mat_list'] = matlist

        varmatlist = [None] * var_matrix.size

        # setup up sparsity of var matrix
        idx = 0
        for i, of in enumerate(data['oflabels']):
            for j, wrt in enumerate(data['wrtlabels']):
                val = None if (of, wrt) not in nonempty_submats else var_matrix[i, j]
                varmatlist[idx] = [i, j, val]
                idx += 1

        data['var_mat_list'] = varmatlist

        print("var_matrix")
        print(norm_mat)
        print("----")
        print(var_matrix)
        print("var_mat_list")
        import pprint
        pprint.pprint(varmatlist)
        print("obj", list(obj_vals))
        print("con", list(con_vals))
        print("dv", list(dv_vals))

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

    with open(os.path.join(libs_dir, 'd3.v6.min.js'), "r") as f:
        d3_src = f.read()

    jsontxt = json.dumps(data, default=default_noraise)

    with open(outfile, 'w') as f:
        s = template.replace("<tabulator_src>", tabulator_src)
        s = s.replace("<tabulator_style>", tabulator_style)
        s = s.replace("<d3_src>", d3_src)
        s = s.replace("<scaling_data>", jsontxt)
        f.write(s)

    if show_browser:
        webview(outfile)


def _scaling_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao driver_scaling' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', default='driver_scaling_report.html', action='store', dest='outfile',
                        help='html output file.')
    parser.add_argument('-t', '--title', action='store', dest='title',
                        help='title of web page.')
    parser.add_argument('--no_browser', action='store_true', dest='no_browser',
                        help="don't display in a browser.")
    parser.add_argument('-p', '--problem', action='store', dest='problem', help='Problem name')
    parser.add_argument('--no-jac', action='store_true', dest='nojac', help="Don't show jacobian info")


def _scaling_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao driver_scaling'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    def _scaling(problem):
        hooks._unregister_hook('final_setup', 'Problem')  # avoid recursive loop
        driver = problem.driver
        if options.title:
            title = options.title
        else:
            title = "Driver scaling for %s" % os.path.basename(options.file[0])
        view_driver_scaling(driver, outfile=options.outfile, show_browser=not options.no_browser,
                            title=title, jac=not options.nojac)
        exit()

    # register the hook
    hooks._register_hook('final_setup', class_name='Problem', inst_id=options.problem,
                         post=_scaling)

    ignore_errors(True)
    _load_and_exec(options.file[0], user_args)
