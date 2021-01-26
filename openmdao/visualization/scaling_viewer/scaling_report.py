
"""Define a function to view driver scaling."""
import os
import sys
import json
from itertools import chain
from collections import defaultdict

import numpy as np

import openmdao
from openmdao.core.constants import _SetupStatus
import openmdao.utils.coloring as coloring_mod
import openmdao.utils.hooks as hooks
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


def _scale(val, scaler, adder, unset=''):
    if val is None:
        return unset
    if adder is not None:
        val = val + adder
    if scaler is not None:
        val = val * scaler
    return val


def _getdef(val, unset):
    if val is None:
        return unset
    if np.isscalar(val) and (val == openmdao.INF_BOUND or val == -openmdao.INF_BOUND):
        return unset
    return val


def _get_val_and_size(val, unset=''):
    # return val (or max abs val) and the size of the value
    val = _getdef(val, unset)
    if np.isscalar(val) or val.size == 1:
        return [val, 1]
    return [np.max(np.abs(val)), val.size]


def _get_flat(val, size, unset=''):
    if val is None:
        return val
    if np.isscalar(val):
        if (val == openmdao.INF_BOUND or val == -openmdao.INF_BOUND):
            val = unset
        return np.full(size, val)
    if val.size > 1:
        return val.flatten()
    return np.full(size, val[0])


def _add_child_rows(row, mval, dval, scaler=None, adder=None, ref=None, ref0=None,
                    lower=None, upper=None, equals=None, inds=None):
    if not (np.isscalar(mval) or mval.size == 1):
        rowchild = row.copy()
        children = row['_children'] = []
        rowchild['name'] = ''
        rowchild['size'] = ''
        dval_flat = dval.flatten()
        mval_flat = mval.flatten()
        scaler_flat = _get_flat(scaler, mval.size)
        adder_flat = _get_flat(adder, mval.size)
        ref_flat = _get_flat(ref, mval.size)
        ref0_flat = _get_flat(ref0, mval.size)
        upper_flat = _get_flat(upper, mval.size)
        lower_flat = _get_flat(lower, mval.size)
        equals_flat = _get_flat(equals, mval.size)

        if inds is None:
            inds = list(range(dval.size))
        else:
            inds = np.atleast_1d(inds).flatten()

        for i, idx in enumerate(inds):
            d = rowchild.copy()
            d['index'] = idx
            d['driver_val'] = [dval_flat[i], 1]
            d['model_val'] = [mval_flat[i], 1]
            if scaler_flat is not None:
                d['scaler'] = [scaler_flat[i], 1]
            if adder_flat is not None:
                d['adder'] = [adder_flat[i], 1]
            if ref_flat is not None:
                d['ref'] = [ref_flat[i], 1]
            if ref0_flat is not None:
                d['ref0'] = [ref0_flat[i], 1]
            if upper_flat is not None:
                d['upper'] = [upper_flat[i], 1]
            if lower_flat is not None:
                d['lower'] = [lower_flat[i], 1]
            if equals_flat is not None:
                d['equals'] = [equals_flat[i], 1]
            children.append(d)


def _compute_jac_view_info(totals, data, dv_vals, response_vals, coloring):
    start = end = 0
    data['ofslices'] = slices = {}
    for n, v in response_vals.items():
        end += v.size
        slices[n] = [start, end]
        start = end

    start = end = 0
    data['wrtslices'] = slices = {}
    for n, v in dv_vals.items():
        end += v.size
        slices[n] = [start, end]
        start = end

    nonempty_submats = set()  # submats with any nonzero values

    var_matrix = np.zeros((len(data['ofslices']), len(data['wrtslices'])))

    matrix = np.abs(totals)

    if coloring is not None:  # factor in the sparsity
        mask = np.zeros(totals.shape, dtype=bool)
        mask[coloring._nzrows, coloring._nzcols] = 1

    for i, of in enumerate(response_vals):
        ofstart, ofend = data['ofslices'][of]
        for j, wrt in enumerate(dv_vals):
            wrtstart, wrtend = data['wrtslices'][wrt]
            # use max of abs value here instead of norm to keep coloring consistent between
            # top level jac and subjacs
            var_matrix[i, j] = np.max(matrix[ofstart:ofend, wrtstart:wrtend])
            if var_matrix[i, j] > 0. or (coloring and
                                         np.any(mask[ofstart:ofend, wrtstart:wrtend])):
                nonempty_submats.add((of, wrt))

    matlist = [None] * matrix.size
    idx = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if coloring and not mask[i, j]:
                val = None
            else:
                if val == 0.:
                    val = 0  # set to int 0
            matlist[idx] = [i, j, val]
            idx += 1

    data['mat_list'] = matlist

    varmatlist = [None] * var_matrix.size

    # setup up sparsity of var matrix
    idx = 0
    for i, of in enumerate(data['oflabels']):
        for j, wrt in enumerate(data['wrtlabels']):
            if coloring is not None and (of, wrt) not in nonempty_submats:
                val = None
            else:
                val = var_matrix[i, j]
            varmatlist[idx] = [of, wrt, val]
            idx += 1

    data['var_mat_list'] = varmatlist


def view_driver_scaling(driver, outfile='driver_scaling_report.html', show_browser=True,
                        title=None, jac=True):
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
    title : str, optional
        Sets the title of the web page.
    jac : bool
        If True, show jacobian information.

    Returns
    -------
    dict
        Data to used to generate html file.
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

    if driver._problem()._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
        raise RuntimeError("Driver scaling report cannot be generated before calling final_setup "
                           "on the Problem.")

    default = ''

    idx = 1  # unique ID for use by Tabulator

    # set up design vars table data
    for name, meta in driver._designvars.items():
        scaler = meta['total_scaler']
        adder = meta['total_adder']
        ref = meta['ref']
        ref0 = meta['ref0']
        lower = meta['lower']
        upper = meta['upper']

        dval = dv_vals[name]
        mval = _unscale(dval, scaler, adder, default)

        if dval.size == 1:
            index = meta['indices']
            if index is not None:
                index = index[0]
            index = _getdef(index, '')
        else:
            index = ''

        dct = {
            'id': idx,
            'name': name,
            'size': meta['size'],
            'driver_val': _get_val_and_size(dval),
            'driver_units': _getdef(meta['units'], default),
            'model_val': _get_val_and_size(mval),
            'model_units': _getdef(mod_meta[meta['ivc_source']]['units'], default),
            'ref': _get_val_and_size(ref, default),
            'ref0': _get_val_and_size(ref0, default),
            'scaler': _get_val_and_size(scaler, default),
            'adder': _get_val_and_size(adder, default),
            'lower': _get_val_and_size(lower, default),  # scaled
            'upper': _get_val_and_size(upper, default),  # scaled
            'index': index,
        }

        dv_table.append(dct)

        _add_child_rows(dct, mval, dval, scaler=scaler, adder=adder, ref=ref, ref0=ref0,
                        lower=lower, upper=upper, inds=meta['indices'])

        idx += 1

    # set up constraints table data
    for name, meta in driver._cons.items():
        scaler = meta['total_scaler']
        adder = meta['total_adder']
        ref = meta['ref']
        ref0 = meta['ref0']
        lower = meta['lower']
        upper = meta['upper']
        equals = meta['equals']

        dval = con_vals[name]
        mval = _unscale(dval, scaler, adder, default)

        if dval.size == 1:
            index = meta['indices']
            if index is not None:
                index = index[0]
            index = _getdef(index, '')
        else:
            index = ''

        dct = {
            'id': idx,
            'name': name,
            'size': meta['size'],
            'index': index,
            'driver_val': _get_val_and_size(dval),
            'driver_units': _getdef(meta['units'], default),
            'model_val': _get_val_and_size(mval),
            'model_units': _getdef(mod_meta[meta['ivc_source']]['units'], default),
            'ref': _get_val_and_size(meta['ref'], default),
            'ref0': _get_val_and_size(meta['ref0'], default),
            'scaler': _get_val_and_size(scaler, default),
            'adder': _get_val_and_size(adder, default),
            'lower': _get_val_and_size(meta['lower'], default),  # scaled
            'upper': _get_val_and_size(meta['upper'], default),  # scaled
            'equals': _get_val_and_size(meta['equals'], default),  # scaled
            'linear': meta['linear'],
        }

        con_table.append(dct)
        _add_child_rows(dct, mval, dval, scaler=scaler, adder=adder, ref=ref, ref0=ref0,
                        lower=lower, upper=upper, equals=equals, inds=meta['indices'])

        idx += 1

    # set up objectives table data
    for name, meta in driver._objs.items():
        scaler = meta['total_scaler']
        adder = meta['total_adder']
        ref = meta['ref']
        ref0 = meta['ref0']

        dval = obj_vals[name]
        mval = _unscale(dval, scaler, adder, default)

        if dval.size == 1:
            index = meta['indices']
            if index is not None:
                index = index[0]
            index = _getdef(index, '')
        else:
            index = ''

        dct = {
            'id': idx,
            'name': name,
            'size': meta['size'],
            'index': index,
            'driver_val': _get_val_and_size(dval),
            'driver_units': _getdef(meta['units'], default),
            'model_val': _get_val_and_size(mval),
            'model_units': _getdef(mod_meta[meta['ivc_source']]['units'], default),
            'ref': _get_val_and_size(meta['ref'], default),
            'ref0': _get_val_and_size(meta['ref0'], default),
            'scaler': _get_val_and_size(scaler, default),
            'adder': _get_val_and_size(adder, default),
        }

        obj_table.append(dct)
        _add_child_rows(dct, mval, dval, scaler=scaler, adder=adder, ref=ref, ref0=ref0,
                        inds=meta['indices'])

        idx += 1

    data = {
        'title': _getdef(title, ''),
        'dv_table': dv_table,
        'con_table': con_table,
        'obj_table': obj_table,
        'oflabels': [],
        'wrtlabels': [],
        'var_mat_list': [],
        'linear': {
            'oflabels': [],
        }
    }

    if jac and not driver._problem().model._use_derivatives:
        print("\nCan't display jacobian because derivatives are turned off.\n")
        jac = False

    if jac:
        # save old totals
        save = driver._total_jac
        driver._total_jac = None

        coloring = driver._get_static_coloring()
        if coloring_mod._use_total_sparsity:
            if coloring is None and driver._coloring_info['dynamic']:
                coloring = coloring_mod.dynamic_total_coloring(driver)

        # assemble data for jacobian visualization
        data['oflabels'] = driver._get_ordered_nl_responses()
        data['wrtlabels'] = list(dv_vals)

        try:
            totals = driver._compute_totals(of=data['oflabels'], wrt=data['wrtlabels'],
                                            return_format='array')
        finally:
            driver._total_jac = save

        data['linear'] = lindata = {}
        lindata['oflabels'] = [n for n, meta in driver._cons.items() if meta['linear']]
        lindata['wrtlabels'] = data['wrtlabels']  # needs to mimic data structure

        # check for separation of linear constraints
        if lindata['oflabels']:
            if set(lindata['oflabels']).intersection(data['oflabels']):
                # linear cons are found in data['oflabels'] so they're not separated
                lindata['oflabels'] = []
                lindata['wrtlables'] = []

        full_response_vals = con_vals.copy()
        full_response_vals.update(obj_vals)
        response_vals = {n: full_response_vals[n] for n in data['oflabels']}

        _compute_jac_view_info(totals, data, dv_vals, response_vals, coloring)

        if lindata['oflabels']:
            # prevent reuse of nonlinear totals
            save = driver._total_jac
            driver._total_jac = None

            try:
                lintotals = driver._compute_totals(of=lindata['oflabels'], wrt=data['wrtlabels'],
                                                   return_format='array')
                lin_response_vals = {n: full_response_vals[n] for n in lindata['oflabels']}
            finally:
                driver._total_jac = save

            _compute_jac_view_info(lintotals, lindata, dv_vals, lin_response_vals, None)

    if driver._problem().comm.rank == 0:

        viewer = 'scaling_table.html'

        code_dir = os.path.dirname(os.path.abspath(__file__))
        libs_dir = os.path.join(os.path.dirname(code_dir), 'common', 'libs')
        style_dir = os.path.join(os.path.dirname(code_dir), 'common', 'style')

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

    return data


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
    parser.add_argument('--no-jac', action='store_true', dest='nojac',
                        help="Don't show jacobian info")


_run_driver_called = False
_run_model_start = False
_run_model_done = False


def _exitfunc():
    if not _run_driver_called:
        print("\n\nNo driver scaling report was generated because run_driver() was not called "
              "on the required Problem.\n")


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
    def _set_run_driver_flag(problem):
        global _run_driver_called
        _run_driver_called = True

    def _set_run_model_start(problem):
        global _run_model_start
        _run_model_start = True

    def _set_run_model_done(problem):
        global _run_model_done
        _run_model_done = True

    def _scaling_check(problem):
        if _run_driver_called:
            # If run_driver has been called, we know no more user changes are coming.
            if not _run_model_start:
                problem.run_model()
            if _run_model_done:
                _scaling(problem)

    def _scaling(problem):
        hooks._unregister_hook('final_setup', 'Problem')  # avoid recursive loop
        hooks._unregister_hook('run_driver', 'Problem')
        hooks._unregister_hook('run_model', 'Problem')
        driver = problem.driver
        if options.title:
            title = options.title
        else:
            title = "Driver scaling for %s" % os.path.basename(options.file[0])
        view_driver_scaling(driver, outfile=options.outfile, show_browser=not options.no_browser,
                            title=title, jac=not options.nojac)
        exit()

    # register the hooks
    hooks._register_hook('final_setup', class_name='Problem', inst_id=options.problem,
                         post=_scaling_check)

    hooks._register_hook('run_model', class_name='Problem', inst_id=options.problem,
                         pre=_set_run_model_start, post=_set_run_model_done)

    hooks._register_hook('run_driver', class_name='Problem', inst_id=options.problem,
                         pre=_set_run_driver_flag)

    # register an atexit function to check if scaling report was triggered during the script
    import atexit
    atexit.register(_exitfunc)

    ignore_errors(True)
    _load_and_exec(options.file[0], user_args)
