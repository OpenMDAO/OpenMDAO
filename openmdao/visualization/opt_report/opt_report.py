import base64
import datetime
import io
import os
import sys
import time

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

from openmdao.core.problem import Problem
from openmdao.core.constants import INF_BOUND

from openmdao.utils.mpi import MPI


MAX_VARNAME_LENGTH = 48
_ABS_BOUNDS_TOL = 1.0E-4


def _prom_name_dict(d, abs2prom):
    """

    Parameters
    ----------
    d : dict
        A dictionary whose keys are absolute names, for which we want the promoted names.
    abs2prom : dict
        A mapping from absolute name to promoted name.

    Returns
    -------
    dict
        A dictionary identical to d, except the keys have been converted to promoted names.
    """
    ret = {}
    for abs_name, meta in d.items():
        if abs_name in abs2prom['input']:
            prom_name = abs2prom['input'][abs_name]
        elif abs_name in abs2prom['output']:
            prom_name = abs2prom['output'][abs_name]
        else:
            prom_name = abs_name
        ret[prom_name] = meta
    return ret


def _make_header_table(prob):
    t = datetime.datetime.now()
    time_stamp = t.strftime("%Y-%m-%d %H:%M:%S %Z")

    rows = []

    rows.append(['Problem:', prob._name])
    rows.append(['Script:', sys.argv[0]])
    rows.append(['Execution Time:', time_stamp])

    runtime = prob.driver.opt_result['runtime']
    runtime_ms = (runtime * 1000.0) % 1000.0
    runtime_formatted =  f"{time.strftime('%H hours %M minutes %S seconds', time.gmtime(runtime))} " \
                     f"{runtime_ms:.1f} milliseconds"


    rows.append(['Run Time:', runtime_formatted])
    rows.append(['Exit Status:', prob.driver.opt_result['exit_status']])
    rows.append(['Driver:', type(prob.driver).__name__])

    return tabulate(rows, tablefmt='html')


# def _make_opt_info_table(driver):
#     # driver_options_table = driver.options.to_table(fmt='html', show_val=True)
#     driver_options_table = driver.options.to_table(fmt='html')
#     opt_settings_table = None
#
#     if hasattr(driver, 'opt_settings') and driver.opt_settings:
#         opt_settings = [(key, val) for key, val in driver.opt_settings.items()]
#         opt_settings_table = tabulate(opt_settings, headers=['Setting', 'Val'], tablefmt='html')
#
#     html = f'{driver_options_table}'
#     if opt_settings_table is not None:
#         html += f'\n<h2>Optimizer Settings</h2>\n{opt_settings_table}'
#
#     return html

def _make_opt_value_table(driver):
    # driver_options_table = driver.options.to_table(fmt='html', show_val=True)
    driver_options_table = driver.options.to_table(fmt='html')
    opt_settings_table = None

    opt_settings = []
    for key, meta in driver.options.items():
        meta = driver.options._dict[key]
        opt_settings.append((key, meta['val'], meta['desc']))
    opt_settings_table = tabulate(opt_settings, headers=['Setting', 'Val', 'Description'], tablefmt='html')

    # if hasattr(driver, 'opt_settings') and driver.opt_settings:
    #     # opt_settings = [(key, val) for key, val in driver.opt_settings.items()]
    #     opt_settings = []
    #     # for key, val in driver.opt_settings.items():
    #     #     meta = driver.options._dict[key]
    #     #     opt_settings.append((key, val, meta['desc']))
    #     for key, meta in driver.options.items():
    #         meta = driver.options._dict[key]
    #         opt_settings.append((key, meta['val'], meta['desc']))
    #     opt_settings_table = tabulate(opt_settings, headers=['Setting', 'Val', 'Description'], tablefmt='html')

    # html = f'{driver_options_table}'
    html = ''
    if opt_settings_table is not None:
        # html += f'\n<h2>Optimizer Settings</h2>\n{opt_settings_table}'
        html += f'\n{opt_settings_table}'

    return html


def _desvar_sparkline(fig, ax, meta, val):
    indices = np.s_[...] if meta['indices'] is None else meta['indices']()

    _val = np.asarray(val).ravel()

    if isinstance(meta['lower'], float):
        _lower = (meta['lower'] * np.ones_like(val)[indices]).ravel()
    else:
        _lower = np.asarray(meta['lower']).ravel()

    if isinstance(meta['upper'], float):
        _upper = (meta['upper'] * np.ones_like(val)[indices]).ravel()
    else:
        _upper = np.asarray(meta['upper']).ravel()

    # Plot the lower/upper/equals lines
    ar = np.arange(_val.size)
    x_min = -1
    x_max = val.size

    ax.scatter(ar, val.ravel(), c='tab:blue', s=1)

    ax.margins(x=0, y=0.1)
    ax.autoscale_view(True)

    plt.autoscale(False)
    if _val.size == 1:
        ax.fill_between([x_min, x_max], _lower, _upper, color='tab:orange', alpha=0.1)
    else:
        ax.fill_between(ar, _lower, _upper, color='tab:orange', alpha=0.1)
        ax.plot(ar, _lower, color='tab:orange', linewidth=0.5)
        ax.plot(ar, _upper, color='tab:orange', linewidth=0.5)

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_xticklines(), visible=False)
    plt.setp(ax.spines.values(), visible=False)

    ax.set_xlim([x_min, x_max])

    y_min = np.min(_lower) if np.min(_lower) > -INF_BOUND else np.min(_val)
    y_max = np.max(_upper) if np.max(_upper) < INF_BOUND else np.max(_val)

    ax.set_yticks([y_min, y_max])

    ax.margins(x=0, y=0.1)
    ax.tick_params(axis='y', labelsize=7)


def _ineq_constraint_sparkline(fig, ax, meta, val):
    indices = np.s_[...] if meta['indices'] is None else meta['indices']()

    _val = np.asarray(val).ravel()

    if isinstance(meta['lower'], float):
        _lower = (meta['lower'] * np.ones_like(val)[indices]).ravel()
    else:
        _lower = np.asarray(meta['lower']).ravel()

    if isinstance(meta['upper'], float):
        _upper = (meta['upper'] * np.ones_like(val)[indices]).ravel()
    else:
        _upper = np.asarray(meta['upper']).ravel()

    # Plot the lower/upper/equals lines
    ar = np.arange(_val.size)
    x_min = -1
    x_max = val.size

    ax.scatter(ar, _val, c='tab:blue', s=1)

    if _val.size == 1:
        ax.fill_between([x_min, x_max], _lower, _upper, color='tab:orange', alpha=0.1)
    else:
        ax.fill_between(ar, _lower, _upper, color='tab:orange', alpha=0.1)
        ax.plot(ar, _lower, color='tab:orange', linewidth=0.5)
        ax.plot(ar, _upper, color='tab:orange', linewidth=0.5)

    ax.set_xlim([x_min, x_max])

    y_min = np.min(_lower) if np.min(_lower) > -INF_BOUND else np.min(_val)
    y_max = np.max(_upper) if np.max(_upper) < INF_BOUND else np.max(_val)

    ax.margins(x=0, y=0.1)
    ax.set_yticks([y_min, y_max])


def _eq_constraint_sparkline(fig, ax, meta, val):
    indices = np.s_[...] if meta['indices'] is None else meta['indices']

    _val = np.asarray(val).ravel()

    if 'equals' not in meta:
        _equals = None
    elif isinstance(meta['equals'], float):
        _equals = (meta['equals'] * np.ones_like(val)[indices]).ravel()
    else:
        _equals = np.asarray(meta['equals']).ravel()

    colors = {'lower': 'tab:orange',
              'upper': 'tab:orange',
              'equals': 'tab:gray',
              'feas': 'tab:blue',
              'infeas': 'tab:red',
              'omit': 'lightgray'}

    scatter_color = [colors['feas']] * int(val.size)

    ar = np.arange(val.size)

    # Plot the lower/upper/equals lines
    x_min = ar - 0.5
    x_max = x_min + 1

    # Error
    err = _val - _equals

    # ax.scatter(ar, err.ravel(), c=scatter_color, s=3)
    colors = []
    for e in err.ravel():
        if np.abs(e) < 1.0E-6:
            colors.append('tab:green')
        else:
            colors.append('tab:red')

    y_min = np.min(err.ravel())
    y_max = np.max(err.ravel())
    ax.bar(ar, err.ravel(), color=colors)
    ax.set_xlim([x_min[0], x_max[-1]])
    ax.set_yticks([y_min, y_max])


def _sparkline(kind, meta, val, width=300):
    """
    Given the metadata and value of a design variable or constraint, make an html-embeddable
    sparkline.

    Parameters
    ----------
    kind : str
        One of 'desvar' or 'constraint' to specify which type of sparkline is being made.
        This has a slight impact on how the bounds are plotted.
    meta : dict-like
        The metadata associated with the design variable or constraint.
    val : np.array
        The value of the design variable or constraint.
    width : int
        The width of the figure in the returned HTML tag.

    Returns
    -------
    str
        An HTML image tag containing the encoded sparkline image.

    """
    # Prepare the matplotlib figure/axes
    _backend = mpl.get_backend()
    plt.style.use('default')
    plt.autoscale(False)
    mpl.use('Agg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, .4))

    if kind == 'desvar':
        _desvar_sparkline(fig, ax, meta, val)
    elif 'equals' not in meta or meta['equals'] is None:
        _ineq_constraint_sparkline(fig, ax, meta, val)
    else:
        _eq_constraint_sparkline(fig, ax, meta, val)

    ax.tick_params(axis='y', labelsize=7)

    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%5.3e'))

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_xticklines(), visible=False)
    plt.setp(ax.spines.values(), visible=False)

    fig.patch.set_facecolor(None)
    fig.patch.set_alpha(0.0)

    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(.5)
    fig.subplots_adjust(left=0.25)

    tmpfile = io.BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html = f'<img width={width} src=\'data:image/png;base64,{encoded}\'>'

    mpl.use(_backend)
    plt.close()

    return html


def _make_obj_table(objs_meta, objs_vals,
                    cols=['size', 'index', 'val', 'ref', 'ref0', 'adder', 'scaler', 'units']):
    _col_names = cols if cols is not None else []
    # col_names = ['name'] + _col_names
    col_names = ['name'] + _col_names

    # Get the values for all the elements in the tables
    rows = []
    for prom_name, meta in objs_meta.items():
        row = {}
        if meta['ref0'] is None and meta['ref'] is None and meta['scaler'] is None and meta[
            'adder'] is None:
            meta['ref'] = 1.0
            meta['ref0'] = 0.0

        for col_name in col_names:
            if col_name == 'name':
                row[col_name] = prom_name
                if len(row[col_name]) > MAX_VARNAME_LENGTH:
                    split_name = row[col_name].split('.')
                    row[col_name] = '.'.join(split_name[:2]) + '...' + split_name[-1]
            elif col_name == 'val':
                row[col_name] = objs_vals[prom_name]
            elif col_name == 'units':
                if meta['units'] is not None:
                    row[col_name] = meta[col_name]
                else:
                    row[col_name] = objs_meta[prom_name][col_name]
            else:
                row[col_name] = meta[col_name]
        rows.append(row)

    return tabulate(rows, headers='keys', tablefmt='html', floatfmt='.4e')


def _make_dvcons_table(meta_dict, vals_dict, kind,
                       cols=['lower', 'upper', 'ref', 'ref0', 'adder', 'scaler', 'units', 'min',
                             'max', 'plot']):
    _col_names = cols if cols is not None else []
    # col_names = ['name', 'size'] + _col_names
    col_names = ['name', 'alias', 'size'] + _col_names

    # Get the values for all the elements in the tables
    rows = []
    for name, meta in meta_dict.items():
        row = {}

        if meta['ref0'] is None and meta['ref'] is None and meta['scaler'] is None and \
                meta['adder'] is None:
            meta['ref'] = 1.0
            meta['ref0'] = 0.0


        alias = meta.get('alias', '')
        if alias:
            name = meta['name']
        for col_name in col_names:
            if col_name == 'name':
                row[col_name] = name
                if len(row[col_name]) > MAX_VARNAME_LENGTH:
                    split_name = row[col_name].split('.')
                    row[col_name] = '.'.join(split_name[:2]) + '...' + split_name[-1]
            elif col_name == 'alias':
                row[col_name] = alias
            elif col_name == 'mean':
                mean_val = np.mean(vals_dict[name])
                row[col_name] = mean_val
            elif col_name == 'min':
                min_val = min(vals_dict[name])
                # Rounding to match float precision to numpy precision
                if np.any(min_val < np.min(meta['lower']) - _ABS_BOUNDS_TOL):
                    row[col_name] = f'<span style="color:red">{min_val}</span>'
                else:
                    row[col_name] = min_val
            elif col_name == 'max':
                max_val = max(vals_dict[name])
                # Rounding to match float precision to numpy precision
                if np.any(max_val > meta['upper'] + _ABS_BOUNDS_TOL):
                    row[col_name] = f'<span style="color:red">{max_val}</span>'
                else:
                    row[col_name] = max_val
            elif col_name == 'lower':
                if np.all(meta[col_name] == -INF_BOUND):
                    row[col_name] = None
                else:
                    row[col_name] = np.mean(meta[col_name])
            elif col_name == 'upper':
                if np.all(meta[col_name] == INF_BOUND):
                    row[col_name] = None
                else:
                    row[col_name] = np.mean(meta[col_name])
            elif col_name == 'equals':
                if 'equals' not in meta or meta['equals'] is None:
                    row[col_name] = ''
                else:
                    row[col_name] = np.mean(meta[col_name])
            elif col_name == 'units':
                if meta['units'] is not None:
                    row[col_name] = meta[col_name]
                else:
                    if alias:
                        row[col_name] = meta_dict[alias][col_name]
                    else:
                        row[col_name] = meta_dict[name][col_name]
            elif col_name == 'plot':
                row['plot'] = _sparkline(meta=meta, val=vals_dict[name], kind=kind)
            else:
                row[col_name] = meta[col_name]

        rows.append(row)

    return tabulate(rows, headers='keys', tablefmt='unsafehtml', floatfmt='.4e')


def opt_report(prob):
    """
    Write a summary of the optimization to the given file.

    Parameters
    ----------
    source : Problem
        The problem for which the optimization report is being generated.
    driver_scaling : bool
        If True, display variable values with driver scaling applied.
    outfile : str or None
        The path to the HTML file to be written.  If None (default), write to the default record
        output path.
    """
    # if MPI is active only display one copy of the viewer
    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    driver_scaling = True
    # outfile = os.path.join(prob.options['report_dir'], 'opt_report.html')
    outfile = 'opt_report.html'

    # Collect data from the problem
    abs2prom = prob.model._var_abs2prom

    # Collect the entire array of array valued desvars and constraints (ignore indices)

    objs_vals = {}
    desvars_vals = {}
    cons_vals = {}

    objs_meta = _prom_name_dict(prob.driver._objs, abs2prom)
    desvars_meta = _prom_name_dict(prob.driver._designvars, abs2prom)
    cons_meta = _prom_name_dict(prob.driver._cons, abs2prom)

    with prob.model._scaled_context_all():
        for abs_name, meta in prob.driver._objs.items():
            prom_name = abs2prom['input'][abs_name] if abs_name in abs2prom['input'] else \
            abs2prom['output'][abs_name]
            objs_vals[prom_name] = prob.driver.get_objective_values(driver_scaling=driver_scaling)[
                abs_name]

        for abs_name, meta in prob.driver._designvars.items():

            if abs_name in abs2prom['input']:
                prom_name = abs2prom['input'][abs_name]
            elif abs_name in abs2prom['output']:
                prom_name = abs2prom['output'][abs_name]
            else:
                prom_name = abs_name

            # prom_name = abs2prom['input'][abs_name] if abs_name in abs2prom['input'] else \
            # abs2prom['output'][abs_name]
            desvars_vals[prom_name] = \
            prob.driver.get_design_var_values(driver_scaling=driver_scaling)[abs_name]

        for abs_name, meta in prob.driver._cons.items():
            #             vname = meta['name'] if meta.get('alias') else name

            if 'alias' in meta and meta['alias'] is not None:
                # check to see if the abs_name is the alias
                if abs_name == meta['alias']:
                    prom_name = meta['name']
                else:
                    raise ValueError("Absolute name of var was expected to be the alias")  # TODO ??
            else:
                prom_name = abs2prom['input'][abs_name] if abs_name in abs2prom['input'] else \
                            abs2prom['output'][abs_name]
                # if abs_name in abs2prom['input']:
                #     prom_name = abs2prom['input'][abs_name]
                # elif abs_name in abs2prom['output']:
                #     prom_name = abs2prom['output'][abs_name]
                # else:   # must be alias


            cons_vals[prom_name] = prob.driver.get_constraint_values(driver_scaling=driver_scaling)[
                abs_name]

    header_html = _make_header_table(prob)

    objs_html = _make_obj_table(objs_meta=objs_meta, objs_vals=objs_vals,
                                # precision=precision,
                                cols=['val', 'ref', 'ref0', 'adder', 'scaler', 'units'])

    desvars_html = _make_dvcons_table(meta_dict=desvars_meta, vals_dict=desvars_vals, kind='desvar',
                                      # precision=precision,
                                      cols=['min', 'max', 'mean', 'lower', 'upper', 'equals', 'ref',
                                            'ref0', 'units', 'plot'])

    cons_html = _make_dvcons_table(meta_dict=cons_meta, vals_dict=cons_vals, kind='constraint',
                                   # precision=precision,
                                   cols=['min', 'max', 'mean', 'lower', 'upper', 'equals', 'ref',
                                         'ref0', 'units', 'plot'])

    # driver_info_html = _make_opt_info_table(prob.driver)
    driver_info_html = _make_opt_value_table(prob.driver)

    this_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(this_dir, 'opt_report_template.html'), 'r') as f:
        template = f.read()


    print("writing opt report")



    with open(outfile, 'w') as f:
        s = template.format(title=prob._name, header=header_html, objs=objs_html,
                                desvars=desvars_html, cons=cons_html, driver=driver_info_html,
                                driver_class=type(prob.driver).__name__)
        f.write(s)
        # f.write(template.format(title=repr(prob), header=header_html, objs=objs_html,
        #                         desvars=desvars_html, cons=cons_html, driver=driver_info_html,
        #                         driver_class=type(prob.driver).__name__))