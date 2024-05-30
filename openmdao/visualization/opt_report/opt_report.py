"""
Generate report on the results of the optimization.
"""
import base64
import datetime
from functools import partial
import io
import os
import pathlib
import sys
import time

import numpy as np

from openmdao.utils.reports_system import register_report

try:
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib import patches
except ImportError:
    mpl = None


from openmdao.core.constants import INF_BOUND
from openmdao.utils.mpi import MPI
from openmdao.utils.om_warnings import issue_warning, DriverWarning
from openmdao.visualization.tables.table_builder import generate_table


# Report file constants
_default_optimizer_report_filename = 'opt_report.html'
_optimizer_report_template = 'opt_report_template.html'

MAX_VARNAME_LENGTH = 48

# value tolerances
_bounds_tolerance = 1.0E-4
_near_bounds_tolerance = 1.0E-4
_equality_constraint_tolerance = 1e-4

# Colors
#     from https://jfly.uni-koeln.de/color/ some colorblind friendly colors
_out_of_bounds_test_color = 'rgb(213, 94, 0)'
_out_of_bounds_plot_color = (0.83529, 0.36862, 0.)
_out_of_bounds_plot_alpha = 0.2

_out_of_bounds_plot_hatch_pattern = 'xxxxx'
_out_of_bounds_constraint_visual_hatch_pattern = 'xxxxxxxxx'

_out_of_bounds_hatch_color = (0., 0., 0.)
_out_of_bounds_hatch_width = 0.3

_in_bounds_plot_color = (0., 0.61960, 0.45098)
_in_bounds_plot_alpha = 0.2
_in_bounds_constraint_visual_alpha = 0.2

_value_plot_color = 'black'
_plot_marker_size = 0.1
_plot_value_linewidth = 0.5
_equality_constraint_dot_size = 3

# overall image parameters and layout
_sparkline_figsize = (3, 0.8)
_scalar_visual_figsize = (2.0, .2)
_plot_dpi = 150
_plot_pad_inches = 0
_plot_padding_fraction = 0.2

# Font sizes
_sparkline_font_size = 7
_scalar_visual_font_size = 4

# variable value formatting
_variable_label_format = '5.3g'

# https://jfly.uni-koeln.de/color/
# scalar visual parameters
_plot_x_max = 1.0
_plot_y_max = 1.0
_plot_y_margin = 0.2
_lower_plot = _plot_x_max / 3.
_upper_plot = 2 * _plot_x_max / 3.
_pointer_half_width = 0.04
_pointer_height = 0.4
_text_height = 0.3
_pointer_line_width = 0.05
_near_bound_highlight_half_width = 0.05
_near_bound_highlight_half_width_y_min = 0.0
_near_bound_highlight_half_width_y_max = _plot_y_max
_near_bound_highlight_alpha = 0.7
_equality_bound_width = 0.01
# For the ellipsis that are displayed when scalar is way outside of bounds
_ellipsis_x_offset = 0.2
_ellipse_width = 0.01
_ellipse_height = 0.15


def _optimizer_report_register():
    register_report('optimizer', opt_report, 'Summary of optimization',
                    'Problem', 'run_driver', 'post')


def opt_report(prob, outfile=None):
    """
    Write a summary of the optimization to the given file.

    Parameters
    ----------
    prob : Problem
        The Problem for which the optimization report is being generated.
    outfile : str or None
        The path to the HTML file to be written.  If None (default), write to the default report
        output path.
    """
    driver = prob.driver
    if not driver.supports['optimization']:
        driver_class = type(driver).__name__
        issue_warning(f"The optimizer report is not applicable for Driver type '{driver_class}', "
                      "which does not support optimization", category=DriverWarning)
        return

    # only create report on rank 0
    create = MPI is None or MPI.COMM_WORLD.rank == 0

    if not outfile:
        outfile = _default_optimizer_report_filename

    outfilepath = pathlib.Path(prob.get_reports_dir(force=True)).joinpath(outfile)

    driver_scaling = True

    get_prom_name = prob.model._get_prom_name

    # Collect the entire array of array valued desvars and constraints (ignore indices)
    objs_vals = {}
    desvars_vals = {}
    cons_vals = {}

    objs_meta = {}
    desvars_meta = {}
    cons_meta = {}

    with prob.model._scaled_context_all():
        for abs_name, meta in driver._objs.items():
            prom_name = get_prom_name(abs_name)
            objs_meta[prom_name] = meta
            objs_vals[prom_name] = \
                driver.get_objective_values(driver_scaling=driver_scaling)[abs_name]

        for abs_name, meta in driver._designvars.items():
            prom_name = get_prom_name(abs_name)
            desvars_meta[prom_name] = meta
            desvars_vals[prom_name] = \
                driver.get_design_var_values(driver_scaling=driver_scaling)[abs_name]

        for abs_name, meta in prob.driver._cons.items():
            if meta.get('alias') is not None:
                prom_name = abs_name
            else:
                prom_name = get_prom_name(abs_name)
            cons_meta[prom_name] = meta
            cons_vals[prom_name] = \
                driver.get_constraint_values(driver_scaling=driver_scaling)[abs_name]

    header_html = _make_header_table(prob)

    objs_html = _make_obj_table(objs_meta=objs_meta, objs_vals=objs_vals,
                                cols=['val', 'ref', 'ref0', 'adder', 'scaler', 'units'])

    desvars_html = _make_dvcons_table(meta_dict=desvars_meta, vals_dict=desvars_vals, kind='desvar',
                                      cols=['min', 'max', 'mean', 'lower', 'upper', 'equals', 'ref',
                                            'ref0', 'units', 'visual'])

    cons_html = _make_dvcons_table(meta_dict=cons_meta, vals_dict=cons_vals, kind='constraint',
                                   cols=['min', 'max', 'mean', 'lower', 'upper', 'equals', 'ref',
                                         'ref0', 'units', 'visual'])

    driver_info_html = _make_opt_value_table(driver)

    this_dir = os.path.dirname(os.path.abspath(__file__))

    if create:
        with open(os.path.join(this_dir, _optimizer_report_template), 'r', encoding='utf-8') as f:
            template = f.read()

        with open(outfilepath, 'w') as f:
            s = template.format(title=prob._name, header=header_html, objs=objs_html,
                                desvars=desvars_html, cons=cons_html, driver=driver_info_html,
                                driver_class=type(driver).__name__)
            f.write(s)


def _make_header_table(prob):
    """
    Make the HTML table at the top of the report with basic info about the optimization run.

    Parameters
    ----------
    prob : OpenMDAO Problem object
        The report will be run on this Problem.

    Returns
    -------
    str
        HTML table that displays basic optimization run info.
    """
    t = datetime.datetime.now()
    time_stamp = t.strftime("%Y-%m-%d %H:%M:%S %Z")
    runtime = prob.driver.result.runtime
    runtime_ms = (runtime * 1000.0) % 1000.0
    runtime_formatted = \
        f"{time.strftime('%H hours %M minutes %S seconds', time.gmtime(runtime))} " \
        f"{runtime_ms:.1f} milliseconds"

    rows = list()
    rows.append(['Problem:', prob._name])
    rows.append(['Script:', sys.argv[0]])
    rows.append(['Optimizer:', prob.driver._get_name()])
    rows.append(['Number of driver iterations:', prob.driver.result.iter_count])
    rows.append(['Number of model evals:', prob.driver.result.model_evals])
    rows.append(['Number of deriv evals:', prob.driver.result.deriv_evals])
    rows.append(['Execution start time:', time_stamp])
    rows.append(['Wall clock run time:', runtime_formatted])
    rows.append(['Exit status:', prob.driver.result.exit_status])

    return generate_table(rows, tablefmt='html')


def _make_opt_value_table(driver):
    """
    Make the HTML table that summarizes the optimizer settings.

    Parameters
    ----------
    driver : OpenMDAO Driver object
        The table will include info on the settings for this driver.

    Returns
    -------
    str
        HTML table that displays driver settings info.
    """
    opt_settings = []
    for key, meta in driver.options.items():
        meta = driver.options._dict[key]
        opt_settings.append((key, meta['val'], meta['desc']))
    opt_settings_table = generate_table(opt_settings, headers=['Setting', 'Val', 'Description'],
                                        tablefmt='html')

    html = ''
    if opt_settings_table is not None:
        html += f'\n{opt_settings_table}'

    return html


def _make_obj_table(objs_meta, objs_vals,
                    cols=['size', 'index', 'val', 'ref', 'ref0', 'adder', 'scaler', 'units']):
    """
    Make a table of info about the objective.

    Parameters
    ----------
    objs_meta : dict
        Dictionary of metadata about the objectives.
    objs_vals : dict
        Dictionary of values for the objectives.
    cols : list
        List of columns to be displayed in the table.

    Returns
    -------
    str
        An HTML image tag containing the table holding the info about the variable.
    """
    _col_names = cols if cols is not None else []
    col_names = ['name'] + _col_names

    # Get the values for all the elements in the tables
    rows = []
    for prom_name, meta in objs_meta.items():
        row = {}
        if meta['ref0'] is None and meta['ref'] is None and meta['scaler'] is None and \
                meta['adder'] is None:
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

    return generate_table(rows, headers='keys', tablefmt='html')


def _make_dvcons_table(meta_dict, vals_dict, kind,
                       cols=['lower', 'upper', 'ref', 'ref0', 'adder', 'scaler', 'units', 'min',
                             'max', 'visual']):
    """
    Make a table of info about either design variables or constraints.

    Parameters
    ----------
    meta_dict : dict
        Dictionary of metadata about the variables.
    vals_dict : dict
        Dictionary of values for the variables.
    kind : str, must be 'desvar' or 'constraint'
        Indicates whether table is for 'desvar' or 'constraint'.
    cols : list
        List of columns to be displayed in the table.

    Returns
    -------
    str
        An HTML image tag containing the table holding the info about the variable.
    """
    _col_names = cols if cols is not None else []
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

        # the scipy optimizer, when using COBYLA, creates constraints under the hood.
        #  But the values are not given by the driver, so use this as a sign that this
        #  variable should be skipped
        if name not in vals_dict:
            continue

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
                row[col_name] = _indicate_value_is_derived_from_array(mean_val, vals_dict[name])
            elif col_name == 'min':
                if isinstance(vals_dict[name], np.ndarray):
                    min_val = min(vals_dict[name])  # get min. Could be an array
                    min_val_as_str = _indicate_value_is_derived_from_array(min_val, vals_dict[name])
                else:
                    min_val_as_str = str(vals_dict[name])
                comp = (vals_dict[name] - meta['lower']) < _bounds_tolerance
                if np.any(comp):
                    row[col_name] = \
                        f'<span style="color:{_out_of_bounds_test_color}">({min_val_as_str})</span>'
                else:
                    row[col_name] = min_val_as_str
            elif col_name == 'max':
                if isinstance(vals_dict[name], np.ndarray):
                    max_val = max(vals_dict[name])  # get max. Could be an array
                    max_val_as_str = _indicate_value_is_derived_from_array(max_val, vals_dict[name])
                else:
                    max_val_as_str = str(vals_dict[name])
                comp = (meta['upper'] - vals_dict[name]) < _bounds_tolerance
                if np.any(comp):
                    row[col_name] = \
                        f'<span style="color:{_out_of_bounds_test_color}">({max_val_as_str})</span>'
                else:
                    row[col_name] = max_val_as_str
            elif col_name == 'lower':
                if np.all(meta[col_name] == -INF_BOUND):
                    row[col_name] = None
                else:
                    lower_val = np.mean(meta[col_name])
                    row[col_name] = _indicate_value_is_derived_from_array(lower_val, meta[col_name])
            elif col_name == 'upper':
                if np.all(meta[col_name] == INF_BOUND):
                    row[col_name] = None
                else:
                    upper_val = np.mean(meta[col_name])
                    row[col_name] = _indicate_value_is_derived_from_array(upper_val, meta[col_name])
            elif col_name == 'equals':
                if 'equals' not in meta or meta['equals'] is None:
                    row[col_name] = ''
                else:
                    equals_val = np.mean(meta[col_name])
                    row[col_name] = _indicate_value_is_derived_from_array(equals_val,
                                                                          meta[col_name])
            elif col_name == 'units':
                if meta['units'] is not None:
                    row[col_name] = meta[col_name]
                else:
                    if alias:
                        row[col_name] = meta_dict[alias][col_name]
                    else:
                        row[col_name] = meta_dict[name][col_name]
            elif col_name == 'visual':
                if mpl:
                    val = vals_dict[name]
                    if np.isscalar(val) or val.shape == (1,):
                        row['visual'] = _constraint_plot(meta=meta, val=vals_dict[name], kind=kind)
                    else:
                        row['visual'] = _sparkline(meta=meta, val=vals_dict[name], kind=kind)
                else:
                    row['visual'] = \
                        '<span class="plot-unavailable">Visuals require matplotlib</span>'
            elif col_name == 'size':
                row[col_name] = int(meta[col_name])  # sometimes size in the meta data is a numpy
                # array so generate_table does different formatting for that
            elif col_name in ['ref', 'ref0']:
                if meta[col_name] is not None:
                    derived_val = np.mean(meta[col_name])
                    row[col_name] = _indicate_value_is_derived_from_array(derived_val,
                                                                          meta[col_name])
                else:
                    row[col_name] = 'None'
            else:
                row[col_name] = meta[col_name]

        rows.append(row)

    return generate_table(rows, headers='keys', tablefmt='html', safe=False, precision='.4e')


def _sparkline(kind, meta, val, width=300):
    """
    Given the metadata & value of design variable or constraint, make an html-embeddable sparkline.

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
    _backend = mpl.get_backend()  # Save it so we can set it back at the end
    mpl.use('Agg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=_sparkline_figsize,
                           tight_layout=False, dpi=_plot_dpi)

    # settings across all plots
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(f"%{_variable_label_format}"))
    ax.tick_params(axis='x', labelsize=_sparkline_font_size)
    ax.tick_params(axis='y', labelsize=_sparkline_font_size)

    try:
        if kind == 'desvar':
            _desvar_or_ineq_constraint_sparkline(ax, meta, val)
        elif 'equals' not in meta or meta['equals'] is None:
            _desvar_or_ineq_constraint_sparkline(ax, meta, val)
        else:
            _eq_constraint_sparkline(ax, meta, val)
    except (ValueError, IndexError):
        plt.close()
        mpl.use(_backend)  # set it back
        return '<span class="plot-unavailable">Plot unavailable</span>'

    tmpfile = io.BytesIO()
    fig.patch.set_alpha(0.0)  # So that the figures are transparent
    fig.savefig(tmpfile, format='png', bbox_inches='tight', pad_inches=_plot_pad_inches)
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = f'<img width={width} src=\'data:image/png;base64,{encoded}\'>'

    plt.close()
    mpl.use(_backend)  # set it back

    return html


def _desvar_or_ineq_constraint_sparkline(ax, meta, val):
    """
    Use matplotlib to plot a visual showing the values of the desvar or constraint and any bounds.

    Parameters
    ----------
    ax : Matplotlib Axes instance
        Contains most of the figure elements.
    meta : dict
        The metadata associated with the design variable.
    val : np.array
        The value of the design variable.
    """
    indices = np.s_[...] if meta['indices'] is None else meta['indices']()

    _val = np.asarray(val).ravel()

    # get array for the lower and upper bounds
    if isinstance(meta['lower'], float):
        _lower = (meta['lower'] * np.ones_like(val)[indices]).ravel()
    else:
        _lower = np.asarray(meta['lower']).ravel()

    if isinstance(meta['upper'], float):
        _upper = (meta['upper'] * np.ones_like(val)[indices]).ravel()
    else:
        _upper = np.asarray(meta['upper']).ravel()

    # Get the x values and their range
    ar = np.arange(_val.size)
    x_min = 0
    x_max = val.size - 1

    _lower, _lower_no_inf_min, _lower_no_inf_max = _get_bound_array_min_max(_lower, indices)
    _upper, _upper_no_inf_min, _upper_no_inf_max = _get_bound_array_min_max(_upper, indices)

    # get y min and max
    y_min = min(_lower_no_inf_min, _upper_no_inf_min, np.min(_val))
    y_max = max(_lower_no_inf_max, _upper_no_inf_max, np.max(_val))

    # add a little to the top and bottom of the plot
    yrange = y_max - y_min
    ymin_plot = y_min - yrange * _plot_padding_fraction
    ymax_plot = y_max + yrange * _plot_padding_fraction

    xrange = x_max - x_min
    xmin_plot = x_min - xrange * _plot_padding_fraction
    xmax_plot = x_max + xrange * _plot_padding_fraction

    # set plot limits
    ax.set_xlim([xmin_plot, xmax_plot])
    ax.set_ylim([ymin_plot, ymax_plot])

    # set ticks
    ax.set_xticks([x_min, x_max])
    ax.set_yticks([y_min, y_max])

    # plot actual values
    ax.plot(ar, _val, '-o', markersize=_plot_marker_size, color=_value_plot_color,
            linewidth=_plot_value_linewidth)

    # Need to do this because of a bug in matplotlib. If the upper or lowers include INF_BOUND
    #  it affects how the other side of the fill_between is drawn
    _lower = np.clip(_lower, ymin_plot, ymax_plot)
    _upper = np.clip(_upper, ymin_plot, ymax_plot)

    # plot lower area, if exists
    if not (isinstance(meta['lower'], float) and meta['lower'] == -INF_BOUND):
        ax.fill_between(ar, ymin_plot, _lower, color=_out_of_bounds_plot_color,
                        hatch=_out_of_bounds_plot_hatch_pattern, alpha=_out_of_bounds_plot_alpha)
        ax.plot(ar, _lower, color=_out_of_bounds_plot_color, linewidth=_plot_value_linewidth)

    # plot upper area, if exists
    if not (isinstance(meta['upper'], float) and meta['upper'] == INF_BOUND):
        ax.fill_between(ar, _upper, ymax_plot, color=_out_of_bounds_plot_color,
                        hatch=_out_of_bounds_plot_hatch_pattern, alpha=_out_of_bounds_plot_alpha)
        ax.plot(ar, _upper, color=_out_of_bounds_plot_color, linewidth=_plot_value_linewidth)

    # Plot area where bounds are satisfied
    ax.fill_between(ar, _lower, _upper, color=_in_bounds_plot_color, alpha=_in_bounds_plot_alpha)


def _eq_constraint_sparkline(ax, meta, val):
    """
    Plot to matplotlib a visual showing the values of the equality constraint and also the value.

    Parameters
    ----------
    ax : Matplotlib Axes instance
        Contains most of the figure elements.
    meta : dict
        The metadata associated with the variables, including info about the constraint.
    val : np.array
        The value of the variable.
    """
    indices = np.s_[...] if meta['indices'] is None else meta['indices']

    # get value array flattened
    _val = np.asarray(val).ravel()

    # get equal constraint array
    if 'equals' not in meta:
        raise ValueError("Equality constraint sparkline cannot be "
                         "drawn without equality constraint value")
    elif isinstance(meta['equals'], float):
        _equals = (meta['equals'] * np.ones_like(val)[indices]).ravel()
    else:
        _equals = np.asarray(meta['equals']).ravel()

    # get x coordinate array and info
    ar = np.arange(val.size)
    x_min = 0
    x_max = val.size - 1
    xrange = x_max - x_min
    xmin_plot = x_min - xrange * _plot_padding_fraction
    xmax_plot = x_max + xrange * _plot_padding_fraction
    ax.set_xlim([xmin_plot, xmax_plot])
    ax.set_xticks([x_min, x_max])

    # Get y coordinate plotting info
    y_min = min(np.min(_val), np.min(_equals))
    y_max = max(np.max(_val), np.max(_equals))
    yrange = y_max - y_min
    ymin_plot = y_min - yrange * _plot_padding_fraction
    ymax_plot = y_max + yrange * _plot_padding_fraction
    ax.set_ylim([ymin_plot, ymax_plot])
    ax.set_yticks([y_min, y_max])

    # plot the constraint as a line
    ax.plot(ar, _equals, color=_value_plot_color, linewidth=_plot_value_linewidth)

    # plot the actual values as dots colored by whether they satisfy the constraint
    err = _val - _equals
    colors = []
    for e in err:
        if np.abs(e) < _equality_constraint_tolerance:
            colors.append(_in_bounds_plot_color)
        else:
            colors.append(_out_of_bounds_plot_color)
    ax.scatter(ar, _val, color=colors, s=_equality_constraint_dot_size)


def _constraint_plot(kind, meta, val, width=300):
    """
    Given the metadata and value of a design variable or constraint, make an html-embeddable plot.

    Only for scalars

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
    if not (np.isscalar(val) or val.shape == (1,)):
        raise ValueError("Value for the _constraint_plot function must be a "
                         f"scalar. Variable {meta['name']} is not a scalar")
    else:
        try:
            val = val.item()
        except AttributeError:
            pass  # handle other than ndarray, e.g. int

    # If lower and upper bounds are None, return an HTML snippet indicating the issue
    if kind == 'constraint' and meta['upper'] == INF_BOUND and meta['lower'] == -INF_BOUND:
        return '<span class="bounds-unavailable">Both lower and upper bounds are None.</span>'

    if kind == 'desvar' and meta['upper'] == INF_BOUND and meta['lower'] == -INF_BOUND:
        return   # nothing to plot

    # Equality constraints are visualized differently
    equals = meta['equals'] if 'equals' in meta else None
    if equals is not None:
        if abs(val - equals) < _equality_constraint_tolerance:
            html = '<span class="equality-constraint equality-constraint-satisfied">&#10003;</span>'
        else:
            html = '<span class="equality-constraint equality-constraint-violated">&#10007;</span>'
        return html

    # If lower and upper are the same value, visualize the same as an equality constraint
    if (
        'lower' in meta and meta['lower'] != -INF_BOUND
        and
        'upper' in meta and meta['upper'] != INF_BOUND
        and
        meta['lower'] == meta['upper']
    ):
        if abs(val - meta['lower']) < _equality_constraint_tolerance:
            html = '<span class="equality-constraint equality-constraint-satisfied">&#10003;</span>'
        else:
            html = '<span class="equality-constraint equality-constraint-violated">&#10007;</span>'
        return html

    _backend = mpl.get_backend()
    mpl.use('Agg')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=_scalar_visual_figsize, dpi=_plot_dpi)

    var_bounds_plot(kind, ax, float(val), meta['lower'], meta['upper'])
    tmpfile = io.BytesIO()
    fig.savefig(tmpfile, format='png', transparent=True, bbox_inches='tight',
                pad_inches=_plot_pad_inches)
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html = f'<img width={width} src=\'data:image/png;base64,{encoded}\'>'

    plt.close()
    mpl.use(_backend)

    return html


def _indicate_value_is_derived_from_array(derived_value, original_value):
    """
    Given a value or bound, & its derived value for use in report, format the output for the tables.

    Parameters
    ----------
    derived_value : float
        The value of the variable to be shown in the table.
    original_value : float or np.ndarray
        The original value of the variable. Need this to know if it was a scalar.

    Returns
    -------
    str
        A formatted string to be used to display the value of this variable in the tables.
    """
    displayed_string = f'{derived_value:{_variable_label_format}}'.strip()
    if isinstance(original_value, np.ndarray) and original_value.size > 1:
        out = '|{}|'.format(displayed_string)
    else:
        out = displayed_string
    return out


def _get_bound_array_min_max(bounds, indices):
    """
    Given a bounds (either lower or upper) & indices, return the full array bounds, min,& max.

    Parameters
    ----------
    bounds : float
        An array representing either lower or upper bounds of a desvar or constraint.
    indices : sequence of int
        If variable is an array, these indicate which entries are of
        interest for this particular response.

    Returns
    -------
    _bounds
        An array representing the bounds.
    _bounds_min
        The minimum of the bounds, excluding any entries in the _bounds array that
          equal -INF_BOUND or INF_BOUND. If all the values are those, return INF_BOUND
          so that when mins are taken that include this value, it doesn't affect the result
    _bounds_max
        The maximum of the bounds, excluding any entries in the _bounds array that
          equal -INF_BOUND or INF_BOUND. If all the values are those, return -INF_BOUND
          so that when maxes are taken that include this value, it doesn't affect the result
    """
    _bounds_no_inf = bounds[np.where(((bounds != -INF_BOUND) & (bounds != INF_BOUND)))]
    if _bounds_no_inf.size > 0:
        _bounds_min = np.min(_bounds_no_inf)
        _bounds_max = np.max(_bounds_no_inf)
    else:  # so that when we do min and max on these they are not involved in getting the min/max
        _bounds_min = INF_BOUND
        _bounds_max = - INF_BOUND

    if isinstance(bounds, float):
        _bounds = (bounds * np.ones_like(bounds)[indices]).ravel()
    else:
        _bounds = np.asarray(bounds).ravel()

    return _bounds, _bounds_min, _bounds_max


def var_bounds_plot(kind, ax, value, lower, upper):
    """
    Make a plot to show where a variable is relative to constraints.

    Parameters
    ----------
    kind : str
        One of 'desvar' or 'constraint' to specify which type of plot is being made.
    ax : Matplotlib Axes instance
        Contains most of the figure elements.
    value : float
        The design var value.
    lower : float or None
        Lower constraint.
    upper : float or None
        Upper constraint.
    """
    # must handle 5 cases if both upper and lower are given:
    #  - value much less than lower
    #  - value a little less than lower
    #  - value between lower and upper
    #  - value a little greater than upper
    #  - value much greater than upper

    # also need to handle one-sided constraints where only one of lower and upper is given

    # Basic plot setup
    plt.rcParams['hatch.linewidth'] = _out_of_bounds_hatch_width  # can't seem to do this any other
    #                                                              way. Cannot control hatch
    #                                                              pattern in Rectangle with
    #                                                              set_linewidth
    plt.rcParams['hatch.color'] = _out_of_bounds_plot_color
    ax.set_axis_off()
    ax.set_xlim([-_pointer_half_width, _plot_x_max + _pointer_half_width])
    ax.set_ylim(-_pointer_height - _text_height - _plot_y_margin,
                _plot_y_max + _text_height + _plot_y_margin)

    func_val_to_plot_coord = partial(_val_to_plot_coord, lower=lower, upper=upper)
    value_in_plot_coord = func_val_to_plot_coord(value)

    if upper == INF_BOUND:  # there is a lower bound
        _draw_in_or_out_bound_section(ax, 0, _plot_x_max / 2, False)
        _draw_in_or_out_bound_section(ax, _plot_x_max / 2, _plot_x_max / 2, True)
        _draw_boundary_label(ax, _plot_x_max / 2,
                             "lower = " + f"{lower:{_variable_label_format}}".strip())

        if abs(value - lower) < _near_bounds_tolerance:
            pointer_plot_coord = _plot_x_max / 2
            _draw_bound_highlight(ax, _plot_x_max / 2)
            pointer_color = _in_bounds_plot_color
        elif value >= lower:
            pointer_plot_coord = 3. * _plot_x_max / 4
            pointer_color = _in_bounds_plot_color
        else:
            pointer_color = _out_of_bounds_plot_color
            pointer_plot_coord = 1. * _plot_x_max / 4
        _draw_pointer_and_label(ax, pointer_plot_coord, pointer_color, value)
        return

    if lower == -INF_BOUND:  # there is an upper bound
        _draw_in_or_out_bound_section(ax, 0, _plot_x_max / 2, True)
        _draw_in_or_out_bound_section(ax, _plot_x_max / 2, _plot_x_max / 2, False)
        _draw_boundary_label(ax, _plot_x_max / 2,
                             "upper = " + f"{upper:{_variable_label_format}}".strip())

        if abs(value - upper) < _near_bounds_tolerance:  # value is close to bound
            pointer_plot_coord = _plot_x_max / 2
            _draw_bound_highlight(ax, _plot_x_max / 2)
            pointer_color = _in_bounds_plot_color
        elif value <= upper:  # value satisfies bound
            pointer_plot_coord = 1. * _plot_x_max / 4
            pointer_color = _in_bounds_plot_color
        else:  # value violates the bound
            pointer_plot_coord = 3. * _plot_x_max / 4
            pointer_color = _out_of_bounds_plot_color

        _draw_pointer_and_label(ax, pointer_plot_coord, pointer_color, value)
        return

    # If we get this far, there are both lower and upper bounds

    # in bounds visual is always the same
    _draw_in_or_out_bound_section(ax, _lower_plot, _upper_plot - _lower_plot, True)

    # draw below bound
    if value_in_plot_coord >= 0.0:
        _draw_in_or_out_bound_section(ax, 0, _lower_plot, False)
    else:  # value is off to the left of the plot
        _draw_in_or_out_bound_section(ax, _lower_plot / 3., 2 * _lower_plot / 3., False)
        _draw_ellipsis(ax, 0.0)

    # draw upper bound
    if value_in_plot_coord <= _plot_x_max:
        _draw_in_or_out_bound_section(ax, _upper_plot, _lower_plot, False)
    else:  # value is off to the right of the plot
        _draw_in_or_out_bound_section(ax, _upper_plot, 2 * _lower_plot / 3., False)
        _draw_ellipsis(ax, _upper_plot + 2 * _lower_plot / 3)

    # draw upper and lower labels
    _draw_boundary_label(ax, func_val_to_plot_coord(lower), str(lower))
    _draw_boundary_label(ax, func_val_to_plot_coord(upper), str(upper))

    # add highlight if value near a bound
    if abs(value - lower) / abs(upper - lower) < _near_bounds_tolerance:
        _draw_bound_highlight(ax, func_val_to_plot_coord(lower))
    elif abs(value - upper) / abs(upper - lower) < _near_bounds_tolerance:
        _draw_bound_highlight(ax, func_val_to_plot_coord(upper))

    # pointer and pointer label
    if value_in_plot_coord < 0.0:
        pointer_plot_coord = 0.0
    elif value_in_plot_coord > _plot_x_max:
        pointer_plot_coord = _plot_x_max
    else:
        pointer_plot_coord = value_in_plot_coord
    pointer_color = _in_bounds_plot_color if (lower <= value <= upper) \
        else _out_of_bounds_plot_color
    _draw_pointer_and_label(ax, pointer_plot_coord, pointer_color, value)


# A series of functions used to draw parts of the scalar constraints visual
def _draw_in_or_out_bound_section(ax, x_left, width, is_in_bound):
    if is_in_bound:
        color = _in_bounds_plot_color
        hatch = None
        alpha = _in_bounds_plot_alpha
    else:
        color = _out_of_bounds_plot_color
        hatch = _out_of_bounds_constraint_visual_hatch_pattern
        # hatch = None
        alpha = _out_of_bounds_plot_alpha
    rectangle = patches.Rectangle((x_left, 0), width, _plot_y_max, facecolor=color,
                                  hatch=hatch, alpha=alpha)
    ax.add_patch(rectangle)


def _draw_bound_highlight(ax, x):
    rectangle = patches.Rectangle((
        x - _near_bound_highlight_half_width, _near_bound_highlight_half_width_y_min),
        2 * _near_bound_highlight_half_width, _near_bound_highlight_half_width_y_max,
        edgecolor='black', facecolor='yellow', alpha=_near_bound_highlight_alpha)
    ax.add_patch(rectangle)


def _draw_ellipsis(ax, x_left):
    # Draw three dots as an ellipsis to show that the value is beyond
    #   either the left or right edge of the plot
    for i in [1, 2, 3]:
        circle = patches.Ellipse((x_left + i * _lower_plot / 12., _ellipsis_x_offset),
                                 _ellipse_width, _ellipse_height,
                                 facecolor=_out_of_bounds_plot_color)
        ax.add_patch(circle)


def _draw_boundary_label(ax, pointer_plot_coord, s):
    ax.text(pointer_plot_coord, _plot_y_max + _text_height,
            s,
            horizontalalignment='center',
            verticalalignment='bottom',
            size=_scalar_visual_font_size
            )


def _draw_pointer_and_label(ax, pointer_plot_coord, pointer_color, value):
    pts = np.array([
        [pointer_plot_coord - _pointer_half_width, -_pointer_height],
        [pointer_plot_coord + _pointer_half_width, -_pointer_height],
        [pointer_plot_coord, 0.0]
    ])
    p = patches.Polygon(pts, closed=True, facecolor=pointer_color, edgecolor='black',
                        linewidth=_pointer_line_width)
    ax.add_patch(p)
    plt.text(pointer_plot_coord, -_pointer_height - _text_height,
             f"{value:{_variable_label_format}}".strip(),
             horizontalalignment='center', verticalalignment='top', size=_scalar_visual_font_size)


def _val_to_plot_coord(value, lower, upper):
    # need to get function that maps actual values to 0.0 to 1.0
    # and where lower maps to 1./3 and upper to 2/3
    # Used with Python's functools to make a Python function out of this
    plot_coord = 1. / 3. + (value - lower) / (upper - lower) * 1. / 3.
    if isinstance(plot_coord, np.ndarray):
        return plot_coord[0]
    return plot_coord
