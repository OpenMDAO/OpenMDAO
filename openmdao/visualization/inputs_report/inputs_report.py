"""
A Viewer for OpenMDAO inputs.
"""
import pathlib
import functools

import numpy as np

try:
    from IPython.display import IFrame, display, HTML
except ImportError:
    IFrame = display = None

from openmdao.core.problem import Problem
from openmdao.utils.mpi import MPI
from openmdao.utils.general_utils import printoptions, issue_warning
from openmdao.utils.om_warnings import OMDeprecationWarning
from openmdao.utils.reports_system import register_report
from openmdao.visualization.tables.table_builder import generate_table


def _unit_str(meta):
    try:
        u = meta['units']
    except KeyError:
        return ''
    return '' if u is None else u


def _get_val_cells(val):
    if isinstance(val, np.ndarray) and val.size > 0:
        minval = np.min(val)
        maxval = np.max(val)
        if val.size > 5:
            val = f"| {np.linalg.norm(val):.4g} |"
        else:
            val = np.array2string(val)
        return val, minval, maxval

    return val, val, val


def _arr_fmt(formatter, string):
    return formatter.format(string)


def inputs_report(prob, outfile=None, display=True, precision=6, title=None,
                  tablefmt='tabulator'):
    """
    Generate a self-contained html file containing a detailed inputs report.

    Optionally pops up a web browser to view the file.

    Parameters
    ----------
    prob : Problem
        View the inputs for this Problem.
    outfile : str, optional
        The name of the output file.  Defaults to 'inputs.html' for web-based
        table formats and stdout for text based ones.
    display : bool, optional
        If True, display the table. Defaults to True.
    precision : int, optional
        Sets the precision for displaying array values.
    title : str, optional
        Sets the title of the web page.
    tablefmt : str, optional
        Format for generated table. Should be one of ['text', 'github', 'rst', 'html', 'tabulator'].
        Defaults to 'tabulator' which generates a sortable, filterable web-based table.
    """
    if isinstance(prob, Problem):
        model = prob.model
    else:
        raise RuntimeError("Input report requires a Problem instance, but got a "
                           f"'{type(prob).__name__}'.")

    connections = model._conn_global_abs_in2out

    if not connections:  # only possible if top level system is a component
        return

    if model._outputs is None:
        raise RuntimeError("Can't generate inputs report. Input values are unknown because "
                           "final_setup has not been called.")

    # get absolute src names of design vars
    desvars = model.get_design_vars(recurse=True, use_prom_ivc=False)

    headers = ['Absolute Name', 'Input Name', 'Source Name', 'Source is IVC', 'Source is DV',
               'Units', 'Shape', 'Tags', 'Val', 'Min Val', 'Max Val', 'Absolute Source', ]
    column_meta = [{'header': h} for h in headers]
    column_meta[9]['format'] = '{:.6g}'
    column_meta[10]['format'] = '{:.6g}'
    column_meta[1]['tooltip'] = 'c0'
    column_meta[2]['tooltip'] = 'c11'
    column_meta[0]['visible'] = False  # abs input is only used for a tooltip, so hide it
    column_meta[11]['visible'] = False  # abs src is only used for a tooltip, so hide it

    rows = []
    with printoptions(formatter={'float': functools.partial(_arr_fmt, '{:.6g}')},
                      suppress=True, threshold=10000):
        for target, meta in model._var_allprocs_abs2meta['input'].items():
            prom = model._var_allprocs_abs2prom['input'][target]
            src = connections[target]
            sprom = model._var_allprocs_abs2prom['output'][src]
            val = model.get_val(target, get_remote=True, from_src=not src.startswith('_auto_ivc.'))
            smeta = model._var_allprocs_abs2meta['output'][src]
            src_is_ivc = 'openmdao:indep_var' in smeta['tags'] or 'indep_var' in smeta['tags']
            if 'indep_var' in smeta['tags'] and 'openmdao:indep_var' not in smeta['tags']:
                issue_warning(f'source output {sprom} is tagged with the deprecated `indep_var`'
                              f' tag. Please change this tag to `openmdao:indep_var` as'
                              f' `indep_var` will be deprecated in a future release.',
                              category=OMDeprecationWarning)

            vcell, mincell, maxcell = _get_val_cells(val)

            rows.append([target, prom, sprom, src_is_ivc, src in desvars, _unit_str(meta),
                         meta['shape'], sorted(meta['tags']), vcell, mincell, maxcell, src])

    for target, meta in model._var_discrete['input'].items():
        prom = model._var_allprocs_abs2prom['input'][target]
        src = connections[target]
        val = model.get_val(target, get_remote=True, from_src=not src.startswith('_auto_ivc.'))
        smeta = model._var_discrete['output'][src]
        src_is_ivc = 'openmdao:indep_var' in smeta['tags'] or 'indep_var' in smeta['tags']
        if 'indep_var' in smeta['tags'] and 'openmdao:indep_var' not in smeta['tags']:
            issue_warning(f'source output {src} is tagged with the deprecated `indep_var` tag.'
                          f' Please change this tag to `openmdao:indep_var` as `indep_var` will be'
                          f' deprecated in a future release.',
                          category=OMDeprecationWarning)

        rows.append([target, prom, src, src_is_ivc, src in desvars, '', None, sorted(meta['tags']),
                     val, None, None, src])

    if not rows:
        column_meta = []

    kwargs = {'rows': rows, 'tablefmt': tablefmt, 'column_meta': column_meta}
    if title is not None and tablefmt in ('html', 'tabulator'):
        kwargs['title'] = title
        kwargs['center'] = True

    table = generate_table(**kwargs)
    if tablefmt == 'tabulator':
        table._table_meta['initialHeaderFilter'] = [
            # set the filter defaults for Source is IVC and Source is DV so that the user will
            # initially see only those inputs that they are responsible for setting.

            # set initial filter for 'Source is IVC' to True
            {'field': 'c3', 'value': True},
            # set initial filter for 'Source is DV' to False
            {'field': 'c4', 'value': False},
        ]

    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    if display:
        if tablefmt in ('html', 'tabulator') and outfile is None:
            outfile = 'inputs.html'
        table.display(outfile)
    elif outfile is not None:
        table.write(outfile)


# inputs report definition
def _run_inputs_report(prob, report_filename='inputs.html'):

    path = str(pathlib.Path(prob.get_reports_dir()).joinpath(report_filename))
    inputs_report(prob, display=False, outfile=path,
                  title=f'Inputs Report for {prob._name}', tablefmt='tabulator')


def _inputs_report_register():
    register_report('inputs', _run_inputs_report, 'Inputs report',
                    'Problem', 'final_setup', 'post')
