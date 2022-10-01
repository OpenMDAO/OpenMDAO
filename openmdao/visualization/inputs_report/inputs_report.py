"""
A Viewer for OpenMDAO inputs.
"""
import pathlib

import numpy as np

try:
    from IPython.display import IFrame, display, HTML
except ImportError:
    IFrame = display = None

from openmdao.core.problem import Problem
from openmdao.utils.units import convert_units
from openmdao.utils.mpi import MPI
from openmdao.utils.webview import webview
from openmdao.utils.general_utils import printoptions
from openmdao.utils.notebook_utils import notebook, colab
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.reports_system import register_report
from openmdao.visualization.tables.table_builder import generate_table


def _unit_str(meta):
    try:
        u = meta['units']
    except KeyError:
        return ''
    return '' if u is None else u


def _get_val_cells(val):
    if isinstance(val, np.ndarray):
        minval = np.min(val)
        maxval = np.max(val)
        if val.size > 5:
            val = f"| {np.linalg.norm(val)} |"
        else:
            val = np.array2string(val)
        return val, minval, maxval

    return val, None, None


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

    rows = []
    with printoptions(precision=precision, suppress=True, threshold=10000):
        for target, meta in model._var_allprocs_abs2meta['input'].items():
            prom = model._var_allprocs_abs2prom['input'][target]
            src = connections[target]
            val = model.get_val(target, get_remote=True, from_src=not src.startswith('_auto_ivc.'))
            smeta = model._var_allprocs_abs2meta['output'][src]
            src_is_ivc = 'indep_var' in smeta['tags']
            vcell, mincell, maxcell = _get_val_cells(val)

            rows.append([target, prom, src, src_is_ivc, src in desvars, _unit_str(meta),
                         meta['shape'], sorted(meta['tags']), vcell, mincell, maxcell])

    for target, meta in model._var_discrete['input'].items():
        prom = model._var_allprocs_abs2prom['input'][target]
        src = connections[target]
        val = model.get_val(target, get_remote=True, from_src=not src.startswith('_auto_ivc.'))
        smeta = model._var_discrete['output'][src]
        src_is_ivc = 'indep_var' in smeta['tags']

        rows.append([target, prom, src, src_is_ivc, src in desvars, '', None, sorted(meta['tags']),
                     val, None, None])

    headers = ['Absolute Name', 'Promoted Name', 'Source Name', 'Source is IVC', 'Source is DV',
               'Units', 'Shape', 'Tags', 'Val', 'Min Val', 'Max Val']
    column_meta = [{'header': h} for h in headers]

    # kwargs = {}

    # if tablefmt == 'tabulator':
    #     column_meta[0]['responsive'] = 10  # hide abs name column first if width too small
    #     column_meta[0]['minWidth'] = 100
    #     column_meta[1]['responsive'] = 0  # don't hide prom name
    #     column_meta[2]['responsive'] = 0  # don't hide src name
    #     column_meta[3]['responsive'] = 0  # don't hide src is IVC
    #     column_meta[4]['responsive'] = 0  # don't hide src is DV
    #     column_meta[5]['responsive'] = 4  # units
    #     column_meta[6]['responsive'] = 5  # shape
    #     column_meta[7]['responsive'] = 9  # tags
    #     column_meta[8]['responsive'] = 6  # val
    #     column_meta[9]['responsive'] = 8  # minval
    #     column_meta[10]['responsive'] = 7  # maxval
    #     kwargs['table_meta'] = {
    #         'layout': 'fitDataTable',
    #         'responsiveLayout': 'hide',
    #     }

    if not rows:
        column_meta = []

    table = generate_table(rows, tablefmt=tablefmt, column_meta=column_meta)  # , **kwargs)
    if tablefmt == 'tabulator':
        table._table_meta['initialHeaderFilter'] = [
            # set the filter defaults for Source is IVC and Source is DV so that the user will
            # see only those inputs that they are responsible for setting.

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
