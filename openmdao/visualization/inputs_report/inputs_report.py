import os
import pathlib
import json
from itertools import chain
from collections import defaultdict

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
from openmdao.utils.table_builder import generate_table


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
        if val.size > 10:
            val = f"|{np.linalg.norm(val)}|"
        else:
            val = np.array2string(val)
        return val, minval, maxval

    return val, None, None


def inputs_report(prob, outfile=None, display=True, precision=6, title=None,
                  tablefmt='tabulate'):
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
    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    # since people will be used to passing the Problem as the first arg to
    # the N2 diagram funct, allow them to pass a Problem here as well.
    if isinstance(prob, Problem):
        model = prob.model
    else:
       raise RuntimeError("Input report requires a Problem instance, but got a "
                          f"'{type(prob).__name__}'.")

    connections = model._conn_global_abs_in2out

    if model._outputs is None:
        raise RuntimeError("Can't generate inputs report. Input values are unknown because "
                           "final_setup has not been called.")

    # get absolute src names of design vars
    desvars =  model.get_design_vars(recurse=True, use_prom_ivc=False)

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

    table = generate_table(rows, tablefmt=tablefmt, headers=headers)
    if display:
        if tablefmt in ('html', 'tabulate') and outfile is None:
            outfile = 'inputs.html'
        table.display(outfile)
    elif outfile is not None:
        table.write(outfile)

    return table


# inputs report definition
def _run_inputs_report(prob, report_filename='inputs.html'):

    path = str(pathlib.Path(prob.get_reports_dir()).joinpath(report_filename))
    inputs_report(prob, display=False, outfile=path,
                  title=f'Inputs Report for {prob._name}', tablefmt='tabulator')


def _inputs_report_register():
    register_report('inputs', _run_inputs_report, 'Inputs report',
                    'Problem', 'final_setup', 'post')
