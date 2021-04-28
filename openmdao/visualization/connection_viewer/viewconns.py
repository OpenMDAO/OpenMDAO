
"""Define a function to view connections."""
import os
import sys
import json
from itertools import chain
from collections import defaultdict

import numpy as np

try:
    from IPython.display import IFrame, display, HTML
except ImportError:
    IFrame = display = None

import openmdao
from openmdao.core.problem import Problem
from openmdao.utils.units import convert_units
from openmdao.utils.mpi import MPI
from openmdao.utils.webview import webview
from openmdao.utils.general_utils import printoptions
from openmdao.utils.notebook_utils import notebook, colab
from openmdao.warnings import issue_warning


def _val2str(val):
    if isinstance(val, np.ndarray):
        if val.size > 5:
            return 'array %s' % str(val.shape)
        else:
            return np.array2string(val)

    return str(val)


def view_connections(root, outfile='connections.html', show_browser=True,
                     show_values=True, precision=6, title=None):
    """
    Generate a self-contained html file containing a detailed connection viewer.

    Optionally pops up a web browser to view the file.

    Parameters
    ----------
    root : System or Problem
        The root for the desired tree.

    outfile : str, optional
        The name of the output html file.  Defaults to 'connections.html'.

    show_browser : bool, optional
        If True, pop up a browser to view the generated html file.
        Defaults to True.

    show_values : bool, optional
        If True, retrieve the values and display them.

    precision : int, optional
        Sets the precision for displaying array values.

    title : str, optional
        Sets the title of the web page.
    """
    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    # since people will be used to passing the Problem as the first arg to
    # the N2 diagram funct, allow them to pass a Problem here as well.
    if isinstance(root, Problem):
        system = root.model
    else:
        system = root

    connections = system._problem_meta['model_ref']()._conn_global_abs_in2out

    src2tgts = defaultdict(list)
    units = defaultdict(lambda: '')
    for io in ('input', 'output'):
        for n, data in system._var_allprocs_abs2meta[io].items():
            u = data.get('units', '')
            if u is not None:
                units[n] = u

    vals = {}

    prefix = system.pathname + '.' if system.pathname else ''
    all_vars = {}
    for io in ('input', 'output'):
        all_vars[io] = chain(system._var_abs2meta[io].items(),
                             [(prefix + n, m) for n, m in system._var_discrete[io].items()])

    if show_values and system._outputs is None:
        issue_warning("Values will not be shown because final_setup has not been called yet.",
                      prefix=system.msginfo)

    with printoptions(precision=precision, suppress=True, threshold=10000):

        for t, meta in all_vars['input']:
            s = connections[t]
            if show_values and system._outputs is not None:
                if s.startswith('_auto_ivc.'):
                    val = system.get_val(t, flat=True, get_remote=True,
                                         from_src=False)
                else:
                    val = system.get_val(t, flat=True, get_remote=True)

                    # if there's a unit conversion, express the value in the
                    # units of the target
                    if units[t] and s in system._outputs:
                        val = system.get_val(t, flat=True, units=units[t], get_remote=True)
                    else:
                        val = system.get_val(t, flat=True, get_remote=True)
            else:
                val = ''

            src2tgts[s].append(t)

            vals[t] = val

    NOCONN = '[NO CONNECTION]'
    vals[NOCONN] = ''

    src_systems = set()
    tgt_systems = set()
    for s, _ in all_vars['output']:
        parts = s.split('.')
        for i in range(len(parts)):
            src_systems.add('.'.join(parts[:i]))

    for t, _ in all_vars['input']:
        parts = t.split('.')
        for i in range(len(parts)):
            tgt_systems.add('.'.join(parts[:i]))

    src_systems = [{'name': n} for n in sorted(src_systems)]
    src_systems.insert(1, {'name': NOCONN})
    tgt_systems = [{'name': n} for n in sorted(tgt_systems)]
    tgt_systems.insert(1, {'name': NOCONN})

    tprom = system._var_allprocs_abs2prom['input']
    sprom = system._var_allprocs_abs2prom['output']

    table = []
    idx = 1  # unique ID for use by Tabulator
    for tgt, src in connections.items():
        usrc = units[src]
        utgt = units[tgt]
        if usrc != utgt:
            # prepend these with '!' so they'll be colored red
            if usrc:
                usrc = '!' + units[src]
            if utgt:
                utgt = '!' + units[tgt]

        row = {'id': idx, 'src': src, 'sprom': sprom[src], 'sunits': usrc,
               'val': _val2str(vals[tgt]), 'tunits': utgt,
               'tprom': tprom[tgt], 'tgt': tgt}
        table.append(row)
        idx += 1

    # add rows for unconnected sources
    for src, _ in all_vars['output']:
        if src not in src2tgts:
            if show_values:
                v = _val2str(system._abs_get_val(src))
            else:
                v = ''
            row = {'id': idx, 'src': src, 'sprom': sprom[src], 'sunits': units[src],
                   'val': v, 'tunits': '', 'tprom': NOCONN, 'tgt': NOCONN}
            table.append(row)
            idx += 1

    if title is None:
        title = ''

    data = {
        'title': title,
        'table': table,
        'show_values': show_values,
    }

    viewer = 'connect_table.html'

    code_dir = os.path.dirname(os.path.abspath(__file__))
    libs_dir = os.path.join(os.path.dirname(code_dir), 'common', 'libs')
    style_dir = os.path.join(os.path.dirname(code_dir), 'common', 'style')

    with open(os.path.join(code_dir, viewer), "r") as f:
        template = f.read()

    with open(os.path.join(libs_dir, 'tabulator.min.js'), "r") as f:
        tabulator_src = f.read()

    with open(os.path.join(style_dir, 'tabulator.min.css'), "r") as f:
        tabulator_style = f.read()

    jsontxt = json.dumps(data)

    with open(outfile, 'w') as f:
        s = template.replace("<connection_data>", jsontxt)
        s = s.replace("<tabulator_src>", tabulator_src)
        s = s.replace("<tabulator_style>", tabulator_style)
        f.write(s)

    if notebook:
        # display in Jupyter Notebook
        if not colab:
            display(IFrame(src=outfile, width=1000, height=1000))
        else:
            display(HTML(outfile))

    elif show_browser:
        # open it up in the browser
        from openmdao.utils.webview import webview
        webview(outfile)
