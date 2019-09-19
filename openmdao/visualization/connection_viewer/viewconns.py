
"""Define a function to view connections."""
import os
import sys
import json
import contextlib
from itertools import chain
from collections import defaultdict

from six import iteritems

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


def view_connections(root, outfile='connections.html', show_browser=True,
                     precision=6, title=None):
    """
    Generate a self-contained html file containing a detailed connection viewer.

    Optionally pops up a web browser to view the file.

    Parameters
    ----------
    root : system or Problem
        The root for the desired tree.

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

    # since people will be used to passing the Problem as the first arg to
    # the N2 diagram funct, allow them to pass a Problem here as well.
    if isinstance(root, Problem):
        system = root.model
    else:
        system = root

    input_srcs = system._conn_global_abs_in2out

    connections = {
        tgt: src for tgt, src in iteritems(input_srcs) if src is not None
    }

    src2tgts = defaultdict(list)
    units = {}
    for n, data in iteritems(system._var_allprocs_abs2meta):
        u = data.get('units', '')
        if u is None:
            u = ''
        units[n] = u

    vals = {}

    with printoptions(precision=precision, suppress=True, threshold=10000):

        for t in system._var_abs_names['input']:
            tmeta = system._var_abs2meta[t]
            idxs = tmeta['src_indices']

            if t in connections:
                s = connections[t]
                val = _get_output(system, s, idxs)

                # if there's a unit conversion, express the value in the
                # units of the target
                if units[t] and s in system._outputs:
                    val = convert_units(val, units[s], units[t])

                src2tgts[s].append(t)
            else:  # unconnected param
                val = _get_input(system, t)

            vals[t] = val

    vals['NO CONNECTION'] = ''

    src_systems = set()
    tgt_systems = set()
    for s in system._var_abs_names['output']:
        parts = s.split('.')
        for i in range(len(parts)):
            src_systems.add('.'.join(parts[:i]))

    for t in system._var_abs_names['input']:
        parts = t.split('.')
        for i in range(len(parts)):
            tgt_systems.add('.'.join(parts[:i]))

    src_systems = [{'name': n} for n in sorted(src_systems)]
    src_systems.insert(1, {'name': "NO CONNECTION"})
    tgt_systems = [{'name': n} for n in sorted(tgt_systems)]
    tgt_systems.insert(1, {'name': "NO CONNECTION"})

    tprom = system._var_allprocs_abs2prom['input']
    sprom = system._var_allprocs_abs2prom['output']

    table = []
    idx = 1  # unique ID for use by Tabulator
    for tgt, src in iteritems(connections):
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

    for t in system._var_abs_names['input']:
        if t not in connections:
            row = {'id': idx, 'src': 'NO CONNECTION', 'sprom': 'NO CONNECTION', 'sunits': '',
                   'val': _val2str(vals[t]), 'tunits': units[t], 'tprom': tprom[t], 'tgt': t}
            table.append(row)
            idx += 1

    for src in system._var_abs_names['output']:
        if src not in src2tgts:
            row = {'id': idx, 'src': src, 'sprom': sprom[src], 'sunits': units[src],
                   'val': _val2str(system._outputs[src]),
                   'tunits': '', 'tprom': 'NO CONNECTION', 'tgt': 'NO CONNECTION'}
            table.append(row)
            idx += 1

    if title is None:
        title = ''

    data = {
        'title': title,
        'table': table,
    }

    viewer = 'connect_table.html'

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
        s = template.replace("<connection_data>", jsontxt)
        s = s.replace("<tabulator_src>", tabulator_src)
        s = s.replace("<tabulator_style>", tabulator_style)
        f.write(s)

    if show_browser:
        webview(outfile)


def _get_input(system, name):
    """
    Return the named value if it's local to the process, else "<on remote proc>".
    """
    if name in system._inputs:
        return system._inputs[name]
    return "<on remote proc>"


def _get_output(system, name, idxs=None):
    """
    Return the named value if it's local to the process, else "<on remote proc>".
    """
    if name in system._outputs:
        val = system._outputs[name]
        if idxs is not None and isinstance(val, np.ndarray):
            val = val.flatten()[idxs]
        return val
    return "<on remote proc>"
