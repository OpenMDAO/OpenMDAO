
import os
import sys
import json
import contextlib
from itertools import chain

from six import iteritems

import numpy as np

from openmdao.core.problem import Problem
from openmdao.utils.units import convert_units
from openmdao.devtools.webview import webview

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)

def view_connections(root, outfile='connections.html', show_browser=True,
                     src_filter='', tgt_filter='', precision=6):
    """
    Generates a self-contained html file containing a detailed connection
    viewer.  Optionally pops up a web browser to view the file.

    Parameters
    ----------
    root : system or Problem
        The root for the desired tree.

    outfile : str, optional
        The name of the output html file.  Defaults to 'connections.html'.

    show_browser : bool, optional
        If True, pop up a browser to view the generated html file.
        Defaults to True.

    src_filter : str, optional
        If defined, use this as the initial value for the source system filter.

    tgt_filter : str, optional
        If defined, use this as the initial value for the target system filter.

    precision : int, optional
        Sets the precision for displaying array values.
    """
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

    src2tgts = {}
    units = {n: data.get('units','')
                for n, data in iteritems(system._var_allprocs_abs2meta)}
    vals = {}

    with printoptions(precision=precision, suppress=True, threshold=10000):

        for t in system._var_abs_names['input']:
            tmeta = system._var_abs2meta[t]
            idxs = tmeta['src_indices']
            flat = tmeta['flat_src_indices']
            if idxs is None:
                idxs = np.arange(tmeta['size'], dtype=int)

            if t in connections:
                s = connections[t]
                val = system._outputs[s]
                if isinstance(val, np.ndarray):
                    val = system._outputs[s].flatten()[idxs]
                else:
                    val = system._outputs[s]

                # if there's a unit conversion, express the value in the
                # units of the target
                if units[t]:
                    val = convert_units(val, units[s], units[t])

                if s not in src2tgts:
                    src2tgts[s] = [t]
                else:
                    src2tgts[s].append(t)
            else: # unconnected param
                val = system._inputs[t]

            if isinstance(val, np.ndarray):
                val = np.array2string(val)
            else:
                val = str(val)

            vals[t] = val

        noconn_srcs = sorted((n for n in system._var_abs_names['output']
                                if n not in src2tgts), reverse=True)
        for s in noconn_srcs:
            vals[s] = str(system._outputs[s])

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

    # reverse sort so that "NO CONNECTION" shows up at the bottom
    src2tgts['NO CONNECTION'] = sorted([t for t in system._var_abs_names['input']
                                    if t not in connections], reverse=True)

    src_systems = [{'name':n} for n in sorted(src_systems)]
    src_systems.insert(1, {'name': "NO CONNECTION"})
    tgt_systems = [{'name':n} for n in sorted(tgt_systems)]
    tgt_systems.insert(1, {'name': "NO CONNECTION"})

    data = {
        'src2tgts': sorted(iteritems(src2tgts)),
        'proms': None,
        'units': units,
        'vals': vals,
        'src_systems': src_systems,
        'tgt_systems': tgt_systems,
        'noconn_srcs': noconn_srcs,
        'src_filter': src_filter,
        'tgt_filter': tgt_filter,
    }

    viewer = 'connect_table.html'

    code_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(code_dir, viewer), "r") as f:
        template = f.read()

    graphjson = json.dumps(data)

    with open(outfile, 'w') as f:
        s = template.replace("<connection_data>", graphjson)
        f.write(s)

    if show_browser:
        webview(outfile)
