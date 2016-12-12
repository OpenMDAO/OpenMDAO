
import os
import sys
import json
import contextlib
from itertools import chain

from six import iteritems

import numpy

from openmdao.api import Problem
from openmdao.devtools.compat import abs_conn_iter, abs_varname_iter, \
                                     abs_meta_iter, abs2prom_map
from openmdao.devtools.webview import webview

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = numpy.get_printoptions()
    numpy.set_printoptions(*args, **kwargs)
    yield
    numpy.set_printoptions(**original)

def view_connections(root, outfile='connections.html', show_browser=True,
                     src_filter='', tgt_filter='', precision=6):
    """
    Generates a self-contained html file containing a detailed connection
    viewer.  Optionally pops up a web browser to view the file.

    Args
    ----
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
        system = root.root
    else:
        system = root

    abs_tgt_names = list(abs_varname_iter(system, 'input', local=True))
    abs_src_names = list(abs_varname_iter(system, 'output', local=True))
    src_vals = [system._outputs[n]
                    for n in system._variable_myproc_names['output']]
    tgt_vals = [system._inputs[n]
                    for n in system._variable_myproc_names['input']]
    connections = dict(abs_conn_iter(system))
    tmetas = dict(abs_meta_iter(system, 'input'))
    to_prom_out = abs2prom_map(system, 'output', local=True)
    to_prom_in = abs2prom_map(system, 'input', local=True)
    to_prom = to_prom_out.copy()
    to_prom.update(to_prom_in)

    src2tgts = {}
    # units = {n: m.get('units','') for n,m in chain(iteritems(system._unknowns_dict),
    #                                                iteritems(system._params_dict))}
    units = {}  # no units yet
    vals = {}

    with printoptions(precision=precision, suppress=True, threshold=10000):

        for idx, t in enumerate(abs_tgt_names):
            tmeta = tmetas[t]
            idxs = tmeta['indices']
            if idxs is None:
                idxs = numpy.arange(numpy.prod(tmeta['shape']), dtype=int)

            if t in connections:
                s = connections[t]
                val = system._outputs[to_prom_out[s]]
                if isinstance(val, numpy.ndarray) and idxs is not None:
                    val = system._outputs[to_prom_out[s]][idxs]
                else:
                    val = system._outputs[to_prom_out[s]]

                # # if there's a unit conversion, express the value in the
                # # units of the target
                # if 'unit_conv' in tmeta:
                #     scale, offset = tmeta['unit_conv']
                #     val = (val + offset) * scale

                if s not in src2tgts:
                    src2tgts[s] = [t]
                else:
                    src2tgts[s].append(t)
            else: # unconnected param
                val = system._inputs[to_prom_in[t]]

            if isinstance(val, numpy.ndarray):
                val = numpy.array2string(val)
            else:
                val = str(val)

            vals[t] = val

        noconn_srcs = sorted((n for n in abs_src_names
                                if n not in src2tgts), reverse=True)
        for s in noconn_srcs:
            vals[s] = str(system._outputs[to_prom_out[s]])

    vals['NO CONNECTION'] = ''

    src_systems = set()
    tgt_systems = set()
    for s in abs_src_names:
        parts = s.split('.')
        for i in range(len(parts)):
            src_systems.add('.'.join(parts[:i]))

    for t in abs_tgt_names:
        parts = t.split('.')
        for i in range(len(parts)):
            tgt_systems.add('.'.join(parts[:i]))

    # reverse sort so that "NO CONNECTION" shows up at the bottom
    src2tgts['NO CONNECTION'] = sorted([t for t in abs_tgt_names
                                    if t not in connections], reverse=True)

    src_systems = [{'name':n} for n in sorted(src_systems)]
    src_systems.insert(1, {'name': "NO CONNECTION"})
    tgt_systems = [{'name':n} for n in sorted(tgt_systems)]
    tgt_systems.insert(1, {'name': "NO CONNECTION"})

    data = {
        'src2tgts': sorted(iteritems(src2tgts)),
        'proms': to_prom,
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
