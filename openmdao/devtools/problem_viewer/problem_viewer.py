from six.moves import range
import numpy as np
import os
import pickle
import json
from six import iteritems
import networkx as nx
import shutil
from collections import OrderedDict

from sqlitedict import SqliteDict

try:
    import h5py
except ImportError:
    # Necessary for the file to parse
    h5py = None

from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.utils.general_utils import warn_deprecation
import base64


def _get_tree_dict(system, component_execution_orders, component_execution_index):
    """Get a dictionary representation of the system hierarchy."""
    tree_dict = OrderedDict()
    tree_dict['name'] = system.name
    tree_dict['type'] = 'subsystem'

    local_prom_dict = OrderedDict()

    from_prom_name = OrderedDict(system._var_allprocs_prom2abs_list['input'])
    from_prom_name.update(system._var_allprocs_prom2abs_list['output'])
    for var_prom_name, var_abs_name_list in iteritems(from_prom_name):
        for var_abs_name in var_abs_name_list:
            if "." in var_prom_name:
                local_prom_dict[var_abs_name] = var_prom_name
        if(len(local_prom_dict) > 0):
            tree_dict['promotions'] = OrderedDict(sorted(local_prom_dict.items())) # sort to make deterministic for testing

    if not isinstance(system, Group):
        tree_dict['subsystem_type'] = 'component'
        component_execution_orders[system.pathname] = component_execution_index[0]
        component_execution_index[0] += 1

        children = []
        for typ in ['input', 'output']:
            for ind, abs_name in enumerate(system._var_abs_names[typ]):
                meta = system._var_abs2meta[typ][abs_name]
                name = system._var_abs2prom[typ][abs_name]

                var_dict = OrderedDict()
                var_dict['name'] = name
                if typ == 'input':
                    var_dict['type'] = 'param'
                elif typ == 'output':
                    isimplicit = isinstance(system, ImplicitComponent)
                    var_dict['type'] = 'unknown'
                    var_dict['implicit'] = isimplicit

                var_dict['dtype'] = type(meta['value']).__name__
                children.append(var_dict)
    else:
        tree_dict['subsystem_type'] = 'group'
        children = [_get_tree_dict(s, component_execution_orders, component_execution_index)
                    for s in system._subsystems_myproc]
        if system.comm.size > 1:
            sub_comm = system._subsystems_myproc[0].comm
            if sub_comm.rank != 0:
                children = []
            children_lists = system.comm.allgather(children)

            children = []
            for children_list in children_lists:
                children.extend(children_list)

    tree_dict['children'] = children

    if not tree_dict['name']:
        tree_dict['name'] = 'root'
        tree_dict['type'] = 'root'

    return tree_dict

def _get_viewer_data(problem_or_rootgroup):
    """Get the data needed by the N2 viewer as a dictionary."""
    if isinstance(problem_or_rootgroup, Problem):
        root_group = problem_or_rootgroup.model
    elif isinstance(problem_or_rootgroup, Group):
        if not problem_or_rootgroup.pathname: # root group
            root_group = problem_or_rootgroup
        else:
            # this function only makes sense when it is at the root
            return {}
    else:
        raise TypeError('get_model_viewer_data only accepts Problems or Groups')

    data_dict = {}
    component_execution_idx = [0] #list so pass by ref
    component_execution_orders = {}
    data_dict['tree'] = _get_tree_dict(root_group, component_execution_orders, component_execution_idx)

    connections_list = []
    sorted_abs_input2src = OrderedDict(sorted(root_group._conn_global_abs_in2out.items())) # sort to make deterministic for testing
    root_group._conn_global_abs_in2out = sorted_abs_input2src
    G = root_group.compute_sys_graph(comps_only=True)
    scc = nx.strongly_connected_components(G)
    scc_list = [s for s in scc if len(s)>1] #list(scc)
    for in_abs, out_abs in iteritems(sorted_abs_input2src):
        if out_abs is None:
            continue
        src_subsystem = out_abs.rsplit('.', 1)[0]
        tgt_subsystem = in_abs.rsplit('.', 1)[0]
        count = 0
        edges_list = []
        for li in scc_list:
            if src_subsystem in li and tgt_subsystem in li:
                count = count+1
                if(count > 1):
                    raise ValueError('Count greater than 1')

                exe_tgt = component_execution_orders[tgt_subsystem]
                exe_src = component_execution_orders[src_subsystem]
                exe_low = min(exe_tgt,exe_src)
                exe_high = max(exe_tgt,exe_src)
                subg = G.subgraph(li)
                for n in subg.nodes():
                    exe_order = component_execution_orders[n]
                    if(exe_order < exe_low or exe_order > exe_high):
                        subg.remove_node(n)


                src_to_tgt_str = src_subsystem + ' ' + tgt_subsystem
                for tup in subg.edges():
                    edge_str = tup[0] + ' ' + tup[1]
                    if edge_str != src_to_tgt_str:
                        edges_list.append(edge_str)

        if(len(edges_list) > 0):
            edges_list.sort() # make deterministic so same .html file will be produced each run
            connections_list.append(OrderedDict([('src', out_abs), ('tgt', in_abs), ('cycle_arrows', edges_list)]))
        else:
            connections_list.append(OrderedDict([('src', out_abs), ('tgt', in_abs)]))


    data_dict['connections_list'] = connections_list

    return data_dict

def view_tree(*args, **kwargs):
    """
    view_tree was renamed to view_model, but left here for backwards compatibility
    """
    warn_deprecation("view_tree is deprecated. Please switch to view_model.")
    view_model(*args, **kwargs)

def view_model(problem_or_filename, outfile_name='visualization', show_browser=True):
    """
    Generates a directory containing a tree viewer Optionally pops up a web browser to
    view the file.

    Parameters
    ----------
    problem_or_filename : Either a Problem() or a string
        Problem() : The Problem (after problem.setup()) for the desired tree.
        string : The filename of the case recorder file containing the data required to build the tree.

    outfile_name : str, optional
        The name of the output html file.  Defaults to 'visualization'.

    show_browser : bool, optional
        If True, pop up the system default web browser to view the generated html file.
        Defaults to True.
    """
    model_data_filename = 'model_data.js'
    folder_name = outfile_name

    code_dir = os.path.dirname(os.path.abspath(__file__))

    if isinstance(problem_or_filename, Problem):
        model_viewer_data = _get_viewer_data(problem_or_filename)

    model_data = json.dumps(model_viewer_data)

    cur_dir = os.getcwd()
    directory = os.path.dirname(cur_dir + '/' + folder_name)

    if os.path.isdir(folder_name):
        shutil.rmtree(folder_name)

    shutil.copytree(src=code_dir + '/visualization', dst=folder_name)

    with open(cur_dir + '/' + folder_name + '/' + model_data_filename, 'w') as f:
        f.write('var modelData = %s' % model_data)

    if show_browser:
        from openmdao.devtools.webview import webview
        webview(cur_dir + '/' + folder_name + '/index.html')
