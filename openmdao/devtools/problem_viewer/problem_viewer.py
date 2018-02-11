from six.moves import range
import numpy as np
import os
import json
from six import iteritems
import networkx as nx
import shutil
from collections import OrderedDict
import base64

try:
    import h5py
except ImportError:
    # Necessary for the file to parse
    h5py = None

from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.utils.general_utils import warn_deprecation
from openmdao.utils.record_util import is_valid_sqlite3_db


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
                meta = system._var_abs2meta[abs_name]
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
            if system._subsystems_myproc:
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

def _get_viewer_data(problem_or_rootgroup_or_filename):
    """Get the data needed by the N2 viewer as a dictionary."""
    if isinstance(problem_or_rootgroup_or_filename, Problem):
        root_group = problem_or_rootgroup_or_filename.model
    elif isinstance(problem_or_rootgroup_or_filename, Group):
        if not problem_or_rootgroup_or_filename.pathname: # root group
            root_group = problem_or_rootgroup_or_filename
        else:
            # this function only makes sense when it is at the root
            return {}
    elif is_valid_sqlite3_db(problem_or_rootgroup_or_filename):
        import sqlite3
        con = sqlite3.connect(problem_or_rootgroup_or_filename, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()
        cur.execute("SELECT model_viewer_data FROM driver_metadata;")
        model_pickle = cur.fetchone()
        from six import PY2, PY3
        if PY2:
            import cPickle
            return cPickle.loads(str(model_pickle[0]))
        if PY3:
            import pickle
            return pickle.loads(model_pickle[0])
    else:
        raise TypeError('get_model_viewer_data only accepts Problems, Groups or filenames')

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
                subg = G.subgraph(li).copy()
                for n in list(subg.nodes()):
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


def view_model(problem_or_filename, outfile='n2.html', show_browser=True, embeddable=False, draw_potential_connections=True):
    """
    Generates an HTML file containing a tree viewer. Optionally pops up a web browser to
    view the file.

    Parameters
    ----------
    problem_or_filename : Either a Problem() or a string
        Problem() : The Problem (after problem.setup()) for the desired tree.
        string : The filename of the case recorder file containing the data required to
         build the tree.

    outfile : str, optional
        The name of the final output file

    show_browser : bool, optional
        If True, pop up the system default web browser to view the generated html file.
        Defaults to True.

    embeddable : bool, optional
        If True, gives a single HTML file that doesn't have the <html>, <DOCTYPE>, <body>
        and <head> tags. If False, gives a single, standalone HTML file for viewing.

    draw_potential_connections : bool, optional
        If true, allows connections to be drawn on the N2 that do not currently exist
        in the model. Defaults to True.
    """
    html_begin_tags = """
                      <html>
                      <head>
                        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
                      </head>
                      <body>\n
                      """
    html_end_tags = """
                    </body>
                    </html>
                    """

    code_dir = os.path.dirname(os.path.abspath(__file__))
    vis_dir = os.path.join(code_dir, "visualization")
    libs_dir = os.path.join(vis_dir, "libs")
    src_dir = os.path.join(vis_dir, "src")
    style_dir = os.path.join(vis_dir, "style")

    #grab the libraries
    with open(os.path.join(libs_dir, "awesomplete.js"), "r") as f:
        awesomplete = f.read()
    with open(os.path.join(libs_dir, "d3.v4.min.js"), "r") as f:
        d3 = f.read()
    with open(os.path.join(libs_dir, "http.js"), "r") as f:
        http = f.read()
    with open(os.path.join(libs_dir, "jquery-3.2.1.min.js"), "r") as f:
        jquery = f.read()
    with open(os.path.join(libs_dir, "vkBeautify.js"), "r") as f:
        vk_beautify = f.read()

    #grab the src
    with open(os.path.join(src_dir, "constants.js"), "r") as f:
        constants = f.read()
    with open(os.path.join(src_dir, "draw.js"), "r") as f:
        draw = f.read()
    with open(os.path.join(src_dir, "legend.js"), "r") as f:
        legend = f.read()
    with open(os.path.join(src_dir, "modal.js"), "r") as f:
        modal = f.read()
    with open(os.path.join(src_dir, "ptN2.js"), "r") as f:
        pt_n2 = f.read()
    with open(os.path.join(src_dir, "search.js"), "r") as f:
        search = f.read()
    with open(os.path.join(src_dir, "svg.js"), "r") as f:
        svg = f.read()

    #grab the style
    with open(os.path.join(style_dir, "awesomplete.css"), "r") as f:
        awesomplete_style = f.read()
    with open(os.path.join(style_dir, "partition_tree.css"), "r") as f:
        partition_tree_style = f.read()
    with open(os.path.join(style_dir, "fontello.woff"), "rb") as f:
        encoded_font = str(base64.b64encode(f.read()).decode("ascii"))

    #grab the index.html
    with open(os.path.join(vis_dir, "index.html"), "r") as f:
        index = f.read()

    #grab the model viewer data
    model_viewer_data = 'var modelData = %s' % json.dumps(_get_viewer_data(problem_or_filename))

    #add the necessary HTML tags if we aren't embedding
    if not embeddable:
        index = html_begin_tags + index + html_end_tags

    #put all style and JS into index
    index = index.replace('{{awesomplete_style}}', awesomplete_style)
    index = index.replace('{{partition_tree_style}}', partition_tree_style)
    index = index.replace('{{fontello}}', encoded_font)
    index = index.replace('{{d3_lib}}', d3)
    index = index.replace('{{awesomplete_lib}}', awesomplete)
    index = index.replace('{{vk_beautify_lib}}', vk_beautify)
    index = index.replace('{{model_data}}', model_viewer_data)
    index = index.replace('{{constants_lib}}', constants)
    index = index.replace('{{modal_lib}}', modal)
    index = index.replace('{{svg_lib}}', svg)
    index = index.replace('{{search_lib}}', search)
    index = index.replace('{{legend_lib}}', legend)
    index = index.replace('{{draw_lib}}', draw)
    index = index.replace('{{ptn2_lib}}', pt_n2)
    if draw_potential_connections:
        index = index.replace('{{draw_potential_connections}}', 'true')
    else:
        index = index.replace('{{draw_potential_connections}}', 'false')

    with open(outfile, 'w') as f:
        f.write(index)

    #open it up in the browser
    if show_browser:
        from openmdao.devtools.webview import webview
        webview(outfile)
