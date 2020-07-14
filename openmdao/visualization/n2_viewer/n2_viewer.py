"""Code for generating N2 diagram."""
import base64
import inspect
import json
import os
import zlib
from collections import OrderedDict
from itertools import chain
import networkx as nx

from openmdao.components.exec_comp import ExecComp
from openmdao.components.meta_model_structured_comp import MetaModelStructuredComp
from openmdao.components.meta_model_unstructured_comp import MetaModelUnStructuredComp
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.core.component import Component
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.drivers.doe_driver import DOEDriver
from openmdao.recorders.case_reader import CaseReader
from openmdao.solvers.nonlinear.newton import NewtonSolver
from openmdao.utils.class_util import overrides_method
from openmdao.utils.general_utils import simple_warning, make_serializable
from openmdao.utils.record_util import check_valid_sqlite3_db
from openmdao.utils.mpi import MPI
from openmdao.visualization.html_utils import read_files, write_script, DiagramWriter
from openmdao.utils.general_utils import warn_deprecation

_IND = 4  # HTML indentation (spaces)


def _get_var_dict(system, typ, name):
    if name in system._var_discrete[typ]:
        meta = system._var_discrete[typ][name]
    else:
        meta = system._var_abs2meta[name]
        name = system._var_abs2prom[typ][name]

    var_dict = OrderedDict()

    var_dict['name'] = name
    if typ == 'input':
        var_dict['type'] = 'param'
    elif typ == 'output':
        isimplicit = isinstance(system, ImplicitComponent)
        var_dict['type'] = 'unknown'
        var_dict['implicit'] = isimplicit

    var_dict['dtype'] = type(meta['value']).__name__
    return var_dict


def _get_tree_dict(system, component_execution_orders, component_execution_index,
                   is_parallel=False):
    """Get a dictionary representation of the system hierarchy."""
    tree_dict = OrderedDict()
    tree_dict['name'] = system.name
    tree_dict['type'] = 'subsystem'
    tree_dict['class'] = system.__class__.__name__
    tree_dict['expressions'] = None

    if not isinstance(system, Group):
        tree_dict['subsystem_type'] = 'component'
        tree_dict['is_parallel'] = is_parallel
        if isinstance(system, ImplicitComponent):
            tree_dict['component_type'] = 'implicit'
        elif isinstance(system, ExecComp):
            tree_dict['component_type'] = 'exec'
            tree_dict['expressions'] = system._exprs
        elif isinstance(system, (MetaModelStructuredComp, MetaModelUnStructuredComp)):
            tree_dict['component_type'] = 'metamodel'
        elif isinstance(system, IndepVarComp):
            tree_dict['component_type'] = 'indep'
        elif isinstance(system, ExplicitComponent):
            tree_dict['component_type'] = 'explicit'
        else:
            tree_dict['component_type'] = None

        component_execution_orders[system.pathname] = component_execution_index[0]
        component_execution_index[0] += 1

        children = []
        for typ in ['input', 'output']:
            for abs_name in system._var_abs_names[typ]:
                children.append(_get_var_dict(system, typ, abs_name))

            for prom_name in system._var_discrete[typ]:
                children.append(_get_var_dict(system, typ, prom_name))

    else:
        if isinstance(system, ParallelGroup):
            is_parallel = True
        tree_dict['component_type'] = None
        tree_dict['subsystem_type'] = 'group'
        tree_dict['is_parallel'] = is_parallel

        children = []
        for s in system._subsystems_myproc:
            if (s.name != '_auto_ivc'):
                children.append(_get_tree_dict(s, component_execution_orders,
                                component_execution_index, is_parallel))

        if system.comm.size > 1:
            if system._subsystems_myproc:
                sub_comm = system._subsystems_myproc[0].comm
                if sub_comm.rank != 0:
                    children = []
            children_lists = system.comm.allgather(children)

            children = []
            for children_list in children_lists:
                children.extend(children_list)

    if isinstance(system, ImplicitComponent):
        if overrides_method('solve_linear', system, ImplicitComponent):
            tree_dict['linear_solver'] = "solve_linear"
        elif system.linear_solver:
            tree_dict['linear_solver'] = system.linear_solver.SOLVER
        else:
            tree_dict['linear_solver'] = ""

        if overrides_method('solve_nonlinear', system, ImplicitComponent):
            tree_dict['nonlinear_solver'] = "solve_nonlinear"
        elif system.nonlinear_solver:
            tree_dict['nonlinear_solver'] = system.nonlinear_solver.SOLVER
        else:
            tree_dict['nonlinear_solver'] = ""
    else:
        if system.linear_solver:
            tree_dict['linear_solver'] = system.linear_solver.SOLVER
        else:
            tree_dict['linear_solver'] = ""

        if system.nonlinear_solver:
            tree_dict['nonlinear_solver'] = system.nonlinear_solver.SOLVER

            if system.nonlinear_solver.SOLVER == NewtonSolver.SOLVER:
                tree_dict['solve_subsystems'] = system._nonlinear_solver.options['solve_subsystems']
        else:
            tree_dict['nonlinear_solver'] = ""

    tree_dict['children'] = children

    if not tree_dict['name']:
        tree_dict['name'] = 'root'
        tree_dict['type'] = 'root'

    return tree_dict


def _get_declare_partials(system):
    """
    Get a list of the declared partials.

    Parameters
    ----------
    system : <System>
        A System in the model.

    Returns
    -------
    list
        A list containing all the declared partials (strings in the form "of > wrt" )
        beginning from the given system on down.
    """
    declare_partials_list = []

    def recurse_get_partials(system, dpl):
        if isinstance(system, Component):
            subjacs = system._subjacs_info
            for abs_key, meta in subjacs.items():
                if abs_key[0] != abs_key[1]:
                    dpl.append("{} > {}".format(abs_key[0], abs_key[1]))
        elif isinstance(system, Group):
            for s in system._subsystems_myproc:
                recurse_get_partials(s, dpl)
        return

    recurse_get_partials(system, declare_partials_list)
    return declare_partials_list


def _get_viewer_data(data_source):
    """
    Get the data needed by the N2 viewer as a dictionary.

    Parameters
    ----------
    data_source : <Problem> or <Group> or str
        A Problem or Group or case recorder file name containing the model or model data.

    Returns
    -------
    dict
        A dictionary containing information about the model for use by the viewer.
    """
    if isinstance(data_source, Problem):
        root_group = data_source.model

        if not isinstance(root_group, Group):
            simple_warning(
                "The model is not a Group, viewer data is unavailable.")
            return {}

        driver = data_source.driver
        driver_name = driver.__class__.__name__
        driver_type = 'doe' if isinstance(
            driver, DOEDriver) else 'optimization'
        driver_options = {k: driver.options[k] for k in driver.options}
        driver_opt_settings = None
        if driver_type == 'optimization' and 'opt_settings' in dir(driver):
            driver_opt_settings = driver.opt_settings

    elif isinstance(data_source, Group):
        if not data_source.pathname:  # root group
            root_group = data_source
            driver_name = None
            driver_type = None
            driver_options = None
            driver_opt_settings = None
        else:
            # this function only makes sense when it is at the root
            return {}

    elif isinstance(data_source, str):
        data_dict = CaseReader(data_source, pre_load=False).problem_metadata

        # Delete the variables key since it's not used in N2
        if 'variables' in data_dict:
            del data_dict['variables']

        return data_dict

    else:
        raise TypeError(
            '_get_viewer_data only accepts Problems, Groups or filenames')

    data_dict = {}
    comp_exec_idx = [0]  # list so pass by ref
    orders = {}
    data_dict['tree'] = _get_tree_dict(root_group, orders, comp_exec_idx)

    connections_list = []

    sys_pathnames_list = []  # list of pathnames of systems found in cycles
    sys_pathnames_dict = {}  # map of pathnames to index of pathname in list

    G = root_group.compute_sys_graph(comps_only=True)

    scc = nx.strongly_connected_components(G)

    for strong_comp in scc:
        if len(strong_comp) > 1:
            # these IDs are only used when back edges are present
            sys_pathnames_list.extend(strong_comp)
            for name in strong_comp:
                sys_pathnames_dict[name] = len(sys_pathnames_dict)

        for src, tgt in G.edges(strong_comp):
            if src in strong_comp and tgt in strong_comp:
                if src in orders:
                    exe_src = orders[src]
                else:
                    exe_src = orders[src] = -1
                if tgt in orders:
                    exe_tgt = orders[tgt]
                else:
                    exe_tgt = orders[tgt] = -1

                if exe_tgt < exe_src:
                    exe_low = exe_tgt
                    exe_high = exe_src
                else:
                    exe_low = exe_src
                    exe_high = exe_tgt

                edges_list = [
                    (sys_pathnames_dict[s], sys_pathnames_dict[t]) for s, t in G.edges(strong_comp)
                    if s in orders and exe_low <= orders[s] <= exe_high and t in orders and
                    exe_low <= orders[t] <= exe_high and
                    not (s == src and t == tgt) and t in sys_pathnames_dict
                ]
                for vsrc, vtgtlist in G.get_edge_data(src, tgt)['conns'].items():
                    for vtgt in vtgtlist:
                        connections_list.append({'src': vsrc, 'tgt': vtgt,
                                                 'cycle_arrows': edges_list})
            else:  # edge is out of the SCC
                for vsrc, vtgtlist in G.get_edge_data(src, tgt)['conns'].items():
                    for vtgt in vtgtlist:
                        connections_list.append({'src': vsrc, 'tgt': vtgt})

    data_dict['sys_pathnames_list'] = sys_pathnames_list
    data_dict['connections_list'] = connections_list
    data_dict['abs2prom'] = root_group._var_abs2prom

    data_dict['driver'] = {'name': driver_name, 'type': driver_type,
                           'options': driver_options, 'opt_settings': driver_opt_settings}
    data_dict['design_vars'] = root_group.get_design_vars(use_prom_ivc=False)
    data_dict['responses'] = root_group.get_responses()

    data_dict['declare_partials_list'] = _get_declare_partials(root_group)

    return data_dict


def n2(data_source, outfile='n2.html', show_browser=True, embeddable=False,
       title=None, use_declare_partial_info=False):
    """
    Generate an HTML file containing a tree viewer.

    Optionally opens a web browser to view the file.

    Parameters
    ----------
    data_source : <Problem> or str
        The Problem or case recorder database containing the model or model data.

    outfile : str, optional
        The name of the final output file

    show_browser : bool, optional
        If True, pop up the system default web browser to view the generated html file.
        Defaults to True.

    embeddable : bool, optional
        If True, gives a single HTML file that doesn't have the <html>, <DOCTYPE>, <body>
        and <head> tags. If False, gives a single, standalone HTML file for viewing.

    title : str, optional
        The title for the diagram. Used in the HTML title.

    use_declare_partial_info : ignored
        This option is no longer used because it is now always true.
        Still present for backwards compatibility.

    """
    # grab the model viewer data
    model_data = _get_viewer_data(data_source)

    # if MPI is active only display one copy of the viewer
    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    options = {}
    model_data['options'] = options

    if use_declare_partial_info:
        warn_deprecation("'use_declare_partial_info' is now the"
                         " default and the option is ignored.")

    raw_data = json.dumps(model_data, default=make_serializable).encode('utf8')
    b64_data = str(base64.b64encode(zlib.compress(raw_data)).decode("ascii"))
    model_data = 'var compressedModel = "%s";' % b64_data

    import openmdao
    openmdao_dir = os.path.dirname(inspect.getfile(openmdao))
    vis_dir = os.path.join(openmdao_dir, "visualization/n2_viewer")
    libs_dir = os.path.join(vis_dir, "libs")
    src_dir = os.path.join(vis_dir, "src")
    style_dir = os.path.join(vis_dir, "style")
    assets_dir = os.path.join(vis_dir, "assets")

    # grab the libraries, src and style
    lib_dct = {
        'd3': 'd3.v5.min',
        'awesomplete': 'awesomplete',
        'vk_beautify': 'vkBeautify',
        'pako_inflate': 'pako_inflate.min'
    }
    libs = read_files(lib_dct.values(), libs_dir, 'js')
    src_names = \
        'modal', \
        'utils', \
        'SymbolType', \
        'N2TreeNode', \
        'ModelData', \
        'N2Style', \
        'N2Layout', \
        'N2MatrixCell', \
        'N2Legend', \
        'N2Matrix', \
        'N2Arrow', \
        'N2Search', \
        'N2Toolbar', \
        'N2Diagram', \
        'N2UserInterface', \
        'defaults', \
        'ptN2'

    srcs = read_files(src_names, src_dir, 'js')

    style_names = \
        'partition_tree', \
        'icon', \
        'toolbar', \
        'nodedata', \
        'legend', \
        'awesomplete'

    styles = read_files((style_names), style_dir, 'css')

    with open(os.path.join(style_dir, "icomoon.woff"), "rb") as f:
        encoded_font = str(base64.b64encode(f.read()).decode("ascii"))

    with open(os.path.join(style_dir, "logo_png.b64"), "r") as f:
        logo_png = str(f.read())

    with open(os.path.join(assets_dir, "spinner.png"), "rb") as f:
        waiting_icon = str(base64.b64encode(f.read()).decode("ascii"))

    if title:
        title = "OpenMDAO Model Hierarchy and N2 diagram: %s" % title
    else:
        title = "OpenMDAO Model Hierarchy and N2 diagram"

    h = DiagramWriter(filename=os.path.join(vis_dir, "index.html"),
                      title=title,
                      styles=styles, embeddable=embeddable)

    if (embeddable):
        h.insert("non-embedded-n2", "embedded-n2")

    # put all style and JS into index
    h.insert('{{fontello}}', encoded_font)
    h.insert('{{logo_png}}', logo_png)
    h.insert('{{waiting_icon}}', waiting_icon)

    for k, v in lib_dct.items():
        h.insert('{{{}_lib}}'.format(k), write_script(libs[v], indent=_IND))

    for name, code in srcs.items():
        h.insert('{{{}_lib}}'.format(name.lower()),
                 write_script(code, indent=_IND))

    h.insert('{{model_data}}', write_script(model_data, indent=_IND))

    # Help
    help_txt = ('Left clicking on a node in the partition tree will navigate to that node. '
                'Right clicking on a node in the model hierarchy will collapse/expand it. '
                'A click on any element in the N2 diagram will allow those arrows to persist.')
    help_diagram_svg_filepath = os.path.join(assets_dir, "toolbar_help.svg")
    h.add_help(help_txt, help_diagram_svg_filepath,
               footer="OpenMDAO Model Hierarchy and N2 diagram")

    # Write output file
    h.write(outfile)

    # open it up in the browser
    if show_browser:
        from openmdao.utils.webview import webview
        webview(outfile)
