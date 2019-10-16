"""Code for generating N2 diagram."""
import base64
import inspect
import json
import os
from collections import OrderedDict
from itertools import chain

import networkx as nx
from six import iteritems, itervalues

try:
    import h5py
except ImportError:
    # Necessary for the file to parse
    h5py = None

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
from openmdao.visualization.html_utils import read_files, write_script, DiagramWriter
from openmdao.utils.class_util import overrides_method
from openmdao.utils.general_utils import warn_deprecation, simple_warning, make_serializable
from openmdao.utils.record_util import check_valid_sqlite3_db
from openmdao.utils.mpi import MPI
from openmdao.recorders.case_reader import CaseReader
from openmdao.drivers.doe_driver import DOEDriver

# Toolbar settings
_FONT_SIZES = [8, 9, 10, 11, 12, 13, 14]
_MODEL_HEIGHTS = [600, 650, 700, 750, 800, 850, 900, 950, 1000, 2000, 3000, 4000]

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

    if not isinstance(system, Group):
        tree_dict['subsystem_type'] = 'component'
        tree_dict['is_parallel'] = is_parallel
        if isinstance(system, ImplicitComponent):
            tree_dict['component_type'] = 'implicit'
        elif isinstance(system, ExecComp):
            tree_dict['component_type'] = 'exec'
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
        children = [_get_tree_dict(s, component_execution_orders, component_execution_index,
                                   is_parallel)
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
            for abs_key, meta in iteritems(subjacs):
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
            simple_warning("The model is not a Group, viewer data is unavailable.")
            return {}

        driver = data_source.driver
        driver_name = driver.__class__.__name__
        driver_type = 'doe' if isinstance(driver, DOEDriver) else 'optimization'
        driver_options = {k: driver.options[k] for k in driver.options}
        driver_opt_settings = None
        if driver_type is 'optimization' and 'opt_settings' in dir(driver):
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
        return CaseReader(data_source, pre_load=False).problem_metadata

    else:
        raise TypeError('_get_viewer_data only accepts Problems, Groups or filenames')

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
                for vsrc, vtgtlist in iteritems(G.get_edge_data(src, tgt)['conns']):
                    for vtgt in vtgtlist:
                        connections_list.append({'src': vsrc, 'tgt': vtgt,
                                                 'cycle_arrows': edges_list})
            else:  # edge is out of the SCC
                for vsrc, vtgtlist in iteritems(G.get_edge_data(src, tgt)['conns']):
                    for vtgt in vtgtlist:
                        connections_list.append({'src': vsrc, 'tgt': vtgt})

    data_dict['sys_pathnames_list'] = sys_pathnames_list
    data_dict['connections_list'] = connections_list
    data_dict['abs2prom'] = root_group._var_abs2prom

    data_dict['driver'] = {'name': driver_name, 'type': driver_type,
                           'options': driver_options, 'opt_settings': driver_opt_settings}
    data_dict['design_vars'] = root_group.get_design_vars()
    data_dict['responses'] = root_group.get_responses()

    data_dict['declare_partials_list'] = _get_declare_partials(root_group)

    return data_dict


def view_tree(*args, **kwargs):
    """
    view_tree was renamed to n2, but left here for backwards compatibility.

    Parameters
    ----------
    *args : dict
        Postional args.
    **kwargs : dict
        Keyword args.
    """
    warn_deprecation("view_tree is deprecated. Please switch to n2.")
    n2(*args, **kwargs)


def view_model(*args, **kwargs):
    """
    view_model was renamed to n2, but left here for backwards compatibility.

    Parameters
    ----------
    *args : dict
        Postional args.
    **kwargs : dict
        Keyword args.
    """
    warn_deprecation("view_model is deprecated. Please switch to n2.")
    n2(*args, **kwargs)


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

    use_declare_partial_info : bool, optional
        If True, in the N2 matrix, component internal connectivity computed using derivative
        declarations, otherwise, derivative declarations ignored, so dense component connectivity
        is assumed.

    """
    # if MPI is active only display one copy of the viewer
    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    # grab the model viewer data
    model_data = _get_viewer_data(data_source)
    options = {}
    options['use_declare_partial_info'] = use_declare_partial_info
    model_data['options'] = options

    model_data = 'var modelData = %s' % json.dumps(model_data, default=make_serializable)

    import openmdao
    openmdao_dir = os.path.dirname(inspect.getfile(openmdao))
    vis_dir = os.path.join(openmdao_dir, "visualization/n2_viewer")
    libs_dir = os.path.join(vis_dir, "libs")
    src_dir = os.path.join(vis_dir, "src")
    style_dir = os.path.join(vis_dir, "style")

    # grab the libraries, src and style
    lib_dct = {'d3': 'd3.v4.min', 'awesomplete': 'awesomplete', 'vk_beautify': 'vkBeautify'}
    libs = read_files(itervalues(lib_dct), libs_dir, 'js')
    src_names = 'constants', 'draw', 'legend', 'modal', 'utils', 'SymbolType', 'N2TreeNode', 'ModelData', 'N2Style', 'N2Layout', 'N2MatrixNode', 'N2Matrix', 'N2Arrow', 'N2Diagram', 'ptN2', 'search'
    srcs = read_files(src_names, src_dir, 'js')
    styles = read_files(('awesomplete', 'partition_tree'), style_dir, 'css')

    with open(os.path.join(style_dir, "fontello.woff"), "rb") as f:
        encoded_font = str(base64.b64encode(f.read()).decode("ascii"))

    if title:
        title = "OpenMDAO Model Hierarchy and N2 diagram: %s" % title
    else:
        title = "OpenMDAO Model Hierarchy and N2 diagram"

    h = DiagramWriter(filename=os.path.join(vis_dir, "index.html"),
                      title=title,
                      styles=styles, embeddable=embeddable)

    # put all style and JS into index
    h.insert('{{fontello}}', encoded_font)

    for k, v in iteritems(lib_dct):
        h.insert('{{{}_lib}}'.format(k), write_script(libs[v], indent=_IND))

    for name, code in iteritems(srcs):
        h.insert('{{{}_lib}}'.format(name.lower()), write_script(code, indent=_IND))

    h.insert('{{model_data}}', write_script(model_data, indent=_IND))

    # Toolbar
    toolbar = h.toolbar
    group1 = toolbar.add_button_group()
    group1.add_button("Return To Root", uid="returnToRootButtonId", disabled="disabled",
                      content="icon-home")
    group1.add_button("Back", uid="backButtonId", disabled="disabled", content="icon-left-big")
    group1.add_button("Forward", uid="forwardButtonId", disabled="disabled",
                      content="icon-right-big")
    group1.add_button("Up One Level", uid="upOneLevelButtonId", disabled="disabled",
                      content="icon-up-big")

    group2 = toolbar.add_button_group()
    group2.add_button("Uncollapse In View Only", uid="uncollapseInViewButtonId",
                      content="icon-resize-full")
    group2.add_button("Uncollapse All", uid="uncollapseAllButtonId",
                      content="icon-resize-full bigger-font")
    group2.add_button("Collapse Outputs In View Only", uid="collapseInViewButtonId",
                      content="icon-resize-small")
    group2.add_button("Collapse All Outputs", uid="collapseAllButtonId",
                      content="icon-resize-small bigger-font")
    group2.add_dropdown("Collapse Depth", button_content="icon-sort-number-up",
                        uid="idCollapseDepthDiv")

    group3 = toolbar.add_button_group()
    group3.add_button("Clear Arrows and Connections", uid="clearArrowsAndConnectsButtonId",
                      content="icon-eraser")
    group3.add_button("Show Path", uid="showCurrentPathButtonId", content="icon-terminal")
    group3.add_button("Show Legend", uid="showLegendButtonId", content="icon-map-signs")
    group3.add_button("Toggle Solver Names", uid="toggleSolverNamesButtonId", content="icon-minus")
    group3.add_dropdown("Font Size", id_naming="idFontSize", options=_FONT_SIZES,
                        option_formatter=lambda x: '{}px'.format(x),
                        button_content="icon-text-height")
    group3.add_dropdown("Vertically Resize", id_naming="idVerticalResize",
                        options=_MODEL_HEIGHTS, option_formatter=lambda x: '{}px'.format(x),
                        button_content="icon-resize-vertical", header="Model Height")

    group4 = toolbar.add_button_group()
    group4.add_button("Save SVG", uid="saveSvgButtonId", content="icon-floppy")

    group5 = toolbar.add_button_group()
    group5.add_button("Help", uid="helpButtonId", content="icon-help")

    # Help
    help_txt = ('Left clicking on a node in the partition tree will navigate to that node. '
                'Right clicking on a node in the model hierarchy will collapse/uncollapse it. '
                'A click on any element in the N^2 diagram will allow those arrows to persist.')

    h.add_help(help_txt, footer="OpenMDAO Model Hierarchy and N^2 diagram")

    # Write output file
    h.write(outfile)

    # open it up in the browser
    if show_browser:
        from openmdao.utils.webview import webview
        webview(outfile)
