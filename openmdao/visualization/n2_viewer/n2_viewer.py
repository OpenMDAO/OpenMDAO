"""Code for generating N2 diagram."""
import inspect
import os
import pathlib
from operator import itemgetter

import networkx as nx
import numpy as np

import openmdao.utils.hooks as hooks
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.core.component import Component
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.constants import _UNDEFINED
from openmdao.components.exec_comp import ExecComp
from openmdao.components.meta_model_structured_comp import MetaModelStructuredComp
from openmdao.components.meta_model_unstructured_comp import MetaModelUnStructuredComp
from openmdao.drivers.doe_driver import DOEDriver
from openmdao.recorders.case_reader import CaseReader
from openmdao.solvers.nonlinear.newton import NewtonSolver
from openmdao.utils.class_util import overrides_method
from openmdao.utils.general_utils import default_noraise
from openmdao.utils.mpi import MPI
from openmdao.utils.notebook_utils import notebook, display, HTML, IFrame, colab
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.reports_system import register_report_hook
from openmdao.utils.file_utils import _load_and_exec, _to_filename
from openmdao.visualization.htmlpp import HtmlPreprocessor
from openmdao import __version__ as openmdao_version

_MAX_ARRAY_SIZE_FOR_REPR_VAL = 1000  # If var has more elements than this do not pass to N2

_default_n2_filename = 'n2.html'


def _convert_nans_in_nested_list(val_as_list):
    """
    Given a list, possibly nested, replace any numpy.nan values with the string "nan".

    This is done since JSON does not handle nan. This code is used to pass variable values
    to the N2 diagram.

    The modifications to the list values are done in-place to avoid excessive copying of lists.

    Parameters
    ----------
    val_as_list : list, possibly nested
        the list whose nan elements need to be converted
    """
    for i, val in enumerate(val_as_list):
        if isinstance(val, list):
            _convert_nans_in_nested_list(val)
        else:
            if np.isnan(val):
                val_as_list[i] = "nan"
            elif np.isinf(val):
                val_as_list[i] = "infinity"
            else:
                val_as_list[i] = val


def _convert_ndarray_to_support_nans_in_json(val):
    """
    Given numpy array of arbitrary dimensions, return the equivalent nested list with nan replaced.

    numpy.nan values are replaced with the string "nan".

    Parameters
    ----------
    val : ndarray
        the numpy array to be converted

    Returns
    -------
    object : list, possibly nested
        The equivalent list with any nan values replaced with the string "nan".
    """
    val = np.asarray(val)

    # do a quick check for any nans or infs and if not we can avoid the slow check
    nans = np.where(np.isnan(val))
    infs = np.where(np.isinf(val))
    if nans[0].size == 0 and infs[0].size == 0:
        return val.tolist()

    val_as_list = val.tolist()
    _convert_nans_in_nested_list(val_as_list)
    return val_as_list


def _get_array_info(system, vec, name, prom, var_dict, from_src=True):
    ndarray_to_convert = vec._abs_get_val(name, flat=False) if vec else \
        system.get_val(prom, from_src=from_src)

    var_dict['val'] = _convert_ndarray_to_support_nans_in_json(ndarray_to_convert)

    # Find the minimum indices and value
    min_indices = np.unravel_index(np.nanargmin(ndarray_to_convert, axis=None),
                                   ndarray_to_convert.shape)
    var_dict['val_min_indices'] = min_indices
    var_dict['val_min'] = ndarray_to_convert[min_indices]

    # Find the maximum indices and value
    max_indices = np.unravel_index(np.nanargmax(ndarray_to_convert, axis=None),
                                   ndarray_to_convert.shape)
    var_dict['val_max_indices'] = max_indices
    var_dict['val_max'] = ndarray_to_convert[max_indices]


def _get_var_dict(system, typ, name, is_parallel, is_implicit):
    if name in system._var_abs2meta[typ]:
        meta = system._var_abs2meta[typ][name]
        prom = system._var_abs2prom[typ][name]
        val = np.asarray(meta['val'])
        is_dist = MPI is not None and meta['distributed']

        var_dict = {
            'name': prom,
            'type': typ,
            'dtype': type(val).__name__,
            'is_discrete': False,
            'distributed': is_dist,
            'shape': str(meta['shape']),
        }

        if typ == 'output':
            var_dict['implicit'] = is_implicit
            vec = system._outputs
        else:  # input
            if MPI:
                # for inputs if we're under MPI, we only retrieve the value currently stored
                # in the input vector and not from the connected source because that source
                # could be remote.
                vec = system._inputs
            else:
                vec = None

        # if 'vec' is not None at this point, we can retrieve the value using vec._abs_get_val,
        # which is a faster call than system.get_val.

        if meta['units'] is None:
            var_dict['units'] = 'None'
        else:
            var_dict['units'] = meta['units']

        try:
            if val.size < _MAX_ARRAY_SIZE_FOR_REPR_VAL:
                if not MPI:
                    # Get the current value
                    _get_array_info(system, vec, name, prom, var_dict, from_src=True)

                elif is_parallel or is_dist:
                    # we can't access non-local values, so just get the initial value
                    var_dict['val'] = val
                    var_dict['initial_value'] = True
                else:
                    # get the current value but don't try to get it from the source,
                    # which could be remote under MPI
                    _get_array_info(system, vec, name, prom, var_dict, from_src=False)

            else:
                var_dict['val'] = None
        except Exception as err:
            issue_warning(str(err))
            var_dict['val'] = None
    else:  # discrete
        meta = system._var_discrete[typ][name]
        val = meta['val']
        var_dict = {
            'name': name,
            'type': typ,
            'dtype': type(val).__name__,
            'is_discrete': True,
        }
        if MPI is None or isinstance(val, (int, str, list, dict, complex, np.ndarray)):
            var_dict['val'] = default_noraise(system.get_val(name))
        else:
            var_dict['val'] = type(val).__name__

    if 'surrogate_name' in meta:
        var_dict['surrogate_name'] = meta['surrogate_name']

    return var_dict


def _serialize_single_option(option):
    """
    Return a json-safe equivalent of the option.

    The default_noraise function performs the datatype serialization, while this function takes
    care of attributes specific to options dicts.

    Parameters
    ----------
    option : object
        Option to be serialized.

    Returns
    -------
    object
       JSON-safe serialized object.
    """
    if not option['recordable']:
        return 'Not Recordable'

    val = option['val']
    if val is _UNDEFINED:
        return str(val)

    return default_noraise(val)


def _get_tree_dict(system, is_parallel=False):
    """Get a dictionary representation of the system hierarchy."""
    tree_dict = {
        'name': system.name if system.name else 'root',
        'type': 'subsystem' if system.name else 'root',
        'class': system.__class__.__name__,
        'expressions': None,
        'nonlinear_solver': "",
        'nonlinear_solver_options': None,
        'linear_solver': "",
        'linear_solver_options': None,
    }
    is_implicit = False

    if isinstance(system, Group):
        if MPI and isinstance(system, ParallelGroup):
            is_parallel = True
        tree_dict['component_type'] = None
        tree_dict['subsystem_type'] = 'group'
        tree_dict['is_parallel'] = is_parallel

        children = [_get_tree_dict(s, is_parallel) for s in system._subsystems_myproc]

        if system.comm.size > 1:
            if system._subsystems_myproc:
                sub_comm = system._subsystems_myproc[0].comm
                if sub_comm.rank != 0:
                    children = []
            children_lists = system.comm.allgather(children)

            children = []
            for children_list in children_lists:
                children.extend(children_list)

        if system.linear_solver:
            tree_dict['linear_solver'] = system.linear_solver.SOLVER
            tree_dict['linear_solver_options'] = {
                k: _serialize_single_option(opt)
                for k, opt in system.linear_solver.options._dict.items()
            }

        if system.nonlinear_solver:
            tree_dict['nonlinear_solver'] = system.nonlinear_solver.SOLVER
            tree_dict['nonlinear_solver_options'] = {
                k: _serialize_single_option(opt)
                for k, opt in system.nonlinear_solver.options._dict.items()
            }

            if system.nonlinear_solver.SOLVER == NewtonSolver.SOLVER:
                tree_dict['solve_subsystems'] = system._nonlinear_solver.options['solve_subsystems']
    else:
        tree_dict['subsystem_type'] = 'component'
        tree_dict['is_parallel'] = is_parallel
        if isinstance(system, ImplicitComponent):
            is_implicit = True
            tree_dict['component_type'] = 'implicit'
            if overrides_method('solve_linear', system, ImplicitComponent):
                tree_dict['linear_solver'] = "solve_linear"
            elif system.linear_solver:
                tree_dict['linear_solver'] = system.linear_solver.SOLVER
                tree_dict['linear_solver_options'] = {
                    k: _serialize_single_option(opt)
                    for k, opt in system.linear_solver.options._dict.items()
                }

            if overrides_method('solve_nonlinear', system, ImplicitComponent):
                tree_dict['nonlinear_solver'] = "solve_nonlinear"
            elif system.nonlinear_solver:
                tree_dict['nonlinear_solver'] = system.nonlinear_solver.SOLVER
                tree_dict['nonlinear_solver_options'] = {
                    k: _serialize_single_option(opt)
                    for k, opt in system.nonlinear_solver.options._dict.items()
                }
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

        children = []
        for typ in ['input', 'output']:
            for abs_name in system._var_abs2meta[typ]:
                children.append(_get_var_dict(system, typ, abs_name, is_parallel, is_implicit))

            for prom_name in system._var_discrete[typ]:
                children.append(_get_var_dict(system, typ, prom_name, is_parallel, is_implicit))

    tree_dict['children'] = children

    options = {}
    slv = {'linear_solver', 'nonlinear_solver'}
    for k, opt in system.options._dict.items():
        if k in slv:
            # need to handle solver option separately because it can be a class, instance or None
            try:
                val = opt['val']
            except KeyError:
                val = opt['value']

            try:
                options[k] = val.SOLVER
            except AttributeError:
                options[k] = val
        else:
            options[k] = _serialize_single_option(opt)

    tree_dict['options'] = options

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


def _get_viewer_data(data_source, case_id=None):
    """
    Get the data needed by the N2 viewer as a dictionary.

    Parameters
    ----------
    data_source : <Problem> or <Group> or str
        A Problem or Group or case recorder filename containing the model or model data.
        If the case recorder file from a parallel run has separate metadata, the
        filenames can be specified with a comma, e.g.: case.sql_0,case.sql_meta

    case_id : int or str or None
        Case name or index of case in SQL file.

    Returns
    -------
    dict
        A dictionary containing information about the model for use by the viewer.
    """
    if isinstance(data_source, Problem):
        root_group = data_source.model

        if not isinstance(root_group, Group):
            # this function only makes sense when the model is a Group
            msg = f"The model is of type {root_group.__class__.__name__}, " \
                  "viewer data is only available if the model is a Group."
            raise TypeError(msg)

        driver = data_source.driver
        driver_name = driver.__class__.__name__
        driver_type = 'doe' if isinstance(driver, DOEDriver) else 'optimization'

        driver_options = {key: _serialize_single_option(driver.options._dict[key])
                          for key in driver.options}

        if driver_type == 'optimization' and hasattr(driver, 'opt_settings'):
            driver_opt_settings = driver.opt_settings
        else:
            driver_opt_settings = None

    elif isinstance(data_source, Group):
        if not data_source.pathname:  # root group
            root_group = data_source
            driver_name = None
            driver_type = None
            driver_options = None
            driver_opt_settings = None
        else:
            # this function only makes sense when it is at the root
            msg = f"Viewer data is not available for sub-Group '{data_source.pathname}'."
            raise TypeError(msg)

    elif isinstance(data_source, str):
        if ',' in data_source:
            filenames = data_source.split(',')
            cr = CaseReader(filenames[0], metadata_filename=filenames[1])
        else:
            cr = CaseReader(data_source)

        data_dict = cr.problem_metadata

        if case_id is not None:
            cases = cr.get_case(case_id)
            print(f"Using source: {cases.source}\nCase: {cases.name}")

            def recurse(children, stack):
                for child in children:
                    # if 'val' in child
                    if child['type'] == 'subsystem':
                        if child['name'] != '_auto_ivc':
                            stack.append(child['name'])
                            recurse(child['children'], stack)
                            stack.pop()
                    elif child['type'] == 'input':
                        if cases.inputs is None:
                            child['val'] = 'N/A'
                        else:
                            path = child['name'] if not stack else '.'.join(stack + [child['name']])
                            child['val'] = cases.inputs[path]
                    elif child['type'] == 'output':
                        if cases.outputs is None:
                            child['val'] = 'N/A'
                        else:
                            path = child['name'] if not stack else '.'.join(stack + [child['name']])
                            try:
                                child['val'] = cases.outputs[path]
                            except KeyError:
                                child['val'] = 'N/A'
            recurse(data_dict['tree']['children'], [])

        # Delete the variables key since it's not used in N2
        if 'variables' in data_dict:
            del data_dict['variables']

        # Older recordings might not have this.
        if 'md5_hash' not in data_dict:
            data_dict['md5_hash'] = None

        return data_dict

    else:
        raise TypeError(f"Viewer data is not available for '{data_source}'."
                        "The source must be a Problem, model or the filename of a recording.")

    data_dict = {}
    data_dict['tree'] = _get_tree_dict(root_group)
    data_dict['md5_hash'] = root_group._generate_md5_hash()

    connections_list = []

    G = root_group.compute_sys_graph(comps_only=True)

    scc = nx.strongly_connected_components(G)

    strongdict = {}
    sys_idx_names = []

    for i, strong_comp in enumerate(scc):
        for c in strong_comp:
            strongdict[c] = i  # associate each comp with a strongly connected component

        if len(strong_comp) > 1:
            # these IDs are only used when back edges are present
            for name in strong_comp:
                sys_idx_names.append(name)

    sys_idx = {}  # map of pathnames to index of pathname in list (systems in cycles only)

    comp_orders = {name: i for i, name in enumerate(root_group._ordered_comp_name_iter())}
    for name in sorted(sys_idx_names):
        sys_idx[name] = len(sys_idx)

    # 1 is added to the indices of all edges in the matrix so that we can use 0 entries to
    # indicate that there is no connection.
    matrix = np.zeros((len(comp_orders), len(comp_orders)), dtype=np.int32)
    edge_ids = []
    for i, edge in enumerate(G.edges()):
        src, tgt = edge
        if strongdict[src] == strongdict[tgt]:
            matrix[comp_orders[src], comp_orders[tgt]] = i + 1  # bump edge index by 1
            edge_ids.append((sys_idx[src], sys_idx[tgt]))
        else:
            edge_ids.append(None)

    for edge_i, (src, tgt) in enumerate(G.edges()):
        if strongdict[src] == strongdict[tgt]:
            start = comp_orders[src]
            end = comp_orders[tgt]
            # get a view here so we can remove this edge from submat temporarily to eliminate
            # an 'if' check inside the nested list comprehension for edges_list
            rem = matrix[start:start + 1, end:end + 1]
            rem[0, 0] = 0

            if end < start:
                start, end = end, start

            submat = matrix[start:end + 1, start:end + 1]
            nz = submat[submat > 0]

            rem[0, 0] = edge_i + 1  # put removed edge back

            if nz.size > 1:
                nz -= 1  # convert back to correct edge index
                edges_list = [edge_ids[i] for i in nz]
                edges_list = sorted(edges_list, key=itemgetter(0, 1))

                for vsrc, vtgtlist in G.get_edge_data(src, tgt)['conns'].items():
                    for vtgt in vtgtlist:
                        connections_list.append({'src': vsrc, 'tgt': vtgt,
                                                 'cycle_arrows': edges_list})
                continue

        for vsrc, vtgtlist in G.get_edge_data(src, tgt)['conns'].items():
            for vtgt in vtgtlist:
                connections_list.append({'src': vsrc, 'tgt': vtgt})

    connections_list = sorted(connections_list, key=itemgetter('src', 'tgt'))

    data_dict['sys_pathnames_list'] = list(sys_idx)
    data_dict['connections_list'] = connections_list
    data_dict['abs2prom'] = root_group._var_abs2prom

    data_dict['driver'] = {
        'name': driver_name,
        'type': driver_type,
        'options': driver_options,
        'opt_settings': driver_opt_settings
    }
    data_dict['design_vars'] = root_group.get_design_vars(use_prom_ivc=False)
    data_dict['responses'] = root_group.get_responses()

    data_dict['declare_partials_list'] = _get_declare_partials(root_group)

    return data_dict


def n2(data_source, outfile=_default_n2_filename, case_id=None, show_browser=True, embeddable=False,
       title=None, display_in_notebook=True):
    """
    Generate an HTML file containing a tree viewer.

    Optionally opens a web browser to view the file.

    Parameters
    ----------
    data_source : <Problem> or str
        The Problem or case recorder database containing the model or model data.
    outfile : str, optional
        The name of the final output file.
    case_id : int, str, or None
        Case name or index of case in SQL file if data_source is a database.
    show_browser : bool, optional
        If True, pop up the system default web browser to view the generated html file.
        Defaults to True.
    embeddable : bool, optional
        If True, gives a single HTML file that doesn't have the <html>, <DOCTYPE>, <body>
        and <head> tags. If False, gives a single, standalone HTML file for viewing.
    title : str, optional
        The title for the diagram. Used in the HTML title.
    display_in_notebook : bool, optional
        If True, display the N2 diagram in the notebook, if this is called from a notebook.
        Defaults to True.
    """
    # grab the model viewer data
    try:
        model_data = _get_viewer_data(data_source, case_id=case_id)
        err_msg = ''
    except TypeError as err:
        model_data = {}
        err_msg = str(err)
        issue_warning(err_msg)

    # if MPI is active only display one copy of the viewer
    if MPI and MPI.COMM_WORLD.rank != 0:
        return

    options = {}
    model_data['options'] = options

    import openmdao
    openmdao_dir = os.path.dirname(inspect.getfile(openmdao))
    vis_dir = os.path.join(openmdao_dir, "visualization/n2_viewer")

    if title:
        title = f"OpenMDAO Model Hierarchy and N2 diagram: {title}"
    else:
        title = "OpenMDAO Model Hierarchy and N2 diagram"

    html_vars = {
        'title': title,
        'embeddable': "embedded-diagram" if embeddable else "non-embedded-diagram",
        'openmdao_version': openmdao_version,
        'model_data': model_data
    }

    if err_msg:
        with open(outfile, 'w') as f:
            f.write(err_msg)
    else:
        HtmlPreprocessor(os.path.join(vis_dir, "index.html"),
                         outfile, allow_overwrite=True, var_dict=html_vars,
                         json_dumps_default=default_noraise, verbose=False).run()

    if notebook:
        if display_in_notebook:
            # display in Jupyter Notebook
            outfile = os.path.relpath(outfile)
            if not colab:
                display(IFrame(src=outfile, width="100%", height=700))
            else:
                display(HTML(outfile))
    elif show_browser:
        # open it up in the browser
        from openmdao.utils.webview import webview
        webview(outfile)


# N2 report definition
def _run_n2_report(prob, report_filename=_default_n2_filename):

    n2_filepath = str(pathlib.Path(prob.get_reports_dir()).joinpath(report_filename))
    try:
        n2(prob, show_browser=False, outfile=n2_filepath, display_in_notebook=False)
    except RuntimeError as err:
        # We ignore this error
        if str(err) != "Can't compute total derivatives unless " \
                       "both 'of' or 'wrt' variables have been specified.":
            raise err


def _run_n2_report_w_errors(prob, report_filename=_default_n2_filename):
    errs = prob._metadata['saved_errors']
    if errs:
        n2_filepath = str(pathlib.Path(prob.get_reports_dir()).joinpath(report_filename))
        # only run the n2 here if we've had setup errors. Normally we'd wait until
        # after final_setup in order to have correct values for all of the I/O variables.
        try:
            n2(prob, show_browser=False, outfile=n2_filepath, display_in_notebook=False)
        except RuntimeError as err:
            # We ignore this error
            if str(err) != "Can't compute total derivatives unless " \
                           "both 'of' or 'wrt' variables have been specified.":
                prob.model._collect_error(str(err))
        # errors will result in exit at the end of the _check_collected_errors method


def _n2_report_register():
    register_report_hook('n2', 'final_setup', 'Problem', post=_run_n2_report,
                         description='N2 diagram', report_filename=_default_n2_filename)
    register_report_hook('n2', '_check_collected_errors', 'Problem', pre=_run_n2_report_w_errors,
                         description='N2 diagram')


def _n2_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao n2' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1,
                        help='Python script or recording containing the model. '
                        'If metadata from a parallel run was recorded in a separate file, '
                        'specify both database filenames delimited with a comma.')
    parser.add_argument('-o', default=_default_n2_filename, action='store', dest='outfile',
                        help='html output file.')
    parser.add_argument('--no_browser', action='store_true', dest='no_browser',
                        help="don't display in a browser.")
    parser.add_argument('--embed', action='store_true', dest='embeddable',
                        help="create embeddable version.")
    parser.add_argument('--title', default=None,
                        action='store', dest='title', help='diagram title.')


def _n2_cmd(options, user_args):
    """
    Process command line args and call n2 on the specified file.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Command line options after '--' (if any).  Passed to user script.
    """
    filename = _to_filename(options.file[0])

    if filename.endswith('.py'):
        def _view_model_w_errors(prob):
            errs = prob._metadata['saved_errors']
            if errs:
                # only run the n2 here if we've had setup errors. Normally we'd wait until
                # after final_setup in order to have correct values for all of the I/O variables.
                n2(prob, outfile=options.outfile, show_browser=not options.no_browser,
                   title=options.title, embeddable=options.embeddable)
                # errors will result in exit at the end of the _check_collected_errors method

        def _view_model_no_errors(prob):
            n2(prob, outfile=options.outfile, show_browser=not options.no_browser,
               title=options.title, embeddable=options.embeddable)

        hooks._register_hook('_check_collected_errors', 'Problem', pre=_view_model_w_errors)
        hooks._register_hook('final_setup', 'Problem', post=_view_model_no_errors, exit=True)

        from openmdao.utils.reports_system import _register_cmdline_report
        # tell report system not to duplicate effort
        _register_cmdline_report('n2')

        _load_and_exec(options.file[0], user_args)
    else:
        # assume the file is a recording, run standalone
        n2(filename, outfile=options.outfile, title=options.title,
           show_browser=not options.no_browser, embeddable=options.embeddable)
