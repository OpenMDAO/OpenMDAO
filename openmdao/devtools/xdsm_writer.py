"""
XDSM writer using the pyXDSM package or XDSMjs.

The pyXDSM package is available at https://github.com/mdolab/pyXDSM.
XDSMjs is available at https://github.com/OneraHub/XDSMjs.
"""
from __future__ import print_function

import json

from openmdao.devtools.problem_viewer.problem_viewer import _get_viewer_data
from openmdao.devtools.webview import webview

try:
    from pyxdsm.XDSM import XDSM
except ImportError:
    msg = ('The pyxdsm package should be installed. You can download the package '
           'from https://github.com/mdolab/pyXDSM')
    raise ImportError(msg)

from six import iteritems


# Character substitutions in labels
_CHAR_SUBS = (('_', '~'), (')', ' '), ('(', '_'))


class AbstractXDSMWriter(object):
    """
    Abstract class to define methods for XDSM writers.

    All methods should be implemented in child classes.
    """
    def __init__(self):
        self.comps = []
        self.connections = []
        self.left_outs = {}
        self.right_outs = {}
        self.ins = {}
        self.processes = []
        self.process_arrows = []

    def add_solver(self, label, name='solver', **kwargs):
        pass  # Implement in child class

    def add_comp(self, name, label=None, **kwargs):
        pass  # Implement in child class

    def add_func(self, name, **kwargs):
        pass  # Implement in child class

    def add_optimizer(self, label, name='opt', **kwargs):
        pass  # Implement in child class

    def add_input(self, name, label, style='DataIO', stack=False):
        pass  # Implement in child class

    def add_output(self, name, label, style='DataIO', stack=False, side="left"):
        pass  # Implement in child class


class XDSMWriter(XDSM):
    """
    XDSM with some additional semantics.
    Creates a TeX file and TiKZ file, and converts it to PDF.

    On Windows it might be necessary to add the second line in the :class:`~pyxdsm.XDSM.XDSM`::

        diagram_styles_path = os.path.join(module_path, 'diagram_styles')
        diagram_styles_path = diagram_styles_path.replace('\\', '/')  # Add this line on Windows

    """

    def add_solver(self, label, name='solver', **kwargs):
        """
        Add a solver.

        Parameters
        ----------
        label : str
            Label in the XDSM
        name : str
            Name of the solver
        kwargs : dict
            Keyword args
        """
        self.add_system(name, 'MDA', '\\text{%s}' % label, **kwargs)

    def add_comp(self, name, label=None, **kwargs):
        """
        Add a component.

        Parameters
        ----------
        label : str
            Label in the XDSM, defaults to the name of the component.
        name : str
            Name of the component
        kwargs : dict
            Keyword args
        """
        if label is None:
            label = name
        self.add_system(name, 'Analysis', label, **kwargs)

    def add_func(self, name, **kwargs):
        """
        Add a function

        Parameters
        ----------
        name : str
            Name of the function
        kwargs : dict
            Keyword args
        """
        self.add_system(name, 'Function', name, **kwargs)

    def add_optimizer(self, label, name='opt', **kwargs):
        """
        Add an optimizer.

        Parameters
        ----------
        label : str
            Label in the XDSM
        name : str
            Name of the optimizer.
        kwargs : dict
            Keyword args
        """
        self.add_system(name, 'Optimization', '\\text{%s}' % label, **kwargs)


class XDSMjsWriter(AbstractXDSMWriter):
    """
    JSON input file writer for XDSMjs.

    XDSMjs is available at https://github.com/OneraHub/XDSMjs
    """
    def __init__(self):
        super(XDSMjsWriter, self).__init__()
        self.optimizer = 'opt'
        self.comp_names = []
        self.components = []
        self.reserved_words = '_U_',

    def _format_id(self, name, subs=(('_', ''),)):
        if name not in self.reserved_words:
            return _replace_chars(name, subs)
        else:
            return name

    def connect(self, src, target, label, style='DataInter', stack=False, faded=False):
        edge = {'to': self._format_id(target),
                'from': self._format_id(src),
                'name': label}
        self.connections.append(edge)

    def add_solver(self, label, name='solver', **kwargs):
        raise NotImplementedError()

    def add_comp(self, name, label=None, **kwargs):
        self.comp_names.append(self._format_id(name))
        self.add_system(name, 'analysis', label, **kwargs)

    def add_func(self, name, **kwargs):
        pass

    def add_optimizer(self, label, name='opt', **kwargs):
        self.optimizer = self._format_id(name)
        self.add_system(name, 'optimization', label, **kwargs)

    def add_system(self, node_name, style, label=None, stack=False, faded=False):
        if label is None:
            label = node_name
        dct = {"type": style, "id": self._format_id(node_name), "name": label}
        self.components.append(dct)

    def add_workflow(self):
        wf = ["_U_",
              [
                self.optimizer, self.comp_names
              ]
            ]
        self.processes = wf

    def add_input(self, name, label=None, style='DataIO', stack=False):
        self.connect(src='_U_', target=name, label=label)

    def add_output(self, name, label=None, style='DataIO', stack=False, side="left"):
        self.connect(src=name, target='_U_', label=label)

    def write(self, filename='xdsmjs', ext='json', *args, **kwargs):
        self.add_workflow()
        data = {'edges': self.connections, 'nodes': self.components, 'workflow': self.processes}

        if ext is not None:
            filename = '.'.join([filename, ext])
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)
        print('XDSM output file written to: {}'.format(filename))


def write_xdsm(problem, filename, model_path=None, recurse=True,
               include_external_outputs=True, out_format='tex',
               include_solver=False, subs=_CHAR_SUBS):
    """
    Writes XDSM diagram of an optimization problem.

    Parameters
    ----------
    problem : Problem
        Problem
    filename : str
        Name of the output files (do not provide file extension)
    model_path : str or None
        Path to the subsystem to be transcribed to XDSM.  If None, use the model root.
    recurse : bool
        If False, treat the top level of each name as the source/target component.
    include_external_outputs : bool
        If True, show externally connected outputs when transcribing a subsystem.
    out_format : str
        Output format, one of "tex" (pyXDSM) or "json" (XDSMjs)
    include_solver : bool
        Include or not the problem model's nonlinear solver in the XDSM.
    subs : tuple(str, str)
        Characters to be replaced
    Returns
    -------
       XDSM
    """

    viewer_data = _get_viewer_data(problem)
    driver = problem.driver
    if model_path is None:
        _model = problem.model
    else:
        _model = problem.model._get_subsystem(model_path)

    if driver:
        driver_name = _get_cls_name(driver)
    else:
        driver_name = None
    if include_solver:
        solver_name = _get_cls_name(_model.nonlinear_solver)
    else:
        solver_name = None
    design_vars = _model.get_design_vars()
    responses = _model.get_responses()

    filename = filename.replace('\\', '/')  # Needed for LaTeX
    return _write_xdsm(filename, viewer_data=viewer_data,
                       optimizer=driver_name, solver=solver_name, model_path=model_path,
                       design_vars=design_vars, responses=responses, out_format=out_format,
                       recurse=recurse, subs=subs,
                       include_external_outputs=include_external_outputs)


def _write_xdsm(filename, viewer_data, optimizer=None, solver=None, cleanup=True,
                design_vars=None, responses=None, residuals=None, model_path=None, recurse=True,
                include_external_outputs=True, subs=_CHAR_SUBS, out_format='tex',
                show_browser=True, **kwargs):
    """
    XDSM writer. Components are extracted from the connections of the problem.

    In the diagram the connections are marked with the source name.

    Parameters
    ----------
    filename : str
        Filename (absolute path without extension)
    connections : list[(str, str)]
        Connections list
    optimizer : str or None
        Optimizer name
    solver:  str or None
        Solver name
    cleanup : bool
        Clean-up temporary files after making the diagram
    design_vars : OrderedDict or None
        Design variables
    responses : OrderedDict or None
        Responses
    model_path : str or None
        Path to the subsystem to be transcribed to XDSM.  If None, use the model root.
    recurse : bool
        If False, treat the top level of each name as the source/target component.
    include_external_outputs : bool
        If True, show externally connected outputs when transcribing a subsystem.
    subs : tuple
       Character pairs to be substituted. Forbidden characters or just for the sake of nicer names.
    kwargs : dict
        Keyword arguments

    Returns
    -------
        XDSM
    """
    # TODO implement residuals
    connections = viewer_data['connections_list']
    tree = viewer_data['tree']

    conns1, external_inputs1, external_outputs1 = _prune_connections(connections,
                                                                     model_path=model_path)

    conns2 = _process_connections(conns1, recurse=recurse, subs=subs)
    external_inputs2 = _process_connections(external_inputs1, recurse=recurse, subs=subs)
    external_outputs2 = _process_connections(external_outputs1, recurse=recurse, subs=subs)

    conns3 = _accumulate_connections(conns2)
    external_inputs3 = _accumulate_connections(external_inputs2)
    external_outputs3 = _accumulate_connections(external_outputs2)

    if out_format == 'tex':
        x = XDSMWriter()
    elif out_format == 'json':
        x = XDSMjsWriter()
    else:
        msg = 'The "out_format" should be "tex" or "json", instead it is "{}"'
        raise ValueError(msg.format(out_format))

    if optimizer is not None:
        x.add_optimizer(optimizer)

    if solver is not None:
        x.add_solver(solver)

    design_vars2 = _collect_connections(design_vars)
    responses2 = _collect_connections(responses)

    # Feed forward
    for comp, conn_vars in iteritems(design_vars2):
        x.connect('opt', comp, conn_vars)
        opt_con_vars = [_opt_var_str(var) for var in conn_vars]
        x.add_output(comp, ', '.join(opt_con_vars), side='left')

    # Feedback
    for comp, conn_vars in iteritems(responses2):
        x.connect(comp, 'opt', conn_vars)
        opt_con_vars = [_opt_var_str(var) for var in conn_vars]
        x.add_output(comp, ', '.join(opt_con_vars), side='left')

    # Get the top level system to be transcripted to XDSM
    comps = _get_comps(tree, model_path=model_path, recurse=recurse)

    # Add components
    for comp in comps:
        x.add_comp(name=comp['name'], label=_replace_chars(comp['name'], substitutes=subs))

    # Add the connections
    for src, dct in iteritems(conns3):
        for tgt, conn_vars in iteritems(dct):
            x.connect(src, tgt, ', '.join(conn_vars))

    # Add the externally sourced inputs
    for src, tgts in iteritems(external_inputs3):
        for tgt, conn_vars in iteritems(tgts):
            x.add_input(tgt, conn_vars)

    # Add the externally connected outputs
    if include_external_outputs:
        for src, tgts in iteritems(external_outputs3):
            output_vars = set()
            for tgt, conn_vars in iteritems(tgts):
                output_vars |= set(conn_vars)
            x.add_output(src, list(output_vars), side='right')

    x.write(filename, cleanup=cleanup, **kwargs)

    if show_browser:
        if out_format == 'tex':
            ext = 'pdf'
        elif out_format == 'json':
            ext = 'html'
        else:
            err_msg = '"{}" is an invalid output format.'
            raise ValueError(err_msg.format(out_format))
        path = '.'.join([filename, ext])
        webview(path)

    return x


def _get_cls_name(obj):
    return obj.__class__.__name__


def _residual_str(name):
    """Makes a residual symbol."""
    return '\\mathcal{R}(%s)' % name


def _opt_var_str(name):
    """Puts an asterisk superscript on a string."""
    return '{}^*'.format(name)


def _process_connections(conns, recurse=True, subs=None):
    conns_new = [{k: _convert_name(v, recurse=recurse, subs=subs) for k, v in iteritems(conn)} for conn in conns]
    return conns_new


def _accumulate_connections(conns):
    # Makes a dictionary with source and target components and with the connection sources
    conns_new = dict()
    for conn in conns:  # list
        src_comp = conn['src']['comp']
        tgt_comp = conn['tgt']['comp']
        if src_comp == tgt_comp:
            # When recurse is False, ignore connections within the same subsystem.
            continue
        var = conn['src']['var']
        conns_new.setdefault(src_comp, {})
        conns_new[src_comp].setdefault(tgt_comp, []).append(var)
    return conns_new


def _collect_connections(variables):
    conv_vars = [_convert_name(v) for v in variables]
    connections = dict()
    for conv_var in conv_vars:
        connections.setdefault(conv_var['comp'], []).append(conv_var['var'])
    return connections


def _convert_name(name, recurse=True, subs=None):
    """
    From an absolute path returns the variable name and its owner component in a dict.

    Parameters
    ----------
    name : str
        Connection absolute path and name
    recurse : bool
        If False, treat the top level of each name as the source/target component.
    subs: tuple or None
        Character pairs with old and substitute characters

    Returns
    -------
        dict(str, str)
    """

    def convert(name):
        name = name.split('.')
        if recurse:
            comp = name[-2]
        else:
            comp = name[0]
        var = name[-1]
        var = _replace_chars(var, substitutes=subs)
        return {'comp': comp, 'var': var}

    if isinstance(name, list):  # If a source has multiple targets
        return [convert(n) for n in name]
    else:  # string
        return convert(name)


def _prune_connections(conns, model_path=None):
    """
    Remove connections that don't involve components within model.

    Parameters
    ----------
    conns : list
        A list of connections from viewer_data
    model_path : str or None
        The path in model to the system to be transcribed to XDSM.

    Returns
    -------
    internal_conns : list(dict)
        A list of the connections with sources and targets inside the given model path.
    external_inputs : list(dict)
        A list of the connections where the target is inside the model path but is connected
        to an external source.
    external_outputs : list(dict)
        A list of the connections where the source is inside the model path but is connected
        to an external target.

    """
    internal_conns = []
    external_inputs = []
    external_outputs = []

    if model_path is None:
        return conns, external_inputs, external_outputs

    for conn in conns:
        src = conn['src']
        rel_src = src.replace(model_path + '.', '')
        tgt = conn['tgt']
        rel_tgt = tgt.replace(model_path + '.', '')

        if src.startswith(model_path) and tgt.startswith(model_path):
            # Internal connections
            internal_conns.append({'src': rel_src, 'tgt': rel_tgt})
        elif not src.startswith(model_path) and tgt.startswith(model_path):
            # Externally connected input
            external_inputs.append({'src': rel_src, 'tgt': rel_tgt})
        elif src.startswith(model_path) and not tgt.startswith(model_path):
            # Externally connected output
            external_outputs.append({'src': rel_src, 'tgt': rel_tgt})

    return internal_conns, external_inputs, external_outputs


def _get_comps(tree, model_path=None, recurse=True):
    """
    Return the components in the tree, optionally only those within the given model_path.

    Parameters
    ----------
    tree : list(OrderedDict)
        The model tree as returned by viewer_data.
    model_path : str or None
        The path of the model within the tree to be transcribed to XDSM. If None, transcribe
        the entire tree.
    recurse : bool
        If True, return individual components within the model_path.  If False, treat
        Groups as black-box components and don't show their internal components.

    Returns
    -------
    components : list
        A list of the components within the model_path in tree.  If recurse is False, this
        list may contain groups.

    """
    # Components are ordered in the tree, so they can be collected by walking through the tree.
    components = list()

    def get_children(tree_branch, recurse=recurse):
        for ch in tree_branch['children']:
            if ch['subsystem_type'] == 'component':
                components.append(ch)
            elif recurse:
                get_children(ch)
            else:
                components.append(ch)

    top_level_tree = tree
    if model_path is not None:
        path_list = model_path.split('.')
        while path_list:
            next_path = path_list.pop(0)
            children = [child for child in top_level_tree['children']]
            top_level_tree = [c for c in children if c['name'] == next_path][0]

    get_children(top_level_tree)

    return components


def _replace_chars(name, substitutes):
    """
    Replaces characters in `name` with the substitute characters. If some of the characters are
    both to be replaced or other characters are replaced with them (e.g.: ? -> !, ! ->#), than
    it is not safe to give a dictionary as the `substitutes` (because it is unordered).

    .. warning::

       Order matters, because otherwise some characters could be replaced more than once.

    Parameters
    ----------
    name : str
       Name
    substitutes: tuple or None
       Character pairs with old and substitute characters

    Returns
    -------
       str
    """
    if substitutes:
        for (k, v) in substitutes:
            name = name.replace(k, v)
    return name