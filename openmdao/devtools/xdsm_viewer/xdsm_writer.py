"""
XDSM writer using the pyXDSM package or XDSMjs.

The XDSM (eXtended Design Structure Matrix) is a tool used to visualize MDO processes.
It is an extension of the classical Design Structure Matrix commonly used in systems engineering to
describe the interfaces among components of a complex system.

Theoretical background:
Lambe, AB and Martins, JRRA (2012): Extensions to the Design Structure Matrix for the Description of
Multidisciplinary Design, Analysis, and Optimization Processes.
In: Structural and Multidisciplinary Optimization.

The pyXDSM package is available at https://github.com/mdolab/pyXDSM.
XDSMjs is available at https://github.com/OneraHub/XDSMjs.
"""

# TODO implement "stack" boxes for parallel components

from __future__ import print_function

import json
import os

from openmdao.devtools.problem_viewer.problem_viewer import _get_viewer_data
from openmdao.devtools.webview import webview
from openmdao.devtools.xdsm_viewer.html_writer import write_html

try:
    from pyxdsm.XDSM import XDSM
except ImportError:
    msg = ('The pyxdsm package should be installed. You can download the package '
           'from https://github.com/mdolab/pyXDSM')
    raise ImportError(msg)

from six import iteritems

_DIR = os.path.dirname(os.path.abspath(__file__))
_XDSMJS_PATH = os.path.join(_DIR, 'XDSMjs')

# Writer is chosen based on the output format
_OUT_FORMATS = {'tex': 'pyxdsm', 'pdf': 'pyxdsm', 'json': 'xdsmjs', 'html': 'xdsmjs'}

# Character substitutions in labels
# pyXDSM:
# Interpreted as TeX syntax
# Underscore is replaced with a skipped underscore
# Round parenthesis is replaced with subscript syntax, e.g. x(1) --> x_{1}
_CHAR_SUBS = {
    'pyxdsm': (('_', '\_'), ('(', '_{'), (')', '}'),),
    'xdsmjs': (),
}

# Default file names in XDSMjs

_XDSMJS_DATA = 'xdsm.json'  # data file, used only if data is not embedded
_XDSMJS_FILENAME = 'xdsm.html'  # output file


# Settings for pyXDSM

# The box width can be set by the user:
# _DEFAULT_BOX_WIDTH or _DEFAULT_BOX_CHAR_LIMIT can be overwritten with keyword argument "box_width"
_DEFAULT_BOX_WIDTH = 3.  # Width of boxes [cm]. Depends on other settings, weather it is used or not
# Maximum characters for line breaking.
# The line can be longer, if a variable name is longer.
_DEFAULT_BOX_CHAR_LIMIT = 25
# Controls the appearance of boxes
# Can be set with keyword argument "box_stacking"
# Options: horizontal, vertical, max_chars
_DEFAULT_BOX_STACKING = 'max_chars'
# Show arrowheads in processes
_PROCESS_ARROWS = False
_MAX_BOX_LINES = None


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

    def add_process(self, systems, arrow=True):
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

    def add_workflow(self):
        comp_names = [c[0] for c in self.comps]
        comp_names.append(comp_names[0])  # close the loop
        self.add_process(comp_names, arrow=_PROCESS_ARROWS)


class XDSMjsWriter(AbstractXDSMWriter):
    """
    Creates an interactive diagram with XDSMjs, which can be opened with a web browser.

    XDSMjs was created by Remi Lafage. The code and documentation is available at
    https://github.com/OneraHub/XDSMjs
    """
    def __init__(self):
        super(XDSMjsWriter, self).__init__()
        self.optimizer = 'opt'
        self.comp_names = []  # Component names
        self.components = []
        self.reserved_words = '_U_',  # Ignored at text formatting

    def _format_id(self, name, subs=(('_', ''),)):
        if name not in self.reserved_words:
            return _replace_chars(name, subs)
        else:
            return name

    def connect(self, src, target, label, **kwargs):
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

    def add_system(self, node_name, style, label=None, **kwargs):
        if label is None:
            label = node_name
        dct = {"type": style, "id": self._format_id(node_name), "name": label}
        self.components.append(dct)

    def add_workflow(self):
        wf = ["_U_", [self.optimizer, self.comp_names]]
        self.processes = wf

    def add_input(self, name, label=None, style='DataIO', stack=False):
        self.connect(src='_U_', target=name, label=label)

    def add_output(self, name, label=None, style='DataIO', stack=False, side="left"):
        self.connect(src=name, target='_U_', label=label)

    def collect_data(self):
        """
        Makes a dictionary with the structure of an XDSMjs JSON file.

        Returns
        -------
            dict
        """
        data = {'edges': self.connections, 'nodes': self.components, 'workflow': self.processes}
        return data

    def write(self, filename='xdsmjs', embed_data=True, **kwargs):
        """
        Writes HTML output file, and depending on the value of "embed_data" a JSON file with the
        data.

        If "embed_data" is true, a single standalone HTML file will be generated, which includes
        the data of the XDSM diagram.

        Parameters
        ----------
        filename : str, optional
            Output file name (without extension).
            Defaults to "xdsmjs".
        embed_data : bool, optional
            Embed XDSM data into the HTML file.
            If False, a JSON file will be also written.
            Defaults to True.
        """
        data = self.collect_data()

        html_filename = '.'.join([filename, 'html'])

        embeddable = kwargs.pop('embeddable', False)
        if embed_data:
            # Write HTML file
            write_html(outfile=html_filename, source_data=data, embeddable=embeddable)
        else:
            json_filename = '.'.join([filename, 'json'])
            with open(json_filename, 'w') as f:
                json.dump(data, f)

            # Write HTML file
            write_html(outfile=html_filename, data_file=json_filename, embeddable=embeddable)
        print('XDSM output file written to: {}'.format(html_filename))


def write_xdsm(problem, filename, model_path=None, recurse=True,
               include_external_outputs=True, out_format='tex',
               include_solver=False, subs=_CHAR_SUBS, show_browser=True,
               add_process_conns=True, **kwargs):
    """
    Writes XDSM diagram of an optimization problem.

    With the 'tex' or 'pdf' output format it uses the pyXDSM package, with 'json' or 'HTML'
    output format it uses XDSMjs.

    If a component (or group) name is not unique in the diagram, the systems absolute path is
    used as a label. If the component (or group) name is unique, the relative name of the
    system is the label.

    In the diagram the connections are marked with the source name.

    Writer specific settings and default:

    pyXDSM
    ~~~~~~

    * The appearance of the boxes can be controlled with "box_stacking" and "box_width" arguments.
      The box stacking can be "horizontal", "vertical", "cut_chars" or "max_chars".
      With "cut_chars" the text in the box will be one line with the maximum number of characters
      limited by "box_width". In the latter case the "box_width" argument is used to determine
      the maximum allowed width of boxes (in characters).
      A default value is taken, if not specified.
    * By default the part of variable names following underscores (_)
      are not converted to subscripts.
      To write in subscripts wrap that part of the name into a round bracket.
      Example: To write :math:`x_12` the variable name should be "x(12)"
    * "box_lines" can be used to limit the number of lines, if the box stacking is vertical

    XDSMjs
    ~~~~~~

    * If "embed_data" is true, a single standalone HTML file will be generated, which includes
      the data of the XDSM diagram.
    * variable names with exactly one underscore have a subscript.
      Example: "x_12" will be :math:`x_12`
    * If "embeddable" is True, gives a single HTML file that doesn't have the <html>, <DOCTYPE>,
      <body> and <head> tags. If False, gives a single, standalone HTML file for viewing.

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
        Defaults to True.
    out_format : str, optional
        Output format, one of "tex" (pyXDSM) or "json"/"html" (XDSMjs)
        Defaults to "tex".
    include_solver : bool
        Include or not the problem model's nonlinear solver in the XDSM.
    subs : dict(str, tuple), tuple(str, str), optional
        Characters to be replaced. Dictionary with writer names and character pairs or just the
        character pairs.
    show_browser : bool, optional
        If True, pop up a browser to view the generated html file.
        Defaults to True.
    add_process_conns: bool
        Add process connections (thin black lines)
        Defaults to True
    kwargs : dict
        Keyword arguments
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

    # Name is None if the driver is not specified
    driver_name = _get_cls_name(driver) if driver else None
    solver_name = _get_cls_name(_model.nonlinear_solver) if include_solver else None

    design_vars = _model.get_design_vars()
    responses = _model.get_responses()

    filename = filename.replace('\\', '/')  # Needed for LaTeX

    try:
        out_formats = _OUT_FORMATS
        writer = out_formats[out_format]
    except KeyError:
        msg = 'Invalid output format "{}", choose from: {}'
        raise ValueError(msg.format(out_format, out_formats.keys()))
    writer_name = writer.lower()  # making it case insensitive
    if isinstance(subs, dict):
        subs = subs[writer_name]  # Getting the character substitutes of the chosen writer
    return _write_xdsm(filename, viewer_data=viewer_data,
                       optimizer=driver_name, solver=solver_name, model_path=model_path,
                       design_vars=design_vars, responses=responses, writer=writer,
                       recurse=recurse, subs=subs,
                       include_external_outputs=include_external_outputs, show_browser=show_browser,
                       add_process_conns=add_process_conns, **kwargs)


def _write_xdsm(filename, viewer_data, optimizer=None, solver=None, cleanup=True,
                design_vars=None, responses=None, residuals=None, model_path=None, recurse=True,
                include_external_outputs=True, subs=_CHAR_SUBS, writer='pyXDSM',
                show_browser=False, add_process_conns=True, **kwargs):
    """
    XDSM writer. Components are extracted from the connections of the problem.

    Parameters
    ----------
    filename : str
        Filename (absolute path without extension)
    connections : list[(str, str)]
        Connections list
    optimizer : str or None, optional
        Optimizer name
    solver:  str or None, optional
        Solver name
    cleanup : bool, optional
        Clean-up temporary files after making the diagram.
        Defaults to True.
    design_vars : OrderedDict or None
        Design variables
    responses : OrderedDict or None, , optional
        Responses
    model_path : str or None, optional
        Path to the subsystem to be transcribed to XDSM.  If None, use the model root.
    recurse : bool, optional
        If False, treat the top level of each name as the source/target component.
    include_external_outputs : bool, optional
        If True, show externally connected outputs when transcribing a subsystem.
        Defaults to True.
    subs : tuple, optional
       Character pairs to be substituted. Forbidden characters or just for the sake of nicer names.
    writer: str, optional
        Writer is pyXDSM or XDSMjs.
        Defaults to pyXDSM.
    show_browser : bool, optional
        If True, pop up a browser to view the generated html file.
        Defaults to False.
    add_process_conns: bool
        Add process connections (thin black lines)
        Defaults to True
    kwargs : dict
        Keyword arguments

    Returns
    -------
        XDSM
    """
    # TODO implement residuals

    # Box appearance
    box_stacking = kwargs.pop('box_stacking', _DEFAULT_BOX_STACKING)
    box_width = kwargs.pop('box_width', _DEFAULT_BOX_WIDTH)
    box_lines = kwargs.pop('box_lines', _DEFAULT_BOX_WIDTH)

    def format_block(names, **kwargs):
        if writer == 'pyxdsm':
            return _format_block_string(var_names=names, stacking=box_stacking,
                                        box_width=box_width, box_lines=box_lines, **kwargs)
        else:
            return names

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

    writer_name = writer.lower()  # making it case insensitive

    if writer_name == 'pyxdsm':  # pyXDSM
        x = XDSMWriter()
    elif writer_name == 'xdsmjs':  # XDSMjs
        x = XDSMjsWriter()

    if optimizer is not None:
        x.add_optimizer(optimizer)

    if solver is not None:
        x.add_solver(solver)

    design_vars2 = _collect_connections(design_vars)
    responses2 = _collect_connections(responses)

    # Design variables
    for comp, conn_vars in iteritems(design_vars2):
        conn_vars = [_replace_chars(var, subs) for var in conn_vars]  # Format var names
        opt_con_vars = [_opt_var_str(var) for var in conn_vars]   # Optimal var names
        init_con_vars = [_init_var_str(var, writer_name) for var in conn_vars]   # Optimal var names
        x.connect('opt', comp, format_block(conn_vars))  # Connection from optimizer
        x.add_output(comp, format_block(opt_con_vars), side='left')  # Optimal design variables
        x.add_output('opt', format_block(opt_con_vars), side='left')  # Optimal design variables
        x.add_input('opt', format_block(init_con_vars))  # Initial design variables

    # Responses
    for comp, conn_vars in iteritems(responses2):
        conn_vars = [_replace_chars(var, subs) for var in conn_vars]  # Optimal var names
        opt_con_vars = [_opt_var_str(var) for var in conn_vars]
        x.connect(comp, 'opt', conn_vars)  # Connection to optimizer
        x.add_output(comp, format_block(opt_con_vars), side='left')  # Optimal output

    # Get the top level system to be transcripted to XDSM
    comps = _get_comps(tree, model_path=model_path, recurse=recurse)

    # Add components
    for comp in comps:
        x.add_comp(name=comp['abs_name'], label=_replace_chars(comp['name'], substitutes=subs))

    # Add the connections
    for src, dct in iteritems(conns3):
        for tgt, conn_vars in iteritems(dct):
            x.connect(src, tgt, format_block(conn_vars))

    # Add the externally sourced inputs
    for src, tgts in iteritems(external_inputs3):
        for tgt, conn_vars in iteritems(tgts):
            formatted_conn_vars = [_replace_chars(o, substitutes=subs) for o in conn_vars]
            x.add_input(tgt, format_block(formatted_conn_vars))

    # Add the externally connected outputs
    if include_external_outputs:
        for src, tgts in iteritems(external_outputs3):
            output_vars = set()
            for tgt, conn_vars in iteritems(tgts):
                output_vars |= set(conn_vars)
                formatted_outputs = [_replace_chars(o, subs) for o in output_vars]
            x.add_output(src, formatted_outputs, side='right')

    if add_process_conns:
        x.add_workflow()
    x.write(filename, cleanup=cleanup, **kwargs)

    if show_browser:
        # path will be specified based on the "out_format", if all required inputs where
        # provided for showing the results.
        if writer_name == 'pyxdsm':  # pyXDSM
            ext = 'pdf'
        elif writer_name == 'xdsmjs':  # XDSMjs
            ext = 'html'
        else:
            err_msg = '"{}" is an invalid writer name.'
            raise ValueError(err_msg.format(writer))
        path = '.'.join([filename, ext])
        webview(path)  # Can open also PDFs

    return x


def _get_cls_name(obj):
    return obj.__class__.__name__


def _residual_str(name):
    """Makes a residual symbol."""
    return '\\mathcal{R}(%s)' % name


def _opt_var_str(name):
    """Puts an asterisk superscript on a string."""
    return '{}^*'.format(name)


def _init_var_str(name, writer):
    """Puts a 0 superscript on a string."""
    if writer == 'pyxdsm':
        return '{}^{{(0)}}'.format(name)
    elif writer == 'xdsmjs':
        return '{}^(0)'.format(name)


def _process_connections(conns, recurse=True, subs=None):

    def convert(x):
        return _convert_name(x, recurse=recurse, subs=subs)

    conns_new = [{k: convert(v) for k, v in iteritems(conn)} for conn in conns]
    return conns_new


def _accumulate_connections(conns):
    # Makes a dictionary with source and target components and with the connection sources
    name_type = 'path'
    conns_new = dict()
    for conn in conns:  # list
        src_comp = conn['src'][name_type]
        tgt_comp = conn['tgt'][name_type]
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
        connections.setdefault(conv_var['path'], []).append(conv_var['var'])
    return connections


def _convert_name(name, recurse=True, subs=None):
    """
    From an absolute path returns the variable name and its owner component in a dict.
    Names are also formatted.

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
        name_items = name.split('.')
        if recurse:
            comp = name_items[-2]  # -1 is variable name, before that -2 is the component name
            path = name.rsplit('.', 1)[0]
        else:
            comp = name_items[0]
            path = comp
        var = name_items[-1]
        var = _replace_chars(var, substitutes=subs)
        return {'comp': comp, 'var': var,
                'abs_name': _format_name(name), 'path': _format_name(path)}

    if isinstance(name, list):  # If a source has multiple targets
        return [convert(n) for n in name]
    else:  # string
        return convert(name)


def _format_name(x):
    # Character to replace dot (.) in names for pyXDSM component and connection names
    return x.replace('.', '@')


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
    else:
        for conn in conns:
            src = conn['src']
            src_path = _format_name(src.rsplit('.', 1)[0])
            tgt = conn['tgt']
            tgt_path = _format_name(tgt.rsplit('.', 1)[0])

            if src.startswith(model_path) and tgt.startswith(model_path):
                # Internal connections
                internal_conns.append({'src': src_path, 'tgt': tgt_path})
            elif not src.startswith(model_path) and tgt.startswith(model_path):
                # Externally connected input
                external_inputs.append({'src': src_path, 'tgt': tgt_path})
            elif src.startswith(model_path) and not tgt.startswith(model_path):
                # Externally connected output
                external_outputs.append({'src': src_path, 'tgt': tgt_path})

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
    comp_names = set()

    def get_children(tree_branch, path=''):
        for ch in tree_branch['children']:
            ch['path'] = path
            name = ch['name']
            if path:
                ch['abs_name'] = _format_name('.'.join([path, name]))
            else:
                ch['abs_name'] = _format_name(name)
            ch['rel_name'] = name
            if ch['subsystem_type'] == 'component':
                if name in comp_names:  # There is already a component with the same name
                    ch['name'] = '.'.join([path, name])  # Replace with absolute name
                    for comp in components:
                        if comp['name'] == name:  # replace in the other component to abs. name
                            comp['name'] = '.'.join([comp['path'], name])
                components.append(ch)
                comp_names.add(ch['rel_name'])
            elif recurse:
                if path:
                    new_path = '.'.join([path, ch['name']])
                else:
                    new_path = ch['name']
                get_children(ch, new_path)
            else:
                components.append(ch)
                comp_names.add(ch['rel_name'])

    top_level_tree = tree
    if model_path is not None:
        path_list = model_path.split('.')
        while path_list:
            next_path = path_list.pop(0)
            children = [child for child in top_level_tree['children']]
            top_level_tree = [c for c in children if c['name'] == next_path][0]

    get_children(top_level_tree)
    return components


def _format_block_string(var_names, stacking='vertical', **kwargs):
    max_lines = kwargs.pop('box_lines', _MAX_BOX_LINES)
    if stacking == 'vertical':
        if (max_lines is None) or (max_lines >= len(var_names)):
            return var_names
        else:
            names = var_names[0:max_lines]
            names[-1] = names[-1] + ', ...'
            return names

    elif stacking == 'horizontal':
        return ', '.join(var_names)
    elif stacking in ('max_chars', 'cut_chars'):
        max_chars = kwargs.pop('box_width', _DEFAULT_BOX_CHAR_LIMIT)
        if len(var_names) < 2:
            return var_names
        else:
            lengths = 0
            lines = list()
            line = ''
            for name in var_names:
                lengths += len(name)
                if lengths <= max_chars:
                    if line:  # there are already var names on the line
                        line += ', ' + name
                    else:  # it will be the first var name on the line
                        line = name
                else:  # make new line
                    if stacking == 'max_chars':
                        lines.append(line)
                        line = name
                        lengths = len(name)
                    else:  # 'cut_chars'
                        lines.append(line + ', ...')
                        line = ''  # No new line
                        break
            if line:  # it will be the last line, if var_names was not empty
                lines.append(line)
            if len(lines) > 1:
                return lines
            else:
                return lines[0]  # return the string instead of a list
    else:
        msg = 'Invalid block stacking option "{}".'
        raise ValueError(msg.format(stacking))


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
