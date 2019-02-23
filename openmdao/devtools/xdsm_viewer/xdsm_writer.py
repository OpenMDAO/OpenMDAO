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

# TODO solvers: also include solvers of groups, not just for the root. Include connections between
#  component inputs & outputs and the solver.
# TODO show parallel blocks also in XDSMjs

from __future__ import print_function

import json
import os
import warnings

from six import iteritems

from openmdao.devtools.problem_viewer.problem_viewer import _get_viewer_data
from openmdao.devtools.webview import webview
from openmdao.devtools.xdsm_viewer.html_writer import write_html

from numpy.distutils.exec_command import find_executable

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
# Variable formatting settings
_SUPERSCRIPTS = {'optimal': '*', 'initial': '(0)', 'target': 't'}
# Default solver, if no solver is added to a group.
_DEFAULT_SOLVER_NAMES = {'linear': 'LN: RUNONCE', 'nonlinear': 'NL: RUNONCE'}

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
        self.comps = []
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

    def add_solver(self, name, label=None, **kwargs):
        self.comp_names.append(self._format_id(name))
        self.add_system(name, 'mda', label, **kwargs)

    def add_comp(self, name, label=None, **kwargs):
        self.comp_names.append(self._format_id(name))
        self.add_system(name, 'analysis', label, **kwargs)

    def add_func(self, name, label=None, **kwargs):
        pass

    def add_optimizer(self, label, name='opt', **kwargs):
        self.optimizer = self._format_id(name)
        self.add_system(name, 'optimization', label, **kwargs)

    def add_system(self, node_name, style, label=None, **kwargs):
        if label is None:
            label = node_name
        dct = {"type": style, "id": self._format_id(node_name), "name": label}
        self.comps.append(dct)

    def add_workflow(self, comp_names=None):
        if comp_names is None:
            comp_names = self.comp_names

        # FIXME now it does not work as expected, because second process might be inserted
        #  into another process (like optimizer and MDA)
        if len(self.processes) < 2:
            self.processes.append([self.optimizer, comp_names])
        else:
            new_proc = [comp_names[0], comp_names[1:]]
            self.processes[1].insert(1, new_proc)

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
        data = {'edges': self.connections, 'nodes': self.comps, 'workflow': self.processes}
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


try:
    from pyxdsm.XDSM import XDSM
except ImportError:
    XDSM = None
else:

    class XDSMWriter(XDSM):
        """
        XDSM with some additional semantics.
        Creates a TeX file and TiKZ file, and converts it to PDF.

        .. note:: On Windows it might be necessary to add the second line in the
           :class:`~pyxdsm.XDSM.XDSM`, if an older version of the package is installed::

            diagram_styles_path = os.path.join(module_path, 'diagram_styles')
            diagram_styles_path = diagram_styles_path.replace('\\', '/')  # Add this line on Windows

           This issue is resolved in the latest version of pyXDSM.

        """

        def write(self, filename=None, **kwargs):
            """
            Write the output file.

            This just wraps the XDSM version and throws out incompatible arguments.

            Parameters
            ----------
            filename : str
                Name of the file to be written.
            kwargs : dict
                Keyword args
            """
            build = kwargs.pop('build', False)
            cleanup = kwargs.pop('cleanup', True)

            super(XDSMWriter, self).write(file_name=filename, build=build, cleanup=cleanup, **kwargs)

        def add_solver(self, name, label=None, **kwargs):
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
            if label is None:
                label = name
            self.add_system(node_name=name, style='MDA', label='\\text{%s}' % label, **kwargs)

        def add_comp(self, name, label=None, stack=False, **kwargs):
            """
            Add a component.

            Parameters
            ----------
            label : str
                Label in the XDSM, defaults to the name of the component.
            name : str
                Name of the component
            stack : bool
                True for parallel components.
                Defaults to False.
            kwargs : dict
                Keyword args
            """
            if label is None:
                label = name
            self.add_system(node_name=name, style='Analysis', label='\\text{%s}' % label,
                            stack=stack, **kwargs)

        def add_func(self, name, label=None, stack=False, **kwargs):
            """
            Add a function

            Parameters
            ----------
            label : str
                Label in the XDSM, defaults to the name of the component.
            name : str
                Name of the component
            stack : bool
                True for parallel.
                Defaults to False.
            kwargs : dict
                Keyword args
            """
            if label is None:
                label = name
            self.add_system(node_name=name, style='Function', label='\\text{%s}' % label,
                            stack=stack, **kwargs)

        def add_optimizer(self, name, label=None, **kwargs):
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

        def add_workflow(self, comp_names=None):
            """
            Add a workflow. If not specified, all components will be included.

            Parameters
            ----------
            comp_names : list(str) or None, optional
                List of component names.
                Defaults to None.
            """
            if comp_names is None:
                comp_names = [c[0] for c in self.comps]
            comps = comp_names + [comp_names[0]]  # close the loop
            self.add_process(comps, arrow=_PROCESS_ARROWS)


def write_xdsm(problem, filename, model_path=None, recurse=True,
               include_external_outputs=True, out_format='tex',
               include_solver=False, subs=_CHAR_SUBS, show_browser=True,
               add_process_conns=True, show_parallel=True, **kwargs):
    """
    Writes XDSM diagram of an optimization problem.

    With the 'tex' or 'pdf' output format it uses the pyXDSM package, with 'html'
    output format it uses XDSMjs.

    If a component (or group) name is not unique in the diagram, the systems absolute path is
    used as a label. If the component (or group) name is unique, the relative name of the
    system is the label.

    In the diagram the connections are marked with the source name.

    Writer specific settings and default:

    pyXDSM
    ~~~~~~

    * The appearance of the boxes can be controlled with "box_stacking" and "box_width" arguments.
      The box stacking can be:

      * "horizontal" - All variables in one line
      * "vertical" - All variables in one column
      * "cut_chars" - The text in the box will be one line with the maximum number of characters
        limited by "box_width".
      * "max_chars" - The "box_width" argument is used to determine
        the maximum allowed width of boxes (in characters).
      * "empty" - There are no variable names in the data block. Good for large diagrams.

      A default value is taken, if not specified.
    * By default the part of variable names following underscores (_)
      are not converted to subscripts.
      To write in subscripts wrap that part of the name into a round bracket.
      Example: To write :math:`x_12` the variable name should be "x(12)"
    * "box_lines" can be used to limit the number of lines, if the box stacking is vertical
    * "numbered_comps": bool, If True, components are numbered. Defaults to True.
    * "number_alignment": str, Horizontal or vertical. Defaults to horizontal. If "numbered_comps"
      is True, it positions the number either above or in front of the component label.

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
        Output format, one of "tex" or "pdf" (pyXDSM) or "html" (XDSMjs).
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
    show_parallel : bool
        Show parallel components with stacked blocks.
        Defaults to True.
    kwargs : dict
        Keyword arguments
    Returns
    -------
       XDSM or AbstractXDSMWriter
    """
    build_pdf = False
    if out_format in ('tex', 'pdf'):
        if XDSM is None:
            print('\nThe "tex" and "pdf" formats require the pyxdsm package. You can download the '
                  'package from https://github.com/mdolab/pyXDSM, or install it directly from '
                  'github using:  pip install git+https://github.com/mdolab/pyXDSM.git')
            return
        elif out_format == 'pdf':
            if not find_executable('pdflatex'):
                print("Can't find pdflatex, so a pdf can't be generated.")
            else:
                build_pdf = True

    viewer_data = _get_viewer_data(problem)
    driver = problem.driver
    if model_path is None:
        _model = problem.model
    else:
        _model = problem.model._get_subsystem(model_path)
        if _model is None:
            msg = 'Model path "{}" does not exist in problem "{}".'
            raise ValueError(msg.format(model_path, problem))

    # Name is None if the driver is not specified
    driver_name = _get_cls_name(driver) if driver else None

    design_vars = _model.get_design_vars()
    responses = _model.get_responses()

    filename = filename.replace('\\', '/')  # Needed for LaTeX

    try:
        writer = _OUT_FORMATS[out_format]
    except KeyError:
        msg = 'Invalid output format "{}", choose from: {}'
        raise ValueError(msg.format(out_format, _OUT_FORMATS.keys()))
    writer_name = writer.lower()  # making it case insensitive
    if isinstance(subs, dict):
        subs = subs[writer_name]  # Getting the character substitutes of the chosen writer
    return _write_xdsm(filename, viewer_data=viewer_data,
                       optimizer=driver_name, include_solver=include_solver, model_path=model_path,
                       design_vars=design_vars, responses=responses, writer=writer,
                       recurse=recurse, subs=subs,
                       include_external_outputs=include_external_outputs, show_browser=show_browser,
                       add_process_conns=add_process_conns, build_pdf=build_pdf,
                       show_parallel=show_parallel, **kwargs)


def _write_xdsm(filename, viewer_data, optimizer=None, include_solver=False, cleanup=True,
                design_vars=None, responses=None, residuals=None, model_path=None, recurse=True,
                include_external_outputs=True, subs=_CHAR_SUBS, writer='pyXDSM', show_browser=False,
                add_process_conns=True, show_parallel=True, quiet=False, build_pdf=False, **kwargs):
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
    include_solver:  bool, optional
        Defaults to False.
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
    show_parallel : bool
        Show parallel components with stacked blocks.
        Defaults to True.
    quiet : bool
        Set to True to suppress output from pdflatex
    build_pdf : bool
        If True and a .tex file is generated, create a .pdf file from the .tex.

    kwargs : dict
        Keyword arguments

    Returns
    -------
        XDSM or AbstractXDSMWriter
    """
    # TODO implement residuals

    writer_name = writer.lower()  # making it case insensitive

    # Box appearance
    box_stacking = kwargs.pop('box_stacking', _DEFAULT_BOX_STACKING)
    box_width = kwargs.pop('box_width', _DEFAULT_BOX_WIDTH)
    box_lines = kwargs.pop('box_lines', _MAX_BOX_LINES)
    # In XDSMjs components are numbered by default, so only add for pyXDSM as an option
    add_component_indices = kwargs.pop('numbered_comps', True) and (writer_name == 'pyxdsm')
    number_alignment = kwargs.pop('number_alignment', 'horizontal')  # nothing, space or new line

    def format_block(names, **kwargs):
        if writer_name == 'pyxdsm':
            return _format_block_string(var_names=names, stacking=box_stacking,
                                        box_width=box_width, box_lines=box_lines, **kwargs)
        else:
            return names

    def format_var_str(name, var_type):
        sup = _SUPERSCRIPTS[var_type]
        if writer_name == 'pyxdsm':
            return '{}^{{{}}}'.format(name, sup)
        else:
            return '{}^{}'.format(name, sup)

    def number_label(number, text, alignment):
        # Adds an index to the label either above or on the left side.
        number_str = '{}: '.format(number)
        if alignment == 'horizontal':
            txt = '{}{}'.format(number_str, text)
            if box_stacking == 'vertical':
                return _multiline_block(txt)
            else:
                return txt
        elif alignment == 'vertical':
            return _multiline_block(number_str, text)
        else:
            return text  # In case of a wrong setting

    connections = viewer_data['connections_list']
    tree = viewer_data['tree']

    # Get the top level system to be transcripted to XDSM
    comps = _get_comps(tree, model_path=model_path, recurse=recurse)
    comps_dct = {comp['abs_name']: comp for comp in comps}
    solvers = []

    conns1, external_inputs1, external_outputs1 = _prune_connections(connections,
                                                                     model_path=model_path)

    conns2 = _process_connections(conns1, recurse=recurse, subs=subs)
    external_inputs2 = _process_connections(external_inputs1, recurse=recurse, subs=subs)
    external_outputs2 = _process_connections(external_outputs1, recurse=recurse, subs=subs)

    conns3 = _accumulate_connections(conns2)
    external_inputs3 = _accumulate_connections(external_inputs2)
    external_outputs3 = _accumulate_connections(external_outputs2)

    def add_solver(comps, first=0):
        # Adds a solver.
        # Uses some vars from the outer scope.
        comp_names = [_format_name(c['abs_name']) for c in comps]
        solver_label = _format_solver_str(tree,
                                          stacking=box_stacking,
                                          add_indices=add_component_indices)
        solver_label = _replace_chars(solver_label, subs)
        solver_name = _format_solver_str(tree, stacking='horizontal', add_indices=False)
        solver_name = _format_name(solver_name)

        if solver_label:  # At least one non-default solver (default solvers are ignored)
            nr_components = len(comps)
            if add_component_indices:
                solver_index = _make_loop_str(first=first,
                                              last=nr_components, start_index=1)
                solver_label = number_label(solver_index, solver_label, number_alignment)
            solvers.append(solver_label)
            x.add_solver(name=solver_name, label=solver_label)
            if add_process_conns:
                x.add_workflow([solver_name] + comp_names)

            # Add the connections
            for src, dct in iteritems(conns3):
                for tgt, conn_vars in iteritems(dct):
                    formatted_cons = format_block(conn_vars)
                    if (src in comp_names) and (tgt in comp_names):
                        formatted_targets = [format_var_str(c, 'target') for c in formatted_cons]
                        # From solver to components (targets)
                        x.connect(solver_name, tgt, formatted_targets)
                        # From components to solver
                        x.connect(src, solver_name, formatted_cons)

    if writer_name == 'pyxdsm':  # pyXDSM
        x = XDSMWriter()
    elif writer_name == 'xdsmjs':  # XDSMjs
        x = XDSMjsWriter()
    else:
        msg = 'Undefined XDSM writer "{}"'
        raise ValueError(msg.format(writer_name))

    if optimizer is not None:
        optimizer_label = optimizer
        optimizer_name = _format_name(optimizer)
        if add_component_indices:
            opt_index = len(comps) + 1  # index of last block + 1
            if include_solver:
                opt_index += len(solvers)
            nr_comps = len(x.comps)
            index_str = _make_loop_str(first=nr_comps, last=opt_index, start_index=1)
            optimizer_label = number_label(index_str, optimizer_label, number_alignment)
        x.add_optimizer(name=optimizer_name, label=optimizer_label)

        design_vars2 = _collect_connections(design_vars, recurse=recurse)
        responses2 = _collect_connections(responses, recurse=recurse)

        # Design variables
        for comp, conn_vars in iteritems(design_vars2):
            conn_vars = [_replace_chars(var, subs) for var in conn_vars]  # Format var names
            opt_con_vars = [format_var_str(var, 'optimal') for var in conn_vars]   # Optimal var names
            init_con_vars = [format_var_str(var, 'initial') for var in conn_vars]   # Optimal var names
            x.connect(optimizer_name, comp, format_block(conn_vars))  # Connection from optimizer
            x.add_output(comp, format_block(opt_con_vars), side='left')  # Optimal design variables
            x.add_output(optimizer_name, format_block(opt_con_vars), side='left')  # Optimal design variables
            x.add_input(optimizer_name, format_block(init_con_vars))  # Initial design variables

        # Responses
        for comp, conn_vars in iteritems(responses2):
            conn_vars = [_replace_chars(var, subs) for var in conn_vars]  # Optimal var names
            opt_con_vars = [format_var_str(var, 'optimal') for var in conn_vars]
            x.connect(comp, optimizer_name, conn_vars)  # Connection to optimizer
            x.add_output(comp, format_block(opt_con_vars), side='left')  # Optimal output

    if include_solver:
        # Default "run once" solvers are ignored
        # Nonlinear solver has precedence
        msg = "Solvers in the XDSM diagram are not fully supported yet, and needs manual editing."
        warnings.warn(msg)

        add_solver(comps=comps, first=1)

    # Add components
    for comp in comps:  # Driver is 1, so starting from 2
        i = len(x.comps) + 1
        label = _replace_chars(comp['name'], substitutes=subs)
        if add_component_indices:
            label = number_label(i, label, number_alignment)
        stack = comp['is_parallel'] and show_parallel
        x.add_comp(name=comp['abs_name'], label=label, stack=stack)

    # Add the connections
    for src, dct in iteritems(conns3):
        for tgt, conn_vars in iteritems(dct):
            stack = (comps_dct[src]['is_parallel'] or comps_dct[tgt]['is_parallel']) and show_parallel
            x.connect(src, tgt, label=format_block(conn_vars), stack=stack)

    # Add the externally sourced inputs
    for src, tgts in iteritems(external_inputs3):
        for tgt, conn_vars in iteritems(tgts):
            formatted_conn_vars = [_replace_chars(o, substitutes=subs) for o in conn_vars]
            stack = comps_dct[tgt]['is_parallel'] and show_parallel
            x.add_input(tgt, format_block(formatted_conn_vars), stack=stack)

    # Add the externally connected outputs
    if include_external_outputs:
        for src, tgts in iteritems(external_outputs3):
            output_vars = set()
            for tgt, conn_vars in iteritems(tgts):
                output_vars |= set(conn_vars)
            formatted_outputs = [_replace_chars(o, subs) for o in output_vars]
            stack = comps_dct[src]['is_parallel'] and show_parallel
            x.add_output(src, formatted_outputs, side='right', stack=stack)

    if add_process_conns:
        x.add_workflow()

    x.write(filename, cleanup=cleanup, quiet=quiet, build=build_pdf, **kwargs)

    if show_browser and (build_pdf or writer_name == 'xdsmjs'):
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
        if var not in conns_new[src_comp].setdefault(tgt_comp, []):  # Avoid duplicates
            conns_new[src_comp][tgt_comp].append(var)
    return conns_new


def _collect_connections(variables, recurse):
    conv_vars = [_convert_name(v, recurse) for v in variables]
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
        sep = '.'
        name = name.replace('@', sep)
        name_items = name.split(sep)
        if recurse:
            if len(name_items) > 1:
                comp = name_items[-2]  # -1 is variable name, before that -2 is the component name
                path = name.rsplit(sep, 1)[0]
            else:
                msg = ('The name "{}" cannot be processed. The separator character is "{}", '
                       'which does not occur in the name.')
                raise ValueError(msg.format(name, sep))
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


def _format_name(name):
    # Replaces illegal characters in names for pyXDSM component and connection names
    # This does not effect the labels, only reference names TikZ
    if isinstance(name, str):
        for char in ('.', ' ', '-', '_', ':'):
            name = name.replace(char, '@')
    return name


def _prune_connections(conns, model_path=None, sep='.'):
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
            src = src0 = conn['src']
            if src.startswith(model_path):
                src = src[len(model_path):]
            src_path = _format_name(src.rsplit(sep, 1)[0])
            tgt = tgt0 = conn['tgt']
            if tgt.startswith(model_path):
                tgt = tgt[len(model_path):]
            tgt_path = _format_name(tgt.rsplit(sep, 1)[0])

            if src0.startswith(model_path) and tgt0.startswith(model_path):
                # Internal connections
                internal_conns.append({'src': src_path, 'tgt': tgt_path})
            elif not src0.startswith(model_path) and tgt0.startswith(model_path):
                # Externally connected input
                external_inputs.append({'src': src_path, 'tgt': tgt_path})
            elif src0.startswith(model_path) and not tgt0.startswith(model_path):
                # Externally connected output
                external_outputs.append({'src': src_path, 'tgt': tgt_path})
        print('CONNNNNS', internal_conns, external_inputs, external_outputs)
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
    components = list()  # Components will be collected to this list
    comp_names = set()  # To check if names are unique
    sep = '.'

    def get_children(tree_branch, path=''):
        for ch in tree_branch['children']:
            ch['path'] = path
            name = ch['name']
            if path:
                ch['abs_name'] = _format_name(sep.join([path, name]))
            else:
                ch['abs_name'] = _format_name(name)
            ch['rel_name'] = name
            if ch['subsystem_type'] == 'component':
                if name in comp_names:  # There is already a component with the same name
                    ch['name'] = sep.join([path, name])  # Replace with absolute name
                    for comp in components:
                        if comp['name'] == name:  # replace in the other component to abs. name
                            comp['name'] = sep.join([comp['path'], name])
                components.append(ch)
                comp_names.add(ch['rel_name'])
            elif recurse:
                if path:
                    new_path = sep.join([path, ch['name']])
                else:
                    new_path = ch['name']
                get_children(ch, new_path)
            else:
                components.append(ch)
                comp_names.add(ch['rel_name'])

    top_level_tree = tree
    if model_path is not None:
        path_list = model_path.split(sep)
        while path_list:
            next_path = path_list.pop(0)
            children = [child for child in top_level_tree['children']]
            top_level_tree = [c for c in children if c['name'] == next_path][0]

    get_children(top_level_tree)
    return components


def _format_block_string(var_names, stacking='vertical', **kwargs):
    end_str = ', ...'
    max_lines = kwargs.pop('box_lines', _MAX_BOX_LINES)
    if stacking == 'vertical':
        if (max_lines is None) or (max_lines >= len(var_names)):
            return var_names
        else:
            names = var_names[0:max_lines]
            names[-1] = names[-1] + end_str
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
                        lines.append(line + end_str)
                        line = ''  # No new line
                        break
            if line:  # it will be the last line, if var_names was not empty
                lines.append(line)
            if len(lines) > 1:
                return lines
            else:
                return lines[0]  # return the string instead of a list
    elif stacking == 'empty':  # No variable names in the data block, good for big diagrams
        return ''
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


def _format_solver_str(dct, stacking='horizontal', solver_types=('nonlinear', 'linear'),
                       add_indices=False):
    """
    Format solver string.

    Parameters
    ----------
    dct : dict
        Dictionary, which contains keys for the solver names
    stacking : str
        Box stacking
    solver_types : tuple(str)
        Solver types, e.g. "linear"

    Returns
    -------
        str
    """
    stacking = stacking.lower()

    solvers = []
    for solver_type in solver_types:
        solver_name = dct['{}_solver'.format(solver_type)]
        if solver_name != _DEFAULT_SOLVER_NAMES[solver_type]:
            solvers.append(solver_name)
    if stacking == 'vertical':
        # Make multiline comp if not numbered
        if add_indices:  # array is already created for the numbering
            return '} \\\\ \\text{'.join(solvers)
        else:  # Goes into an array environment
            return _multiline_block(*solvers)
    elif stacking in ('horizontal', 'max_chars', 'cut_chars'):
        return ' '.join(solvers)
    else:
        msg = ('Invalid stacking "{}". Choose from: "vertical", "horizontal", "max_chars", '
               '"cut_chars"')
        raise ValueError(msg.format(stacking))


def _multiline_block(*texts, **kwargs):
    """
    Makes a string for a multiline block.

    texts : iterable(str)
        Text strings, each will go to new line
    kwargs : dict
        Unused keywords are ignored.
        "end_char" is the separator at the end of line. Defaults to '' (no separator).
    Returns
    -------
       str
    """
    end_char = kwargs.pop('end_char', '')
    texts = ['\\text{{{}{}}}'.format(t, end_char) for t in texts]
    template = '$\\begin{{array}}{{{pos}}} {text} \\end{{array}}$'
    new_line = ' \\\\ '
    return template.format(text=new_line.join(texts), pos='c'*len(texts))


def _make_loop_str(first, last, start_index=0):
    i = start_index
    txt = '{}, {}$ \\rightarrow $ {}'
    return txt.format(first+i, last+i, first+i+1)


##### openmdao command line setup


def _xdsm_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao xdsm' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', '--outfile', default='xdsm_out', action='store', dest='outfile',
                        help='XDSM output file. (use pathname without extension)')
    parser.add_argument('-f', '--format', default='html', action='store', dest='format',
                        choices=['html', 'pdf', 'tex'], help='format of XSDM output.')
    parser.add_argument('-m', '--model_path', action='store', dest='model_path',
                        help='Path to system to transcribe to XDSM.')
    parser.add_argument('-r', '--recurse', action='store_true', dest='recurse',
                        help="Don't treat the top level of each name as the source/target component.")
    parser.add_argument('--no_browser', action='store_true', dest='no_browser',
                        help="Don't display in a browser.")
    parser.add_argument('--no_parallel', action='store_true', dest='no_parallel',
                        help="don't show stacked parallel blocks. Only active for 'pdf' and 'tex' "
                             "formats.")
    parser.add_argument('--no_ext', action='store_true', dest='no_extern_outputs',
                        help="Don't show externally connected outputs.")
    parser.add_argument('-s', '--include_solver', action='store_true', dest='include_solver',
                        help="Include the problem model's solver in the XDSM.")
    parser.add_argument('--no_process_conns', action='store_true', dest='no_process_conns',
                        help="Don't add process connections (thin black lines).")
    parser.add_argument('--box_stacking', action='store', default=_DEFAULT_BOX_STACKING,
                        choices=['max_chars', 'vertical', 'horizontal', 'cut_chars', 'empty'],
                        dest='box_stacking', help='Controls the appearance of boxes.')
    parser.add_argument('--box_width', action='store', default=_DEFAULT_BOX_WIDTH,
                        dest='box_width', type=int, help='Controls the width of boxes.')
    parser.add_argument('--box_lines', action='store', default=_MAX_BOX_LINES,
                        dest='box_lines', type=int,
                        help='Limits number of vertical lines in box if box_stacking is vertical.')
    parser.add_argument('--numbered_comps', action='store_true', dest='numbered_comps',
                        help="Display components with numbers.  Only active for 'pdf' and 'tex' "
                        "formats.")
    parser.add_argument('--number_alignment', action='store', dest='number_alignment',
                        choices=['horizontal', 'vertical'], default='horizontal',
                        help='Positions the number either above or in front of the component label '
                        'if numbered_comps is true.')


def _xdsm_cmd(options):
    """
    Return the post_setup hook function for 'openmdao xdsm'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    def _xdsm(prob):
        kwargs = {}
        for name in ['box_stacking', 'box_width', 'box_lines', 'numbered_comps', 'number_alignment']:
            val = getattr(options, name)
            if val is not None:
                kwargs[name] = val

        write_xdsm(prob, filename=options.outfile, model_path=options.model_path,
                   recurse=options.recurse,
                   include_external_outputs=not options.no_extern_outputs,
                   out_format=options.format,
                   include_solver=options.include_solver, subs=_CHAR_SUBS,
                   show_browser=not options.no_browser, show_parallel=not options.no_parallel,
                   add_process_conns=not options.no_process_conns, **kwargs)
        exit()
    return _xdsm
