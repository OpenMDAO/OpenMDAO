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

from __future__ import print_function

import json
import os
from distutils.version import LooseVersion

from numpy.distutils.exec_command import find_executable
from six import iteritems, string_types

from openmdao.core.problem import Problem
from openmdao.utils.general_utils import simple_warning
from openmdao.utils.webview import webview
from openmdao.visualization.n2_viewer.n2_viewer import _get_viewer_data
from openmdao.visualization.xdsm_viewer.html_writer import write_html

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
    'pyxdsm': (('_', r'\_'), ('(', '_{'), (')', '}'),),
    'xdsmjs': ((' ', '-'), (':', ''), ('_', r'\_'),),
}
# Variable formatting settings
_SUPERSCRIPTS = {'optimal': '*', 'initial': '(0)', 'target': 't', 'consistency': 'c'}
# Default solver, if no solver is added to a group.
_DEFAULT_SOLVER_NAMES = {'linear': 'LN: RUNONCE', 'nonlinear': 'NL: RUNONCE'}
# On which side to place outputs? One of "left", "right"
_DEFAULT_OUTPUT_SIDE = 'left'
# Default writer, this will be used if settings are not found for a custom writer
_DEFAULT_WRITER = 'pyxdsm'

# Maps OpenMDAO component types with the available block styling options in the writer.
# For pyXDSM check the "diagram_styles" file for style definitions.
# For XDSMjs check the CSS style sheets.
_COMPONENT_TYPE_MAP = {
    'pyxdsm': {  # Newest release
        'indep': 'Function',
        'explicit': 'Function',
        'implicit': 'ImplicitFunction',
        'exec': 'Function',
        'metamodel': 'Metamodel',
        'group': 'Group',
        'implicit_group': 'ImplicitGroup',
        'optimization': 'Optimization',
        'doe': 'DOE',
        'solver': 'MDA',
    },
    'pyxdsm 1.0': {  # Legacy color scheme
        'indep': 'Function',
        'explicit': 'Function',
        'implicit': 'ImplicitAnalysis',
        'exec': 'Function',
        'metamodel': 'Metamodel',
        'group': 'Function',
        'implicit_group': 'ImplicitAnalysis',
        'optimization': 'Optimization',
        'doe': 'DOE',
        'solver': 'MDA',
    },
    'xdsmjs': {
        'indep': 'function',
        'explicit': 'function',
        'implicit': 'analysis',
        'exec': 'function',
        'metamodel': 'metamodel',
        'group': 'function',
        'implicit_group': 'analysis',
        'optimization': 'optimization',
        'doe': 'doe',
        'solver': 'mda',
    }
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
# Options: horizontal, vertical, max_chars, cut_chars, empty
_DEFAULT_BOX_STACKING = 'max_chars'
# Show arrowheads in process connection lines
_PROCESS_ARROWS = False
# Maximum number of lines in a box. No limit, if None.
_MAX_BOX_LINES = None
# If components are indexed, this will be the first index. 0 or 1
_START_INDEX = 0
# Place the number one row above ("vertical") or in the same row as text ("horizontal")
_DEFAULT_NUMBER_ALIGNMENT = 'horizontal'


class BaseXDSMWriter(object):
    """
    All XDSM writers have to inherit from this base class.

    Attributes
    ----------
    name : str
        Name of XDSM writer.
    extension : str
        Output file saved with this extension.
    type_map : str
        XDSM component type.
    """

    def __init__(self, name, options={}):
        """
        Initialize.

        Parameters
        ----------
        name : str
            Name of this XDSM writer
        options : dict
            Writer options.
        """
        self.name = name
        # This should be a dictionary mapping OpenMDAO system types to XDSM component types.
        # See for example any value in _COMPONENT_TYPE_MAP
        self.type_map = None
        self.extension = None  # Implement in child class as string file extension


class AbstractXDSMWriter(BaseXDSMWriter):
    """
    Abstract class to define methods for XDSM writers.

    All methods should be implemented in child classes.

    Attributes
    ----------
    comps : list of dicts
        List of systems where the list items are dicts indicating type, id, and name.
    connections : list of dicts
        List of connections where the list items are dicts indicating 'to', 'from', 'name' of edge.
    processes : list
        List of process.
    """

    def __init__(self, name='abstract_xdsm_writer'):
        """
        Initialize.

        Parameters
        ----------
        name : str
            Name of XDSM writer.
        """
        super(AbstractXDSMWriter, self).__init__(name=name)
        self.comps = []
        self.connections = []
        self.processes = []

    def add_solver(self, label, name='solver', **kwargs):
        """
        Add a solver.

        Parameters
        ----------
        label : str
            Label in the XDSM
        name : str
            Name of the solver
        **kwargs : dict
            Keyword args
        """
        pass  # Implement in child class

    def add_comp(self, name, label=None, stack=False, comp_type=None, **kwargs):
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
        comp_type : str or None
            Component type, e.g. explicit, implicit or metamodel
        **kwargs : dict
            Keyword args
        """
        pass  # Implement in child class

    def add_driver(self, label, name='opt', driver_type='optimization', **kwargs):
        """
        Add a driver.

        Parameters
        ----------
        label : str
            Label in the XDSM.
        name : str
            Name of the driver.
        driver_type : str
            Optimization or DOE.
            Defaults to "optimization".
        **kwargs : dict
            Keyword args
        """
        pass  # Implement in child class

    def add_input(self, name, label, style='DataIO', stack=False):
        """
        Add input connection.

        Parameters
        ----------
        name : str
            Target name.
        label : str
            Label for connection.
        style : str
            Formatting style.
        stack : bool
            True for parallel.
            Defaults to False.
        """
        pass  # Implement in child class

    def add_output(self, name, label, style='DataIO', stack=False, side=_DEFAULT_OUTPUT_SIDE):
        """
        Add output connection.

        Parameters
        ----------
        name : str
            Target name.
        label : str
            Label for connection.
        style : str
            Formatting style.
        stack : bool
            True for parallel.
            Defaults to False.
        side : str
            Location of output, either 'left' or 'right'.
        """
        pass  # Implement in child class

    def add_process(self, systems, arrow=True):
        """
        Add process.

        Parameters
        ----------
        systems : list
            List of systems.
        arrow : bool
            Show process arrow.
        """
        pass  # Implement in child class

    @staticmethod
    def format_block(names, **kwargs):
        """
        Reimplement this method to format the names in a data block.

        Parameters
        ----------
        names : list
            List of items in the block
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
            list(str)
        """
        return names

    @staticmethod
    def format_var_str(name, var_type, superscripts=None):
        """
        Format a variable name to include a superscript for the variable type.

        Parameters
        ----------
        name : str
            Name (label in the block) of the variable.
        var_type : str
            Variable type.
        superscripts : dict or None, optional
            A dictionary mapping variable types to their superscript notation

        Returns
        -------
        str
            Formatted var string.
        """
        if superscripts is None:
            superscripts = _SUPERSCRIPTS
        sup = superscripts[var_type]
        return '{}^{}'.format(name, sup)

    @staticmethod
    def _make_loop_str(first, last, start_index=0):
        # Implement, so number is formatted as follows: start, end --> next
        # Where start = first + start_index, end = last + start_index, next = start + 1
        return ''


class XDSMjsWriter(AbstractXDSMWriter):
    """
    Creates an interactive diagram with XDSMjs, which can be opened with a web browser.

    XDSMjs was created by Remi Lafage. The code and documentation is available at
    https://github.com/OneraHub/XDSMjs

    Attributes
    ----------
    driver : str
        Driver default name.
    comp_names : list
        Component names.
    _ul : str
        Name of the virtual first element.
    _br : str
        Name of the virtual last component.
    _multi_suffix : str
        If component ends with this string, it will be treated as a parallel component.
    reserved_words : tuple
        Ignored at text formatting.
    extension : str
        Output file saved with this extension. Value fixed at 'html' for this class.
    type_map : str
        XDSM component type.
    class_names : bool
        Include class names of components in diagonal blocks.
    """

    def __init__(self, name='xdsmjs', class_names=False, options={}):
        """
        Initialize.

        Parameters
        ----------
        name : str
            Name of this XDSM writer
        class_names : bool
            Include class names of the components in the diagonal
        options : dict
            Writer options.
        """
        super(XDSMjsWriter, self).__init__(name=name)
        self.driver = 'opt'  # Driver default name
        self.comp_names = []  # Component names
        self._ul = '_U_'  # Name of the virtual first element
        # If component ends with this string, it will be treated as a parallel component
        self._multi_suffix = '_multi'
        self.reserved_words = self._ul,  # Ignored at text formatting
        # Output file saved with this extension
        self.extension = 'html'
        if self.name in _COMPONENT_TYPE_MAP:
            self.type_map = _COMPONENT_TYPE_MAP[self.name]
        else:  # Use default
            self.type_map = _COMPONENT_TYPE_MAP[_DEFAULT_WRITER]
            msg = 'Name "{}" not found in component type mapping, will default to "{}"'
            simple_warning(msg.format(self.name, _DEFAULT_WRITER))
        self.class_names = class_names

    def _format_id(self, name, subs=(('_', ''),)):
        # Changes forbidden characters in the "id" of a component
        if name not in self.reserved_words:
            return _replace_chars(name, subs)
        else:
            return name

    def connect(self, src, target, label, **kwargs):
        """
        Connect to system block.

        Parameters
        ----------
        src : str
            Source system name.
        target : str
            Target system name.
        label : str
            Label to be displayed in the XDSM data block.
        **kwargs : dict
            Keyword args
        """
        edge = {'to': self._format_id(target), 'from': self._format_id(src), 'name': label}
        self.connections.append(edge)

    def add_solver(self, name, label=None, **kwargs):
        """
        Add a solver.

        Parameters
        ----------
        name : str
            Name of the solver
        label : str
            Label in the XDSM
        **kwargs : dict
            Keyword args
        """
        self.comp_names.append(self._format_id(name))
        style = self.type_map['solver']
        self.add_system(node_name=name, style=style, label=label, **kwargs)

    def add_comp(self, name, label=None, stack=False, comp_type=None, **kwargs):
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
        comp_type : str or None
            Component type, e.g. explicit, implicit or metamodel
        **kwargs : dict
            Keyword args
        """
        style = self.type_map.get(comp_type, 'function')
        self.comp_names.append(self._format_id(name))
        self.add_system(node_name=name, style=style, label=label, stack=stack, **kwargs)

    def add_driver(self, label, name='opt', driver_type='optimization', **kwargs):
        """
        Add a driver.

        Parameters
        ----------
        label : str
            Label in the XDSM.
        name : str
            Name of the driver.
        driver_type : str
            Optimization or DOE.
            Defaults to "optimization".
        **kwargs : dict
            Keyword args
        """
        self.driver = self._format_id(name)
        style = self.type_map.get(driver_type, 'optimization')
        self.add_system(node_name=name, style=style, label=label, **kwargs)

    def add_system(self, node_name, style, label=None, stack=False, cls=None, **kwargs):
        """
        Add a system.

        Parameters
        ----------
        node_name : str
            Name of the system
        style : str
            Block formatting style.
        label : str
            Label in the XDSM, defaults to the name of the component.
        stack : bool
            True for parallel.
            Defaults to False.
        **kwargs : dict
            Keyword args
        """
        if label is None:
            label = node_name
        if stack:  # Parallel block
            style += self._multi_suffix  # Block will be stacked in XDSMjs, if ends with this string
        if cls is not None:
            label += '-{}'.format(cls)  # Append class name
        dct = {"type": style, "id": self._format_id(node_name), "name": label}
        self.comps.append(dct)

    def add_workflow(self, solver=None):
        """
        Add a workflow. If "comp_names" is None, all components will be included.

        Parameters
        ----------
        solver : dict or None, optional
            Solver info.
        """
        def recurse(solv, nr, process):
            for i, cmp in enumerate(process, start=1):
                if cmp == solv:
                    process[i:i + nr] = [process[i:i + nr]]
                    return
                elif isinstance(cmp, list):
                    recurse(solv, nr, cmp)
                    break

        if solver is None:
            comp_names = self.comp_names
            solver_name = None
        else:
            solver_name = solver['abs_name']
            comp_names = [c['abs_name'] for c in solver['comps']]
        nr_comps = len(comp_names)

        if not self.processes:  # If no process was added yet, add the process of the driver
            self.processes = [self.driver, list(self.comp_names)]
        recurse(solver_name, nr_comps, self.processes)  # Mutates self.processes

    def add_input(self, name, label=None, style='DataIO', stack=False):
        """
        Add input connection.

        Parameters
        ----------
        name : str
            Target name.
        label : str
            Label for connection.
        style : str
            Formatting style.
        stack : bool
            True for parallel.
            Defaults to False.
        """
        self.connect(src=self._ul, target=name, label=label)

    def add_output(self, name, label=None, style='DataIO', stack=False, side=_DEFAULT_OUTPUT_SIDE):
        """
        Add output connection.

        Parameters
        ----------
        name : str
            Target name.
        label : str
            Label for connection.
        style : str
            Formatting style.
        stack : bool
            True for parallel.
            Defaults to False.
        side : str
            Location of output, either 'left' or 'right'.
        """
        if side == "left":
            self.connect(src=name, target=self._ul, label=label)
        else:
            simple_warning('Right side outputs not implemented for XDSMjs.')
            self.connect(src=name, target=self._ul, label=label)

    def collect_data(self):
        """
        Make a dictionary with the structure of an XDSMjs JSON file.

        Returns
        -------
            dict
        """
        data = {'edges': self.connections, 'nodes': self.comps, 'workflow': self.processes}
        return data

    def write(self, filename='xdsmjs', embed_data=True, **kwargs):
        """
        Write HTML output file, and depending on value of "embed_data" a JSON file with the data.

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
        **kwargs : dict
            Keyword args
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

    class XDSMWriter(XDSM, BaseXDSMWriter):
        r"""
        XDSM with some additional semantics.

        Creates a TeX file and TiKZ file, and converts it to PDF.

        .. note:: On Windows it might be necessary to add the second line in the
           :class:`~pyxdsm.XDSM.XDSM`, if an older version of the package is installed::

            diagram_styles_path = os.path.join(module_path, 'diagram_styles')
            diagram_styles_path = diagram_styles_path.replace('\\', '/')  # Add this line on Windows

           This issue is resolved in the latest version of pyXDSM.

        Attributes
        ----------
        name : str
            Name of XDSM writer.
        box_stacking : str
            Controls the appearance of boxes. Possible values are: 'max_chars','vertical',
            'horizontal','cut_chars','empty'.
        number_alignment : str
            Position of number relative to the component label. Possible values are: 'horizontal',
            'vertical'.
        add_component_indices : bool
            If true, display components with numbers.
        has_legend : bool
            If true, a legend will be added to the diagram.
        class_names : bool
            If true, appends class name of groups/components to the component blocks of diagram.
        extension : str
            Output file saved with this extension. Value fixed at 'pdf' for this class.
        type_map : str
            XDSM component type.
        _comp_indices : dict
            Maps the component names to their index (position on the matrix diagonal).
        _styles_used : set
            Styles in use (needed for legend).
        _comps : list
            List of component dictionaries.
        _loop_ends : list
            Index of last components in a process.
        _nr_comps : int
            Number of components.
        _pyxdsm_version : str
            Version of the installed pyXDSM package.
        """

        def __init__(self, name='pyxdsm', box_stacking=_DEFAULT_BOX_STACKING,
                     number_alignment=_DEFAULT_NUMBER_ALIGNMENT, legend=False, class_names=False,
                     add_component_indices=True, options={}):
            """
            Initialize.

            Parameters
            ----------
            name : str
                Name of XDSM writer.
            box_stacking : str
                Controls the appearance of boxes. Possible values are: 'max_chars','vertical',
                'horizontal','cut_chars','empty'.
            number_alignment : str
                Position of number relative to the component label. Possible values
                are: 'horizontal', 'vertical'.
            legend : bool
                If true, a legend will be added to the diagram.
            class_names : bool, optional
                If true, appends class name of groups/components to the component blocks of diagram.
                Defaults to False.
            add_component_indices : bool
                If true, display components with numbers.
            options : dict
                Keyword argument options of the XDSM class.
            """
            try:
                from pyxdsm import __version__ as pyxdsm_version
                self._pyxdsm_version = pyxdsm_version
            except ImportError:
                self._pyxdsm_version = pyxdsm_version = '1.0.0'

            if LooseVersion(pyxdsm_version) > LooseVersion('1.0.0'):
                super(XDSMWriter, self).__init__(**options)
            else:
                if options:
                    msg = 'pyXDSM {} does not take keyword arguments. Consider upgrading this ' \
                          'package. Writer options "{}" will be ignored'
                    simple_warning(msg.format(pyxdsm_version, options.keys()))
                super(XDSMWriter, self).__init__()

            self.name = name
            # Formatting options
            self.box_stacking = box_stacking
            self.class_names = class_names
            self.number_alignment = number_alignment
            self.add_component_indices = add_component_indices
            self.has_legend = legend  # If true, a legend will be added to the diagram
            # Output file saved with this extension
            self.extension = 'pdf'

            try:
                type_map_name = self.name
                if LooseVersion(pyxdsm_version) < LooseVersion('2.0.0'):
                    type_map_name += ' 1.0'
                self.type_map = _COMPONENT_TYPE_MAP[type_map_name]
            except KeyError:
                self.type_map = _COMPONENT_TYPE_MAP[_DEFAULT_WRITER]
                msg = 'Name "{}" not found in component type mapping, will default to "{}"'
                simple_warning(msg.format(self.name, _DEFAULT_WRITER))
            # Number of components
            self._nr_comps = 0
            # Maps the component names to their index (position on the matrix diagonal)
            self._comp_indices = {}
            # List of component dictionaries
            self._comps = []
            # Index of last components in a process
            self._loop_ends = []
            # Styles in use (needed for legend)
            self._styles_used = set()

        def write(self, filename=None, **kwargs):
            """
            Write the output file.

            This just wraps the XDSM version and throws out incompatible arguments.

            Parameters
            ----------
            filename : str
                Name of the file to be written.
            **kwargs : dict
                Keyword args
            """
            build = kwargs.pop('build', False)
            if LooseVersion(self._pyxdsm_version) <= LooseVersion('1.0.0'):
                kwargs = {}
            else:
                kwargs.setdefault('cleanup', True)

            for comp in self._comps:
                label = comp['label']
                # If the process steps are included in the labels
                if self.add_component_indices:
                    i = i0 = comp.pop('index', None)
                    step = comp.pop('step', None)
                    # For each closed loop increment the process index by one
                    for loop in self._loop_ends:
                        if loop < i0:
                            i += 1
                    # Step is not None for the driver and solvers, for these a different label
                    # will be made showing the starting end and step and the index of the next
                    # step.
                    if step is not None:
                        i = self._make_loop_str(first=i, last=step, start_index=_START_INDEX)
                else:
                    i = None
                label = self.finalize_label(i, label, self.number_alignment,
                                            class_name=comp['class'])

                # Convert from math mode to regular text, if it is a one liner wrapped in math mode
                if isinstance(label, string_types):
                    label = _textify(label)
                comp['label'] = label  # Now the label is finished.
                # Now really add the system with the XDSM class' method
                self.add_system(**comp)

            super(XDSMWriter, self).write(file_name=filename, build=build, **kwargs)

        def add_system(self, node_name, style, label, stack=False, faded=False, **kwargs):
            """
            Add a system.

            Parameters
            ----------
            node_name : str
                Name of the system.
            style : str
                Block formatting style, e.g. Analysis
            label : str
                Label of system in XDSM.
            stack : bool
                Defaults to False.
            faded : bool
                Defaults to False.
            **kwargs : dict
                Keyword arguments.
            """
            super(XDSMWriter, self).add_system(node_name=node_name, style=style, label=label,
                                               stack=stack, faded=faded)

        def _add_system(self, node_name, style, label, stack=False, faded=False, cls=None):
            # Adds a system dictionary to the components.
            # This dictionary can be modified by other methods.
            self._styles_used.add(style)

            if label is None:
                label = node_name
            self._comp_indices[node_name] = self._nr_comps
            sys_dct = {'node_name': node_name, 'style': style, 'label': label, 'stack': stack,
                       'faded': faded, 'index': self._nr_comps, 'class': cls}
            self._nr_comps += 1
            self._comps.append(sys_dct)

        def add_solver(self, name, label=None, **kwargs):
            """
            Add a solver.

            Parameters
            ----------
            label : str
                Label in the XDSM
            name : str
                Name of the solver
            **kwargs : dict
                Keyword args
            """
            style = self.type_map['solver']
            self._add_system(node_name=name, style=style, label=label, **kwargs)

        def add_comp(self, name, label=None, stack=False, comp_type=None, **kwargs):
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
            comp_type : str or None
                Component type, e.g. explicit, implicit or metamodel
            **kwargs : dict
                Keyword args
            """
            style = self.type_map.get(comp_type, 'Function')
            self._add_system(node_name=name, style=style, label=label, stack=stack, **kwargs)

        def add_driver(self, name, label=None, driver_type='Optimization', **kwargs):
            """
            Add an optimizer.

            Parameters
            ----------
            label : str
                Label in the XDSM
            name : str
                Name of the optimizer.
            driver_type : str
                Driver type can be "Optimizer" or "DOE".
                Defaults to "Optimizer"
            **kwargs : dict
                Keyword args
            """
            style = self.type_map.get(driver_type, 'Optimization')
            self._add_system(node_name=name, style=style, label=label, **kwargs)

        def add_workflow(self, solver=None):
            """
            Add a workflow. If "comp_names" is None, all components will be included.

            Parameters
            ----------
            solver : dict or None, optional
                List of component names.
                Defaults to None.
            """
            if hasattr(self, 'processes'):  # Not available in versions <= 1.0.0
                index_dct = self._comp_indices

                if solver is None:
                    # Add driver
                    idx = 0
                    comp_names = [c['node_name'] for c in self._comps]  # Driver process
                    step = len(self._comps) + 1
                    self._comps[idx]['step'] = step
                else:
                    solver_name = solver['abs_name']
                    comp_names = [c['abs_name'] for c in solver['comps']]
                    nr = len(comp_names)
                    idx = index_dct[solver_name]
                    self._comps[idx]['step'] = nr + idx + 1
                    comp_names = [solver_name] + comp_names
                    # Loop through all processes added so far
                    # Assumes, that processes are added in the right order, first the higher level
                    # processes
                    for proc in self.processes:
                        process_name = proc[0]
                        for i, item in enumerate(proc, start=1):
                            if solver_name == item:  # solver found in an already added process
                                # Delete items belonging to the new process from the others
                                proc[i:i + nr] = []
                                process_index = index_dct[process_name]
                                # There is a process loop inside, this adds plus one step
                                self._comps[process_index]['step'] += 1
                self._loop_ends.append(self._comp_indices[comp_names[-1]])
                # Close the loop by
                comp_names.append(comp_names[0])
                self.add_process(comp_names, arrow=_PROCESS_ARROWS)

        @staticmethod
        def format_block(names, stacking='vertical', **kwargs):
            """
            Format a block.

            Parameters
            ----------
            names : list
                Names to put into block.
            stacking : str
                Controls the appearance of boxes. Possible values are: 'max_chars','vertical',
                'horizontal','cut_chars','empty'.
            **kwargs : dict
                Alternative way to add element attributes. Use with attention, can overwrite
                some built-in python names as "class" or "id" if misused.

            Returns
            -------
            str
                The block string.
            """
            end_str = ', ...'
            max_lines = kwargs.pop('box_lines', _MAX_BOX_LINES)
            if stacking == 'vertical':
                if (max_lines is None) or (max_lines >= len(names)):
                    return names
                else:
                    names = names[0:max_lines]
                    names[-1] = names[-1] + end_str
                    return names
            elif stacking == 'horizontal':
                return ', '.join(names)
            elif stacking in ('max_chars', 'cut_chars'):
                max_chars = kwargs.pop('box_width', _DEFAULT_BOX_CHAR_LIMIT)
                if len(names) < 2:
                    return names
                else:
                    lengths = 0
                    lines = list()
                    line = ''
                    for name in names:
                        lengths += len(name)
                        if lengths <= max_chars:
                            if line:  # there are already var names on the line
                                line += ', ' + name
                            else:  # it will be the first var name on the line
                                line = name
                        else:  # make new line
                            if stacking == 'max_chars':
                                if line:
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

        @staticmethod
        def format_var_str(name, var_type, superscripts=None):
            """
            Format string displaying variable name.

            Parameters
            ----------
            name : str
                Name (label in the block) of the variable.
            var_type : str
                Variable type.
            superscripts : dict or None, optional
                A dictionary mapping variable types to their superscript notation

            Returns
            -------
            str
                Formatted var string.
            """
            if superscripts is None:
                superscripts = _SUPERSCRIPTS
            sup = superscripts[var_type]
            return '{}^{{{}}}'.format(name, sup)

        @staticmethod
        def _make_loop_str(first, last, start_index=0):
            # Start index shifts all numbers
            i = start_index
            txt = '{}, {} $ \\rightarrow $ {}'
            return txt.format(first + i, last + i, first + i + 1)

        def finalize_label(self, number, txt, alignment, class_name=None):
            """
            Add an index to the label either above or on the left side.

            Parameters
            ----------
            number : None or empty string or int
                Number value for the label.
            txt : str
                Text appended to the number string.
            alignment : str
                Indicates alignment of label. Either 'horizontal' or 'vertical'.
            class_name : str or None, optional
                Class name.
                Defaults to None.

            Returns
            -------
            str or list(str)
                Label to be used for this item. List, if it is multiline.
            """
            if isinstance(txt, string_types):
                txt = [txt]  # Make iterable, it will be converted back if there is only 1 line.

            if self.class_names and (class_name is not None):
                cls_name = r'\textit{{{}}}'.format(class_name)  # Makes it italic
                txt.append(cls_name)  # Class name goes to a new line
            if number:  # If number is None or empty string, it won't be inserted
                number_str = '{}: '.format(number)
                if alignment == 'horizontal':
                    txt[0] = number_str + txt[0]  # Number added to first line
                elif alignment == 'vertical':
                    txt.insert(0, number_str)  # Number added to new line
                else:
                    msg = '"{}" is an invalid option for number_alignment, it will be ignored.'
                    simple_warning(msg.format(alignment))
            return _multiline_block(*txt)

        def _make_legend(self, title="Legend"):
            """
            Add a legend row to the matrix. The labels of this row show the used component types.

            Parameters
            ----------
            title : str, optional
                Defaults to "Legend".

            Returns
            -------
                str
            """
            node_str = r'\node [{style}] ({name}) {{{label}}};'
            styles = sorted(self._styles_used)  # Alphabetical sort
            for i, style in enumerate(styles):
                super(XDSMWriter, self).add_system(node_name="style{}".format(i), style=style,
                                                   label=style)
            style_strs = [node_str.format(name="style{}".format(i), style=style, label=style)
                          for i, style in enumerate(styles)]
            title_str = r'\node (legend_title) {{\LARGE \textbf{{{title}}}}};\\'
            return title_str.format(title=title) + '  &\n'.join(style_strs) + r'\\'

        def _build_node_grid(self):
            """
            Optionally appends the legend to the node grid.

            Returns
            -------
            str
                A grid of the nodes.
            """
            node_grid = super(XDSMWriter, self)._build_node_grid()
            if self.has_legend:
                node_grid += self._make_legend()
            return node_grid


def write_xdsm(data_source, filename, model_path=None, recurse=True,
               include_external_outputs=True, out_format='tex',
               include_solver=False, subs=_CHAR_SUBS, show_browser=True,
               add_process_conns=True, show_parallel=True, output_side=_DEFAULT_OUTPUT_SIDE,
               legend=False, class_names=True, equations=False,
               writer_options={}, **kwargs):
    """
    Write XDSM diagram of an optimization problem.

    With the 'tex' or 'pdf' output format it uses the pyXDSM package, with 'html'
    output format it uses XDSMjs.

    If a component (or group) name is not unique in the diagram, the systems absolute path is
    used as a label. If the component (or group) name is unique, the relative name of the
    system is the label.

    In the diagram the connections are marked with the source name.

    Writer specific settings and default:

    pyXDSM

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

    * If "embed_data" is true, a single standalone HTML file will be generated, which includes
      the data of the XDSM diagram.
    * variable names with exactly one underscore have a subscript.
      Example: "x_12" will be :math:`x_12`
    * If "embeddable" is True, gives a single HTML file that doesn't have the <html>, <DOCTYPE>,
      <body> and <head> tags. If False, gives a single, standalone HTML file for viewing.

    Parameters
    ----------
    data_source : Problem or str
        The Problem or case recorder database containing the model or model data.
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
    add_process_conns : bool
        Add process connections (thin black lines)
        Defaults to True.
    show_parallel : bool
        Show parallel components with stacked blocks.
        Defaults to True.
    output_side : str or dict(str, str)
        Left or right, or a dictionary with component types as keys. Component type key can
        be 'optimization', 'doe' or 'default'.
        Defaults to "left".
    legend : bool, optional
        If true, it adds a legend to the diagram.
        Defaults to False.
    class_names : bool, optional
        If true, appends class name of the groups/components to the component blocks of the diagram.
        Defaults to False.
    equations : bool, optional
        If true, for ExecComps their equations are shown in the diagram
        Defaults to False.
    writer_options : dict, optional
        Options passed to the writer class at initialization.
    **kwargs : dict
        Keyword arguments

    Returns
    -------
       XDSM or AbstractXDSMWriter
    """
    build_pdf = False
    writer = kwargs.pop('writer', None)

    if out_format in ('tex', 'pdf') and (writer is None):
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

    viewer_data = _get_viewer_data(data_source)

    driver = viewer_data.get('driver', None)
    if driver:
        driver_name = driver.get('name', None)
        driver_type = driver.get('type', 'optimization')
    else:
        driver_name = None
        driver_type = 'optimization'

    design_vars = viewer_data.get('design_vars', None)
    responses = viewer_data.get('responses', None)

    if model_path is not None:
        if isinstance(data_source, Problem):
            _model = data_source.model._get_subsystem(model_path)
            if _model is None:
                msg = 'Model path "{}" does not exist in problem "{}".'
                raise ValueError(msg.format(model_path, data_source))
            design_vars = _model.get_design_vars()
            responses = _model.get_responses()
        else:
            msg = 'Model path is not supported when data source is "{}".'
            raise ValueError(msg.format(type(data_source)))

    if design_vars is None:
        simple_warning('The XDSM diagram will show only the model hierarchy, '
                       'as the driver, design variables and responses are not '
                       'available.')

    filename = filename.replace('\\', '/')  # Needed for LaTeX

    # If the "writer" argument not provided, the output format is used to choose the writer
    if writer is None:
        try:
            writer = _OUT_FORMATS[out_format]
        except KeyError:
            msg = 'Invalid output format "{}", choose from: {}'
            raise ValueError(msg.format(out_format, _OUT_FORMATS.keys()))
        writer_name = writer.lower()  # making it case insensitive
        if isinstance(subs, dict):
            subs = subs[writer_name]  # Getting the character substitutes of the chosen writer
    else:
        if isinstance(writer, BaseXDSMWriter):
            try:
                subs = subs[writer.name]
            except KeyError:
                msg = 'Writer name "{0}" not found, there will be no character ' \
                      'substitutes used. Add "{0}" to your settings, or provide a tuple for' \
                      'character substitutes.'
                simple_warning(msg.format(writer.name, subs))
                subs = ()
        else:
            msg = 'Custom XDSM writer should be an instance of BaseXDSMWriter, now it is a "{}".'
            raise TypeError(msg.format(type(writer)))

    return _write_xdsm(filename, viewer_data=viewer_data,
                       driver=driver_name, include_solver=include_solver, model_path=model_path,
                       design_vars=design_vars, responses=responses, writer=writer,
                       recurse=recurse, subs=subs,
                       include_external_outputs=include_external_outputs, show_browser=show_browser,
                       add_process_conns=add_process_conns, build_pdf=build_pdf,
                       show_parallel=show_parallel, driver_type=driver_type,
                       output_side=output_side, legend=legend, class_names=class_names,
                       writer_options=writer_options, equations=equations, **kwargs)


def _write_xdsm(filename, viewer_data, driver=None, include_solver=False, cleanup=True,
                design_vars=None, responses=None, residuals=None, model_path=None, recurse=True,
                include_external_outputs=True, subs=_CHAR_SUBS, writer='pyXDSM', show_browser=False,
                add_process_conns=True, show_parallel=True, quiet=False, build_pdf=False,
                output_side=_DEFAULT_OUTPUT_SIDE, driver_type='optimization', legend=False,
                class_names=False, equations=False, writer_options={}, **kwargs):
    """
    XDSM writer. Components are extracted from the connections of the problem.

    Parameters
    ----------
    filename : str
        Filename (absolute path without extension)
    connections : list[(str, str)]
        Connections list
    driver : str or None, optional
        Driver name
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
    writer: str or BaseXDSMWriter, optional
        Writer is either a string ("pyXDSM" or "XDSMjs") or a custom writer.
        Defaults to "pyXDSM".
    show_browser : bool, optional
        If True, pop up a browser to view the generated html file.
        Defaults to False.
    add_process_conns : bool
        Add process connections (thin black lines)
        Defaults to True.
    show_parallel : bool
        Show parallel components with stacked blocks.
        Defaults to True.
    quiet : bool
        Set to True to suppress output from pdflatex
    build_pdf : bool, optional
        If True and a .tex file is generated, create a .pdf file from the .tex.
        Defaults to False.
    output_side : str or dict(str, str), optional
        Left or right, or a dictionary with component types as keys. Component type key can
        be 'optimization', 'doe' or 'default'.
        Defaults to "left".
    driver_type : str, optional
        Optimization or DOE.
        Defaults to "optimization".
    legend : bool, optional
        If true, it adds a legend to the diagram.
        Defaults to False.
    class_names : bool, optional
        If true, appends class name of the groups/components to the component blocks of the diagram.
        Defaults to False.
    equations : bool, optional
        If true, for ExecComps their equations are shown in the diagram
        Defaults to False.
    writer_options : dict, optional
        Options passed to the writer class at initialization.
    **kwargs : dict
        Keyword arguments, includes writer specific options.

    Returns
    -------
        BaseXDSMWriter
    """
    # TODO implement residuals
    # Box appearance
    box_stacking = kwargs.pop('box_stacking', _DEFAULT_BOX_STACKING)
    box_width = kwargs.pop('box_width', _DEFAULT_BOX_WIDTH)
    box_lines = kwargs.pop('box_lines', _MAX_BOX_LINES)
    # In XDSMjs components are numbered by default, so only add for pyXDSM as an option
    add_component_indices = kwargs.pop('numbered_comps', True)
    # Number alignment can be "horizontal" or "vertical"
    number_alignment = kwargs.pop('number_alignment', _DEFAULT_NUMBER_ALIGNMENT)

    error_msg = ('Undefined XDSM writer "{}". '
                 'Provide  a valid name or a BaseXDSMWriter instance.')
    if isinstance(writer, string_types):  # Standard writers (XDSMjs or pyXDSM)
        if writer.lower() == 'pyxdsm':  # pyXDSM
            x = XDSMWriter(box_stacking=box_stacking,
                           number_alignment=number_alignment,
                           add_component_indices=add_component_indices,
                           legend=legend,
                           class_names=class_names,
                           options=writer_options)
        elif writer.lower() == 'xdsmjs':  # XDSMjs
            x = XDSMjsWriter(options=writer_options)
        else:
            raise ValueError(error_msg.format(writer))
    elif isinstance(writer, BaseXDSMWriter):  # Custom writer
        x = writer
    else:
        raise TypeError(error_msg.format(writer))

    def format_block(names, **kwargs):
        # Sets the width, number of lines and other string formatting for a block.
        return x.format_block(names=names, box_width=box_width, box_lines=box_lines,
                              box_stacking=box_stacking, **kwargs)

    def get_output_side(component_name):
        if isinstance(output_side, string_types):
            return output_side
        elif isinstance(output_side, dict):
            # Gets the specified key, or the default in the dictionary, or the global default
            # if both of them are missing from the dictionary.
            side = output_side.get(component_name, output_side.get('default', _DEFAULT_OUTPUT_SIDE))
            return side
        else:
            msg = 'Output side argument should be string or dictionary, instead it is a {}.'
            raise ValueError(msg.format(type(output_side)))

    connections = viewer_data['connections_list']
    tree = viewer_data['tree']

    # Get the top level system to be transcripted to XDSM
    comps = _get_comps(tree, model_path=model_path, recurse=recurse, include_solver=include_solver)
    if include_solver:
        # Add the top level solver
        top_level_solver = dict(tree)
        top_level_solver.update({'comps': list(comps), 'abs_name': 'root@solver', 'index': 0,
                                 'type': 'solver'})
        comps.insert(0, top_level_solver)  # Add top level solver
    comps_dct = {comp['abs_name']: comp for comp in comps if comp['type'] != 'solver'}

    solvers = []  # Solver labels

    conns1, external_inputs1, external_outputs1 = _prune_connections(connections,
                                                                     model_path=model_path)

    conns2 = _process_connections(conns1, recurse=recurse, subs=subs)
    external_inputs2 = _process_connections(external_inputs1, recurse=recurse, subs=subs)
    external_outputs2 = _process_connections(external_outputs1, recurse=recurse, subs=subs)

    def add_solver(solver_dct):
        # Adds a solver.
        # Uses some vars from the outer scope.
        # Returns True, if it is a non-default linear or nonlinear solver
        comp_names = [_format_name(c['abs_name']) for c in solver_dct['comps']]
        solver_label = _format_solver_str(solver_dct, stacking=box_stacking)

        if isinstance(solver_label, string_types):
            solver_label = _replace_chars(solver_label, subs)
        else:
            solver_label = [_replace_chars(i, subs) for i in solver_label]
        solver_name = _format_name(solver_dct['abs_name'])

        if solver_label:  # At least one non-default solver (default solvers are ignored)
            # If there is a driver, the start index is increased by one.
            solvers.append(solver_label)
            x.add_solver(name=solver_name, label=solver_label)

            # Add the connections
            for src, dct in iteritems(conns2):
                for tgt, conn_vars in iteritems(dct):
                    formatted_conns = format_block(conn_vars)
                    if (src in comp_names) and (tgt in comp_names):
                        formatted_targets = format_block([x.format_var_str(c, 'target')
                                                          for c in conn_vars])
                        # From solver to components (targets)
                        x.connect(solver_name, tgt, formatted_targets)
                        # From components to solver
                        x.connect(src, solver_name, formatted_conns)
            return True
        else:
            return False

    if driver is not None:
        driver_label = driver
        driver_name = _format_name(driver)
        x.add_driver(name=driver_name, label=driver_label, driver_type=driver_type.lower())

        design_vars2 = _collect_connections(design_vars, recurse=recurse, model_path=model_path)
        responses2 = _collect_connections(responses, recurse=recurse, model_path=model_path)

        # Design variables
        for comp, conn_vars in iteritems(design_vars2):
            # Format var names
            conn_vars = [_replace_chars(var, subs) for var in conn_vars]
            # Optimal var names
            opt_con_vars = [x.format_var_str(var, 'optimal') for var in conn_vars]
            # Initial var names
            init_con_vars = [x.format_var_str(var, 'initial') for var in conn_vars]
            # Connection from optimizer
            x.connect(driver_name, comp, format_block(conn_vars))
            # Optimal design variables
            x.add_output(comp, format_block(opt_con_vars), side=get_output_side('default'))
            x.add_output(driver_name, format_block(opt_con_vars), side=get_output_side(driver_type))
            # Initial design variables
            x.add_input(driver_name, format_block(init_con_vars))

        # Responses
        for comp, conn_vars in iteritems(responses2):
            # Optimal var names
            conn_vars = [_replace_chars(var, subs) for var in conn_vars]
            opt_con_vars = [x.format_var_str(var, 'optimal') for var in conn_vars]
            # Connection to optimizer
            x.connect(comp, driver_name, conn_vars)
            # Optimal output
            x.add_output(comp, format_block(opt_con_vars), side=get_output_side('default'))

    # Add components
    solver_dcts = []
    if equations:
        try:
            from pytexit import py2tex
        except ImportError:
            equations = False
            msg = 'The LaTeX equation formatting requires the pytexit package.' \
                  'The "equations" options was turned off.' \
                  'To enable this option install the package with "pip install pytexit".'
            simple_warning(msg)

    for comp in comps:  # Driver is 1, so starting from 2
        # The second condition is for backwards compatibility with older data.
        if equations and comp.get('expressions', None) is not None:
            # One of the $ signs has to be removed to correctly parse it
            if isinstance(x, XDSMWriter):
                def parse(expr):
                    for (ch, rep) in (('$$', '$'), (r'[', '_'), (r']', '')):
                        expr = expr.replace(ch, rep)
                    # One of the $ signs has to be removed to correctly parse it
                    return py2tex(expr).replace('$$', '$')

                expression = comp['expressions']
                try:
                    label = ', '.join(map(parse, expression))
                except TypeError:
                    label = _replace_chars(comp['name'], substitutes=subs)
                    simple_warning('Could not parse "{}"'.format(expression))
            else:
                msg = 'The "equations" option is available only with pyXDSM. Set the output ' \
                      'format to "tex" or "pdf" to enable this option.'
                simple_warning(msg)
                label = _replace_chars(comp['name'], substitutes=subs)
        else:
            label = _replace_chars(comp['name'], substitutes=subs)
        stack = comp['is_parallel'] and show_parallel
        if include_solver and comp['type'] == 'solver':  # solver
            if add_solver(comp):  # Return value is true, if solver is not the default
                # If not default solver, add to the solver dictionary
                solver_dcts.append(comp)
        else:  # component or group
            cls_name = comp.get('class', None) if class_names else None
            comp_type = comp['component_type']
            if comp.get('subsystem_type', None) == 'group':
                comp_type = 'group'
            x.add_comp(name=comp['abs_name'], label=label, stack=stack,
                       comp_type=comp_type, cls=cls_name)

    # Add process connections
    if add_process_conns:
        if driver is not None:
            x.add_workflow()  # Driver workflow
        for s in solver_dcts:
            x.add_workflow(s)  # Solver workflows

    # Add the connections
    for src, dct in iteritems(conns2):
        for tgt, conn_vars in iteritems(dct):
            if src and tgt:
                stack = show_parallel and \
                    (comps_dct[src]['is_parallel'] or comps_dct[tgt]['is_parallel'])
                x.connect(src, tgt, label=format_block(conn_vars), stack=stack)
            else:  # Source or target missing
                msg = 'Connection "{conn}" from "{src}" to "{tgt}" ignored.'
                simple_warning(msg.format(src=src, tgt=tgt, conn=conn_vars))

    # Add the externally sourced inputs
    for src, tgts in iteritems(external_inputs2):
        for tgt, conn_vars in iteritems(tgts):
            formatted_conn_vars = [_replace_chars(o, substitutes=subs) for o in conn_vars]
            if tgt:
                stack = comps_dct[tgt]['is_parallel'] and show_parallel
                x.add_input(tgt, format_block(formatted_conn_vars), stack=stack)
            else:  # Target missing
                msg = 'External input to "{tgt}" ignored.'
                simple_warning(msg.format(tgt=tgt, conn=conn_vars))

    # Add the externally connected outputs
    if include_external_outputs:
        for src, tgts in iteritems(external_outputs2):
            output_vars = set()
            for tgt, conn_vars in iteritems(tgts):
                output_vars |= set(conn_vars)
            formatted_outputs = [_replace_chars(o, subs) for o in output_vars]
            if src:
                stack = comps_dct[src]['is_parallel'] and show_parallel
                x.add_output(src, formatted_outputs, side='right', stack=stack)
            else:  # Source or target missing
                msg = 'External output "{conn}" from "{src}" ignored.'
                simple_warning(msg.format(src=src, conn=output_vars))

    x.write(filename, cleanup=cleanup, quiet=quiet, build=build_pdf, **kwargs)

    if show_browser and (build_pdf or x.name == 'xdsmjs'):
        # path will be specified based on the "out_format", if all required inputs where
        # provided for showing the results.
        ext = x.extension
        if not isinstance(ext, string_types):
            err_msg = '"{}" is an invalid extension.'
            raise ValueError(err_msg.format(writer))
        path = '.'.join([filename, ext])
        webview(path)  # Can open also PDFs

    return x  # Returns the writer instance


def _get_cls_name(obj):
    return obj.__class__.__name__


def _residual_str(name):
    """Make a residual symbol."""
    return '\\mathcal{R}(%s)' % name


def _process_connections(conns, recurse=True, subs=None):

    def convert(x):
        return _convert_name(x, recurse=recurse, subs=subs)

    conns_new = [
        {k: convert(v) for k, v in iteritems(conn) if k in ('src', 'tgt')} for conn in conns
    ]
    return _accumulate_connections(conns_new)


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


def _collect_connections(variables, recurse, model_path=None):
    conv_vars = [_convert_name(v, recurse) for v in variables]
    connections = dict()
    for conv_var in conv_vars:
        path = _make_rel_path(conv_var['path'], model_path=model_path)
        connections.setdefault(path, []).append(conv_var['var'])
    return connections


def _get_path(name, sep='.'):
    # Returns path until the last separator in the name
    return name.rsplit(sep, 1)[0]


def _make_rel_path(full_path, model_path, sep='.'):
    # Path will be cut from this character. Length of model path + separator after it.
    # If path does not contain the model path, the full path will be returned.
    if model_path is not None:
        path = model_path + sep  # Add separator character
        first_char = len(path)
        if full_path.startswith(path):
            return full_path[first_char:]
    return full_path  # No model path, so return the original


def _convert_name(name, recurse=True, subs=None):
    """
    From an absolute path returns the variable name and its owner component in a dict.

    Names are also formatted.

    Parameters
    ----------
    name : str or list(str)
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
        return map(convert, name)
    else:  # string
        return convert(name)


def _format_name(name):
    # Replaces illegal characters in names for pyXDSM component and connection names
    # This does not effect the labels, only reference names TikZ
    if isinstance(name, string_types):  # from an SQL reader the name will be in unicode
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
    model_path : str or None, optional
        The path in model to the system to be transcribed to XDSM.
        Defaults to None.
    sep : str, optional
        Separator character.
        Defaults to '.'.

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
    external_inputs = []
    external_outputs = []

    if model_path is None:
        return conns, external_inputs, external_outputs
    else:
        internal_conns = []

        path = model_path + sep  # Add separator character
        for conn in conns:
            src = conn['src']
            src_path = _make_rel_path(src, model_path=model_path)
            tgt = conn['tgt']
            tgt_path = _make_rel_path(tgt, model_path=model_path)
            conn_dct = {'src': src_path, 'tgt': tgt_path}

            if src.startswith(path):
                if tgt.startswith(path):
                    internal_conns.append(conn_dct)  # Internal connections
                else:
                    external_outputs.append(conn_dct)  # Externally connected output
            elif tgt.startswith(path):
                external_inputs.append(conn_dct)  # Externally connected input
        return internal_conns, external_inputs, external_outputs


def _get_comps(tree, model_path=None, recurse=True, include_solver=False):
    """
    Return the components in the tree, optionally only those within the given model_path.

    It also includes the solvers of the subsystems, if "include_solver" is True and not the
    default solvers are assigned to the subsystems.

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
    include_solver : bool, optional
        Defaults to False.

    Returns
    -------
    components : list
        A list of the components within the model_path in tree.  If recurse is False, this
        list may contain groups. If "include_solver" is True, it may include solvers.

    """
    # Components are ordered in the tree, so they can be collected by walking through the tree.
    components = list()  # Components will be collected to this list
    comp_names = set()  # To check if names are unique
    sep = '.'

    def get_children(tree_branch, path=''):
        local_comps = []

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
                local_comps.append(ch)
            else:  # Group
                # Add a solver to the component list, if this group has a linear or nonlinear
                # solver.
                has_solver = False
                if include_solver:
                    solver_names = []
                    solver_dct = {}
                    for solver_typ, default_solver in iteritems(_DEFAULT_SOLVER_NAMES):
                        k = '{}_solver'.format(solver_typ)
                        if ch[k] != default_solver:
                            solver_names.append(ch[k])
                            has_solver = True
                        solver_dct[k] = ch[k]
                    if has_solver:
                        i_solver = len(components)
                        name_str = ch['abs_name'] + '@solver'
                        # "comps" will be filled later
                        solver = {'abs_name': _format_name(name_str), 'rel_name': solver_names,
                                  'type': 'solver', 'name': name_str, 'is_parallel': False,
                                  'component_type': 'MDA', 'index': i_solver}
                        solver.update(solver_dct)
                        components.append(solver)
                        comp_names.add(name_str)
                # Add the group or components in the group
                if recurse:  # it is not a component and recurse is True
                    if path:
                        new_path = sep.join([path, ch['name']])
                    else:
                        new_path = ch['name']
                    local_comps = get_children(ch, new_path)
                else:
                    components.append(ch)
                    comp_names.add(ch['rel_name'])
                    local_comps = [ch]
                # Add to the solver, which components are in its loop.
                if include_solver and has_solver:
                    components[i_solver]['comps'] = local_comps
                    local_comps = []
        return list(local_comps)

    top_level_tree = tree
    if model_path is not None:
        path_list = model_path.split(sep)
        while path_list:
            next_path = path_list.pop(0)
            children = [child for child in top_level_tree['children']]
            top_level_tree = [c for c in children if c['name'] == next_path][0]

    get_children(top_level_tree)
    return components


def _replace_chars(name, substitutes):
    """
    Replace characters in `name` with the substitute characters.

    If some of the characters are both to be replaced or other characters are replaced with them
    (e.g.: ? -> !, ! ->#), than it is not safe to give a dictionary as the `substitutes`
    (because it is unordered).

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
        for k, v in substitutes:
            name = name.replace(k, v)
    return name


def _format_solver_str(dct, stacking='horizontal', solver_types=('nonlinear', 'linear')):
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
    for solver_type in solver_types:  # loop through all solver types
        solver_name = dct['{}_solver'.format(solver_type)]
        if solver_name != _DEFAULT_SOLVER_NAMES[solver_type]:  # Not default solver found
            solvers.append(solver_name)
    if stacking == 'vertical':
        # Make multiline comp if not numbered
        return _multiline_block(*solvers)
    elif stacking in ('horizontal', 'max_chars', 'cut_chars'):
        return ' '.join(solvers)
    else:
        msg = ('Invalid stacking "{}". Choose from: "vertical", "horizontal", "max_chars", '
               '"cut_chars"')
        raise ValueError(msg.format(stacking))


def _multiline_block(*texts, **kwargs):
    """
    Make a string for a multiline block.

    A string is returned, if there would be only 1 line.

    texts : iterable(str)
        Text strings, each will go to new line
    **kwargs : dict
        Unused keywords are ignored.
        "end_char" is the separator at the end of line. Defaults to '' (no separator).

    Returns
    -------
       list(str) or str
    """
    end_char = kwargs.pop('end_char', '')
    out_txts = [_textify(t + end_char) for t in texts]
    if len(out_txts) == 1:
        out_txts = out_txts[0]
    return out_txts


def _textify(name):
    # Uses the LaTeX \text{} command to insert plain text in math mode
    return r'\text{{{}}}'.format(name)
