"""
XDSM writer using the pyXDSM package.
The package is available at https://github.com/mdolab/pyXDSM.
"""
from __future__ import print_function

import json

from openmdao.devtools.problem_viewer.problem_viewer import _get_viewer_data

try:
    from pyxdsm.XDSM import XDSM
except ImportError:
    msg = ('The pyxdsm package should be installed. You can download the package '
           'from https://github.com/mdolab/pyXDSM')
    raise RuntimeError(msg)

from six import iteritems


_CHAR_SUBS = (('_', '~'), (')', ' '), ('(', '_'))


class AbstractXDSMWriter(object):

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


def write_xdsm(problem, filename, out_format='tex', include_solver=False, subs=_CHAR_SUBS):
    """
    Writes XDSM diagram of an optimization problem.

    Parameters
    ----------
    problem : Problem
       Problem
    filename : str
       Name of the output files (do not provide file extension)
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

    def get_cls_name(obj):
        return obj.__class__.__name__

    viewer_data = _get_viewer_data(problem)
    driver = problem.driver
    model = problem.model
    if driver:
        driver_name = get_cls_name(driver)
    else:
        driver_name = None
    if include_solver:
        solver_name = get_cls_name(model.nonlinear_solver)
    else:
        solver_name = None
    design_vars = model.get_design_vars()
    responses = model.get_responses()

    filename = filename.replace('\\', '/')  # Needed for LaTeX
    return _write_xdsm(filename, viewer_data=viewer_data,
                       optimizer=driver_name, solver=solver_name, design_vars=design_vars,
                       responses=responses, out_format=out_format, subs=subs)


def _write_xdsm(filename, viewer_data, optimizer=None, solver=None, cleanup=True,
                design_vars=None, responses=None, residuals=None,
                subs=_CHAR_SUBS, out_format='tex', **kwargs):
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

    def get_comps(tree):
        # Components are ordered in the tree, so they can be collected by walking through the tree.
        components = list()

        def get_children(tree_branch):
            for ch in tree_branch['children']:
                if ch['subsystem_type'] == 'component':
                    components.append(ch['name'])
                else:
                    get_children(ch)

        get_children(tree)
        return components

    def residual_str(name):
        """Makes a residual symbol."""
        return '\\mathcal{R}(%s)' % name

    def opt_var_str(name):
        """Puts an asterisk superscript on a string."""
        return '{}^*'.format(name)

    def convert_name(name):
        """
        From an absolute path returns the variable name and its owner component in a dict.

        Parameters
        ----------
        name : str
           Connection absolute path and name

        Returns
        -------
           dict(str, str)
        """

        def convert(name):
            path, var = name.rsplit('.', 1)
            comp = path.rsplit('.', 1)[-1]
            var = _replace_chars(var, substitutes=subs)
            return {'comp': comp, 'var': var}

        if isinstance(name, list):  # If a source has multiple targets
            return [convert(n) for n in name]
        else:  # string
            return convert(name)

    def process_connections(conns):
        conns_new = [{k: convert_name(v) for k, v in iteritems(conn)} for conn in conns]
        return conns_new

    def accumulate_connections(conns):
        # Makes a dictionary with source and target components and with the connection sources
        conns_new = dict()
        for conn in conns:  # list
            src_comp = conn['src']['comp']
            tgt_comp = conn['tgt']['comp']
            var = conn['src']['var']
            conns_new.setdefault(src_comp, {})
            conns_new[src_comp].setdefault(tgt_comp, []).append(var)

        return conns_new

    conns2 = process_connections(connections)
    conns3 = accumulate_connections(conns2)

    if out_format == 'tex':
        x = XDSMWriter()
    elif out_format == 'json':
        x = XDSMjsWriter()

    if optimizer is not None:
        x.add_optimizer(optimizer)

    if solver is not None:
        x.add_solver(solver)

    def collect_connections(variables):
        conv_vars = [convert_name(v) for v in variables]
        connections = dict()
        for conv_var in conv_vars:
            connections.setdefault(conv_var['comp'], []).append(conv_var['var'])
        return connections

    design_vars2 = collect_connections(design_vars)
    responses2 = collect_connections(responses)

    # Feed forward
    for comp, conn_vars in iteritems(design_vars2):
        x.connect('opt', comp, conn_vars)
        opt_con_vars = [opt_var_str(var) for var in conn_vars]
        x.add_output(comp, ', '.join(opt_con_vars), side='left')

    # Feedback
    for comp, conn_vars in iteritems(responses2):
        x.connect(comp, 'opt', conn_vars)
        opt_con_vars = [opt_var_str(var) for var in conn_vars]
        x.add_output(comp, ', '.join(opt_con_vars), side='left')

    comps = get_comps(tree)

    # Add components
    for comp in comps:
        x.add_comp(name=comp, label=_replace_chars(comp, substitutes=subs))

    for src, dct in iteritems(conns3):
        for tgt, conn_vars in iteritems(dct):
            x.connect(src, tgt, ', '.join(conn_vars))

    x.write(filename, cleanup=cleanup, **kwargs)
    return x


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