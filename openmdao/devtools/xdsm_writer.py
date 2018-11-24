"""
XDSM writer using the pyXDSM package.
The package is available at https://github.com/mdolab/pyXDSM.
"""
from __future__ import print_function

from openmdao.devtools.problem_viewer.problem_viewer import _get_viewer_data

try:
    from pyxdsm.XDSM import XDSM
except ImportError:
    msg = ('The pyxdsm package should be installed. You can download the package '
           'from https://github.com/mdolab/pyXDSM')
    raise ImportError(msg)

from six import iteritems


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

        param str label: Label in the XDSM
        :param str name: Name of
        :param dict kwargs: Keyword args
        :return: None
        """
        self.add_system(name, 'MDA', '\\text{%s}' % label, **kwargs)

    def add_comp(self, name, label=None, **kwargs):
        """
        Add a component.

        param str label: Label in the XDSM
        :param str name: Name of
        :param dict kwargs: Keyword args
        :return: None
        """
        if label is None:
            label = name
        self.add_system(name, 'Analysis', label, **kwargs)

    def add_func(self, name, **kwargs):
        """
        Add a function

        :param str name: Name of
        :param dict kwargs: Keyword args
        :return: None
        """
        self.add_system(name, 'Function', name, **kwargs)

    def add_optimizer(self, label, name='opt', **kwargs):
        """
        Add an optimizer.

        :param str label: Label in the XDSM
        :param str name: Name of
        :param dict kwargs: Keyword args
        :return: None
        """
        self.add_system(name, 'Optimization', '\\text{%s}' % label, **kwargs)


def write_xdsm(problem, filename, include_solver=False):
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
    return _write_xdsm(filename, connections=viewer_data['connections_list'],
                       optimizer=driver_name, solver=solver_name, design_vars=design_vars,
                       responses=responses)


def _write_xdsm(filename, connections, optimizer=None, solver=None, cleanup=True,
                design_vars=None, responses=None, residuals=None,
                subs=(('_', '~'), (')', ' '), ('(', '_')), **kwargs):
    """
    XDSM writer. Components are extracted from the connections.

    .. figure:: _images/xdsm.*
       :width: 90 %
       :align: center

       Extended Design Structure Matrix

    :param str filename: Filename (absolute path without extension)
    :param list[(str, str)] connections: Connections list
    :param str | None optimizer: optimizer name
    :param str | None solver: Solver name
    :param bool cleanup: Clean-up temporary files after making the diagram
    :param OrderedDict | None design_vars: Design variables
    :param OrderedDict | None responses: Responses
    :param tuple subs: Character pairs to be substituted
    :param kwargs:
    :return: XDSM
    """
    # TODO implement residuals

    comps = []

    def residual_str(name):
        """Makes a residual symbol."""
        return '\\mathcal{R}(%s)' % name

    def opt_var_str(name):
        """Puts an asterisk superscript on a string."""
        return '{}^*'.format(name)

    def convert_name(name):
        """
        From an absolute path returns the variable name and its owner component in a dict.

        :param str name: Connection absolute path and name
        :return: dict(str, str)
        """

        def convert(name):
            name = name.split('.')
            comp = name[-2]
            var = name[-1]
            var = _replace_chars(var, substitutes=subs)
            new = {'comp': comp, 'var': var}

            if comp not in comps:
                comps.append(comp)
            return new

        if isinstance(name, list):
            return [convert(n) for n in name]
        else:  # string
            return convert(name)

    def process_connections(conns):
        conns_new = [{k: convert_name(v) for k, v in iteritems(conn)} for conn in conns]
        return conns_new

    def accumulate_connections(conns):
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

    x = XDSMWriter()

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

    :param name:
    :param dict | tuple substitutes: Character pairs with old and substitute characters
    :return: str
    """
    for (k, v) in substitutes:
        name = name.replace(k, v)
    return name