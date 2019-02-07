"""
Automatic differentiation functions using the autograd package.
"""

import sys
from six import itervalues, iteritems, PY2
import inspect
from inspect import getsourcelines, getsource, getmodule
from collections import OrderedDict, defaultdict
from os import remove
from importlib import import_module
import types
import time
import numpy as np
import ast
import astunparse
from itertools import chain

from openmdao.utils.general_utils import str2valid_python_name, unique_name
import openmdao.utils.mod_wrapper as mod_wrapper
from openmdao.vectors.default_vector import Vector, DefaultVector
from openmdao.vectors.petsc_vector import PETScVector
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.devtools.ast_tools import transform_ast_names, dependency_analysis, \
    StringSubscriptVisitor

try:
    import autograd.numpy as agnp
    from autograd import make_jvp, make_vjp, jacobian
    from autograd.differential_operators import make_jvp_reversemode
    from autograd.builtins import tuple as agtuple, list as aglist, dict as agdict, DictBox, \
        DictVSpace, container_take
except ImportError:
    agnp = make_jvp = make_vjp = jacobian = make_jvp_reversemode = agtuple = aglist = agdict = None
    DictBox = DictVSpace = container_take = None
else:
    class _VectorBox(DictBox):
        def get_slice(self, slc):
            return container_take(self._value._data, slc)

    _VectorBox.register(Vector)
    _VectorBox.register(DefaultVector)
    _VectorBox.register(PETScVector)


def _get_arg_replacement_map(comp):
    # Return a mapping of names or subscript expressions to be replaced with simple var names.
    inputs = comp._inputs
    outputs = comp._outputs

    if isinstance(comp, ExplicitComponent):
        compute_method = comp.compute
    else:
        compute_method = comp.apply_nonlinear

    if PY2:
        params = list(inspect.getargspec(compute_method).args)
    else:
        params = list(inspect.signature(compute_method).parameters)

    # lists of local names of inputs and outputs (and maybe resids) to code generate
    pnames = []
    onames = []
    rnames = []

    # mapping to rename variables within the compute method
    to_replace = {}

    # gather transformed input and output names
    for pname in inputs:
        new_name = str2valid_python_name(pname)
        pnames.append(new_name)
        to_replace["%s['%s']" % (params[0], pname)] = new_name
        to_replace['%s["%s"]' % (params[0], pname)] = new_name

    for oname in outputs:
        new_name = str2valid_python_name(oname)
        onames.append(new_name)
        to_replace["%s['%s']" % (params[1], oname)] = new_name
        to_replace['%s["%s"]' % (params[1], oname)] = new_name

    if not isinstance(comp, ExplicitComponent):
        for rname in comp._vectors['residual']['linear']:
            new_name = 'resid__' + str2valid_python_name(rname)
            rnames.append(new_name)
            to_replace["%s['%s']" % (params[2], rname)] = new_name
            to_replace['%s["%s"]' % (params[2], rname)] = new_name

    return to_replace, pnames, onames, rnames


def translate_compute_source_autograd(comp, mode):
    """
    Convert a compute or apply_nonlinear method into a function with individual args for each var.

    Converts  def compute(self, inputs, outputs) to def kompute(self, a, b, c)
    and translated function returns all output values as a tuple.

    Parameters
    ----------
    comp : Component
        The component being AD'd.
    mode : str
        Derivative direction ('fwd', 'rev').

    Returns
    -------
    str
        Converted source code.
    list of str
        Input names.
    list of str
        Output names.
    list of str
        Residual names.
    dict
        Translation map for self.options if revert is True, else empty dict.
    """
    explicit = isinstance(comp, ExplicitComponent)

    if explicit:
        compute_method = comp.compute
    else:
        compute_method = comp.apply_nonlinear

    # get source code of original compute() method
    # ignore blank lines, useful for detecting indentation later
    srclines = [line for line in getsourcelines(compute_method)[0] if line.strip()]

    # get rid of function indent.  astunparse will automatically indent the func body to 4 spaces
    srclines[0] = srclines[0].lstrip()
    src = ''.join(srclines)

    if mode == 'rev':
        # mapping to rename variables within the compute method
        to_replace, pnames, onames, rnames = _get_arg_replacement_map(comp)
        ast2 = ast.parse(src)

        # combine all mappings
        mapping = to_replace.copy()
        src_ast = transform_ast_names(ast2, mapping)
    else:
        src_ast = ast.parse(src)

    # make sure indenting is 4
    src = astunparse.unparse(src_ast)

    params = list(signature(compute_method).parameters)

    # add section of code to create a boxed version of the input dict from the input array
    pre_lines = [
        "    import openmdao.utils.mod_wrapper as mod_wrapper",
        "    np = mod_wrapper.np",
        "    numpy = np",
    ]

    vecnames = ['_invec_', '_outvec_', '_residvec_']
    self_vnames = ['self._inputs_autograd', 'self._outputs_autograd', 'self._resids_autograd']
    vecs = [comp._inputs, comp._outputs, comp._vectors['residual']['nonlinear']]
    groups = [(i, pname, vecnames[i], self_vnames[i], vecs[i]) for i, pname in enumerate(params)]

    input_id = 0
    if explicit:
        output_id = 1
    else:
        output_id = 2

    for i, pname, vecname, self_vname, vec in groups:
        if mode == 'rev':
            if (explicit and i > 0) or (i > 1 and not explicit):
                continue
            start = end = 0
            for n in vec:
                val = vec[n]
                end += val.size
                if isinstance(val, np.ndarray) and len(val.shape) > 1:
                    pre_lines.append('    %s = %s[%d:%d].reshape(%s)' %
                                     (mapping["%s['%s']" % (pname, n)], vecname, start, end,
                                      val.shape))
                else:
                    pre_lines.append('    %s = %s[%d:%d]' % (mapping["%s['%s']" % (pname, n)],
                                                             vecname, start, end))
                start = end

    # remove the original compute() call signature
    src = src.split(":\n", 1)[1]

    if mode == 'fwd':
        if explicit:
            # sigstr = "def %s_trans(self, %s):" % (compute_method.__name__, params[0])
            sigstr = "def %s_trans(self, %s, %s):" % (compute_method.__name__,
                                                      params[0], params[1])
        else:
            sigstr = "def %s_trans(self, %s, %s):" % (compute_method.__name__, params[0], params[1])
        retstr = "    return tuple(%s.values())" % params[output_id]
    else:
        if explicit:
            sigstr = "def %s_trans(self, %s):" % (compute_method.__name__, vecnames[0])
            retstr = "    return np.hstack([%s])" % ', '.join(["%s.flatten()" % n for n in onames])
        else:
            sigstr = "def %s_trans(self, %s, %s):" % (compute_method.__name__,
                                                      vecnames[0], vecnames[1])
            retstr = "    return np.hstack([%s])" % ', '.join(["%s.flatten()" % n for n in rnames])

    # generate string of function to be differentiated
    src = '\n'.join([sigstr, '\n'.join(pre_lines), src, retstr])

    print(src)
    return src


def _get_autograd_ad_func(comp, mode):

    funcstr = translate_compute_source_autograd(comp, mode)

    output_names = list(comp._outputs.keys())
    if isinstance(comp, ExplicitComponent):
        compute_method = comp.compute
        argnum = 1
    else:
        compute_method = comp.apply_nonlinear
        argnum = (1, 2)

    funcname = compute_method.__name__ + '_trans'

    input_names = list(comp._inputs.keys())
    input_args = [comp._inputs[n] for n in input_names]

    comp_mod = sys.modules[comp.__class__.__module__]
    namespace = comp_mod.__dict__.copy()
    namespace['self'] = comp

    namespace['numpy'] = agnp
    namespace['np'] = agnp
    for key in dir(agnp):
        if not key.startswith('_'):
            namespace[key] = getattr(agnp, key)
    namespace['tuple'] = agtuple
    namespace['list'] = aglist
    namespace['dict'] = agdict

    exec(funcstr, namespace)

    func = namespace[funcname]

    # JVP - fwd
    # VJP - rev
    if mode == 'fwd':
        offset = len(comp.pathname) + 1 if comp.pathname else 0
        outs = {n[offset:]: v for n, v in iteritems(comp._outputs._views)}
        if isinstance(comp, ExplicitComponent):
            return make_jvp(func, argnum)(comp, comp._inputs, outs)
        else:
            resids = {n[offset:]: v for n, v in
                      iteritems(comp._vectors['residual']['linear']._views)}
            return make_jvp(func, argnum)(comp, comp._inputs, outs, resids)
    else:
        if isinstance(comp, ExplicitComponent):
            return make_vjp(func, argnum)(comp, comp._inputs._data)[0]
        else:
            return make_vjp(func, argnum)(comp, comp._inputs._data, comp._outputs._data)[0]


def _get_autograd_ad_jac(comp, mode, deriv_func, J):
    if mode == 'fwd':
        return _get_ad_jac_fwd(comp, deriv_func, partials=J)
    else:
        return _get_ad_jac_rev(comp, deriv_func, partials=J)


def _get_ad_jac_fwd(comp, deriv_func, partials):
    inputs = comp._inputs
    outputs = comp._outputs
    dinputs = comp._vectors['input']['linear']
    doutputs = comp._vectors['output']['linear']
    array = dinputs._data

    array = agnp.asarray(array)

    explicit = isinstance(comp, ExplicitComponent)

    if explicit:
        J = np.zeros((outputs._data.size, inputs._data.size))
    else:
        J = np.zeros((outputs._data.size, inputs._data.size + outputs._data.size))

    for idx in range(array.size):
        doutputs._data[:] = 0.0
        array[:] = 0.0
        array[idx] = 1.0

        if explicit:
            grad = deriv_func(dinputs)[1]
        else:
            grad = deriv_func((dinputs, doutputs))[1]
        start = end = 0
        for g in grad:
            if explicit:
                end += g.size
                J[start:end, idx] = g.flat
            else:
                end += g.size
                J[start:end, idx] = g.flat
            start = end

    colstart = colend = 0
    if explicit:
        itr = iteritems(inputs._views)
    else:
        itr = chain(iteritems(inputs._views), iteritems(outputs._views))

    for inp, ival in itr:
        colend += ival.size
        rowstart = rowend = 0
        for out, oval in iteritems(outputs._views):
            rowend += oval.size
            partials[out, inp] = J[rowstart:rowend, colstart:colend]
            rowstart = rowend

        colstart = colend


def _get_ad_jac_rev(comp, deriv_func, partials):
    inputs = comp._inputs
    outputs = comp._outputs
    dresids = comp._vectors['residual']['linear']
    array = dresids._data

    explicit = isinstance(comp, ExplicitComponent)

    array = agnp.asarray(array)

    if explicit:
        J = np.zeros((outputs._data.size, inputs._data.size))
    else:
        J = np.zeros((outputs._data.size, inputs._data.size + outputs._data.size))

    for idx in range(array.size):
        array[:] = 0.0
        array[idx] = 1.0

        grad = deriv_func(array)
        J[idx, :] = np.hstack(grad)  # grad has 2 parts, one for each differentiable input arg
        J[idx, :grad[0].size] = grad[0]
        J[idx, grad[0].size:] = grad[1]

    if explicit:
        itr = iteritems(inputs._views)
    else:
        itr = chain(iteritems(inputs._views), iteritems(outputs._views))

    colstart = colend = 0
    for inp, ival in itr:
        colend += ival.size
        rowstart = rowend = 0
        for out, oval in iteritems(outputs._views):
            rowend += oval.size
            partials[out, inp] = J[rowstart:rowend, colstart:colend]
            rowstart = rowend

        colstart = colend
