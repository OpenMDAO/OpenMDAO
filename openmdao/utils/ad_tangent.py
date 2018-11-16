from collections import Iterable

import sys
import os
import traceback
import tangent
from tangent.utils import register_init_grad
import textwrap
import pprint


from inspect import signature, getsourcelines, getsource, getmodule
from collections import OrderedDict, defaultdict
import inspect
from os import remove
from importlib import import_module, invalidate_caches
import types
import time
import numpy as np
import ast
import astunparse
from six import itervalues, iteritems
from itertools import chain

from openmdao.vectors.default_vector import Vector, DefaultVector
from openmdao.vectors.petsc_vector import PETScVector
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.devtools.ast_tools import transform_ast_names, dependency_analysis, \
    StringSubscriptVisitor, transform_ast_slices


modemap = {
    'fwd': 'forward',
    'rev': 'reverse',
}


class Vec(object):
    def __init__(self, v):
        self._data = np.zeros_like(v._data)
        self._views = {}
        start = end = 0
        for name, absname in zip(v, v._views):
            end += v._views[absname].size
            view = self._data[start:end]
            view.shape = v._views[absname].shape
            self._views[name] = view
            start = end

    def __getitem__(self, name):
        return self._views[name]

    def __setitem__(self, name, val):
        self._views[name][:] = val


def zero_vector(vec):
    return Vec(vec)


register_init_grad(DefaultVector, zero_vector)
register_init_grad(str, lambda s: s)


def _translate_compute_source(comp):
    """
    Convert a compute or apply_nonlinear method into a function with individual args for each var.

    Converts  def compute(self, inputs, outputs) by adding an appropriate return line and
    converting literal slices to slice() calls.

    Parameters
    ----------
    comp : Component
        The component being AD'd.

    Returns
    -------
    str
        Converted source code.
    """
    if isinstance(comp, ExplicitComponent):
        compute_method = comp.compute
    else:
        compute_method = comp.apply_nonlinear

    params = list(signature(compute_method).parameters)
    lines = getsourcelines(compute_method)[0]

    for i, line in enumerate(lines):
        if line.lstrip().startswith('def '):
            lines[i] = line.replace('self,', '')  # TODO: fix this to be more robust

    # add appropriate return line (return either outputs or residuals depending on compute_method)
    lines.append("        return %s" % params[-1])

    src = textwrap.dedent('\n'.join(lines))

    temp_mod_name = '_temp_' + comp.__class__.__name__
    temp_file_name = temp_mod_name + '.py'

    # convert any literal slices to calls to slice (else tangent fwd mode bombs)
    src = astunparse.unparse(transform_ast_slices(ast.parse(src)))

    return src


def _get_imports(mod):
    lines = ["from %s import *" % mod.__name__]
    underscores = [n for n in mod.__dict__ if n.startswith('_') and not n.startswith('__')]
    if underscores:
        lines.append("from %s import %s" % (mod.__name__, ', '.join(underscores)))
    lines.append('')
    return '\n'.join(lines)


def _get_tangent_ad_func(comp, mode, verbose=0, optimize=True, check_dims=False):

    src = _translate_compute_source(comp)

    # start construction of partial derivative functions

    comp_mod = comp.__module__

    sys.path.append(os.getcwd())
    temp_mod_name = 'temp_' + '_'.join((comp_mod.replace('.', '_'), comp.__class__.__name__))
    temp_file_name = temp_mod_name + '.py'

    with open(temp_file_name, "w") as f:
        f.write(_get_imports(sys.modules[comp_mod]))
        f.write("import tangent\n")
        f.write(src)

    invalidate_caches()  # need this to recognize dynamically created modules
    import_module(temp_mod_name)
    mod = sys.modules[temp_mod_name]

    if isinstance(comp, ExplicitComponent):
        wrt = (0,)
        func = getattr(mod, 'compute')
    else:
        wrt = (0, 1)
        func = getattr(mod, 'apply_nonlinear')

    deriv_func = tangent.autodiff(func, wrt=wrt, mode=modemap[mode], verbose=verbose,
                                  optimized=optimize, check_dims=check_dims)

    del sys.modules[temp_mod_name]
    os.remove(temp_file_name)

    deriv_mod_name = temp_mod_name + '_deriv_'
    deriv_file_name = deriv_mod_name + '.py'

    # now put 'self' back in the arg list
    lines = getsourcelines(deriv_func)[0]
    for i, line in enumerate(lines):
        if line.lstrip().startswith('def '):
            lines[i] = line.replace('(', '(self,')
    deriv_src = '\n'.join(lines)

    # write the derivative func into a module file so that tracebacks will show us the
    # correct line of source where the problem occurred, and allow us to step into the
    # function with a debugger.
    with open(deriv_file_name, "w") as f:
        f.write("import numpy\n")  # tangent templates use 'numpy' so make sure it's here
        # get all of the comp module globals
        f.write(_get_imports(sys.modules[comp_mod]))
        f.write("import tangent\n")
        f.write(deriv_src)

    invalidate_caches()  # need this to recognize dynamically created modules
    import_module(deriv_mod_name)
    mod = sys.modules[deriv_mod_name]
    sys.path = sys.path[:-1]

    return getattr(mod, deriv_func.__name__), mod


def _get_tangent_ad_jac(comp, mode, deriv_func, partials):
    inputs = comp._inputs
    outputs = comp._outputs
    dinputs = comp._vectors['input']['linear']
    doutputs = comp._vectors['output']['linear']
    resids = comp._vectors['residual']['nonlinear']
    dresids = comp._vectors['residual']['linear']

    explicit = isinstance(comp, ExplicitComponent)

    if explicit:
        J = np.zeros((outputs._data.size, inputs._data.size))
    else:
        J = np.zeros((outputs._data.size, inputs._data.size + outputs._data.size))

    to_zero = [doutputs._data, dresids._data, dinputs._data]

    if mode == 'fwd':
        array = dinputs._data
    else:
        if explicit:
            array = doutputs._data
        else:
            array = dresids._data

    for idx in range(array.size):
        for zvec in to_zero:
            zvec[:] = 0.0
        array[idx] = 1.0

        if mode == 'fwd':
            if explicit:
                grad = deriv_func(comp, inputs, outputs, dinputs)
            else:
                grad = deriv_func(comp, inputs, outputs, resids, dinputs, doutputs)
            J[:, idx] = grad._data
        else:  # rev
            if explicit:
                grad = deriv_func(comp, inputs, outputs, doutputs)
                J[idx, :] = grad._data
            else:
                grad = deriv_func(comp, inputs, outputs, resids, dresids)
                J[idx, :grad[0]._data.size] = grad[0]._data
                J[idx, grad[0]._data.size:] = grad[1]._data

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

