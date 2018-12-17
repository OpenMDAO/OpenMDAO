"""
Automatic differentiation functions using the google's tangent package.
"""

from collections import Iterable

import sys
import os
import traceback
import tangent
from tangent.utils import register_init_grad
import textwrap
import pprint
from numbers import Number


from inspect import signature, getsourcelines, getsource, getmodule
from collections import OrderedDict, defaultdict
import inspect
from os import remove
import importlib
import types
import time
import numpy as np
import ast
import astunparse
from six import itervalues, iteritems, PY3
from itertools import chain

from openmdao.vectors.default_vector import Vector, DefaultVector
from openmdao.vectors.petsc_vector import PETScVector
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.devtools.ast_tools import transform_ast_names, dependency_analysis, \
    StringSubscriptVisitor, transform_ast_slices, get_external_vars
from openmdao.utils.general_utils import print_line_numbers
from openmdao.utils.options_dictionary import OptionsDictionary


modemap = {
    'fwd': 'forward',
    'rev': 'reverse',
}


class _Vec(object):
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


def _zero_vector(vec):
    return _Vec(vec)


class _Opt(object):
    def __init__(self, opt):
        self._dct = {}
        for key in opt:
            val = opt[key]
            if isinstance(val, Number):
                self._dct[key] = type(val)(0)
            elif isinstance(val, np.ndarray):
                self._dct[key] = np.zeros_like(val)
            else:  # punt
                self._dct[key] = val

    def __getitem__(self, name):
        return self._dct[name]

    def __setitem__(self, name, val):
        self._dct[name] = val


def _zero_opt(opt):
    return _Opt(opt)


register_init_grad(DefaultVector, _zero_vector)
register_init_grad(OptionsDictionary, _zero_opt)
register_init_grad(str, lambda s: s)


def _translate_compute_source(comp, verbose=0):
    """
    Convert a compute or apply_nonlinear method into a function with individual args for each var.

    Converts compute or apply_nonlinear by adding an appropriate return line and
    converting literal slices to slice() calls.

    Parameters
    ----------
    comp : Component
        The component being AD'd.
    verbose : bool
        Verbosity flag.

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
            break

    # add appropriate return line (return either outputs or residuals depending on compute_method)
    lines.append("        return %s\n" % params[-1])

    src = textwrap.dedent(''.join(lines))

    temp_mod_name = '_temp_' + comp.__class__.__name__
    temp_file_name = temp_mod_name + '.py'

    # convert any literal slices to calls to slice (else tangent fwd mode bombs)
    if verbose == 1:
        print("SRC:")
        print_line_numbers(src)
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

    src = _translate_compute_source(comp, verbose=verbose)

    # start construction of partial derivative functions

    comp_mod = comp.__module__

    sys.path.append(os.getcwd())
    temp_mod_name = 'temp_' + '_'.join((comp_mod.replace('.', '_'), comp.__class__.__name__))
    temp_file_name = temp_mod_name + '.py'

    with open(temp_file_name, "w") as f:
        f.write(_get_imports(sys.modules[comp_mod]))
        f.write("import tangent\n")
        f.write(src)

    if PY3:
        importlib.invalidate_caches()  # need this to recognize dynamically created modules
    importlib.import_module(temp_mod_name)
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

    # actual tangent-generated deriv file (might contain more than one function)
    lines = open(deriv_func.__code__.co_filename, 'r').readlines()

    # now put 'self' back in the arg list
    # lines = getsourcelines(deriv_func)[0]
    # funcs = set()
    nreplaced = 0
    for i, line in enumerate(lines):
        lstrip = line.lstrip()
        if lstrip.startswith('def '):
            if nreplaced == 0:  # add self for top level function only
                lines[i] = line.replace('(', '(self, ')
                nreplaced += 1

    deriv_src = ''.join(lines)

    # write the derivative func into a module file so that tracebacks will show us the
    # correct line of source where the problem occurred, and allow us to step into the
    # function with a debugger.
    with open(deriv_file_name, "w") as f:
        f.write("import numpy\n")  # tangent templates use 'numpy' so make sure it's here
        # get all of the comp module imports
        f.write(_get_imports(sys.modules[comp_mod]))
        f.write("import tangent\n")
        f.write(deriv_src)

    if PY3:
        importlib.invalidate_caches()  # need this to recognize dynamically created modules

    importlib.import_module(deriv_mod_name)
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
        if explicit:
            arrays = [dinputs._data]
        else:
            arrays = [dinputs._data, doutputs._data]
    else:
        if explicit:
            arrays = [doutputs._data]
        else:
            arrays = [dresids._data]

    col_offset = 0
    for array in arrays:
        for idx in range(array.size):
            for zvec in to_zero:
                zvec[:] = 0.0
            array[idx] = 1.0
    
            if mode == 'fwd':
                if explicit:
                    grad = deriv_func(comp, inputs, outputs, dinputs)
                else:
                    grad = deriv_func(comp, inputs, outputs, resids, dinputs, doutputs)
                J[:, idx + col_offset] = grad._data
            else:  # rev
                if explicit:
                    grad = deriv_func(comp, inputs, outputs, doutputs)
                    J[idx, :] = grad._data
                else:
                    grad = deriv_func(comp, inputs, outputs, resids, dresids)
                    J[idx, :grad[0]._data.size] = grad[0]._data
                    J[idx, grad[0]._data.size:] = grad[1]._data

        col_offset += array.size

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
