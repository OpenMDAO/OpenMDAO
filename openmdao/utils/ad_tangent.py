"""
Automatic differentiation functions using the google's tangent package.
"""

from collections import Iterable

import sys
import os
import traceback
import tangent
from tangent.transformers import TreeTransformer
from tangent.utils import register_init_grad, register_add_grad
from tangent import naming
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
import gast
from six import itervalues, iteritems, PY3
from itertools import chain

from openmdao.vectors.vector import Vector, set_vec
from openmdao.vectors.default_vector import DefaultVector
from openmdao.vectors.petsc_vector import PETScVector
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.devtools.ast_tools import transform_ast_names, dependency_analysis, \
    StringSubscriptVisitor
from openmdao.utils.general_utils import print_line_numbers
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.devtools.debug import compute_approx_jac, compare_jacs
from openmdao.devtools.ast_tools import add_prints


modemap = {
    'fwd': 'forward',
    'rev': 'reverse',
}


# Use this within AD function in place of Vector
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

    def __str__(self):
        return str(self._data)


# use this in AD function in place of OptionsDictionary
class _OptionsAD(object):
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


register_init_grad(DefaultVector, lambda vec: _Vec(vec))
register_init_grad(OptionsDictionary, lambda opt: _OptionsAD(opt))
register_init_grad(_OptionsAD, lambda opt: opt)
register_init_grad(str, lambda s: 0.0)
register_add_grad(float, _OptionsAD, lambda l, r: l)
register_add_grad(_OptionsAD, float, lambda l, r: r)
register_add_grad(np.ndarray, _OptionsAD, lambda l, r: l.copy())
register_add_grad(_OptionsAD, np.ndarray, lambda l, r: r.copy())


@tangent.tangent_(set_vec)
def tset_vec(z, x, y):
    d[z] = d[x]

@tangent.grads.adjoint(set_vec)
def svec(z, x, y):
    d[x] = d[z]
    d[y] = d[z]


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

    src = textwrap.dedent(''.join(lines))

    # add appropriate return line (return either outputs or residuals depending on compute_method)
    src += "    return %s\n" % params[-1]

    temp_mod_name = '_temp_' + comp.__class__.__name__
    temp_file_name = temp_mod_name + '.py'

    # convert any literal slices to calls to slice (else tangent fwd mode bombs)
    #src = astunparse.unparse(transform_ast_slices(ast.parse(src)))

    if verbose == 1:
        print("SRC:")
        print_line_numbers(src)

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
    with open(deriv_func.__code__.co_filename, 'r') as f:
        lines = f.readlines()

    # now put 'self' back in the arg list
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

    dfunc = getattr(mod, deriv_func.__name__)
    if verbose > 2:
        dfunc = add_prints()(dfunc)

    return dfunc, mod


def _dot_prod_test(comp, fwd_func, rev_func):
    inputs = comp._inputs
    outputs = comp._outputs
    dinputs = comp._vectors['input']['linear']
    doutputs = comp._vectors['output']['linear']
    resids = comp._vectors['residual']['nonlinear']
    dresids = comp._vectors['residual']['linear']
    explicit = isinstance(comp, ExplicitComponent)

    in_save = dinputs._data.copy()
    out_save = doutputs._data.copy()
    resid_save = dresids._data.copy()

    rand_inputs = np.random.random(in_save.size) + 1.0
    rand_outputs = np.random.random(out_save.size) + 1.0

    try:
        dinputs._data[:] = rand_inputs
        if explicit:
            fwd_grad = fwd_func(comp, inputs, outputs, dinputs)
            doutputs._data[:] = rand_outputs
            rev_grad = rev_func(comp, inputs, outputs, doutputs)
            dpy = rand_outputs.dot(fwd_grad._data)
            dpx = rev_grad._data.dot(rand_inputs)
        else:
            doutputs._data[:] = rand_outputs
            fwd_grad = fwd_func(comp, inputs, outputs, resids, dinputs, doutputs)
            dresids._data[:] = rand_outputs
            rev_grad = rev_func(comp, inputs, outputs, resids, dresids)
            dpy = rand_outputs.dot(fwd_grad._data)
            dpx = rev_grad[0]._data.dot(rand_inputs) + rev_grad[1]._data.dot(rand_outputs)
    finally:
        # restore old vector values
        dinputs._data[:] = in_save
        doutputs._data[:] = out_save
        dresids._data[:] = resid_save

    print("y.dot(dy)", dpy)
    print("dx.dot(x)", dpx)
    return abs(dpy - dpx)


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


def check_tangent_ad(comp, failtol=1.e-6, mode=None, verbose=0, optimize=True, raise_exc=True,
                     **kwargs):
    """
    Compare AD jac for the given component with its appoximate jac (either fd or cs).

    The model must be in a valid state before running this, else the FD will give bad
    results.

    Parameters
    ----------
    comp : Component
        The component to be checked.
    failtol : float (1.e-6)
        Maximum difference allowed between subjacs before failure.
    mode : str or None
        AD mode used, 'fwd' or 'rev'.  Both are checked if None.
    verbose : int (0)
        If 1 the source code of the generated functions will be
        output to stdout at various stages of the process for debugging
        purposes. If > 1, all intermediate code generation steps will print.
    optimize : bool (True)
        If True, allow tangent to perform optimizations on the generated code.
    raise_exc : bool (True)
        If True, raise an exception if difference in derivs is > failtol.
    **kwargs : dict
        Other named args passed to compute_approx_jac.

    Returns
    -------
    float
        Max difference found between AD and FD (or CS) derivatives.
    """
    save_inputs = comp._inputs._data.copy()
    Japprox, no_cs = compute_approx_jac(comp, method='cs', **kwargs)

    if mode is None:
        modes = ['fwd', 'rev']
    else:
        modes = [mode]

    max_diff = 0.0
    for mode in modes:
        comp._inputs._data[:] = save_inputs
        deriv_func, dmod = _get_tangent_ad_func(comp, mode, verbose=verbose,
                                                optimize=optimize)
        Jad = {}
        _get_tangent_ad_jac(comp, mode, deriv_func, Jad)

        del sys.modules[dmod.__name__]
        os.remove(dmod.__file__)
        try:
            os.remove(dmod.__file__ + 'c')
        except FileNotFoundError:
            pass

        rel_offset = len(comp.pathname) + 1 if comp.pathname else 0

        for key, diff, diff_type in compare_jacs(Japprox, Jad):
            o, i = key
            relkey = (o[rel_offset:], i[rel_offset:])

            if raise_exc and diff > failtol:
                raise RuntimeError("Max diff for subjac %s is %g (%s)" % (relkey, diff,
                                                                          diff_type))

            if diff > max_diff:
                max_diff = diff

    return max_diff
