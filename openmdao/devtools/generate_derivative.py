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

from numpy.testing import assert_almost_equal
from openmdao.core.problem import Problem
from openmdao.vectors.default_vector import Vector, DefaultVector
from openmdao.vectors.petsc_vector import PETScVector
from openmdao.core.explicitcomponent import Component, ExplicitComponent
from openmdao.devtools.ast_tools import transform_ast_names, dependency_analysis, \
    StringSubscriptVisitor, transform_ast_slices
from openmdao.utils.general_utils import str2valid_python_name, unique_name
import openmdao.utils.mod_wrapper as mod_wrapper

try:
    import autograd.numpy as agnp
    from autograd import make_jvp, make_vjp, jacobian
    from autograd.differential_operators import make_jvp_reversemode
    from autograd.builtins import tuple as agtuple, list as aglist, dict as agdict, DictBox, DictVSpace, container_take
except ImportError:
    pass
else:
    class VectorBox(DictBox):
        def get_slice(self, slc): return container_take(self._value._data, slc)

    VectorBox.register(Vector)
    VectorBox.register(DefaultVector)
    VectorBox.register(PETScVector)


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


def _get_arg_replacement_map(comp):
    """
    Return a mapping of names or subscript expressions to be replaced with simple var names.

    Parameters
    ----------
    comp : Component
        Component that is being AD'd.

    Returns
    -------

    """
    inputs = comp._inputs
    outputs = comp._outputs

    if isinstance(comp, ExplicitComponent):
        compute_method = comp.compute
    else:
        compute_method = comp.apply_nonlinear

    params = list(signature(compute_method).parameters)

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


def translate_compute_source_tangent(comp):
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

    # add appropriate return line (return either outputs or residuals depending on compute_method)
    lines.append("        return %s" % params[-1])

    src = textwrap.dedent('\n'.join(lines))

    temp_mod_name = '_temp_' + comp.__class__.__name__
    temp_file_name = temp_mod_name + '.py'

    # convert any literal slices to calls to slice (else tangent fwd mode bombs)
    src = astunparse.unparse(transform_ast_slices(ast.parse(src)))

    return src # , pnames, onames, rnames, to_revert


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
            sigstr = "def %s_trans(self, %s, %s):" % (compute_method.__name__, params[0], params[1])
        else:
            sigstr = "def %s_trans(self, %s, %s):" % (compute_method.__name__, params[0], params[1])
        retstr = "    return tuple(%s.values())" % params[output_id]
    else:
        if explicit:
            sigstr = "def %s_trans(self, %s):" % (compute_method.__name__, vecnames[0])
            retstr = "    return np.hstack([%s])" % ', '.join(["%s.flatten()" % n for n in onames])
        else:
            sigstr = "def %s_trans(self, %s, %s):" % (compute_method.__name__, vecnames[0], vecnames[1])
            retstr = "    return np.hstack([%s])" % ', '.join(["%s.flatten()" % n for n in rnames])

    # generate string of function to be differentiated
    src = '\n'.join([
        sigstr,
        '\n'.join(pre_lines),
        src,
        retstr
        ])

    print(src)
    return src


def _get_imports(mod):
    lines = ["from %s import *" % mod.__name__]
    underscores = [n for n in mod.__dict__ if n.startswith('_') and not n.startswith('__')]
    if underscores:
        lines.append("from %s import %s" % (mod.__name__, ', '.join(underscores)))
    lines.append('')
    return '\n'.join(lines)


def _get_tangent_ad_func(comp, mode, verbose=0, check_dims=False):

    # src, pnames, onames, rnames, to_revert = translate_compute_source_tangent(comp)
    src = translate_compute_source_tangent(comp)

    # start construction of partial derivative functions

    comp_mod = comp.__module__

    sys.path.append(os.getcwd())
    temp_mod_name = 'temp_' + '_'.join((comp_mod.replace('.', '_'), comp.__class__.__name__))
    temp_file_name = temp_mod_name + '.py'

    with open(temp_file_name, "w") as f:
        f.write(_get_imports(sys.modules[comp_mod]))
        f.write("import tangent\n")
        f.write(src)

    if verbose:
        with open(temp_file_name, "r") as f:
            print("Starting source for %s:" % comp.__class__.__name__)
            print(f.read())

    invalidate_caches()  # need this to recognize dynamically created modules
    import_module(temp_mod_name)
    mod = sys.modules[temp_mod_name]

    if isinstance(comp, ExplicitComponent):
        wrt = (1,)
        func = getattr(mod, 'compute')
    else:
        wrt = (1,2)
        func = getattr(mod, 'apply_nonlinear')

    deriv_func = tangent.autodiff(func, wrt=wrt, mode=modemap[mode], verbose=verbose,
                                  check_dims=check_dims)

    print("DERIV MODULE:", deriv_func.__module__)


    del sys.modules[temp_mod_name]
    os.remove(temp_file_name)

    deriv_mod_name = temp_mod_name + '_deriv_'
    deriv_file_name = deriv_mod_name + '.py'

    # write the derivative func into a module file so that tracebacks will show us the
    # correct line of source where the problem occurred, and allow us to step into the
    # function with a debugger.
    with open(deriv_file_name, "w") as f:
        # get all of the comp module globals
        f.write(_get_imports(sys.modules[comp_mod]))
        f.write("import tangent\n")
        f.write(getsource(deriv_func))

    # if verbose:
    #     with open(deriv_file_name, "r") as f:
    #         print("\nDeriv module source for %s:" % comp.__class__.__name__)
    #         print(f.read(),'\n')

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
        array = doutputs._data

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


# def _get_tangent_ad_jac(comp, mode, df, J):
#     if mode == 'fwd':
#         return _get_ad_jac_fwd(comp, df, ad_method='tangent', partials=J)
#     else:
#         return _get_ad_jac_rev(comp, df, ad_method='tangent', partials=J)


def _get_autograd_ad_func(comp, mode):

    funcstr = translate_compute_source_autograd(comp, mode)

    output_names = list(comp._outputs.keys())
    if isinstance(comp, ExplicitComponent):
        compute_method = comp.compute
        argnum = 1
    else:
        compute_method = comp.apply_nonlinear
        argnum = (1,2)

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
            resids = {n[offset:]: v for n, v in iteritems(comp._vectors['residual']['linear']._views)}
            return make_jvp(func, argnum)(comp, comp._inputs, outs, resids)
    else:
        if isinstance(comp, ExplicitComponent):
            return make_vjp(func, argnum)(comp, comp._inputs._data)[0]
        else:
            return make_vjp(func, argnum)(comp, comp._inputs._data, comp._outputs._data)[0]


def _get_autograd_ad_jac(comp, mode, deriv_func, J):
    if mode == 'fwd':
        return _get_ad_jac_fwd(comp, deriv_func, ad_method='autograd', partials=J)
    else:
        return _get_ad_jac_rev(comp, deriv_func, ad_method='autograd', partials=J)


def _get_ad_jac_fwd(comp, deriv_func, ad_method, partials):
    inputs = comp._inputs
    outputs = comp._outputs
    dinputs = comp._vectors['input']['linear']
    doutputs = comp._vectors['output']['linear']
    array = dinputs._data

    agrad = ad_method == 'autograd'
    if agrad:
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

        if agrad:
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
        else:
            params = [inputs[name] for name in inputs] + [dinputs[name] for name in dinputs]
            grad = deriv_func(comp, *params)
            J[:, idx] = grad

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


def _get_ad_jac_rev(comp, deriv_func, ad_method, partials):
    inputs = comp._inputs
    outputs = comp._outputs
    dresids = comp._vectors['residual']['linear']
    array = dresids._data

    explicit = isinstance(comp, ExplicitComponent)

    agrad = ad_method == 'autograd'
    if agrad:
        array = agnp.asarray(array)

    if explicit:
        J = np.zeros((outputs._data.size, inputs._data.size))
    else:
        J = np.zeros((outputs._data.size, inputs._data.size + outputs._data.size))

    for idx in range(array.size):
        array[:] = 0.0
        array[idx] = 1.0

        if agrad:
            grad = deriv_func(array)
            J[idx, :] = np.hstack(grad)  # grad has 2 parts, one for each differentiable input arg
        else:
            params = [inputs[name] for name in inputs] + list(dresids.values())
            grad = deriv_func(comp, *params)
            J[idx, :] = grad

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


def _ad_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao ad' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs='?', help='Python file containing the model.')
    parser.add_argument('-o', default=None, action='store', dest='outfile',
                        help='Output file name. By default, output goes to stdout.')
    parser.add_argument('-m', '--method', default='autograd', action='store', dest='ad_method',
                        help='AD method (autograd, tangent).')
    parser.add_argument('-c', '--class', action='append', dest='classes', default=[],
                        help='Specify component class(es) to run AD on.')


def _ad_exec(options):
    """
    Process command line args and perform postprocessing on the specified memory dump file.
    """
    if options.file:
        from openmdao.utils.om import _post_setup_exec
        options.file = [options.file]
        _post_setup_exec(options)
    else:
        _ad(None, options)


def _get_class(classpath):
    modpath, cname = classpath.rsplit('.', 1)
    import_module(modpath)
    mod = sys.modules[modpath]
    return getattr(mod, cname)


def _comp_iter(classes, prob):
    """
    """
    if classes:
        insts = [_get_class(cpath)() for cpath in classes]
        for obj in insts:
            prob.model.add_subsystem(obj.__class__.__name__.lower() + '_', obj)

        prob.setup()
        prob.run_model()
        invec = prob.model._inputs._data
        invec[:] = np.random.random(invec.size)

        print("\nChecking partials:")
        check_dct = prob.check_partials(out_stream=None)
        prob.run_model()

        for obj in insts:
            print("\nClass:", obj.__class__.__name__)
            yield obj, check_dct


    else:  # find an instance of each Component class in the model
        prob.run_model()
        print("\nChecking partials:")
        check_dct = prob.check_partials(out_stream=None)
        prob.run_model()
        seen = set(('IndepVarComp', 'ExecComp'))
        for s in prob.model.system_iter(recurse=True, include_self=True, typ=Component):
            cname = s.__class__.__name__
            if cname not in seen and (cname in classes or not classes):
                seen.add(cname)
                print("\nClass:", cname)
                print("Instance:", s.pathname)
                yield s, check_dct

            if classes and (len(seen) == len(classes) + 2):
                break

        not_found = classes - seen
        if not_found:
            raise RuntimeError("Couldn't find an instance of the following classes: %s." %
                                not_found)


def _ad(prob, options):
    """
    Compute the fwd and rev AD for the compute or apply_nonlinear method of the given class.
    """
    if options.outfile is None:
        out = sys.stdout
    else:
        out = open(options.outfile, 'w')

    classes = set(options.classes)

    Problem._post_setup_func = None  # prevent infinite recursion

    if prob is None:
        prob = Problem()

    summary = {}

    for s, check_dct in _comp_iter(classes, prob):

        summary[s.__class__.__name__] = summ = {}

        rel_offset = len(s.pathname) + 1 if s.pathname else 0

        type_ = 'Explicit' if isinstance(s, ExplicitComponent) else 'Implicit'
        summ['type'] = type_
        print("Type:", type_)

        if options.ad_method == 'autograd':
            import autograd.numpy as agnp
            mod_wrapper.np = mod_wrapper.numpy = agnp

        for mode in ('fwd', 'rev'):
            summ[mode] = {}
            try:
                J = {}
                if options.ad_method == 'autograd':
                    func = _get_autograd_ad_func(s, mode)
                    _get_autograd_ad_jac(s, mode, func, J)
                elif options.ad_method == 'tangent':
                    func, deriv_mod = _get_tangent_ad_func(s, mode, verbose=1)
                    _get_tangent_ad_jac(s, mode, func, J)

                    del sys.modules[deriv_mod.__name__]
                    os.remove(deriv_mod.__file__)

                mx_diff = 0.0
                print("\n%s J:" % mode.upper())
                for key in sorted(J):
                    o, i = key
                    rel_o = o[rel_offset:]
                    rel_i = i[rel_offset:]
                    if np.any(J[key]) or (rel_o, rel_i) in check_dct[s.pathname]:
                        if (rel_o, rel_i) not in check_dct[s.pathname]:
                            check_dct[s.pathname][rel_o, rel_i] = d = {}
                            d['J_fwd'] = np.zeros(J[key].shape)
                        print("(%s, %s)" % (rel_o, rel_i), end='')
                        try:
                            assert_almost_equal(J[key], check_dct[s.pathname][rel_o, rel_i]['J_fwd'], decimal=5)
                        except:
                            max_diff = np.max(np.abs(J[key] - check_dct[s.pathname][rel_o, rel_i]['J_fwd']))
                            if max_diff > mx_diff:
                                mx_diff = max_diff
                            print("  MAX DIFF:", max_diff)
                        else:
                            print(" ok")
                summ[mode]['diff'] = mx_diff
                summ[mode]['ran'] = True
                print()
            except:
                traceback.print_exc(file=sys.stdout)
                summ[mode]['ran'] = False
                summ[mode]['diff'] = float('nan')
                print("\n")
            finally:
                if options.ad_method == 'autograd':
                    mod_wrapper.np = mod_wrapper.numpy = np

    max_cname = max(len(s) for s in summary) + 2
    max_diff = 16
    bothgood = []
    fwdgood = []
    revgood = []
    bad = []

    toptemplate = "{cname:<{cwidth}}{typ:<10}{fdiff:<{dwidth}}{rdiff:<{dwidth}}"
    template = "{cname:<{cwidth}}{typ:<10}{fdiff:<{dwidth}.4}{rdiff:<{dwidth}.4}"
    print(toptemplate.format(cname='Class', typ='Type', fdiff='Max Diff (fwd)', rdiff='Max Diff (rev)', cwidth=max_cname, dwidth=max_diff))
    print('--------- both derivs ok ------------')
    for cname in sorted(summary):
        s = summary[cname]
        typ = s['type']
        fwdran = s['fwd']['ran']
        fwdmax = s['fwd']['diff']
        revran = s['rev']['ran']
        revmax = s['rev']['diff']
        line = template.format(cname=cname, typ=typ, fdiff=fwdmax, rdiff=revmax, cwidth=max_cname, dwidth=max_diff)
        if fwdran and revran and fwdmax == 0.0 and revmax == 0.0:
            bothgood.append(line)
            print(line)
        elif fwdran and fwdmax == 0.0:
            fwdgood.append(line)
        elif revran and revmax == 0.0:
            revgood.append(line)
        else:
            bad.append(line)

    if fwdgood:
        print('--------- fwd derivs ok ------------')
        for b in fwdgood:
            print(b)

    if revgood:
        print('--------- rev derivs ok ------------')
        for b in revgood:
            print(b)

    if bad:
        print('--------- both derivs bad ------------')
        for b in bad:
            print(b)

    print('\nSummary:  %d both good,  %d fwd good,  %d rev good' % (len(bothgood), len(fwdgood),
                                                                    len(revgood)))
    exit()


def _ad_cmd(options):
    """
    Return the post_setup hook function for 'openmdao ad'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    return lambda prob: _ad(prob, options)

