from collections import Iterable

import sys
import tangent
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
from six import itervalues

from openmdao.core.explicitcomponent import Component, ExplicitComponent
from openmdao.devtools.ast_tools import transform_ast_names, dependency_analysis, \
    StringSubscriptVisitor
from openmdao.utils.general_utils import str2valid_python_name, unique_name


def generate_gradient(comp, mode, body_src, to_replace, pnames, onames, rnames):
    """
    Given the string representing the source code of a python function,
    use the tangent library to generate source code of its gradient.

    Parameters
    ----------
    comp: Component
        The component we're AD'ing.
    mode : str
        Derivative direction ('forward' or 'reverse')
    body_src : str
        The source code of the body of the compute/apply_nonlinear function, converted
        into non-member form.

    Returns
    -------
    function
        The generated derivative function.
    str
        The source of the function actually being AD'd.
    """
    lines = ["%s = self._inputs['%s']\n" % (str2valid_python_name(n), n) for n in comp._inputs]
    lines.extend(["%s = self._outputs['%s']\n" %
                  (str2valid_python_name(n), n) for n in comp._outputs])
    lines.extend(["resid__%s = self._vectors['residual']['nonlinear']['%s']\n" %
                  (str2valid_python_name(n), n) for n in comp._vectors['residual']['nonlinear']])
    lines.extend(["_opt__%s = self.options['%s']\n" %
                  (str2valid_python_name(n), n) for n in comp.options])

    lines.append('\nif True:\n')  # body src is indented, so put in conditional block

    lines.append(body_src)

    full_src = ''.join(lines)

    comp_mod = sys.modules[comp.__class__.__module__]
    namespace = comp_mod.__dict__.copy()
    namespace['self'] = comp

    # exec function body in namespace to define all of the variables
    exec(full_src, namespace)

    # to avoid having to unindent all of the body_src, just put it in an 'if True' block to keep ast happy
    if_body_src = 'if True:\n' + body_src
    fdviz, f_ast = dependency_analysis(if_body_src)

    fmap = {}
    modules = set()

    # this allows us to substitute defined non-member functions with tangent adjoints/tangents
    # for member functions
    for s in fdviz.calls:
        if s.startswith('np.') or s.startswith('numpy.') or s in __builtins__:
            continue
        parts = s.split('.')
        obj = namespace[parts[0]]
        for p in parts[1:]:
            obj = getattr(obj, p)

        if type(obj) is types.MethodType:
            klass = obj.__self__.__class__
            # look for AD mappings in method's module
            mod = sys.modules[klass.__module__]
            name = klass.__name__
        elif type(obj) is types.FunctionType:
            mod = sys.modules[obj.__module__]
            name = obj.__name__
        else:
            mod = None

        if mod is not None and hasattr(mod, '_tangent_ad_mapping'):
            mapping = getattr(mod, '_tangent_ad_mapping', None)
            if mapping is not None and name in mapping:
                tan_func = mapping[name]
                if isinstance(tan_func, dict):
                    tan_func = tan_func[parts[-1]]
                fmap[s] = '.'.join([tan_func.__module__, tan_func.__name__])
                modules.add(tan_func.__module__)

    funcname = 'compute_ad'

    # generate string of function to be differentiated by the tangent library
    func_source = '\n'.join([
        "def %s(%s):" % (funcname, ', '.join(pnames)),
        body_src,
        "    return %s" % ', '.join(onames + rnames)
        ])

    lines = ['import %s' % modname for modname in modules]
    lines.append(func_source)

    new_src = '\n'.join(lines)

    if fmap:
        ast2 = ast.parse(new_src)
        new_ast = transform_ast_names(ast2, fmap)
        new_src = astunparse.unparse(new_ast)

    # needs to have an associated source file for tangent's use of inspect functions to work

    temp_mod_name = '_temp_' + comp.pathname.replace('.', '_')
    temp_file_name = temp_mod_name + '.py'

    with open(temp_file_name, "w") as f:
        f.write(new_src)

    import_module(temp_mod_name)
    mod = sys.modules[temp_mod_name]

    # populate module with globals from component's module
    comp_mod = sys.modules[comp.__class__.__module__]
    new_glob = set(comp_mod.__dict__) - set(mod.__dict__)
    for name in new_glob:
        setattr(mod, name, getattr(comp_mod, name))

    mod.self = comp

    # now set other comp values in globals that we've replaced with non-member variables
    for rhs, name in to_replace.items():
        setattr(mod, name, eval(rhs, mod.__dict__))

    func = getattr(mod, funcname)
    sig = signature(func)
    params = sig.parameters

    # generate AD code for the method for all inputs
    df = tangent.autodiff(func, wrt=tuple(range(len(params))), mode=mode,
                          verbose=0, check_dims=False)

    # cleanup temp file and module
    remove(temp_file_name)
    del sys.modules[temp_mod_name]

    return df, func_source


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

    Converts  def compute(self, inputs, outputs) to def kompute(self, a, b, c)
    and translated function returns all output values as a tuple.

    Parameters
    ----------
    comp : Component
        The component being AD'd.

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
    if isinstance(comp, ExplicitComponent):
        compute_method = comp.compute
    else:
        compute_method = comp.apply_nonlinear

    # mapping to rename variables within the compute method
    to_replace, pnames, onames, rnames = _get_arg_replacement_map(comp)

    to_revert = {"self.options['%s']" % name: '_opt__%s' % name for name in comp.options}

    # get source code of original compute() method
    # ignore blank lines, useful for detecting indentation later
    srclines = [line for line in getsourcelines(compute_method)[0] if line.strip()]

    # get rid of function indent.  astunparse will automatically indent the func body to 4 spaces
    srclines[0] = srclines[0].lstrip()
    src = ''.join(srclines)
    ast2 = ast.parse(src)

    # combine all mappings
    mapping = to_replace.copy()
    mapping.update(to_revert)
    ast3 = transform_ast_names(ast2, mapping)
    src = astunparse.unparse(ast3)

    # remove the original compute() call signature
    src = src.split(":\n", 1)[1]

    return src, pnames, onames, rnames, to_revert


def translate_compute_source_autograd(comp):
    """
    Convert a compute or apply_nonlinear method into a function with individual args for each var.

    Converts  def compute(self, inputs, outputs) to def kompute(self, a, b, c)
    and translated function returns all output values as a tuple.

    Parameters
    ----------
    comp : Component
        The component being AD'd.

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
    if isinstance(comp, ExplicitComponent):
        compute_method = comp.compute
    else:
        compute_method = comp.apply_nonlinear

    # mapping to rename variables within the compute method
    to_replace, pnames, onames, rnames = _get_arg_replacement_map(comp)

    # get source code of original compute() method
    # ignore blank lines, useful for detecting indentation later
    srclines = [line for line in getsourcelines(compute_method)[0] if line.strip()]

    # get rid of function indent.  astunparse will automatically indent the func body to 4 spaces
    srclines[0] = srclines[0].lstrip()
    src = ''.join(srclines)
    ast2 = ast.parse(src)

    # visitor = StringSubscriptVisitor()
    # visitor.visit(ast2)
    # print("found", visitor.subscripts.items())

    # combine all mappings
    mapping = to_replace.copy()
    ast3 = transform_ast_names(ast2, mapping)
    src = astunparse.unparse(ast3)

    # add section of code to equate internal vars to views of input vec
    pre_lines = []
    start = end = 0
    for n in comp._inputs:
        end += comp._inputs[n].size
        val = comp._inputs[n]
        if isinstance(val, np.ndarray):
            if len(val.shape) > 1:
                pre_lines.append('    %s = _invec_[%d:%d].reshape(%s)' %
                                 (str2valid_python_name(n), start, end, val.shape))
            else:
                pre_lines.append('    %s = _invec_[%d:%d]' % (str2valid_python_name(n), start, end))
        else:
            pre_lines.append('    %s = _invec_[%d]' % (str2valid_python_name(n), start))
        start = end

    # remove the original compute() call signature
    src = src.split(":\n", 1)[1]

    # generate string of function to be differentiated
    src = '\n'.join([
        "def %s_trans(_invec_):" % compute_method.__name__,
        '\n'.join(pre_lines),
        src,
        "    return tuple([%s])" % ', '.join(onames + rnames)
        ])

    return src, pnames, onames, rnames


def revert_deriv_source(deriv_func, to_revert):

    # now translate back the member vars we substituted for earlier
    deriv_ast = ast.parse(getsource(deriv_func))

    # reverse the to_revert dict
    revert = {val: key for key, val in to_revert.items()}

    deriv_ast = transform_ast_names(deriv_ast, revert)
    deriv_src = astunparse.unparse(deriv_ast)
    deriv_lines = deriv_src.split('\n')

    # convert function signature to def _compute_derivs_ad(self, ...)
    for i, line in enumerate(deriv_lines):
        if line.startswith('def '):
            deriv_lines[i] = _fix_func_def(line, '_compute_derivs_ad')
            break

    return deriv_lines


def generate_tangent_gradient(comp, mode):

    src, pnames, onames, rnames, to_revert = translate_compute_source_tangent(comp)

    # start construction of partial derivative functions

    comp_mod = sys.modules[comp.__class__.__module__]
    namespace = comp_mod.__dict__.copy()

    # gather generated gradient source code
    df, func_source = generate_gradient(comp, mode, src, to_revert, pnames, onames, rnames)

    deriv_src = '\n'.join(revert_deriv_source(df, to_revert))

    namespace['tangent'] = tangent

    # create an actual function object by exec'ing the source
    exec(deriv_src, namespace)

    return func_source, deriv_src, namespace['_compute_derivs_ad']


def _get_tangent_ad_jac(comp, mode, J):
    src, dsrc, df = generate_tangent_gradient(comp, mode)

    if mode == 'forward':
        return _get_ad_jac_fwd(comp, df, ad_method='tangent', J=J)
    else:
        return _get_ad_jac_rev(comp, df, ad_method='tangent', J=J)


def _get_autograd_ad_jac(comp, mode, J):
    import autograd.numpy as agnp
    from autograd import make_jvp, make_vjp
    from autograd.differential_operators import make_jvp_reversemode
    from autograd.builtins import tuple as agtuple, list as aglist, dict as agdict

    output_names = list(comp._outputs.keys())
    if isinstance(comp, ExplicitComponent):
        compute_method = comp.compute
    else:
        compute_method = comp.apply_nonlinear

    input_names = list(comp._inputs.keys())
    input_args = [comp._inputs[n] for n in input_names]

    funcname = compute_method.__name__ + '_trans'
    funcstr, pnames, onames, rnames = translate_compute_source_autograd(comp)

    comp_mod = sys.modules[comp.__class__.__module__]
    namespace = comp_mod.__dict__.copy()
    namespace['self'] = comp

    namespace['numpy'] = agnp
    namespace['np'] = agnp
    namespace['tuple'] = agtuple
    namespace['list'] = aglist
    namespace['dict'] = agdict

    exec(funcstr, namespace)

    func = namespace[funcname]

    # JVP - forward
    # VJP - reverse
    if mode == 'forward':
        deriv_func = make_jvp(func)(agnp.array(comp._inputs._data))
        return _get_ad_jac_fwd(comp, deriv_func, ad_method='autograd', J=J)
    else:
        deriv_func = make_vjp(func)(agnp.array(comp._inputs._data))[0]
        return _get_ad_jac_rev(comp, deriv_func, ad_method='autograd', J=J)


def _get_ad_jac_fwd(comp, deriv_func, ad_method, J):
    prefix = comp.pathname + '.' if comp.pathname else ''

    inputs = comp._inputs
    outputs = comp._outputs

    vec = inputs
    array = comp._vectors['input']['linear']._data

    agrad = ad_method == 'autograd'

    idx2arrname = [None] * array.size  # map array index to array var name
    idx2loc = np.zeros(array.size, dtype=int)

    views = []
    start = end = 0
    for n in vec:
        end += vec[n].size
        views.append(array[start:end])
        for i in range(start, end):
            idx2arrname[i] = n
        idx2loc[start:end] = np.arange(start, end, dtype=int) - start
        start = end

    if not agrad:
        params = [inputs[name] for name in inputs] + views

    for idx in range(array.size):
        array[:] = 0.0
        array[idx] = 1.0
        iname = idx2arrname[idx]
        locidx = idx2loc[idx]
        abs_in = prefix + iname

        if agrad:
            grad = deriv_func(array)[1]
        else:
            grad = deriv_func(comp, *params)
        for i, oname in enumerate(outputs):
            abs_out = prefix + oname
            key = (abs_out, abs_in)
            if key in comp._subjacs_info:
                if key not in J:
                    J[key] = np.zeros((outputs[oname].size, inputs[iname].size))
                J[key][:, locidx] = grad[i]

    return J


def _get_ad_jac_rev(comp, deriv_func, ad_method, J):
    prefix = comp.pathname + '.' if comp.pathname else ''

    inputs = comp._inputs
    vec = outputs = comp._outputs
    array = comp._vectors['output']['linear']._data

    agrad = ad_method == 'autograd'

    idx2arrname = [None] * array.size  # map array index to array var name
    idx2loc = np.zeros(array.size, dtype=int)

    views = []
    start = end = 0
    for n in vec:
        end += vec[n].size
        views.append(array[start:end])
        for i in range(start, end):
            idx2arrname[i] = n
        idx2loc[start:end] = np.arange(start, end, dtype=int) - start
        start = end

    if not agrad:
        params = [inputs[name] for name in inputs] + [views]

    for oidx in range(array.size):
        array[:] = 0.0
        array[oidx] = 1.0
        oname = idx2arrname[oidx]
        locidx = idx2loc[oidx]
        abs_out = prefix + oname

        if agrad:
            grad = deriv_func(array)
        else:
            grad = deriv_func(comp, *params)
        for i, iname in enumerate(inputs):
            abs_in = prefix + iname
            key = (abs_out, abs_in)
            if key in comp._subjacs_info:
                if key not in J:
                    J[key] = np.zeros((outputs[oname].size, inputs[iname].size))
                J[key][locidx:] = grad[i]

    return J


def _ad_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao ad' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', default=None, action='store', dest='outfile',
                        help='Output file name. By default, output goes to stdout.')
    parser.add_argument('-m', '--method', default='autograd', action='store', dest='ad_method',
                        help='AD method (autograd, tangent).')
    parser.add_argument('-c', '--class', action='store', dest='class_',
                        help='Specify component class to run AD on.')


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
    if options.outfile is None:
        out = sys.stdout
    else:
        out = open(options.outfile, 'w')

    def _ad(prob):
        for s in prob.model.system_iter(recurse=True, include_self=True, typ=Component):
            if s.__class__.__name__ == options.class_:
                break
        else:
            raise RuntimeError("Couldn't find an instance of class '%s'." % options.class_)

        if options.ad_method == 'tangent':
            Jrev = {}
            _get_tangent_ad_jac(s, 'reverse', Jrev)
            Jfwd = {}
            _get_tangent_ad_jac(s, 'forward', Jfwd)
        elif options.ad_method == 'autograd':
            Jrev = {}
            _get_autograd_ad_jac(s, 'reverse', Jrev)
            Jfwd = {}
            _get_autograd_ad_jac(s, 'forward', Jfwd)

        import pprint

        print("\n\nReverse J:")
        pprint.pprint(Jrev)
        print("Forward J:")
        pprint.pprint(Jfwd)

        exit()

    return _ad


def _fix_func_def(line, rename=None):
    # line is assumed to be of the form   def foo(a, b, c)
    # it will be converted to  def _compute_derivs_ad(self, a, b, c)
    parts = line.split('(', 1)
    if rename:
        fstart = 'def %s(' % rename
    else:
        fstart = parts[0]

    return 'def _compute_derivs_ad(' + 'self, ' + parts[1]


def _translate_vars_from_meta(srclines):
    pass
    # # find any blocks that parameterize by iterating over metadata
    # # these need to be explicitly flattened
    # for k, line in enumerate(src):
    #     """
    #     For example, the lines:
    #         for BN in self.metadata['bleed_names']:
    #             outputs['ht_out'] += inputs[BN + ':W']
    #     will automatically be transformed into something like
    #         b1_W = inputs['b1:W']
    #         b2_W = inputs['b2:W']
    #         outputs['ht_out'] += b1_W
    #         outputs['ht_out'] += b2_W
    #     if the iteration is over empty metadata, the block is omitted
    #     """
    #     # key assumption: iteration over metadata corresponds to
    #     # variable-length input or output
    #     tabs0 = len(line) - len(line.lstrip())
    #     if 'in self.metadata' in line:
    #         # determine the iteration variable name
    #         # e.g. 'BN' in the example above
    #         vname = line.split().index("for")
    #         vname = line.split()[vname + 1]

    #         # allows us to break out based on indentation
    #         block_ind = len(line) - len(line.lstrip())

    #         # actually execute the block to get the metadata attribute
    #         # this will expose how many vars are present for this instance
    #         exc = line.replace(":", "").replace("self", "comp")
    #         exc = ("[%s " % vname) + exc + "]"
    #         # this would be ['b1', 'b2'], etc.
    #         varlist = eval(exc)

    #         # code lines for flattened iteration
    #         unravelled = []
    #         # line number of end of iteration block
    #         stop_idx = k
    #         # if there are no variables to iterate over, just get the end
    #         # line number of the block to remove it
    #         if not varlist:
    #             for k2, future_line in enumerate(src[k+1:]):
    #                 if len(future_line) - len(future_line.lstrip()) == block_ind:
    #                     stop_idx = k2
    #                     break
    #         # otherwise, pass over the block for each variable value to
    #         # re-write it
    #         else:
    #             for varname in varlist:
    #                 # search future lines for end of block
    #                 for k2, future_line in enumerate(src[k+1:]):
    #                     if len(future_line) - len(future_line.lstrip()) == block_ind:
    #                         stop_idx = k2
    #                         break
    #                     elif vname in future_line:
    #                         # search and replace each instance in each line
    #                         # e.g. find instances of things that look like
    #                         # inputs[BN + ':W']
    #                         # and replace with inputs['b1:W'], inputs['b2:W'], etc.
    #                         # these will get replaced with local vars b1_W, b2_W later
    #                         while True:
    #                             try:
    #                                 i = future_line.index(vname)
    #                             except ValueError:
    #                                 break
    #                             left = future_line[:i][::-1].index("[")
    #                             right = future_line[i + 1:].index("]")

    #                             left = i - left
    #                             right = i + right + 1

    #                             substr = future_line[left : right]

    #                             repl = substr.replace(vname, '"%s"' % varname)
    #                             repl = '"%s"' % eval(repl)
    #                             future_line = repl.join(future_line.split(substr))[4:]

    #                             #print("!", future_line, "?")
    #                             future_line = future_line.lstrip()
    #                             future_line = tabs0*' ' + future_line

    #                     # save edited lines
    #                     unravelled.append(future_line)

    #         # remove original iteration block
    #         for i in range(k, k + stop_idx + 1):
    #             src[i] = ''
    #         # add in all unravelled iterations to source
    #         src[k] = ''.join(unravelled)
