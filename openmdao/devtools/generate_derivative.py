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

from openmdao.core.explicitcomponent import Component, ExplicitComponent
from openmdao.devtools.ast_tools import transform_ast_names, function_static_analysis


def generate_gradient(comp, src, func_name, mode, nofunc_src, to_replace):
    """
    Given the string representing the source code of a python function,
    use the tangent library to generate source code of its gradient.

    comp: Component instance
    src: str, the source code to be analyzed
    func_name: str, the name of the function

    Returns: the generated derivative function.
    """

    comp_mod = sys.modules[comp.__class__.__module__]
    namespace = comp_mod.__dict__.copy()
    namespace['self'] = comp

    # exec function body in namespace to define all of the variables

    header = ''.join(["%s = self._inputs['%s']\n" % (n.replace(':', '_'), n) for n in comp._inputs])
    header += ''.join(["%s = self._outputs['%s']\n" % (n.replace(':', '_'), n) for n in comp._outputs])
    header += ''.join(["resid__%s = self._vectors['residual']['linear']['%s']\n" %
                       (n.replace(':', '_'), n) for n in comp._vectors['residual']['linear']])
    header += ''.join(["_opt__%s = self.options['%s']\n" %
                       (n.replace(':', '_'), n) for n in comp.options])
    lines = header[:]
    lines += '\nif True:\n'  # body src is indented, so put in conditional block
    lines += nofunc_src

    pre = set(namespace)

    # print("exec'ing the following source:")
    # for l in lines.split('\n'):
    #     print(l)

    exec(lines, namespace)

    post = set(namespace)

    new_attrs = post - pre

    fdviz, f_ast = function_static_analysis(src)

    fmap = {}
    modules = set()

    # this allows us to substitute defined non-member functions with tangent adjoints/tangents
    # for member functions
    for s in fdviz.calls:
        if s.startswith('np.') or s.startswith('numpy.'):
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

    lines = []
    for modname in modules:
        lines.append('import %s' % modname)

    lines.append(src)

    new_src = '\n'.join(lines)

    ast2 = ast.parse(new_src)

    new_ast = transform_ast_names(ast2, fmap)
    new_src = astunparse.unparse(new_ast)

    # print("NEW SRC:")
    # for l in new_src.split('\n'):
    #     print(l)

    # needs to be read in as a proper module for tangent to work
    # in case we're doing this several times, need to reload
    # need to know the number of inputs to the loaded function

    temp_mod_name = '_temp_' + comp.pathname.replace('.', '_')
    temp_file_name = temp_mod_name + '.py'

    with open(temp_file_name, "w") as f:
        f.write(new_src)

    #invalidate_caches()
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

    func = getattr(mod, func_name)
    sig = signature(func)
    params = sig.parameters

    # generate AD code for the method for all inputs
    #print(func)
    df = tangent.autodiff(func, wrt=tuple(range(len(params))), mode=mode,
                          verbose=0, check_dims=False)

    # cleanup temp file and module
    remove(temp_file_name)
    del sys.modules[temp_mod_name]

    # return the generated source as a list of strings
    #print(''.join(src))
    return df


def generate_component_gradient(comp, mode):
    inputs = comp._inputs
    outputs = comp._outputs

    if isinstance(comp, ExplicitComponent):
        compute_method = comp.compute
    else:
        compute_method = comp.apply_nonlinear

    params = list(signature(compute_method).parameters)

    # get the name of the component type
    class_name = comp.__class__.__name__

    # get source code of original compute() method
    # ignore blank lines, useful for detecting indentation later
    src = [line for line in getsourcelines(compute_method)[0] if line.strip()]

    # lists of local names of inputs and outputs (and maybe resids) to code generate
    pnames = []
    onames = []
    rnames = []

    # mapping to rename variables within the compute method
    to_replace = {}
    to_revert = {}

    # gather transformed input and output names
    for pname in inputs:
        new_name = pname.replace(":", "_")
        pnames.append(new_name)
        to_replace["%s['%s']" % (params[0], pname)] = new_name
        to_replace['%s["%s"]' % (params[0], pname)] = new_name

    for oname in outputs:
        new_name = oname.replace(":", "_")
        onames.append(new_name)
        to_replace["%s['%s']" % (params[1], oname)] = new_name
        to_replace['%s["%s"]' % (params[1], oname)] = new_name

    for name in comp.options:
        to_replace["self.options['%s']" % name] = '_opt__%s' % name
        to_revert["self.options['%s']" % name] = '_opt__%s' % name

    if not isinstance(comp, ExplicitComponent):
        for rname in comp._vectors['residual']['linear']:
            new_name = 'resid__' + rname.replace(":", "_")
            rnames.append(new_name)
            to_replace["%s['%s']" % (params[2], rname)] = new_name
            to_replace['%s["%s"]' % (params[2], rname)] = new_name

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

    # replace all input references with local vars
    # e.g. change inputs['x'] to x and inputs['x:y'] to x_y
    # for pattern in to_replace:
    #     print("replacing '%s' with '%s'" % (pattern, to_replace[pattern]))
    #     src = src.replace(pattern, to_replace[pattern])

    # get rid of function indent.  astunparse will automatically indent the func body to 4 spaces
    src[0] = src[0].lstrip()

    src = ''.join(src)

    ast2 = ast.parse(src)
    new_ast = transform_ast_names(ast2, to_replace)
    src = astunparse.unparse(new_ast)
    indent = '    '

    # remove the original compute() call signature
    src = src.split(":", 1)[1]

    # start construction of partial derivative functions

    funcname = 'compute_ad'

    source = []

    # generate string of function to be analyzed by the tangent library
    source.append("def %s(%s):" % (funcname, ', '.join(pnames)))
    source.append(src)
    source.append("%sreturn %s" % (indent, ', '.join(onames + rnames)))
    source = '\n'.join(source)

    # print("Function source:")
    # print(source)

    # gather generated gradient source code
    df = generate_gradient(comp, source, funcname, mode, src, to_revert)
    dsrc = getsource(df)

    # now translate back the member vars we substituted for earlier
    d_ast = ast.parse(dsrc)
    # reverse the to_revert dict
    revert = {}
    for key, val in to_revert.items():
        revert[val] = key
    final_ast = transform_ast_names(d_ast, revert)
    dsrc = astunparse.unparse(final_ast)
    dlines = dsrc.split('\n')
    for i, line in enumerate(dlines):
        if line.startswith('def '):
            dlines[i] = _fix_func_def(line)
            break

    comp_mod = sys.modules[comp.__class__.__module__]
    namespace = comp_mod.__dict__.copy()
    namespace['tangent'] = tangent
    dsrc = '\n'.join(dlines)
    exec(dsrc, namespace)
    df = namespace['_compute_derivs_ad']

    return source, dsrc, df


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
    # parser.add_argument('-m', '--mode', default='forward', action='store', dest='mode',
    #                     help='AD mode (forward, reverse).')
    parser.add_argument('-c', '--class', action='store', dest='class_',
                        help='Specify component class to run AD on.')


def _do_grad_check(comp, mode, show_orig=True, out=sys.stdout):
    src, dsrc, df = generate_component_gradient(comp, mode)

    if show_orig:
        print("ORIG function:", file=out)
        print(src, file=out)

    print("%s grad function:" % mode, file=out)
    print(dsrc, file=out)

    prefix = comp.pathname + '.' if comp.pathname else ''

    J = {}
    inputs = comp._inputs
    outputs = comp._outputs

    if mode == 'forward':
        array = comp._vectors['input']['linear']._data
        vec = inputs
    else:  # reverse
        array = comp._vectors['output']['linear']._data
        vec = outputs

    idx2output = [None] * array.size  # map array index to output name
    idx2loc = np.zeros(array.size, dtype=int)
    views = []
    start = end = 0
    for n in vec:
        end += vec[n].size
        views.append(array[start:end])
        for i in range(start, end):
            idx2output[i] = n
        idx2loc[start:end] = np.arange(start, end, dtype=int) - start
        start = end

    if mode == 'forward':
        params = [inputs[name] for name in inputs] + views
        for idx in range(array.size):
            array[:] = 0.0
            array[idx] = 1.0
            iname = idx2output[idx]
            locidx = idx2loc[idx]
            abs_in = prefix + iname

            grad = df(comp, *params)
            for i, oname in enumerate(outputs):
                abs_out = prefix + oname
                key = (abs_out, abs_in)
                if key in comp._subjacs_info:
                    if key not in J:
                        J[key] = np.zeros((outputs[oname].size, inputs[iname].size))
                    J[key][:, locidx] = grad[i]

    else:  # reverse
        params = [inputs[name] for name in inputs] + [views]

        for oidx in range(array.size):
            array[:] = 0.0
            array[oidx] = 1.0
            oname = idx2output[oidx]
            locidx = idx2loc[oidx]
            abs_out = prefix + oname

            grad = df(comp, *params)
            for i, iname in enumerate(inputs):
                abs_in = prefix + iname
                key = (abs_out, abs_in)
                if key in comp._subjacs_info:
                    if key not in J:
                        J[key] = np.zeros((outputs[oname].size, inputs[iname].size))
                    J[key][locidx:] = grad[i]

    import pprint
    pprint.pprint(J, stream=out)


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

        _do_grad_check(s, 'forward', show_orig=True, out=out)
        _do_grad_check(s, 'reverse', show_orig=False, out=out)

        exit()

    return _ad


def _fix_func_def(line):
    # line is assumed to be of the form   def foo(a, b, c)
    # it will be converted to  def _compute_derivs_ad(self, a, b, c)
    parts = line.split('(', 1)

    return 'def _compute_derivs_ad(' + 'self, ' + parts[1]
