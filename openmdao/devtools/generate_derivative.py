from collections import Iterable

import sys
import tangent
from inspect import signature, getsourcelines, getmodule
import inspect
from os import remove
from importlib import reload
import types
import time
import numpy as np

from openmdao.core.explicitcomponent import Component, ExplicitComponent

def generate_gradient(src, method_name, mode):
    """
    Given the string representing the source code of a python function,
    use the tangent library to generate source code of its gradient.

    src: str, the source code to be analyzed
    method_name: str, the name of the function

    Returns: [str], list of strings of the generated derivative function.
    """

    # needs to be read in as a proper module for tangent to work
    # in case we're doing this several times, need to reload
    # need to know the number of parameters to the loaded function
    method = None
    while method == None:
        try:
            del sys.modules['_temp']
        except:
            pass
        try:
            with open("_temp.py", "w") as f:
                f.write(src)
            import _temp
            method = getattr(_temp, method_name)
        except AttributeError as e:
            #print(e, method_name)
            time.sleep(0.1)
            _temp = reload(_temp)

    sig = signature(method)
    params = sig.parameters

    # generate AD code for the method for each parameter
    #print(method)
    df = tangent.autodiff(method, wrt=range(len(params)), mode=mode,
                          # motion='split',
                          verbose=0, check_dims=False)
    src = getsourcelines(df)[0]

    # cleanup temp file
    #remove("_temp.py")
    # give it a more friendly method name
    longname = src[0][4:-2]

    src[0] = src[0].replace(longname, "d%0s" % method_name)

    # return the generated source as a list of strings
    #print(''.join(src))
    return src, df


def generate_gradient_code(comp, mode): #, inputs, outputs, local_vars, compute_method):
    inputs = comp._inputs
    outputs = comp._outputs

    if isinstance(comp, ExplicitComponent):
        compute_method = comp.compute
    else:
        compute_method = comp.apply_nonlinear

    # get the name of the component type

    # declared = []
    # for outp in comp._declared_partials:
    #     #print(outp)
    #     if isinstance(outp[1], str):
    #         declared.append((outp[0], outp[1]))
    #     else:
    #         for inp in outp[1]:
    #             declared.append((outp[0], inp))
    # comp._declared = declared
    comp_class = comp.__class__
    class_type = comp_class.__name__

    # AD code to be written out to file/augment original component
    final_code = []

    # start by generating code for a sub-class of the original component
    # this is useful in case the original component was general to the
    # point of having variable number if inputs/outputs determined on
    # instantiation. AD can only generate partial derivative code for
    # fixed-sized versions of these components.
    s = "import tangent\n"
    s += "class %sAD(%s):\n" % (class_type, class_type)
    final_code.append(s)
    s = "    def setup(self):\n"
    s += "        super(%sAD, self).setup()\n" % class_type
    s += "        self.declare_partials('*', '*')\n\n"
    final_code.append(s)

    # get source code of original compute() method
    # ignore blank lines, useful for detecting indentation later
    src = [line for line in getsourcelines(compute_method)[0] if line.strip()]

    # source code of generated AD per-output
    source_funcs = {}
    # lists of local names of inputs and outputs to code generate
    pnames = []
    onames = []
    # mapping to rename variables within the compute method
    to_replace = {}
    # for lname in local_vars:
    #     to_replace[lname] = str(local_vars[lname])
    output_name_map = {}
    # gather transformed input and output names
    for pname in inputs:
        new_name = pname.replace(":", "_")
        pnames.append(new_name)
        to_replace["inputs['%s']" % pname] = new_name
        to_replace['inputs["%s"]' % pname] = new_name

    for oname in outputs:
        new_name = oname.replace(":", "_")
        onames.append(new_name)
        to_replace["outputs['%s']" % oname] = new_name
        to_replace['outputs["%s"]' % oname] = new_name
        output_name_map[new_name] = oname

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

    # remove the original compute() call signature
    src = ''.join(src).split(":", 1)[1]

    # replace all input references with local vars
    # e.g. change inputs['x'] to x and inputs['x:y'] to x_y
    for pattern in to_replace:
        src = src.replace(pattern, to_replace[pattern])

    # start construction of partial derivative functions
    indent = "       "
    s = "    def compute_partials(self, inputs, J):\n"
    final_code.append(s)

    # unpack all inputs at top of generated compute_partials()
    for i, name in enumerate(inputs):
        s = indent + " %s = inputs['%s']\n" % (pnames[i], name)
        final_code.append(s)
    final_code.append("\n")

    # now for each output, generate gradient functions and partial
    # derivative assignments
    gradient_funcs = {}
    fname = 'compute_outputs'
    #for oname in onames:
    # generate string of function to be analyzed by the tangent library
    source = "import numpy as np\ndef %s(" % fname
    source += ", ".join(pnames) + "):\n"
    source += src + "\n"
    outs = str(tuple(outputs))
    source += "%s return %s\n" % (indent, outs.replace("'", ""))

    # gather generated gradient source code
    dsrc, df = generate_gradient(source, fname, mode)
    gradient_funcs[oname] = df
    return source, dsrc, df, output_name_map


def _vec2name_map(vec):
    """
    Convert invalid python names to valid ones and return a dict that maps invalid to valid.
    """
    name_map = {}
    for name in vec:
        new_name = name.replace(":", "_")
        if new_name != name:
            new_name = '_%s__' % new_name
        name_map[name] = new_name
    return name_map


def _vec2args(vec):
    """Return a list of values from vec."""
    return [vec[k] for k in vec]


def _gen_ad_deriv(comp, mode, stream=sys.stdout):
    return generate_gradient_code(comp, mode)


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
    src, dsrc, df, foutput_name_map = generate_gradient_code(comp, mode)

    if show_orig:
        print("ORIG function:", file=out)
        print(''.join(src), file=out)

    print("%s grad function:" % mode, file=out)
    print(''.join(getsourcelines(df)[0]), file=out)

    prefix = comp.pathname + '.' if comp.pathname else ''

    J = {}
    inputs = comp._inputs
    outputs = comp._outputs

    if mode == 'forward':
        array = comp._vectors['output']['linear']._data
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

            grad = df(*params)
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

            grad = df(*params)
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
                inst = s
                break
        else:
            raise RuntimeError("Couldn't find an instance of class '%s'." % options.class_)

        _do_grad_check(s, 'forward', show_orig=True, out=out)
        _do_grad_check(s, 'reverse', show_orig=False, out=out)

        exit()

    return _ad
