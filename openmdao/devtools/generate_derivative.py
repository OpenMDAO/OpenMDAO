from collections import Iterable

import tangent
from inspect import signature, getsourcelines, getmodule
import inspect
from os import remove
from importlib import reload
import types
import time
from openmdao.core.explicitcomponent import ExplicitComponent

def generate_gradient(src, method_name):
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
    df = tangent.autodiff(method, wrt=range(len(params)), verbose=0, check_dims=False)
    src = getsourcelines(df)[0]
    # cleanup temp file
    #remove("_temp.py")
    # give it a more friendly method name
    longname = src[0][4:-2]
    src[0] = src[0].replace(longname, "d%0s" % method_name)
    # return the generated source as a list of strings
    return src, df


def GenerateDerivative(compute_method):
    """
    - compute method needs call signature compute(self, inputs, outputs)
    
    """

    local_vars = {}
    for frame in inspect.stack():
        lvars = frame[0].f_locals
        for varname in lvars:
            var = lvars[varname]
            if isinstance(var, float) or isinstance(var, int):
                local_vars[varname] = var

    def generate_gradient_code(comp_instance, inputs, outputs):
        # get the name of the component type
        
        if hasattr(comp_instance, 'tangent_ad_generated'):
            return compute_method(comp_instance, inputs, outputs)

        declared = []
        for outp in comp_instance._declared_partials:
            #print(outp)
            if isinstance(outp[1], str):
                declared.append((outp[0], outp[1]))
            else:
                for inp in outp[1]:
                    declared.append((outp[0], inp))
        comp_instance._declared = declared
        comp_class = comp_instance.__class__
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
        src = getsourcelines(compute_method)[0]
        # ignore blank lines, useful for detecting indentation later
        src = [line for line in src if line.split()]

        # source code of generated AD per-output
        source_funcs = {}
        # lists of local names of inputs and outputs to code generate
        pnames = []
        onames = []
        # mapping to rename variables within the compute method
        to_replace = {}
        for lname in local_vars:
            to_replace[lname] = str(local_vars[lname])
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
        comp_instance.output_name_map = output_name_map

        # find any blocks that parameterize by iterating over metadata
        # these need to be explicitly flattened
        for k, line in enumerate(src):
            """
            For example, the lines:
                for BN in self.metadata['bleed_names']:
                    outputs['ht_out'] += inputs[BN + ':W']
            will automatically be transformed into something like
                b1_W = inputs['b1:W']
                b2_W = inputs['b2:W']
                outputs['ht_out'] += b1_W
                outputs['ht_out'] += b2_W
            if the iteration is over empty metadata, the block is omitted
            """
            # key assumption: iteration over metadata corresponds to 
            # variable-length input or output   
            tabs0 = len(line) - len(line.lstrip())
            if 'in self.metadata' in line:
                # determine the iteration variable name
                # e.g. 'BN' in the example above
                vname = line.split().index("for")
                vname = line.split()[vname + 1]

                # allows us to break out based on indentation
                block_ind = len(line) - len(line.lstrip())

                # actually execute the block to get the metadata attribute
                # this will expose how many vars are present for this instance
                exc = line.replace(":", "").replace("self", "comp_instance")
                exc = ("[%s " % vname) + exc + "]"
                # this would be ['b1', 'b2'], etc.
                varlist = eval(exc)

                # code lines for flattened iteration
                unravelled = []
                # line number of end of iteration block
                stop_idx = k
                # if there are no variables to iterate over, just get the end 
                # line number of the block to remove it
                if not varlist:
                    for k2, future_line in enumerate(src[k+1:]):
                        if len(future_line) - len(future_line.lstrip()) == block_ind:
                            stop_idx = k2
                            break
                # otherwise, pass over the block for each variable value to
                # re-write it
                else:
                    for varname in varlist:
                        # search future lines for end of block
                        for k2, future_line in enumerate(src[k+1:]):
                            if len(future_line) - len(future_line.lstrip()) == block_ind:
                                stop_idx = k2
                                break
                            elif vname in future_line:
                                # search and replace each instance in each line
                                # e.g. find instances of things that look like
                                # inputs[BN + ':W']
                                # and replace with inputs['b1:W'], inputs['b2:W'], etc.
                                # these will get replaced with local vars b1_W, b2_W later
                                while True:
                                    try:
                                        i = future_line.index(vname)
                                    except ValueError:
                                        break
                                    left = future_line[:i][::-1].index("[")
                                    right = future_line[i + 1:].index("]")

                                    left = i - left
                                    right = i + right + 1
                                    
                                    substr = future_line[left : right]

                                    repl = substr.replace(vname, '"%s"' % varname)
                                    repl = '"%s"' % eval(repl)
                                    future_line = repl.join(future_line.split(substr))[4:]

                                    #print("!", future_line, "?")
                                    future_line = future_line.lstrip()
                                    future_line = tabs0*' ' + future_line

                            # save edited lines
                            unravelled.append(future_line)

                # remove original iteration block
                for i in range(k, k + stop_idx + 1):
                    src[i] = ''
                # add in all unravelled iterations to source
                src[k] = ''.join(unravelled)
        
        # remove the original compute() call signature
        src = ''.join(src).split(":")[1:]
        src = ':'.join(src)

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
        for oname in onames:
            # generate string of function to be analyzed by the tangent library
            dname = "d" + oname
            source = "import numpy as np\ndef %s(" % oname
            source += ", ".join(pnames) + "):\n"
            source += src + "\n"
            source += "%s return %s\n" % (indent, oname)
            
            # gather generated gradient source code
            dsrc, df = generate_gradient(source, oname)
            gradient_funcs[oname] = df

        # add gradient calc funcs as instance attribute
        comp_instance.gradient_funcs = gradient_funcs

        # now create a replace compute_partials
        def replace_compute_partials(self, inputs, J):
            params = [inputs[name] for name in inputs] + [1.0]
            for oname in self.output_name_map:
                var = self.output_name_map[oname]
                grad = self.gradient_funcs[oname](*params)
                for i, pname in enumerate(inputs):
                    #print()
                    #print(var, pname, self._declared)
                    if ("*", "*") in self._declared or (var, "*") in self._declared or (var, pname) in self._declared:
                        #print(var, pname)
                        J[var, pname] = grad[i]

        # inject new check_partials method
        comp_instance.compute_partials = types.MethodType(replace_compute_partials, 
                                                          comp_instance)
        #comp_instance.compute_partials = replace_compute_partials

        # mark instance as having AD generated so it isn't re-computed
        comp_instance.tangent_ad_generated = True

        # go ahead and run original compute
        return compute_method(comp_instance, inputs, outputs)

    return generate_gradient_code