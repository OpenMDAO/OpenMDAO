"""Define the ExecComp class, a component that evaluates an expression."""
import math
import cmath
import re
from collections import OrderedDict
from itertools import product

import numpy
from numpy import ndarray, imag, complex as npcomplex

from six import string_types

from openmdao.core.component import ExplicitComponent

# regex to check for variable names.
VAR_RGX = re.compile('([_a-zA-Z]\w*(?::[_a-zA-Z]\w*)*[ ]*\(?)')

def array_idx_iter(shape):
    """Return an iterator over the indices into a n-dimensional array.

    Args
    ----
    shape : tuple
        shape of the array.
    """
    for p in product(*[range(s) for s in shape]):
        yield p

def _parse_for_vars(s):
    return set([x.strip() for x in re.findall(VAR_RGX, s)
                if not x.endswith('(') and x.strip() not in _expr_dict])

def _valid_name(s, exprs):
    """Replace colons with numbers such that the new name does not exist in any
    of the given expressions.
    """
    i = 0
    check = ' '.join(exprs)
    while True:
        n = s.replace(':', '%d'%i)
        if n not in check:
            return n
        i += 1

class ExecComp(ExplicitComponent):
    """
    Given a list of assignment statements, this component creates
    input and output variables at construction time.  All variables
    appearing on the left-hand side of an assignment are outputs,
    and the rest are inputs.  Each variable is assumed to be of
    type float unless the initial value for that variable is supplied
    in \*\*kwargs or inits.  Derivatives are calculated using complex step.

    Args
    ----
    exprs : str or list of str
        An assignment statement or iter of them. These express how the
        outputs are calculated based on the inputs.

    inits : dict, optional
        A mapping of names to initial values, primarily for variables with
        names that are not valid python names, e.g., a:b:c.

    units : dict, optional
        A mapping of variable names to their units.

    \*\*kwargs : dict of named args
        Initial values of variables can be set by setting a named
        arg with the var name.

    Options
    -------
    deriv_options['type'] :  str('user')
        Derivative calculation type ('user', 'fd', 'cs')
        Default is 'user', where derivative is calculated from
        user-supplied derivatives. Set to 'fd' to finite difference
        this system. Set to 'cs' to perform the complex step
        if your components support it.
    deriv_options['form'] :  str('forward')
        Finite difference mode. (forward, backward, central)
    deriv_options['step_size'] :  float(1e-06)
        Default finite difference stepsize
    deriv_options['step_calc'] :  str('absolute')
        Set to absolute, relative
    deriv_options['check_type'] :  str('fd')
        Type of derivative check for check_partial_derivatives. Set
        to 'fd' to finite difference this system. Set to
        'cs' to perform the complex step method if
        your components support it.
    deriv_options['check_form'] :  str('forward')
        Finite difference mode: ("forward", "backward", "central")
        During check_partial_derivatives, the difference form that is used
        for the check.
    deriv_options['check_step_calc'] : str('absolute',)
        Set to 'absolute' or 'relative'. Default finite difference
        step calculation for the finite difference check in check_partial_derivatives.
    deriv_options['check_step_size'] :  float(1e-06)
        Default finite difference stepsize for the finite difference check
        in check_partial_derivatives"
    deriv_options['linearize'] : bool(False)
        Set to True if you want linearize to be called even though you are using FD.

    Notes
    -----
    If a variable has an initial value that is anything other than 0.0,
    either because it has a different type than float or just because its
    initial value is nonzero, you must use a keyword arg or the 'inits' dict
    to set the initial value.  For example, let's say we have an ExecComp that
    takes an array 'x' as input and outputs a float variable 'y' which is the
    sum of the entries in 'x'.

    >>> import numpy
    >>> from openmdao.api import ExecComp
    >>> excomp = ExecComp('y=numpy.sum(x)', x=numpy.ones(10,dtype=float))

    In this example, 'y' would be assumed to be the default type of float
    and would be given the default initial value of 0.0, while 'x' would be
    initialized with a size 10 float array of ones.
    """

    def __init__(self, exprs, inits=None, units=None, **kwargs):
        super(ExecComp, self).__init__(**kwargs)

        # if complex step is used for derivatives, this is the stepsize
        self.complex_stepsize = 1.e-6

        if isinstance(exprs, string_types):
            exprs = [exprs]

        self._exprs = exprs[:]
        self._codes = None
        self._non_pbo_outputs = None
        self._to_colons = None
        self._from_colons = None
        self._colon_names = None

        outs = self._outs = set()
        self._allvars = allvars = set()

        # find all of the variables and which ones are outputs
        for expr in exprs:
            lhs, _ = expr.split('=', 1)
            outs.update(_parse_for_vars(lhs))
            allvars.update(_parse_for_vars(expr))

        if inits is not None:
            self.metadata.update(inits)

        # make sure all kwargs are legit
        for kwarg in kwargs:
            if kwarg not in allvars:
                raise RuntimeError("Arg '%s' in call to ExecComp() "
                                   "does not refer to any variable in the "
                                   "expressions %s" % (kwarg, exprs))

        # make sure units are legit
        self._units_dict = units_dict = units if units is not None else {}
        for unit_var in units_dict:
            if unit_var not in allvars:
                raise RuntimeError("Units specific for variable {0} "
                                   "in call to ExecComp() but {0} does "
                                   "not appear in the expression "
                                   "{1}".format(unit_var, exprs))

    def initialize_variables(self):
        """Add inputs and outputs based on the contents of our expression
        strings.
        """
        allvars = self._allvars
        kwargs = self.metadata
        units_dict = self._units_dict
        outs = self._outs
        exprs = self._exprs

        for var in sorted(allvars):
            # if user supplied an initial value, use it, otherwise set to 0.0
            val = kwargs.get(var, 0.0)
            new_kwargs = {'units':units_dict[var]} if var in units_dict else {}

            if var in outs:
                self.add_output(var, val, **new_kwargs)
            else:
                self.add_input(var, val, **new_kwargs)

        # need to exclude any non-pbo outputs (like case_rank in ExecComp4Test)
        # TODO: for now, assume all outputs are non-pbo
        self._non_pbo_outputs = self._variable_myproc_names['output']

        self._to_colons = {}
        from_colons = self._from_colons = {}
        for n in allvars:
            if ':' in n:
                no_colon = _valid_name(n, exprs)
            else:
                no_colon = n
            self._to_colons[no_colon] = n
            from_colons[n] = no_colon

        self._colon_names = {n for n in allvars if ':' in n}

        self._codes = self._compile_exprs(self._exprs)

    def _compile_exprs(self, exprs):
        for name in self._colon_names:
            exprs = [expr.replace(name, self._from_colons[name])
                     for expr in exprs]

        return [compile(expr, expr, 'exec') for expr in exprs]

    def __getstate__(self):
        """ Returns state as a dict. """
        state = self.__dict__.copy()
        del state['_codes']
        return state

    def __setstate__(self, state):
        """ Restore state from `state`. """
        self.__dict__.update(state)
        self._codes = self._compile_exprs(self._exprs)

    def compute(self, inputs, outputs):
        """
        Executes this component's assignment statemens.

        Args
        ----
        inputs : `Vector`
            `Vector` containing inputs.

        outputs : `Vector`
            `Vector` containing outputs.

        """
        for expr in self._codes:
            exec(expr, _expr_dict, _UPDict(outputs, inputs, self._to_colons))

    def linearize(self, params, unknowns, resids):
        """
        Uses complex step method to calculate a Jacobian dict.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        Returns
        -------
        dict
            Dictionary whose keys are tuples of the form ('unknown', 'param')
            and whose values are ndarrays.
        """

        # our complex step
        step = self.complex_stepsize * 1j

        J = OrderedDict()
        non_pbo_outputs = self._non_pbo_outputs

        for param in params:

            pwrap = _TmpDict(params)

            pval = params[param]
            if isinstance(pval, ndarray):
                # replace the param array with a complex copy
                pwrap[param] = numpy.asarray(pval, npcomplex)
                idx_iter = array_idx_iter(pwrap[param].shape)
                psize = pval.size
            else:
                pwrap[param] = npcomplex(pval)
                idx_iter = (None,)
                psize = 1

            for i, idx in enumerate(idx_iter):
                # set a complex param value
                if idx is None:
                    pwrap[param] += step
                else:
                    pwrap[param][idx] += step

                uwrap = _TmpDict(unknowns, return_complex=True)

                # solve with complex param value
                self._residuals.set_val(0.0)
                self.compute(pwrap, uwrap)

                for u in non_pbo_outputs:
                    jval = imag(uwrap[u] / self.complex_stepsize)
                    if (u, param) not in J: # create the dict entry
                        J[(u, param)] = numpy.zeros((jval.size, psize))

                    # set the column in the Jacobian entry
                    J[(u, param)][:, i] = jval.flat

                # restore old param value
                if idx is None:
                    pwrap[param] -= step
                else:
                    pwrap[param][idx] -= step

        return J


class _TmpDict(object):
    """
    A wrapper for a dictionary that will allow getting of values
    from its inner dict unless those values get modified via
    __setitem__.  After values have been modified they are managed
    thereafter by the wrapper.  This protects the inner dict from
    modification.

    Args
    ----
    inner : dict-like
        The dictionary to be wrapped.

    return_complex : bool, optional
        If True, return a complex version of values from __getitem__
    """
    def __init__(self, inner, return_complex=False):
        self._inner = inner
        self._changed = {}
        self._complex = return_complex

    def __getitem__(self, name):
        if name in self._changed:
            return self._changed[name]
        elif self._complex:
            val = self._inner[name]
            if isinstance(val, ndarray):
                self._changed[name] = numpy.asarray(val, dtype=npcomplex)
            else:
                self._changed[name] = npcomplex(val)
            return self._changed[name]
        else:
            return self._inner[name]

    def __setitem__(self, name, value):
        self._changed[name] = value

    def __contains__(self, name):
        return name in self._inner or name in self._changed

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _UPDict(object):
    """
    A dict-like wrapper for the outputs and inputs
    objects.  Items are first looked for in the outputs
    and then the inputs.

    Args
    ----
    outputs : dict-like
        The outputs object to be wrapped.

    inputs : dict-like
        The inputs object to be wrapped.
    """
    def __init__(self, outputs, inputs, to_colons):
        self._outputs = outputs
        self._inputs = inputs
        self._to_colons = to_colons

    def __getitem__(self, name):
        name = self._to_colons[name]
        try:
            return self._outputs[name]
        except KeyError:
            return self._inputs[name]

    def __setitem__(self, name, value):
        name = self._to_colons[name]
        if name in self._outputs:
            self._outputs[name] = value
        elif name in self._inputs:
            self._inputs[name] = value
        else:
            self._outputs[name] = value # will raise KeyError


def _import_functs(mod, dct, names=None):
    """
    Maps attributes attrs from the given module into the given dict.

    Args
    ----
    dct : dict
        Dictionary that will contain the mapping

    names : iter of str, optional
        If supplied, only map attrs that match the given names
    """
    if names is None:
        names = dir(mod)
    for name in names:
        if isinstance(name, tuple):
            name, alias = name
        else:
            alias = name
        if not name[0] == '_':
            dct[name] = getattr(mod, name)
            dct[alias] = dct[name]


# this dict will act as the local scope when we eval our expressions
_expr_dict = {}

# Note: no function in the math module supports complex args, so the following can only be used
#       in ExecComps if derivatives are not required.  The functions below don't have numpy
#       versions (which do support complex args), otherwise we'd just use those.  Some of these
#       will be overridden if scipy is found.
_import_functs(math, _expr_dict,
               names=['factorial', 'fsum', 'lgamma', 'erf', 'erfc', 'gamma'])

_import_functs(numpy, _expr_dict,
               names=['cosh', 'ldexp', 'hypot', 'tan', 'isnan', 'log', 'fabs',
                      'floor', 'sqrt', 'frexp', 'degrees', 'pi', 'log10', 'modf',
                      'copysign', 'cos', 'ceil', 'isinf', 'sinh', 'trunc',
                      'expm1', 'e', 'tanh', 'radians', 'sin', 'fmod', 'exp', 'log1p',
                      ('arcsin', 'asin'), ('arcsinh', 'asinh'), ('arctanh', 'atanh'),
                      ('arctan', 'atan'), ('arctan2', 'atan2'),
                      ('arccosh', 'acosh'), ('arccos', 'acos'),
                      ('power', 'pow')])

# Note: adding cmath here in case someone wants to have an ExecComp that
# performs some complex operation during solve_nonlinear. cmath functions
# generally return complex numbers even if the args are floats.
_expr_dict['cmath'] = cmath

_expr_dict['numpy'] = numpy


# if scipy is available, add some functions
try:
    import scipy.special
except ImportError:
    pass
else:
    _import_functs(scipy.special, _expr_dict,
                   names=['gamma', 'polygamma', 'erf', 'erfc'])


# Put any functions here that need special versions to work under
# complex step

def _cs_abs(x):
    if isinstance(x, ndarray):
        return x*numpy.sign(x)
    elif x.real < 0.0:
        return -x
    return x

_expr_dict['abs'] = _cs_abs
