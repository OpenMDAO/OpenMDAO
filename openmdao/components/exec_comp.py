"""Define the ExecComp class, a component that evaluates an expression."""
import math
import cmath
import re
from collections import OrderedDict
from itertools import product

import numpy as np
from numpy import ndarray, imag, complex as npcomplex

from six import string_types
from six.moves import range

from openmdao.core.explicitcomponent import ExplicitComponent

# regex to check for variable names.
VAR_RGX = re.compile('([.]*[_a-zA-Z]\w*[ ]*\(?)')

# Names of metadata entries allowed for ExecComp variables.
_allowed_meta = {'value', 'shape', 'units', 'res_units', 'desc', 'var_set',
                 'ref', 'ref0', 'res_ref', 'lower', 'upper', 'src_indices',
                 'flat_src_indices'}


def array_idx_iter(shape):
    """
    Return an iterator over the indices into a n-dimensional array.

    Parameters
    ----------
    shape : tuple
        shape of the array.
    """
    for p in product(*[range(s) for s in shape]):
        yield p


class ExecComp(ExplicitComponent):
    """
    A component defined by an expression string.
    """

    def __init__(self, exprs, **kwargs):
        r"""
        Create a <Component> using only an expression string.

        Given a list of assignment statements, this component creates
        input and output variables at construction time.  All variables
        appearing on the left-hand side of an assignment are outputs,
        and the rest are inputs.  Each variable is assumed to be of
        type float unless the initial value for that variable is supplied
        in \*\*kwargs.  Derivatives are calculated using complex step.

        Parameters
        ----------
        exprs : str, tuple of str or list of str
            An assignment statement or iter of them. These express how the
            outputs are calculated based on the inputs.

        \*\*kwargs : dict of named args
            Initial values of variables can be set by setting a named
            arg with the var name.  If the value is a dict it is assumed
            to contain metadata.  To set the initial value in addition to
            other metadata, assign the initial value to the 'value' entry
            of the dict.

        Notes
        -----
        If a variable has an initial value that is anything other than 0.0,
        either because it has a different type than float or just because its
        initial value is nonzero, you must use a keyword arg
        to set the initial value.  For example, let's say we have an
        ExecComp that takes an array 'x' as input and outputs a float variable
        'y' which is the sum of the entries in 'x'.

        ::

            import numpy
            from openmdao.api import ExecComp
            excomp = ExecComp('y=numpy.sum(x)', x=numpy.ones(10,dtype=float))

        In this example, 'y' would be assumed to be the default type of float
        and would be given the default initial value of 0.0, while 'x' would be
        initialized with a size 10 float array of ones.

        If you want to assign certain metadata for 'x' in addition to its
        initial value, you can do it as follows:

        ::

            excomp = ExecComp('y=numpy.sum(x)',
                              x={'value': numpy.ones(10,dtype=float),
                                 'units': 'ft',
                                 'var_set': 3})
        """
        super(ExecComp, self).__init__()

        # if complex step is used for derivatives, this is the stepsize
        self.complex_stepsize = 1.e-6

        if isinstance(exprs, string_types):
            exprs = [exprs]

        self._exprs = exprs[:]
        self._codes = None
        self._kwargs = kwargs

    def setup(self):
        """
        Set up variable name and metadata lists.
        """
        outs = set()
        allvars = set()
        exprs = self._exprs
        kwargs = self._kwargs

        # find all of the variables and which ones are outputs
        for expr in exprs:
            lhs, _ = expr.split('=', 1)
            outs.update(self._parse_for_out_vars(lhs))
            allvars.update(self._parse_for_vars(expr))

        kwargs2 = {}
        init_vals = {}

        # make sure all kwargs are legit
        for arg, val in kwargs.items():
            if arg not in allvars:
                raise RuntimeError("%s: arg '%s' in call to ExecComp() "
                                   "does not refer to any variable in the "
                                   "expressions %s" % (self.pathname,
                                                       arg, exprs))
            if isinstance(val, dict):
                diff = set(val.keys()) - _allowed_meta
                if diff:
                    raise RuntimeError("%s: the following metadata names were not "
                                       "recognized for variable '%s': %s" %
                                       (self.pathname, arg, sorted(diff)))

                kwargs2[arg] = val.copy()
                if 'value' in val:
                    init_vals[arg] = val['value']
                    del kwargs2[arg]['value']
            else:
                init_vals[arg] = val

        for var in sorted(allvars):
            # if user supplied an initial value, use it, otherwise set to 0.0
            val = init_vals.get(var, 0.0)
            meta = kwargs2.get(var, {})

            if var in outs:
                self.add_output(var, val, **meta)
            else:
                self.add_input(var, val, **meta)

        self._codes = self._compile_exprs(self._exprs)

        # All derivatives are defined.
        self.declare_partials(of='*', wrt='*')

    def _compile_exprs(self, exprs):
        compiled = []
        for i, expr in enumerate(exprs):
            try:
                compiled.append(compile(expr, expr, 'exec'))
            except Exception:
                raise RuntimeError("%s: failed to compile expression '%s'." %
                                   (self.pathname, exprs[i]))
        return compiled

    def _parse_for_out_vars(self, s):
        vnames = set([x.strip() for x in re.findall(VAR_RGX, s)
                      if not x.endswith('(') and not x.startswith('.')])
        for v in vnames:
            if v in _expr_dict:
                raise NameError("%s: cannot assign to variable '%s' "
                                "because it's already defined as an internal "
                                "function or constant." % (self.pathname, v))
        return vnames

    def _parse_for_vars(self, s):
        vnames = set([x.strip() for x in re.findall(VAR_RGX, s)
                      if not x.endswith('(') and not x.startswith('.')])
        to_remove = []
        for v in vnames:
            if v in _expr_dict:
                expvar = _expr_dict[v]
                if callable(expvar):
                    raise NameError("%s: cannot use '%s' as a variable because "
                                    "it's already defined as an internal "
                                    "function." % (self.pathname, v))
                else:
                    to_remove.append(v)
        return vnames.difference(to_remove)

    def __getstate__(self):
        """
        Return state as a dict.
        """
        state = self.__dict__.copy()
        del state['_codes']
        return state

    def __setstate__(self, state):
        """
        Restore state from `state`.
        """
        self.__dict__.update(state)
        self._codes = self._compile_exprs(self._exprs)

    def compute(self, inputs, outputs):
        """
        Execute this component's assignment statements.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.

        outputs : `Vector`
            `Vector` containing outputs.
        """
        for expr in self._codes:
            exec(expr, _expr_dict, _IODict(outputs, inputs))

    def compute_partials(self, inputs, partials):
        """
        Use complex step method to update the given Jacobian.

        Parameters
        ----------
        inputs : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        partials : `Jacobian`
            Contains sub-jacobians.
        """
        # our complex step
        step = self.complex_stepsize * 1j
        out_names = self._var_allprocs_prom2abs_list['output']

        for param in inputs:

            pwrap = _TmpDict(inputs)

            pval = inputs[param]
            if isinstance(pval, ndarray):
                # replace the param array with a complex copy
                pwrap[param] = np.asarray(pval, npcomplex)
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

                uwrap = _TmpDict(self._outputs, return_complex=True)

                # solve with complex param value
                self._residuals.set_const(0.0)
                self.compute(pwrap, uwrap)

                for u in out_names:
                    jval = imag(uwrap[u] / self.complex_stepsize)
                    if (u, param) not in partials:  # create the dict entry
                        partials[(u, param)] = np.zeros((jval.size, psize))

                    # set the column in the Jacobian entry
                    partials[(u, param)][:, i] = jval.flat

                # restore old param value
                if idx is None:
                    pwrap[param] -= step
                else:
                    pwrap[param][idx] -= step


class _TmpDict(object):
    """
    Dict wrapper that allows modification without changing the wrapped dict.

    It will allow getting of values
    from its inner dict unless those values get modified via
    __setitem__.  After values have been modified they are managed
    thereafter by the wrapper.  This protects the inner dict from
    modification.
    """

    def __init__(self, inner, return_complex=False):
        """
        Construct the dictionary object.

        Parameters
        ----------
        inner : dict-like
            The dictionary to be wrapped.

        return_complex : bool, optional
            If True, return a complex version of values from __getitem__
        """
        self._inner = inner
        self._changed = {}
        self._complex = return_complex

    def __getitem__(self, name):
        if name in self._changed:
            return self._changed[name]
        elif self._complex:
            val = self._inner[name]
            if isinstance(val, ndarray):
                self._changed[name] = np.asarray(val, dtype=npcomplex)
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


class _IODict(object):
    """
    A dict wrapper that contains 2 different dicts.

    Items are first looked for in the outputs
    and then the inputs.
    """

    def __init__(self, outputs, inputs):
        """
        Create the dict wrapper.

        Parameters
        ----------
        outputs : dict-like
            The outputs object to be wrapped.

        inputs : dict-like
            The inputs object to be wrapped.
        """
        self._outputs = outputs
        self._inputs = inputs

    def __getitem__(self, name):
        if name in self._outputs:
            return self._outputs[name]
        else:
            return self._inputs[name]

    def __setitem__(self, name, value):
        if name in self._outputs:
            self._outputs[name] = value
        elif name in self._inputs:
            self._inputs[name] = value
        else:
            self._outputs[name] = value  # will raise KeyError

    def __contains__(self, name):
        return name in self._outputs or name in self._inputs


def _import_functs(mod, dct, names=None):
    """
    Map attributes attrs from the given module into the given dict.

    Parameters
    ----------
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

# Note: no function in the math module supports complex args, so the following
# can only be used in ExecComps if derivatives are not required.  The functions
# below don't have numpy versions (which do support complex args), otherwise
# we'd just use those.  Some of these will be overridden if scipy is found.
_import_functs(math, _expr_dict,
               names=['factorial', 'fsum', 'lgamma', 'erf', 'erfc', 'gamma'])

_import_functs(np, _expr_dict,
               names=['cosh', 'ldexp', 'hypot', 'tan', 'isnan', 'log', 'fabs',
                      'floor', 'sqrt', 'frexp', 'degrees', 'pi', 'log10',
                      'modf', 'copysign', 'cos', 'ceil', 'isinf', 'sinh',
                      'trunc', 'expm1', 'e', 'tanh', 'radians', 'sin', 'fmod',
                      'exp', 'log1p', ('arcsin', 'asin'), ('arcsinh', 'asinh'),
                      ('arctanh', 'atanh'), ('arctan', 'atan'),
                      ('arctan2', 'atan2'), ('arccosh', 'acosh'),
                      ('arccos', 'acos'), ('power', 'pow')])

# Note: adding cmath here in case someone wants to have an ExecComp that
# performs some complex operation during solve_nonlinear. cmath functions
# generally return complex numbers even if the args are floats.
_expr_dict['cmath'] = cmath

_expr_dict['numpy'] = np


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
        return x * np.sign(x)
    elif x.real < 0.0:
        return -x
    return x


_expr_dict['abs'] = _cs_abs
