"""Define the ExecComp class, a component that evaluates an expression."""
import re
from itertools import product

import numpy as np
from numpy import ndarray, imag, complex as npcomplex

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.units import valid_units
from openmdao.utils.general_utils import warn_deprecation

# regex to check for variable names.
VAR_RGX = re.compile(r'([.]*[_a-zA-Z]\w*[ ]*\(?)')

# Names of metadata entries allowed for ExecComp variables.
_allowed_meta = {'value', 'shape', 'units', 'res_units', 'desc',
                 'ref', 'ref0', 'res_ref', 'lower', 'upper', 'src_indices',
                 'flat_src_indices', 'tags'}

# Names that are not allowed for input or output variables (keywords for options)
_disallowed_names = {'has_diag_partials', 'units', 'shape'}


def check_option(option, value):
    """
    Check option for validity.

    Parameters
    ----------
    option : str
        The name of the option
    value : any
        The value of the option

    Raises
    ------
    ValueError
    """
    if option == 'units' and value is not None and not valid_units(value):
        raise ValueError("The units '%s' are invalid." % value)


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

    Attributes
    ----------
    _kwargs : dict of named args
        Initial values of variables.
    _exprs : list
        List of expressions.
    _codes : list
        List of code objects.
    _has_diag_partials : bool
        If True, treat all array/array partials as diagonal if both arrays have size > 1.
        All arrays with size > 1 must have the same flattened size or an exception will be raised.
    _units : str or None
        Units to be assigned to all variables in this component.
        Default is None, which means units are provided for variables individually.
    complex_stepsize : double
        Step size used for complex step which is used for derivatives.

    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('has_diag_partials', types=bool, default=False,
                             desc='If True, treat all array/array partials as diagonal if both '
                                  'arrays have size > 1. All arrays with size > 1 must have the '
                                  'same flattened size or an exception will be raised.')

        self.options.declare('units', types=str, allow_none=True, default=None,
                             desc='Units to be assigned to all variables in this component. '
                                  'Default is None, which means units are provided for variables '
                                  'individually.',
                             check_valid=check_option)

        self.options.declare('shape', types=(int, tuple, list), allow_none=True, default=None,
                             desc='Shape to be assigned to all variables in this component. '
                                  'Default is None, which means shape is provided for variables '
                                  'individually.')

    def __init__(self, exprs=[], **kwargs):
        r"""
        Create a <Component> using only an expression string.

        Given a list of assignment statements, this component creates
        input and output variables at construction time.  All variables
        appearing on the left-hand side of an assignment are outputs,
        and the rest are inputs.  Each variable is assumed to be of
        type float unless the initial value for that variable is supplied
        in \*\*kwargs.  Derivatives are calculated using complex step.

        The following functions are available for use in expressions:

        =========================  ====================================
        Function                   Description
        =========================  ====================================
        abs(x)                     Absolute value of x
        acos(x)                    Inverse cosine of x
        acosh(x)                   Inverse hyperbolic cosine of x
        arange(start, stop, step)  Array creation
        arccos(x)                  Inverse cosine of x
        arccosh(x)                 Inverse hyperbolic cosine of x
        arcsin(x)                  Inverse sine of x
        arcsinh(x)                 Inverse hyperbolic sine of x
        arctan(x)                  Inverse tangent of x
        asin(x)                    Inverse sine of x
        asinh(x)                   Inverse hyperbolic sine of x
        atan(x)                    Inverse tangent of x
        cos(x)                     Cosine of x
        cosh(x)                    Hyperbolic cosine of x
        dot(x, y)                  Dot product of x and y
        e                          Euler's number
        erf(x)                     Error function
        erfc(x)                    Complementary error function
        exp(x)                     Exponential function
        expm1(x)                   exp(x) - 1
        factorial(x)               Factorial of all numbers in x
                                   (DEPRECATED, not available with SciPy >=1.5)
        fmax(x, y)                 Element-wise maximum of x and y
        fmin(x, y)                 Element-wise minimum of x and y
        inner(x, y)                Inner product of arrays x and y
        isinf(x)                   Element-wise detection of np.inf
        isnan(x)                   Element-wise detection of np.nan
        kron(x, y)                 Kronecker product of arrays x and y
        linspace(x, y, N)          Numpy linear spaced array creation
        log(x)                     Natural logarithm of x
        log10(x)                   Base-10 logarithm of x
        log1p(x)                   log(1+x)
        matmul(x, y)               Matrix multiplication of x and y
        maximum(x, y)              Element-wise maximum of x and y
        minimum(x, y)              Element-wise minimum of x and y
        ones(N)                    Create an array of ones
        outer(x, y)                Outer product of x and y
        pi                         Pi
        power(x, y)                Element-wise x**y
        prod(x)                    The product of all elements in x
        sin(x)                     Sine of x
        sinh(x)                    Hyperbolic sine of x
        sum(x)                     The sum of all elements in x
        tan(x)                     Tangent of x
        tanh(x)                    Hyperbolic tangent of x
        tensordot(x, y)            Tensor dot product of x and y
        zeros(N)                   Create an array of zeros
        =========================  ====================================

        Parameters
        ----------
        exprs : str, tuple of str or list of str
            An assignment statement or iter of them. These express how the
            outputs are calculated based on the inputs.  In addition to
            standard Python operators, a subset of numpy and scipy functions
            is supported.

        **kwargs : dict of named args
            Initial values of variables can be set by setting a named
            arg with the var name.  If the value is a dict it is assumed
            to contain metadata.  To set the initial value in addition to
            other metadata, assign the initial value to the 'value' entry
            of the dict.

        Notes
        -----
        If a variable has an initial value that is anything other than 1.0,
        either because it has a different type than float or just because its
        initial value is != 1.0, you must use a keyword arg
        to set the initial value.  For example, let's say we have an
        ExecComp that takes an array 'x' as input and outputs a float variable
        'y' which is the sum of the entries in 'x'.

        .. code-block:: python

            import numpy
            import openmdao.api as om
            excomp = om.ExecComp('y=sum(x)', x=numpy.ones(10,dtype=float))

        In this example, 'y' would be assumed to be the default type of float
        and would be given the default initial value of 1.0, while 'x' would be
        initialized with a size 10 float array of ones.

        If you want to assign certain metadata for 'x' in addition to its
        initial value, you can do it as follows:

        .. code-block:: python

            excomp = ExecComp('y=sum(x)',
                              x={'value': numpy.ones(10,dtype=float),
                                 'units': 'ft'})
        """
        options = {}
        for name in _disallowed_names:
            if name in kwargs:
                options[name] = kwargs.pop(name)

        super(ExecComp, self).__init__(**options)

        # if complex step is used for derivatives, this is the stepsize
        self.complex_stepsize = 1.e-40

        if isinstance(exprs, str):
            exprs = [exprs]

        self._exprs = exprs[:]
        self._codes = None
        self._kwargs = kwargs

    def setup(self):
        """
        Set up variable name and metadata lists.
        """
        if not self._exprs:
            raise RuntimeError("%s: No valid expressions provided to ExecComp(): %s."
                               % (self.msginfo, self._exprs))
        outs = set()
        allvars = set()
        exprs = self._exprs
        kwargs = self._kwargs

        units = self.options['units']
        shape = self.options['shape']

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
                                   "expressions %s" % (self.msginfo,
                                                       arg, exprs))
            if isinstance(val, dict):
                diff = set(val.keys()) - _allowed_meta
                if diff:
                    raise RuntimeError("%s: the following metadata names were not "
                                       "recognized for variable '%s': %s" %
                                       (self.msginfo, arg, sorted(diff)))

                kwargs2[arg] = val.copy()

                if units is not None:
                    if 'units' in val and val['units'] != units:
                        raise RuntimeError("%s: units of '%s' have been specified for "
                                           "variable '%s', but units of '%s' have been "
                                           "specified for the entire component." %
                                           (self.msginfo, val['units'], arg, units))
                    else:
                        kwargs2[arg]['units'] = units

                if shape is not None:
                    if 'shape' in val and val['shape'] != shape:
                        raise RuntimeError("%s: shape of %s has been specified for "
                                           "variable '%s', but shape of %s has been "
                                           "specified for the entire component." %
                                           (self.msginfo, val['shape'], arg, shape))
                    elif 'value' in val and np.atleast_1d(val['value']).shape != shape:
                        raise RuntimeError("%s: value of shape %s has been specified for "
                                           "variable '%s', but shape of %s has been "
                                           "specified for the entire component." %
                                           (self.msginfo, np.atleast_1d(val['value']).shape,
                                            arg, shape))
                    else:
                        init_vals[arg] = np.ones(shape)

                if 'value' in val:
                    init_vals[arg] = val['value']
                    del kwargs2[arg]['value']

                if 'shape' in val:
                    if arg not in init_vals:
                        init_vals[arg] = np.ones(val['shape'])
                    elif np.atleast_1d(init_vals[arg]).shape != val['shape']:
                        raise RuntimeError("%s: shape of %s has been specified for variable "
                                           "'%s', but a value of shape %s has been provided." %
                                           (self.msginfo, str(val['shape']), arg,
                                            str(np.atleast_1d(init_vals[arg]).shape)))
                    del kwargs2[arg]['shape']
            else:
                init_vals[arg] = val

        for var in sorted(allvars):
            # if user supplied an initial value, use it, otherwise set to 1.0
            if var in init_vals:
                val = init_vals[var]
            else:
                init_vals[var] = val = 1.0

            meta = kwargs2.get(var, {'units': units, 'shape': shape})

            if var in outs:
                self.add_output(var, val, **meta)
            else:
                self.add_input(var, val, **meta)

        if self.options['has_diag_partials']:
            # check that sizes of any input/output vars match or one of them is size 1
            osorted = sorted(self._var_rel_names['output'])
            for inp in sorted(self._var_rel_names['input']):
                ival = init_vals[inp]
                iarray = isinstance(ival, ndarray) and ival.size > 1
                for out in osorted:
                    oval = init_vals[out]
                    if iarray and isinstance(oval, ndarray) and oval.size > 1:
                        if oval.size != ival.size:
                            raise RuntimeError("%s: has_diag_partials is True but partial(%s, %s) "
                                               "is not square (shape=(%d, %d))." %
                                               (self.msginfo, out, inp, oval.size, ival.size))
                        # partial will be declared as diagonal
                        inds = np.arange(oval.size, dtype=int)
                    else:
                        inds = None
                    self.declare_partials(of=out, wrt=inp, rows=inds, cols=inds)
        else:
            # All derivatives are defined as dense
            self.declare_partials(of='*', wrt='*')

        self._codes = self._compile_exprs(self._exprs)

    def _compile_exprs(self, exprs):
        compiled = []
        for i, expr in enumerate(exprs):
            try:
                compiled.append(compile(expr, expr, 'exec'))
            except Exception:
                raise RuntimeError("%s: failed to compile expression '%s'." %
                                   (self.msginfo, exprs[i]))
        return compiled

    def _parse_for_out_vars(self, s):
        vnames = set([x.strip() for x in re.findall(VAR_RGX, s)
                      if not x.endswith('(') and not x.startswith('.')])
        for v in vnames:
            if v in _expr_dict:
                raise NameError("%s: cannot assign to variable '%s' "
                                "because it's already defined as an internal "
                                "function or constant." % (self.msginfo, v))
        return vnames

    def _parse_for_vars(self, s):
        vnames = set([x.strip() for x in re.findall(VAR_RGX, s)
                      if not x.endswith('(') and not x.startswith('.')])
        to_remove = []
        for v in vnames:
            if v in _disallowed_names:
                raise NameError("%s: cannot use variable name '%s' because "
                                "it's a reserved keyword." % (self.msginfo, v))
            if v in _expr_dict:
                expvar = _expr_dict[v]
                if callable(expvar):
                    raise NameError("%s: cannot use '%s' as a variable because "
                                    "it's already defined as an internal "
                                    "function." % (self.msginfo, v))
                else:
                    to_remove.append(v)
        return vnames.difference(to_remove)

    def __getstate__(self):
        """
        Return state as a dict.

        Returns
        -------
        dict
            State to get.
        """
        state = self.__dict__.copy()
        del state['_codes']
        return state

    def __setstate__(self, state):
        """
        Restore state from `state`.

        Parameters
        ----------
        state : dict
            State to restore.
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
        for i, expr in enumerate(self._codes):
            try:
                exec(expr, _expr_dict, _IODict(outputs, inputs))
            except Exception as err:
                raise RuntimeError("%s: Error occurred evaluating '%s'\n%s"
                                   % (self.msginfo, self._exprs[i], str(err)))

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
        step = self.complex_stepsize * 1j
        out_names = self._var_rel_names['output']
        inv_stepsize = 1.0 / self.complex_stepsize
        has_diag_partials = self.options['has_diag_partials']

        for param in inputs:

            pwrap = _TmpDict(inputs)
            pval = inputs[param]
            psize = pval.size
            pwrap[param] = np.asarray(pval, npcomplex)

            if has_diag_partials or psize == 1:
                # set a complex param value
                pwrap[param] += step

                uwrap = _TmpDict(self._outputs, return_complex=True)

                # solve with complex param value
                self._residuals.set_const(0.0)
                self.compute(pwrap, uwrap)

                for u in out_names:
                    partials[(u, param)] = imag(uwrap[u] * inv_stepsize).flat

                # restore old param value
                pwrap[param] -= step
            else:
                for i, idx in enumerate(array_idx_iter(pwrap[param].shape)):
                    # set a complex param value
                    pwrap[param][idx] += step

                    uwrap = _TmpDict(self._outputs, return_complex=True)

                    # solve with complex param value
                    self._residuals.set_const(0.0)
                    self.compute(pwrap, uwrap)

                    for u in out_names:
                        # set the column in the Jacobian entry
                        partials[(u, param)][:, i] = imag(uwrap[u] * inv_stepsize).flat

                    # restore old param value
                    pwrap[param][idx] -= step


class _TmpDict(object):
    """
    Dict wrapper that allows modification without changing the wrapped dict.

    It will allow getting of values
    from its inner dict unless those values get modified via
    __setitem__.  After values have been modified they are managed
    thereafter by the wrapper.  This protects the inner dict from
    modification.

    Attributes
    ----------
    _inner : dict-like
        The dictionary to be wrapped.
    _changed : dict-like
        The key names for the values that were changed.
    _complex : bool
        If True, return a complex version of values from __getitem__.
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

    Attributes
    ----------
    _inputs : dict-like
        The inputs object to be wrapped.
    _outputs : dict-like
        The outputs object to be wrapped.
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
    mod : object
        Module to check.
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


_import_functs(np, _expr_dict,
               names=['arange', 'ones', 'zeros', 'linspace',  # Array creation
                      'e', 'pi',  # Constants
                      'isinf', 'isnan',  # Logic
                      'log', 'log10', 'log1p', 'power',  # Math operations
                      'exp', 'expm1', 'fmax',
                      'fmin', 'maximum', 'minimum',
                      'sum', 'dot', 'prod',  # Reductions
                      'tensordot', 'matmul',  # Linear algebra
                      'outer', 'inner', 'kron',
                      'sin', 'cos', 'tan', ('arcsin', 'asin'),  # Trig
                      ('arccos', 'acos'), ('arctan', 'atan'),
                      'sinh', 'cosh', 'tanh', ('arcsinh', 'asinh'),  # Hyperbolic trig
                      ('arccosh', 'acosh')])


# if scipy is available, add some functions
try:
    import scipy.special
except ImportError:
    pass
else:
    _import_functs(scipy.special, _expr_dict, names=['erf', 'erfc'])

    from distutils.version import LooseVersion
    if LooseVersion(scipy.__version__) >= LooseVersion("1.5.0"):
        def factorial(*args):
            """
            Raise a RuntimeError stating that the factorial function is not supported.
            """
            raise RuntimeError("The 'factorial' function is not supported for SciPy "
                               f"versions >= 1.5, current version: {scipy.__version__}")
    else:
        def factorial(*args):
            """
            Raise a warning stating that the factorial function is deprecated.
            """
            warn_deprecation("The 'factorial' function is deprecated. "
                             "It is no longer supported for SciPy versions >= 1.5.")
            return scipy.special.factorial(*args)

    _expr_dict['factorial'] = factorial


# Put any functions here that need special versions to work under
# complex step

def _cs_abs(x):
    if isinstance(x, ndarray):
        return x * np.sign(x)
    elif x.real < 0.0:
        return -x
    return x


_expr_dict['abs'] = _cs_abs


class _NumpyMsg(object):
    """
    A class that will raise an error if an attempt is made to access any attribute/function.
    """

    def __init__(self, namespace):
        """
        Construct the _NumpyMsg object.

        Parameters
        ----------
        namespace : str
            The numpy namespace (e.g. 'numpy' or 'np).
        """
        self.namespace = namespace

    def __getattr__(self, name):
        """
        Attempt to access an attribute/function.

        Parameters
        ----------
        name : str
            The name of the attribute/function.

        Raises
        ------
        RuntimeError
            When an attempt is made to access any attribute/function.
        """
        raise RuntimeError('\n'.join([
            "    ExecComp supports a subset of numpy functions directly, without the '%s' prefix.",
            "    '%s' is %ssupported (See the documentation)."
        ]) % (self.namespace, name, '' if name in _expr_dict else 'not '))


_expr_dict['np'] = _NumpyMsg('np')
_expr_dict['numpy'] = _NumpyMsg('numpy')
