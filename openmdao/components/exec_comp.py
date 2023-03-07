"""Define the ExecComp class, a component that evaluates an expression."""
import re
import time
from itertools import product
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
from numpy import ndarray, imag, complex128 as npcomplex

from openmdao.core.system import _DEFAULT_COLORING_META
from openmdao.utils.coloring import _ColSparsityJac, _compute_coloring
from openmdao.core.constants import INT_DTYPE
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.units import valid_units
from openmdao.utils import cs_safe
from openmdao.utils.om_warnings import issue_warning, DerivativesWarning, warn_deprecation, \
    SetupWarning
from openmdao.utils.array_utils import get_random_arr


# regex to check for variable names.
VAR_RGX = re.compile(r'([.]*[_a-zA-Z]\w*[ ]*\(?)')

# Names of metadata entries allowed for ExecComp variables.
_allowed_meta = {'value', 'val', 'shape', 'units', 'res_units', 'desc',
                 'ref', 'ref0', 'res_ref', 'lower', 'upper', 'src_indices',
                 'flat_src_indices', 'tags', 'shape_by_conn', 'copy_shape', 'constant'}

# Names that are not allowed for input or output variables (keywords for options)
_disallowed_names = {'has_diag_partials', 'units', 'shape', 'shape_by_conn', 'run_root_only',
                     'constant', 'do_coloring'}


def check_option(option, value):
    """
    Check option for validity.

    Parameters
    ----------
    option : str
        The name of the option.
    value : any
        The value of the option.

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
        Shape of the array.

    Yields
    ------
    int
    """
    for p in product(*[range(s) for s in shape]):
        yield p


class ExecComp(ExplicitComponent):
    """
    A component defined by an expression string.

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
        other metadata, assign the initial value to the 'val' entry
        of the dict.

    Attributes
    ----------
    _kwargs : dict of named args
        Initial values of variables.
    _exprs : list
        List of expressions.
    _codes : list
        List of code objects.
    _exprs_info : list
        List of tuples containing output and inputs for each expression.
    complex_stepsize : double
        Step size used for complex step which is used for derivatives.
    _manual_decl_partials : bool
        If True, at least one partial has been declared by the user.
    _requires_fd : dict
        Contains a mapping of 'of' variables to a tuple of the form (wrts, functs) for those
        'of' variables that require finite difference to be used to compute their derivatives.
    _constants : dict of dicts
        Constants defined in the expressions. The key is the name of the constant and the value
        is a dict of metadata.
    _coloring_declared : bool
        If True, coloring has been declared manually.
    _inarray : ndarray or None
        If using internal CS, this is a complex array containing input values.
    _outarray : ndarray or None
        If using internal CS, this is a complex array containing output values.
    _indict : dict or None
        If using internal CS, this maps input variable views in _inarray to input names.
    _viewdict : dict or None
        If using internal CS, this maps input, output, and constant names to their corresponding
        views/values.
    """

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
        arctan2(y, x)              4-quadrant arctangent function of y and x
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
            excomp = om.ExecComp('y=sum(x)', x=numpy.ones(10, dtype=float))

        In this example, 'y' would be assumed to be the default type of float
        and would be given the default initial value of 1.0, while 'x' would be
        initialized with a size 10 float array of ones.

        If you want to assign certain metadata for 'x' in addition to its
        initial value, you can do it as follows:

        .. code-block:: python

            excomp = ExecComp('y=sum(x)',
                              x={'val': numpy.ones(10, dtype=float),
                                 'units': 'ft'})
        """
        options = {}
        for name in _disallowed_names:
            if name in kwargs:
                options[name] = kwargs.pop(name)

        super().__init__(**options)

        # change default coloring values
        self._coloring_info['method'] = 'cs'
        self._coloring_info['num_full_jacs'] = 2

        # if complex step is used for derivatives, this is the stepsize
        self.complex_stepsize = 1.e-40

        if isinstance(exprs, str):
            exprs = [exprs]

        self._exprs = exprs[:]
        self._exprs_info = []
        self._codes = []
        self._kwargs = kwargs

        self._manual_decl_partials = False
        self._no_check_partials = True

        self._constants = {}
        self._coloring_declared = False
        self._inarray = None
        self._outarray = None
        self._indict = None
        self._viewdict = None

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
                                  'Default is None, which means units may be provided for variables'
                                  ' individually.',
                             check_valid=check_option)

        self.options.declare('shape', types=(int, tuple, list), allow_none=True, default=None,
                             desc='Shape to be assigned to all variables in this component. '
                                  'Default is None, which means shape may be provided for variables'
                                  ' individually.')

        self.options.declare('shape_by_conn', types=bool, default=False,
                             desc='If True, shape all inputs and outputs based on their '
                                  'connection. Default is False.')

        self.options.declare('do_coloring', types=bool, default=True,
                             desc='If True (the default), compute the partial jacobian '
                             'coloring for this component.')

    @classmethod
    def register(cls, name, callable_obj, complex_safe):
        """
        Register a callable to be usable within ExecComp expressions.

        Parameters
        ----------
        name : str
            Name of the callable.
        callable_obj : callable
            The callable.
        complex_safe : bool
            If True, the given callable works correctly with complex numbers.
        """
        global _expr_dict, _not_complex_safe

        if not callable(callable_obj):
            raise TypeError(f"{cls.__name__}: '{name}' passed to register() of type "
                            f"'{type(callable_obj).__name__}' is not callable.")
        if name in _expr_dict:
            raise NameError(f"{cls.__name__}: '{name}' has already been registered.")

        if name in _disallowed_names:
            raise NameError(f"{cls.__name__}: cannot register name '{name}' because "
                            "it's a reserved keyword.")

        if '.' in name:
            raise NameError(f"{cls.__name__}: cannot register name '{name}' because "
                            "it contains '.'.")
        _expr_dict[name] = callable_obj

        if not complex_safe:
            _not_complex_safe.add(name)

    def setup(self):
        """
        Set up variable name and metadata lists.
        """
        if self._exprs:
            self._setup_expressions()

    def _setup_expressions(self):
        """
        Set up the expressions.

        This is called during setup_procs and after each call to "add_expr" from configure.
        """
        global _not_complex_safe

        exprs = self._exprs
        kwargs = self._kwargs

        shape = self.options['shape']
        shape_by_conn = self.options['shape_by_conn']

        if shape is not None and shape_by_conn:
            raise RuntimeError(f"{self.msginfo}: Can't set both shape and shape_by_conn.")

        self._exprs_info = exprs_info = []
        outs = set()
        allvars = set()
        self._requires_fd = {}

        for expr in exprs:
            lhs, _, rhs = expr.partition('=')
            onames = self._parse_for_out_vars(lhs)
            vnames, fnames = self._parse_for_names(rhs)

            # remove constants
            vnames = vnames.difference(
                [n for n, val in kwargs.items()
                 if isinstance(val, dict) and 'constant' in val and val['constant']]
            )

            allvars.update(vnames)
            outs.update(onames)

            if onames.intersection(allvars):
                # we have a used-before-calculated output
                violators = sorted([n for n in onames if n in allvars])
                raise RuntimeError(f"{self.msginfo}: Outputs {violators} are used before "
                                   "being calculated, so this ExecComp is not a valid explicit "
                                   "component.")

            exprs_info.append((onames, vnames, fnames))
            if _not_complex_safe.intersection(fnames):
                for o in onames:
                    self._requires_fd[o] = (vnames, fnames)

        allvars.update(outs)

        if self._requires_fd:
            inps = []
            for out, (rhsvars, funcs) in self._requires_fd.items():
                iset = rhsvars.difference(outs)
                self._requires_fd[out] = (iset, funcs)
                inps.extend(iset)
            self._no_check_partials = False
            self.set_check_partial_options(wrt=inps, method='fd')

        kwargs2 = {}
        init_vals = {}
        units = self.options['units']
        warned = False

        # make sure all kwargs are legit
        for arg, val in kwargs.items():

            if isinstance(val, dict) and 'constant' in val and val['constant']:
                if 'val' not in val:
                    raise RuntimeError(f"{self.msginfo}: arg '{arg}' in call to ExecComp() "
                                       "is a constant but no value is given")
                for ignored_meta in ['units', 'shape']:
                    if ignored_meta in val:
                        issue_warning(f"arg '{arg}' in call to ExecComp() "
                                      f"is a constant. The {ignored_meta} will be ignored",
                                      prefix=self.msginfo, category=SetupWarning)

                self._constants[arg] = val['val']
                continue  # TODO should still do some checking here!

            if arg not in allvars:
                msg = f"{self.msginfo}: arg '{arg}' in call to ExecComp() " \
                      f"does not refer to any variable in the expressions {exprs}"
                if arg in ('promotes', 'promotes_inputs', 'promotes_outputs'):
                    msg += ". Did you intend to promote variables in the 'add_subsystem' call?"
                raise RuntimeError(msg)

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
                    elif 'val' in val and np.atleast_1d(val['val']).shape != shape:
                        raise RuntimeError("%s: value of shape %s has been specified for "
                                           "variable '%s', but shape of %s has been "
                                           "specified for the entire component." %
                                           (self.msginfo, np.atleast_1d(val['val']).shape,
                                            arg, shape))
                    else:
                        init_vals[arg] = np.ones(shape)

                if 'val' in val:
                    init_vals[arg] = val['val']
                    del kwargs2[arg]['val']

                if shape_by_conn or 'shape_by_conn' in val or 'copy_shape' in val:
                    if val.get('shape') is not None or val.get('val') is not None:
                        raise RuntimeError(f"{self.msginfo}: Can't set 'shape' or 'val' for "
                                           f"variable '{arg}' along with 'copy_shape' or "
                                           "'shape_by_conn'.")

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

        if self._static_mode:
            var_rel2meta = self._static_var_rel2meta
        else:
            var_rel2meta = self._var_rel2meta

        for var in sorted(allvars):
            meta = kwargs2.get(var, {
                'units': units,
                'shape': shape,
                'shape_by_conn': shape_by_conn})

            # if user supplied an initial value, use it, otherwise set to 1.0
            if var in init_vals:
                val = init_vals[var]
            else:
                val = 1.0

            if var in var_rel2meta:
                # Input/Output already exists, but we may be setting defaults for the first time.
                # Note that there is only one submitted dictionary of defaults.
                current_meta = var_rel2meta[var]

                for kname, kvalue in meta.items():
                    if kvalue is not None:
                        current_meta[kname] = kvalue

                new_val = kwargs[var].get('val')
                if new_val is not None:
                    # val is normally ensured to be a numpy array in add_input/add_output,
                    # do the same here...
                    current_meta['val'] = np.atleast_1d(new_val)
            else:
                # new input and/or output.
                if var in outs:
                    current_meta = self.add_output(var, val, **meta)
                else:
                    if 'constant' in meta:
                        meta.pop('constant', None)
                    current_meta = self.add_input(var, val, **meta)

            if var not in init_vals:
                init_vals[var] = current_meta['val']

        self._codes = self._compile_exprs(self._exprs)

    def add_expr(self, expr, **kwargs):
        """
        Add an expression to the ExecComp.

        Parameters
        ----------
        expr : str
            An assignment statement that expresses how the outputs are calculated based on the
            inputs. In addition to standard Python operators, a subset of numpy and scipy
            functions is supported.
        **kwargs : dict of named args
            Initial values of variables can be set by setting a named arg with the var name.  If
            the value is a dict it is assumed to contain metadata.  To set the initial value in
            addition to other metadata, assign the initial value to the 'val' entry of the dict.
            Do not include for inputs whose default kwargs have been declared on previous
            expressions.
        """
        if not isinstance(expr, str):
            typ = type(expr).__name__
            msg = f"Argument 'expr' must be of type 'str', but type '{typ}' was found."
            raise TypeError(msg)

        self._exprs.append(expr)
        for name in kwargs:
            if name in self._kwargs:
                raise NameError(f"Defaults for '{name}' have already been defined in a previous "
                                "expression.")

        self._kwargs.update(kwargs)

        if not self._static_mode:
            self._setup_expressions()

    def _compile_exprs(self, exprs):
        compiled = []
        outputs = set()
        for i, expr in enumerate(exprs):

            # Quick dupe check.
            lhs_name = expr.partition('=')[0].strip()
            if lhs_name in outputs:
                # Can't add two equations with the same output.
                raise RuntimeError(f"{self.msginfo}: The output '{lhs_name}' has already been "
                                   "defined by an expression.")
            else:
                outputs.add(lhs_name)

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

    def _parse_for_names(self, s):
        names = [x.strip() for x in re.findall(VAR_RGX, s) if not x.startswith('.')]
        vnames = set()
        for n in names:
            if n.endswith('('):
                continue
            vnames.add(n)
        fnames = [n[:-1] for n in names if n[-1] == '(']
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
                                    "function or constant." % (self.msginfo, v))
                else:
                    to_remove.append(v)

        for f in fnames:
            if f not in _expr_dict:
                raise NameError(f"{self.msginfo}: can't use '{f}' as a function because "
                                "it hasn't been registered.")

        return vnames.difference(to_remove), fnames

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

    def declare_partials(self, *args, **kwargs):
        """
        Declare information about this component's subjacobians.

        Parameters
        ----------
        *args : list
            Positional args to be passed to base class version of declare_partials.
        **kwargs : dict
            Keyword args  to be passed to base class version of declare_partials.

        Returns
        -------
        dict
            Metadata dict for the specified partial(s).
        """
        if 'method' not in kwargs or kwargs['method'] not in ('cs', 'fd'):
            raise RuntimeError(f"{self.msginfo}: declare_partials must be called with method='cs' "
                               "or method='fd'.")
        if self.options['has_diag_partials']:
            raise RuntimeError(f"{self.msginfo}: declare_partials cannot be called manually if "
                               "has_diag_partials has been set.")

        self._manual_decl_partials = True
        return super().declare_partials(*args, **kwargs)

    def _get_coloring(self):
        """
        Get the Coloring for this system.

        If necessary, load the Coloring from a file or dynamically generate it.

        Returns
        -------
        Coloring or None
            Coloring object, possible loaded from a file or dynamically generated, or None
        """
        if self.options['do_coloring']:
            return super()._get_coloring()

    def _setup_partials(self):
        """
        Check that all partials are declared.
        """
        has_diag_partials = self.options['has_diag_partials']
        if not self._manual_decl_partials:
            if self.options['do_coloring'] and not has_diag_partials:
                rank = self.comm.rank
                sizes = self._var_sizes
                if not self._has_distrib_vars and (sum(sizes['input'][rank]) > 1 and
                                                   sum(sizes['output'][rank]) > 1):
                    if not self._coloring_declared:
                        super().declare_coloring(wrt=None, method='cs')
                        self._coloring_info['dynamic'] = True
                        self._manual_decl_partials = False  # this gets reset in declare_partials
                        self._declared_partials = defaultdict(dict)
                else:
                    self.options['do_coloring'] = False
                    self._coloring_info['dynamic'] = False

            meta = self._var_rel2meta
            decl_partials = super().declare_partials
            for outs, vs, _ in self._exprs_info:
                ins = sorted(set(vs).difference(outs))
                for out in sorted(outs):
                    for inp in ins:
                        if has_diag_partials:
                            ival = meta[inp]['val']
                            oval = meta[out]['val']
                            iarray = isinstance(ival, ndarray) and ival.size > 1
                            if iarray and isinstance(oval, ndarray) and oval.size > 1:
                                if oval.size != ival.size:
                                    raise RuntimeError(
                                        "%s: has_diag_partials is True but partial(%s, %s) "
                                        "is not square (shape=(%d, %d))." %
                                        (self.msginfo, out, inp, oval.size, ival.size))
                                # partial will be declared as diagonal
                                inds = np.arange(oval.size, dtype=INT_DTYPE)
                            else:
                                inds = None
                            decl_partials(of=out, wrt=inp, rows=inds, cols=inds)
                        else:
                            decl_partials(of=out, wrt=inp)

        super()._setup_partials()

        if self._manual_decl_partials:
            undeclared = []
            for outs, vs, _ in self._exprs_info:
                ins = sorted(set(vs).difference(outs))
                for out in sorted(outs):
                    out = '.'.join((self.pathname, out)) if self.pathname else out
                    for inp in ins:
                        inp = '.'.join((self.pathname, inp)) if self.pathname else inp
                        if (out, inp) not in self._subjacs_info:
                            undeclared.append((out, inp))
            if undeclared:
                idx = len(self.pathname) + 1 if self.pathname else 0
                undeclared = ', '.join([' wrt '.join((f"'{of[idx:]}'", f"'{wrt[idx:]}'"))
                                        for of, wrt in undeclared])
                issue_warning(f"The following partial derivatives have not been "
                              f"declared so they are assumed to be zero: [{undeclared}].",
                              prefix=self.msginfo, category=DerivativesWarning)

    def _setup_vectors(self, root_vectors):
        """
        Compute all vectors for all vec names.

        Parameters
        ----------
        root_vectors : dict of dict of Vector
            Root vectors: first key is 'input', 'output', or 'residual'; second key is vec_name.
        """
        super()._setup_vectors(root_vectors)

        if not self._use_derivatives:
            self._manual_decl_partials = True  # prevents attempts to use _viewdict in compute

        self._iodict = _IODict(self._outputs, self._inputs, self._constants)

        self._relcopy = False

        if not self._manual_decl_partials:
            if self._force_alloc_complex:
                # we can use the internal Vector complex arrays

                # set complex_step_mode so we'll get the full complex array
                self._inputs.set_complex_step_mode(True)
                self._outputs.set_complex_step_mode(True)

                self._indict = self._inputs._get_local_views()
                outdict = self._outputs._get_local_views()

                self._inarray = self._inputs.asarray(copy=False)
                self._outarray = self._outputs.asarray(copy=False)

                self._inputs.set_complex_step_mode(False)
                self._outputs.set_complex_step_mode(False)

            else:
                # we make our own complex 'copy' of the Vector arrays
                self._inarray = np.zeros(len(self._inputs), dtype=complex)
                self._outarray = np.zeros(len(self._outputs), dtype=complex)

                self._indict = self._inputs._get_local_views(self._inarray)
                outdict = self._outputs._get_local_views(self._outarray)

                self._relcopy = True

            # combine lookup dicts for faster exec calls
            viewdict = self._indict.copy()
            viewdict.update(outdict)
            viewdict.update(self._constants)
            self._viewdict = _ViewDict(viewdict)

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
        if not self._manual_decl_partials:
            if self._relcopy:
                self._inarray[:] = self._inputs.asarray(copy=False)
                self._exec()
                outs = outputs.asarray(copy=False)
                if outs.dtype == self._outarray.dtype:
                    outs[:] = self._outarray
                else:
                    outs[:] = self._outarray.real
            else:
                self._exec()

            return

        if self._iodict._inputs is not inputs:
            self._iodict = _IODict(outputs, inputs, self._constants)

        for i, expr in enumerate(self._codes):
            try:
                #  inputs, outputs, and _constants are vectors
                exec(expr, _expr_dict, self._iodict)  # nosec:
                # limited to _expr_dict
            except Exception as err:
                raise RuntimeError(f"{self.msginfo}: Error occurred evaluating '{self._exprs[i]}':"
                                   f"\n{err}")

    def _linearize(self, jac=None, sub_do_ln=False):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            Ignored.
        sub_do_ln : bool
            Flag indicating if the children should call linearize on their linear solvers.
        """
        if self._requires_fd:
            if 'fd' in self._approx_schemes:
                fdins = {wrt.rsplit('.', 1)[1] for wrt in self._approx_schemes['fd']._wrt_meta}
            else:
                fdins = set()

            for _, (inps, funcs) in self._requires_fd.items():
                diff = inps.difference(fdins)
                if diff:
                    raise RuntimeError(f"{self.msginfo}: expression contains functions "
                                       f"{sorted(funcs)} that are not complex safe. To fix this, "
                                       f"call declare_partials('*', {sorted(diff)}, method='fd') "
                                       f"on this component prior to setup.")
            self._requires_fd = False  # only need to do this check the first time around

        super()._linearize(jac, sub_do_ln)

    def declare_coloring(self,
                         wrt=_DEFAULT_COLORING_META['wrt_patterns'],
                         method=_DEFAULT_COLORING_META['method'],
                         form=None,
                         step=None,
                         per_instance=_DEFAULT_COLORING_META['per_instance'],
                         num_full_jacs=_DEFAULT_COLORING_META['num_full_jacs'],
                         tol=_DEFAULT_COLORING_META['tol'],
                         orders=_DEFAULT_COLORING_META['orders'],
                         perturb_size=_DEFAULT_COLORING_META['perturb_size'],
                         min_improve_pct=_DEFAULT_COLORING_META['min_improve_pct'],
                         show_summary=_DEFAULT_COLORING_META['show_summary'],
                         show_sparsity=_DEFAULT_COLORING_META['show_sparsity']):
        """
        Set options for deriv coloring of a set of wrt vars matching the given pattern(s).

        Parameters
        ----------
        wrt : str or list of str
            The name or names of the variables that derivatives are taken with respect to.
            This can contain input names, output names, or glob patterns.
        method : str
            Method used to compute derivative: "fd" for finite difference, "cs" for complex step.
        form : str
            Finite difference form, can be "forward", "central", or "backward". Leave
            undeclared to keep unchanged from previous or default value.
        step : float
            Step size for finite difference. Leave undeclared to keep unchanged from previous
            or default value.
        per_instance : bool
            If True, a separate coloring will be generated for each instance of a given class.
            Otherwise, only one coloring for a given class will be generated and all instances
            of that class will use it.
        num_full_jacs : int
            Number of times to repeat partial jacobian computation when computing sparsity.
        tol : float
            Tolerance used to determine if an array entry is nonzero during sparsity determination.
        orders : int
            Number of orders above and below the tolerance to check during the tolerance sweep.
        perturb_size : float
            Size of input/output perturbation during generation of sparsity.
        min_improve_pct : float
            If coloring does not improve (decrease) the number of solves more than the given
            percentage, coloring will not be used.
        show_summary : bool
            If True, display summary information after generating coloring.
        show_sparsity : bool
            If True, display sparsity with coloring info after generating coloring.
        """
        super().declare_coloring(wrt, method, form, step, per_instance, num_full_jacs,
                                 tol, orders, perturb_size, min_improve_pct,
                                 show_summary, show_sparsity)
        self._coloring_declared = True
        self._manual_decl_partials = True

    def _exec(self):
        for i, expr in enumerate(self._codes):
            try:
                exec(expr, _expr_dict, self._viewdict)  # nosec:
            except Exception as err:
                raise RuntimeError(f"{self.msginfo}: Error occurred evaluating "
                                   f"'{self._exprs[i]}':\n{err}")

    def _compute_coloring(self, recurse=False, **overrides):
        """
        Compute a coloring of the partial jacobian.

        This assumes that the current System is in a proper state for computing derivatives.

        Parameters
        ----------
        recurse : bool
            Ignored.
        **overrides : dict
            Any args that will override either default coloring settings or coloring settings
            resulting from an earlier call to declare_coloring.

        Returns
        -------
        list of Coloring
            The computed colorings.
        """
        if self._manual_decl_partials:
            # use framework approx coloring
            return super()._compute_coloring(recurse=recurse, **overrides)

        info = self._coloring_info
        info.update(**overrides)
        if isinstance(info['wrt_patterns'], str):
            info['wrt_patterns'] = [info['wrt_patterns']]

        if not self._coloring_declared and info['method'] is None:
            info['method'] = 'cs'

        if info['method'] != 'cs':
            raise RuntimeError(f"{self.msginfo}: 'method' for coloring must be 'cs' if partials "
                               "and/or coloring are not declared manually using declare_partials "
                               "or declare_coloring.")

        if info['coloring'] is None and info['static'] is None:
            info['dynamic'] = True

        # match everything
        info['wrt_matches_rel'] = None
        info['wrt_matches'] = None

        sparsity_start_time = time.perf_counter()

        step = self.complex_stepsize * 1j
        inv_stepsize = 1.0 / self.complex_stepsize
        inarr = self._inarray
        oarr = self._outarray

        if self.options['has_diag_partials']:
            # we should never get here
            raise NotImplementedError("has_diag_partials not supported with coloring yet")

        # compute perturbations
        starting_inputs = self._inputs.asarray(copy=not self._relcopy)
        in_offsets = starting_inputs.copy()
        in_offsets[in_offsets == 0.0] = 1.0
        in_offsets *= info['perturb_size']

        # use special sparse jacobian to collect sparsity info
        jac = _ColSparsityJac(self, info)

        for i in range(info['num_full_jacs']):
            inarr[:] = starting_inputs + in_offsets * get_random_arr(in_offsets.size, self.comm)

            for i in range(inarr.size):
                inarr[i] += step
                self._exec()
                jac.set_col(self, i, imag(oarr * inv_stepsize))
                inarr[i] -= step

        if not self._relcopy:
            self._inputs.set_val(starting_inputs)

        sparsity, sp_info = jac.get_sparsity(self)
        sparsity_time = time.perf_counter() - sparsity_start_time

        coloring = _compute_coloring(sparsity, 'fwd')

        if not self._finalize_coloring(coloring, info, sp_info, sparsity_time):
            return [None]

        # compute mapping of col index to wrt varname
        self._col_idx2name = idxnames = [None] * len(self._inputs)
        plen = len(self.pathname) + 1 if self.pathname else 0
        for name, slc in self._inputs.get_slice_dict().items():
            name = name[plen:]
            for i in range(slc.start, slc.stop):
                idxnames[i] = name

        # get slice dicts using relative name keys
        self._out_slices = {n[plen:]: slc for n, slc in self._outputs.get_slice_dict().items()}
        self._in_slices = {n[plen:]: slc for n, slc in self._inputs.get_slice_dict().items()}

        return [coloring]

    def _compute_colored_partials(self, partials):
        """
        Use complex step method with coloring to update the given Jacobian.

        Parameters
        ----------
        partials : `Jacobian`
            Contains sub-jacobians.
        """
        step = self.complex_stepsize * 1j
        inv_stepsize = 1.0 / self.complex_stepsize
        inarr = self._inarray
        oarr = self._outarray
        out_names = self._var_rel_names['output']

        inarr[:] = self._inputs.asarray(copy=False)
        scratch = np.zeros(oarr.size)
        idx2name = self._col_idx2name
        out_slices = self._out_slices
        in_slices = self._in_slices

        for icols, nzrowlists in self._coloring_info['coloring'].color_nonzero_iter('fwd'):
            # set a complex input value
            inarr[icols] += step

            # solve with complex input value
            self._exec()

            imag_oar = imag(oarr * inv_stepsize)

            for icol, rows in zip(icols, nzrowlists):
                scratch[rows] = imag_oar[rows]
                inp = idx2name[icol]
                loc_i = icol - in_slices[inp].start
                for u in out_names:
                    key = (u, inp)
                    if key in self._declared_partials:
                        # set the column in the Jacobian entry
                        part = scratch[out_slices[u]]
                        partials[key][:, loc_i] = part
                        part[:] = 0.

            # restore old input value
            inarr[icols] -= step

    def compute_partials(self, inputs, partials):
        """
        Use complex step method to update the given Jacobian.

        Parameters
        ----------
        inputs : Vector or dict
            Vector containing parameters (p).
        partials : `Jacobian`
            Contains sub-jacobians.
        """
        if self._manual_decl_partials:
            return

        if self.under_complex_step:
            raise RuntimeError(f"{self.msginfo}: Can't compute complex step partials when higher "
                               "level system is using complex step unless you manually call "
                               "declare_partials and/or declare_coloring on this ExecComp.")

        if self._coloring_info['coloring'] is not None:
            self._compute_colored_partials(partials)
            return

        step = self.complex_stepsize * 1j
        out_names = self._var_rel_names['output']
        inv_stepsize = 1.0 / self.complex_stepsize
        has_diag_partials = self.options['has_diag_partials']
        inarr = self._inarray
        indict = self._indict
        vdict = self._viewdict

        inarr[:] = self._inputs.asarray(copy=False)

        for inp, ival in indict.items():
            psize = ival.size

            if has_diag_partials or psize == 1:
                # set a complex inpup value
                ival += step

                # solve with complex input value
                self._exec()

                for u in out_names:
                    if (u, inp) in self._declared_partials:
                        partials[u, inp] = imag(vdict[u] * inv_stepsize).flat

                # restore old input value
                ival -= step
            else:
                for i, idx in enumerate(array_idx_iter(ival.shape)):
                    # set a complex input value
                    ival[idx] += step

                    # solve with complex input value
                    self._exec()

                    for u in out_names:
                        if (u, inp) in self._declared_partials:
                            # set the column in the Jacobian entry
                            partials[u, inp][:, i] = imag(vdict[u] * inv_stepsize).flat

                    # restore old input value
                    ival[idx] -= step


class _ViewDict(object):
    def __init__(self, dct):
        self.dct = dct

    def __getitem__(self, name):
        return self.dct[name]

    def __setitem__(self, name, value):
        try:
            self.dct[name][:] = value
        except ValueError:
            # see if value fits if size 1 dimensions are removed
            sqz = np.squeeze(value)
            if np.squeeze(self.dct[name]).shape == sqz.shape:
                self.dct[name][:] = sqz
            else:
                raise

    # need __contains__ here else we get weird KeyErrors in certain situations when evaluating
    # the compiled expressions.  Non-compiled expressions evaluate just fine, but after compilation,
    # and only in rare circumstances (like running under om trace), KeyErrors for 0, 1, ...
    # are mysteriously generated.
    def __contains__(self, name):
        return name in self.dct


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
    _constants : dict-like
        The constants object to be wrapped.
    """

    def __init__(self, outputs, inputs, constants):
        """
        Create the dict wrapper.

        Parameters
        ----------
        outputs : dict-like
            The outputs object to be wrapped.

        inputs : dict-like
            The inputs object to be wrapped.

        constants : dict-like
            The constants object to be wrapped.
        """
        self._outputs = outputs
        self._inputs = inputs
        self._constants = constants

    def __getitem__(self, name):
        try:
            return self._inputs[name]
        except KeyError:
            try:
                return self._outputs[name]
            except KeyError:
                return self._constants[name]

    def __setitem__(self, name, value):
        try:
            self._outputs[name][:] = value
        except ValueError:
            # see if value fits if size 1 dimensions are removed
            sqz = np.squeeze(value)
            if np.squeeze(self._outputs[name]).shape == sqz.shape:
                self._outputs[name][:] = sqz
            else:
                raise

    def __contains__(self, name):
        return name in self._inputs or name in self._outputs or name in self._constants


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


_expr_dict = {}  # this dict will act as the local scope when we eval our expressions
_not_complex_safe = set()  # this is the set of registered functions that are not complex safe


_import_functs(np, _expr_dict,
               names=['arange', 'ones', 'zeros', 'linspace',  # Array creation
                      'e', 'pi',  # Constants
                      'isinf', 'isnan',  # Logic
                      'log', 'log10', 'log1p', 'power',  # Math operations
                      'exp', 'expm1', 'fmax', 'min', 'max', 'diff',
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

    from packaging.version import Version
    if Version(scipy.__version__) >= Version("1.5.0"):
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


# put any functions that need custom complex-safe versions here

_expr_dict['abs'] = cs_safe.abs
_expr_dict['arctan2'] = cs_safe.arctan2


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


@contextmanager
def _temporary_expr_dict():
    """
    During a test, it's useful to be able to save and restore the _expr_dict.
    """
    global _expr_dict, _not_complex_safe

    save = (_expr_dict.copy(), _not_complex_safe.copy())

    try:
        yield
    finally:
        _expr_dict, _not_complex_safe = save
