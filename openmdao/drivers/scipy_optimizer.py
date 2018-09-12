"""
OpenMDAO Wrapper for the scipy.optimize.minimize family of local optimizers.
"""

from __future__ import print_function
from collections import OrderedDict
import sys

from six import itervalues, iteritems, reraise
from six.moves import range

import numpy as np
from scipy.optimize import minimize

import openmdao
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.general_utils import warn_deprecation
import openmdao.utils.coloring as coloring_mod


_optimizers = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
               'TNC', 'COBYLA', 'SLSQP']
_gradient_optimizers = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC',
                        'SLSQP', 'dogleg', 'trust-ncg']
_bounds_optimizers = ['L-BFGS-B', 'TNC', 'SLSQP']
_constraint_optimizers = ['COBYLA', 'SLSQP']
_constraint_grad_optimizers = ['SLSQP']
_eq_constraint_optimizers = ['SLSQP']

# These require Hessian or Hessian-vector product, so they are not supported
# right now.
_unsupported_optimizers = ['dogleg', 'trust-ncg']


CITATIONS = """
@phdthesis{hwang_thesis_2015,
  author       = {John T. Hwang},
  title        = {A Modular Approach to Large-Scale Design Optimization of Aerospace Systems},
  school       = {University of Michigan},
  year         = 2015
}
"""


class ScipyOptimizeDriver(Driver):
    """
    Driver wrapper for the scipy.optimize.minimize family of local optimizers.

    Inequality constraints are supported by COBYLA and SLSQP,
    but equality constraints are only supported by SLSQP. None of the other
    optimizers support constraints.

    ScipyOptimizeDriver supports the following:
        equality_constraints
        inequality_constraints

    Attributes
    ----------
    fail : bool
        Flag that indicates failure of most recent optimization.
    iter_count : int
        Counter for function evaluations.
    result : OptimizeResult
        Result returned from scipy.optimize call.
    opt_settings : dict
        Dictionary of solver-specific options. See the scipy.optimize.minimize documentation.
    _con_cache : OrderedDict
        Cached result of constraint evaluations because scipy asks for them in a separate function.
    _con_idx : dict
        Used for constraint bookkeeping in the presence of 2-sided constraints.
    _cons : dict
        Contains all constraint info.
    _designvars : dict
        Contains all design variable info.
    _grad_cache : OrderedDict
        Cached result of nonlinear constraint derivatives because scipy asks for them in a separate
        function.
    _objs : dict
        Contains all objective info.
    _exc_info : 3 item tuple
        Storage for exception and traceback information.
    _obj_and_nlcons : list
        List of objective + nonlinear constraints. Used to compute total derivatives
        for all except linear constraints.
    """

    def __init__(self, **kwargs):
        """
        Initialize the ScipyOptimizeDriver.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        super(ScipyOptimizeDriver, self).__init__(**kwargs)

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['two_sided_constraints'] = True
        self.supports['linear_constraints'] = True
        self.supports['simultaneous_derivatives'] = True

        # What we don't support
        self.supports['multiple_objectives'] = False
        self.supports['active_set'] = False
        self.supports['integer_design_vars'] = False

        # The user places optimizer-specific settings in here.
        self.opt_settings = OrderedDict()

        self.result = None
        self.fail = 0
        self._grad_cache = None
        self._con_cache = None
        self._con_idx = {}
        self._obj_and_nlcons = None
        self.fail = False
        self.iter_count = 0
        self._exc_info = None

        self.cite = CITATIONS

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('optimizer', 'SLSQP', values=_optimizers,
                             desc='Name of optimizer to use')
        self.options.declare('tol', 1.0e-6, lower=0.0,
                             desc='Tolerance for termination. For detailed '
                             'control, use solver-specific options.')
        self.options.declare('maxiter', 200, lower=0,
                             desc='Maximum number of iterations.')
        self.options.declare('disp', True,
                             desc='Set to False to prevent printing of Scipy convergence messages')
        self.options.declare('dynamic_simul_derivs', default=False, types=bool,
                             desc='Compute simultaneous derivative coloring dynamically if True')
        self.options.declare('dynamic_derivs_repeats', default=3, types=int,
                             desc='Number of compute_totals calls during dynamic computation of '
                                  'simultaneous derivative coloring')

    def _get_name(self):
        """
        Get name of current optimizer.

        Returns
        -------
        str
            The name of the current optimizer.
        """
        return self.options['optimizer']

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer
        """
        super(ScipyOptimizeDriver, self)._setup_driver(problem)
        opt = self.options['optimizer']

        self.supports['gradients'] = opt in _gradient_optimizers
        self.supports['inequality_constraints'] = opt in _constraint_optimizers
        self.supports['two_sided_constraints'] = opt in _constraint_optimizers
        self.supports['equality_constraints'] = opt in _eq_constraint_optimizers

        # Since COBYLA does not support bounds, we
        #   need to add to the _cons metadata for any bounds that
        #   need to be translated into a constraint
        if opt == 'COBYLA':
            for name, meta in iteritems(self._designvars):
                lower = meta['lower']
                upper = meta['upper']
                if isinstance(lower, np.ndarray) or lower >= -openmdao.INF_BOUND \
                        or isinstance(upper, np.ndarray) or upper <= openmdao.INF_BOUND:
                    d = OrderedDict()
                    d['lower'] = lower
                    d['upper'] = upper
                    d['equals'] = None
                    d['indices'] = None
                    d['adder'] = None
                    d['scaler'] = None
                    d['size'] = meta['size']
                    self._cons[name] = d

    def run(self):
        """
        Optimize the problem using selected Scipy optimizer.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        problem = self._problem
        opt = self.options['optimizer']
        model = problem.model
        self.iter_count = 0
        self._total_jac = None

        # Initial Run
        model._solve_nonlinear()

        self._con_cache = self.get_constraint_values()
        desvar_vals = self.get_design_var_values()
        self._dvlist = list(self._designvars)

        # maxiter and disp get passsed into scipy with all the other options.
        self.opt_settings['maxiter'] = self.options['maxiter']
        self.opt_settings['disp'] = self.options['disp']

        # Size Problem
        nparam = 0
        for param in itervalues(self._designvars):
            nparam += param['size']
        x_init = np.empty(nparam)

        # Initial Design Vars
        i = 0
        use_bounds = (opt in _bounds_optimizers)
        if use_bounds:
            bounds = []
        else:
            bounds = None

        for name, meta in iteritems(self._designvars):
            size = meta['size']
            x_init[i:i + size] = desvar_vals[name]
            i += size

            # Bounds if our optimizer supports them
            if use_bounds:
                meta_low = meta['lower']
                meta_high = meta['upper']
                for j in range(size):

                    if isinstance(meta_low, np.ndarray):
                        p_low = meta_low[j]
                    else:
                        p_low = meta_low

                    if isinstance(meta_high, np.ndarray):
                        p_high = meta_high[j]
                    else:
                        p_high = meta_high

                    bounds.append((p_low, p_high))

        # Constraints
        constraints = []
        i = 1  # start at 1 since row 0 is the objective.  Constraints start at row 1.
        lin_i = 0  # counter for linear constraint jacobian
        lincons = []  # list of linear constraints
        self._obj_and_nlcons = list(self._objs)

        if opt in _constraint_optimizers:
            for name, meta in iteritems(self._cons):
                size = meta['size']
                upper = meta['upper']
                lower = meta['lower']
                if 'linear' in meta and meta['linear']:
                    lincons.append(name)
                    self._con_idx[name] = lin_i
                    lin_i += size
                else:
                    self._obj_and_nlcons.append(name)
                    self._con_idx[name] = i
                    i += size

                # Loop over every index separately, because scipy calls each constraint by index.
                for j in range(0, size):
                    con_dict = {}
                    if meta['equals'] is not None:
                        con_dict['type'] = 'eq'
                    else:
                        con_dict['type'] = 'ineq'
                    con_dict['fun'] = self._confunc
                    if opt in _constraint_grad_optimizers:
                        con_dict['jac'] = self._congradfunc
                    con_dict['args'] = [name, False, j]
                    constraints.append(con_dict)

                    if isinstance(upper, np.ndarray):
                        upper = upper[j]

                    if isinstance(lower, np.ndarray):
                        lower = lower[j]

                    dblcon = (upper < openmdao.INF_BOUND) and (lower > -openmdao.INF_BOUND)

                    # Add extra constraint if double-sided
                    if dblcon:
                        dcon_dict = {}
                        dcon_dict['type'] = 'ineq'
                        dcon_dict['fun'] = self._confunc
                        if opt in _constraint_grad_optimizers:
                            dcon_dict['jac'] = self._congradfunc
                        dcon_dict['args'] = [name, True, j]
                        constraints.append(dcon_dict)

            # precalculate gradients of linear constraints
            if lincons:
                self._lincongrad_cache = self._compute_totals(of=lincons, wrt=self._dvlist,
                                                              return_format='array')
            else:
                self._lincongrad_cache = None

        # Provide gradients for optimizers that support it
        if opt in _gradient_optimizers:
            jac = self._gradfunc
        else:
            jac = None

        # compute dynamic simul deriv coloring if option is set
        if coloring_mod._use_sparsity and self.options['dynamic_simul_derivs']:
            coloring_mod.dynamic_simul_coloring(self, do_sparsity=False)

        # optimize
        try:
            result = minimize(self._objfunc, x_init,
                              # args=(),
                              method=opt,
                              jac=jac,
                              # hess=None,
                              # hessp=None,
                              bounds=bounds,
                              constraints=constraints,
                              tol=self.options['tol'],
                              # callback=None,
                              options=self.opt_settings)

        # If an exception was swallowed in one of our callbacks, we want to raise it
        # rather than the cryptic message from scipy.
        except Exception as msg:
            if self._exc_info is not None:
                self._reraise()
            else:
                raise

        if self._exc_info is not None:
            self._reraise()

        self.result = result
        self.fail = False if self.result.success else True

        if self.fail:
            print('Optimization FAILED.')
            print(result.message)
            print('-' * 35)

        elif self.options['disp']:
            print('Optimization Complete')
            print('-' * 35)

        return self.fail

    def _objfunc(self, x_new):
        """
        Evaluate and return the objective function.

        Model is executed here.

        Parameters
        ----------
        x_new : ndarray
            Array containing parameter values at new design point.

        Returns
        -------
        float
            Value of the objective function evaluated at the new design point.
        """
        model = self._problem.model

        try:

            # Pass in new parameters
            i = 0
            for name, meta in iteritems(self._designvars):
                size = meta['size']
                self.set_design_var(name, x_new[i:i + size])
                i += size

            with RecordingDebugging(self.options['optimizer'], self.iter_count, self) as rec:
                self.iter_count += 1
                model._solve_nonlinear()

            # Get the objective function evaluations
            for name, obj in iteritems(self.get_objective_values()):
                f_new = obj
                break

            self._con_cache = self.get_constraint_values()

        except Exception as msg:
            self._exc_info = sys.exc_info()
            return 0

        # print("Functions calculated")
        # print(x_new)
        # print(f_new)

        return f_new

    def _confunc(self, x_new, name, dbl, idx):
        """
        Return the value of the constraint function requested in args.

        Note that this function is called for each constraint, so the model is only run when the
        objective is evaluated.

        Parameters
        ----------
        x_new : ndarray
            Array containing parameter values at new design point.
        name : string
            Name of the constraint to be evaluated.
        dbl : bool
            True if double sided constraint.
        idx : float
            Contains index into the constraint array.

        Returns
        -------
        float
            Value of the constraint function.
        """
        if self._exc_info is not None:
            self._reraise()

        cons = self._con_cache
        meta = self._cons[name]

        # Equality constraints
        equals = meta['equals']
        if equals is not None:
            if isinstance(equals, np.ndarray):
                equals = equals[idx]
            return (cons[name][idx] - equals)

        # Note, scipy defines constraints to be satisfied when positive,
        # which is the opposite of OpenMDAO.
        upper = meta['upper']
        if isinstance(upper, np.ndarray):
            upper = upper[idx]

        lower = meta['lower']
        if isinstance(lower, np.ndarray):
            lower = lower[idx]

        if dbl or (lower <= -openmdao.INF_BOUND):
            return upper - cons[name][idx]
        else:
            return cons[name][idx] - lower

    def _gradfunc(self, x_new):
        """
        Evaluate and return the gradient for the objective.

        Gradients for the constraints are also calculated and cached here.

        Parameters
        ----------
        x_new : ndarray
            Array containing parameter values at new design point.

        Returns
        -------
        ndarray
            Gradient of objective with respect to parameter array.
        """
        try:
            grad = self._compute_totals(of=self._obj_and_nlcons, wrt=self._dvlist,
                                        return_format='array')
            self._grad_cache = grad

        except Exception as msg:
            self._exc_info = sys.exc_info()
            return np.array([[]])

        # print("Gradients calculated")
        # print(x_new)
        # print(grad[0, :])

        return grad[0, :]

    def _congradfunc(self, x_new, name, dbl, idx):
        """
        Return the cached gradient of the constraint function.

        Note, scipy calls the constraints one at a time, so the gradient is cached when the
        objective gradient is called.

        Parameters
        ----------
        x_new : ndarray
            Array containing parameter values at new design point.
        name : string
            Name of the constraint to be evaluated.
        dbl : bool
            Denotes if a constraint is double-sided or not.
        idx : float
            Contains index into the constraint array.

        Returns
        -------
        float
            Gradient of the constraint function wrt all params.
        """
        if self._exc_info is not None:
            self._reraise()

        meta = self._cons[name]

        if meta['linear']:
            grad = self._lincongrad_cache
        else:
            grad = self._grad_cache
        grad_idx = self._con_idx[name] + idx

        # print("Constraint Gradient returned")
        # print(x_new)
        # print(name, idx, grad[grad_idx, :])

        # Equality constraints
        if meta['equals'] is not None:
            return grad[grad_idx, :]

        # Note, scipy defines constraints to be satisfied when positive,
        # which is the opposite of OpenMDAO.
        lower = meta['lower']
        if isinstance(lower, np.ndarray):
            lower = lower[idx]

        if dbl or (lower <= -openmdao.INF_BOUND):
            return -grad[grad_idx, :]
        else:
            return grad[grad_idx, :]

    def _reraise(self):
        """
        Reraise any exception encountered when scipy calls back into our method.
        """
        exc = self._exc_info
        self._exc_info = None
        reraise(*exc)


class ScipyOptimizer(ScipyOptimizeDriver):
    """
    Deprecated.  Use ScipyOptimizeDriver.
    """

    def __init__(self, **kwargs):
        """
        Initialize attributes.

        Parameters
        ----------
        **kwargs : dict
            Named args.
        """
        super(ScipyOptimizer, self).__init__(**kwargs)
        warn_deprecation("'ScipyOptimizer' provides backwards compatibility "
                         "with OpenMDAO <= 2.2 ; use 'ScipyOptimizeDriver' instead.")
