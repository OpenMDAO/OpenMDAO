"""
OpenMDAO Wrapper for the scipy.optimize.minimize family of local optimizers.
"""

from __future__ import print_function
from collections import OrderedDict
import traceback
import sys

from six import itervalues, iteritems, reraise
from six.moves import range

import numpy as np
from scipy.optimize import minimize

from openmdao.core.driver import Driver
from openmdao.recorders.recording_iteration_stack import Recording


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


class ScipyOptimizer(Driver):
    """
    Driver wrapper for the scipy.optimize.minimize family of local optimizers.

    Inequality constraints are supported by COBYLA and SLSQP,
    but equality constraints are only supported by COBYLA. None of the other
    optimizers support constraints.

    ScipyOptimizer supports the following:
        equality_constraints
        inequality_constraints

    Options
    -------
    options['disp'] :  bool(True)
        Set to False to prevent printing of Scipy convergence messages
    options['maxiter'] : int(200)
        Maximum number of iterations.
    options['optimizer'] : str('SLSQP')
        Name of optimizer to use
    options['tol'] :  float(1e-06)
        Tolerance for termination. For detailed control, use solver-specific options.

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
    _con_idx : OrderedDict
        Used for constraint bookkeeping in the presence of 2-sided constraints.
    _cons : dict
        Contains all constraint info.
    _designvars : dict
        Contains all design variable info.
    _grad_cache : OrderedDict
        Cached result of constraint derivatives because scipy asks for them in a separate function.
    _objs : dict
        Contains all objective info.
    _exc_info : 3 item tuple
        Storage for exception and traceback information.
    """

    def __init__(self):
        """
        Initialize the ScipyOptimizer.
        """
        super(ScipyOptimizer, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['two_sided_constraints'] = True

        # What we don't support
        self.supports['multiple_objectives'] = False
        self.supports['active_set'] = False
        self.supports['integer_design_vars'] = False
        self.supports['linear_constraints'] = False

        # User Options
        self.options.declare('optimizer', 'SLSQP', values=_optimizers,
                             desc='Name of optimizer to use')
        self.options.declare('tol', 1.0e-6, lower=0.0,
                             desc='Tolerance for termination. For detailed '
                             'control, use solver-specific options.')
        self.options.declare('maxiter', 200, lower=0,
                             desc='Maximum number of iterations.')
        self.options.declare('disp', True,
                             desc='Set to False to prevent printing of Scipy convergence messages')

        # The user places optimizer-specific settings in here.
        self.opt_settings = OrderedDict()

        self.result = None
        self.fail = 0
        self._grad_cache = None
        self._con_cache = None
        self._con_idx = OrderedDict()
        self.objs = None
        self.fail = False
        self.iter_count = 0
        self._exc_info = None

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer
        """
        super(ScipyOptimizer, self)._setup_driver(problem)
        opt = self.options['optimizer']

        self.supports['gradients'] = opt in _gradient_optimizers
        self.supports['inequality_constraints'] = opt in _constraint_optimizers
        self.supports['two_sided_constraints'] = opt in _constraint_optimizers
        self.supports['equality_constraints'] = opt in _eq_constraint_optimizers

    def run(self):
        """
        Optimize the problem using selected Scipy optimizer.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        opt = self.options['optimizer']
        problem = self._problem
        model = self._problem.model
        self.iter_count = 0

        # Initial Run
        model._solve_nonlinear()

        self.objs = list(self.get_objective_values())
        self._con_cache = self.get_constraint_values()
        desvar_vals = self.get_design_var_values()

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
                for j in range(0, size):

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
        i = 0
        if opt in _constraint_optimizers:
            for name, meta in iteritems(self._cons):
                size = meta['size']

                # Loop over every index separately, because scipy calls each constraint by index.
                for j in range(0, size):
                    con_dict = OrderedDict()
                    if meta['equals'] is not None:
                        con_dict['type'] = 'eq'
                    else:
                        con_dict['type'] = 'ineq'
                    con_dict['fun'] = self._confunc
                    if opt in _constraint_grad_optimizers:
                        con_dict['jac'] = self._congradfunc
                    con_dict['args'] = [name, j]
                    constraints.append(con_dict)

                    upper = meta['upper']
                    if isinstance(upper, np.ndarray):
                        upper = upper[j]

                    lower = meta['lower']
                    if isinstance(lower, np.ndarray):
                        lower = lower[j]

                    dblcon = (upper < sys.float_info.max) and (lower > -sys.float_info.max)

                    # Add extra constraint if double-sided
                    if dblcon:
                        dblname = '2bl-' + name
                        con_dict = OrderedDict()
                        con_dict['type'] = 'ineq'
                        con_dict['fun'] = self._confunc
                        if opt in _constraint_grad_optimizers:
                            con_dict['jac'] = self._congradfunc
                        con_dict['args'] = [dblname, j]
                        constraints.append(con_dict)

                self._con_idx[name] = i
                i += size

        # Provide gradients for optimizers that support it
        if opt in _gradient_optimizers:
            jac = self._gradfunc
        else:
            jac = None

        # optimize
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

        if self._exc_info is not None:
            self._reraise()

        self.result = result
        self.fail = False if self.result.success else True

        if self.options['disp']:
            print('Optimization Complete')
            print('-' * 35)

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

            with Recording(self.options['optimizer'], self.iter_count, self) as rec:
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

    def _confunc(self, x_new, name, idx):
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
        idx : float
            Contains index into the constraint array.

        Returns
        -------
        float
            Value of the constraint function.
        """
        if self._exc_info is not None:
            self._reraise()

        if name.startswith('2bl-'):
            name = name[4:]
            dbl_side = True
        else:
            dbl_side = False

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

        if (lower == -sys.float_info.max) or dbl_side:
            return upper - cons[name][idx]
        else:
            return cons[name][idx] - lower

    def _gradfunc(self, x_new):
        """
        Evaluate and return the objective function.

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
            quantities = list(self._objs) + list(self._cons)
            grad = self._compute_totals(of=quantities, wrt=list(self._designvars),
                                        return_format='array')
            self._grad_cache = grad

        except Exception as msg:
            self._exc_info = sys.exc_info()
            return np.array([[]])

        # print("Gradients calculated")
        # print(x_new)
        # print(grad[0, :])

        return grad[0, :]

    def _congradfunc(self, x_new, name, idx):
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
        idx : float
            Contains index into the constraint array.

        Returns
        -------
        float
            Gradient of the constraint function wrt all params.
        """
        if self._exc_info is not None:
            self._reraise()

        if name.startswith('2bl-'):
            name = name[4:]
            dbl_side = True
        else:
            dbl_side = False

        grad = self._grad_cache
        meta = self._cons[name]
        grad_idx = self._con_idx[name] + idx + 1

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

        if (lower == -sys.float_info.max) or dbl_side:
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
