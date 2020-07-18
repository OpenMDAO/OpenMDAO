"""
OpenMDAO Wrapper for the scipy.optimize.minimize family of local optimizers.
"""


import sys
from collections import OrderedDict
from distutils.version import LooseVersion

import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import minimize

import openmdao
import openmdao.utils.coloring as coloring_mod
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.general_utils import simple_warning
from openmdao.utils.class_util import weak_method_wrapper
from openmdao.utils.mpi import MPI

# Optimizers in scipy.minimize
_optimizers = {'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
               'TNC', 'COBYLA', 'SLSQP'}
if LooseVersion(scipy_version) >= LooseVersion("1.1"):  # Only available in newer versions
    _optimizers.add('trust-constr')

# For 'basinhopping' and 'shgo' gradients are used only in the local minimization
_gradient_optimizers = {'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg',
                        'trust-ncg', 'trust-constr', 'basinhopping', 'shgo'}
_hessian_optimizers = {'trust-constr', 'trust-ncg'}
_bounds_optimizers = {'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr', 'dual_annealing', 'shgo',
                      'differential_evolution', 'basinhopping'}
_constraint_optimizers = {'COBYLA', 'SLSQP', 'trust-constr', 'shgo'}
_constraint_grad_optimizers = _gradient_optimizers & _constraint_optimizers
_eq_constraint_optimizers = {'SLSQP', 'trust-constr'}
_global_optimizers = {'differential_evolution', 'basinhopping'}
if LooseVersion(scipy_version) >= LooseVersion("1.2"):  # Only available in newer versions
    _global_optimizers |= {'shgo', 'dual_annealing'}

# Global optimizers and optimizers in minimize
_all_optimizers = _optimizers | _global_optimizers

# These require Hessian or Hessian-vector product, so they are not supported
# right now.
# dual-annealing and basinhopping not supported yet
_unsupported_optimizers = {'dogleg', 'trust-ncg'}

# With "old-style" a constraint is a dictionary, with "new-style" an object
# With "old-style" a bound is a tuple, with "new-style" a Bounds instance
# In principle now everything can work with "old-style"
# These settings have no effect to the optimizers implemented before SciPy 1.1
_supports_new_style = {'trust-constr'}
_use_new_style = True  # Recommended to set to True

CITATIONS = """
@article{Hwang_maud_2018
 author = {Hwang, John T. and Martins, Joaquim R.R.A.},
 title = "{A Computational Architecture for Coupling Heterogeneous
          Numerical Models and Computing Coupled Derivatives}",
 journal = "{ACM Trans. Math. Softw.}",
 volume = {44},
 number = {4},
 month = jun,
 year = {2018},
 pages = {37:1--37:39},
 articleno = {37},
 numpages = {39},
 doi = {10.1145/3182393},
 publisher = {ACM},
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
    _con_cache : dict
        Cached result of constraint evaluations because scipy asks for them in a separate function.
    _con_idx : dict
        Used for constraint bookkeeping in the presence of 2-sided constraints.
    _grad_cache : OrderedDict
        Cached result of nonlinear constraint derivatives because scipy asks for them in a separate
        function.
    _exc_info : 3 item tuple
        Storage for exception and traceback information.
    _obj_and_nlcons : list
        List of objective + nonlinear constraints. Used to compute total derivatives
        for all except linear constraints.
    _dvlist : list
        Copy of _designvars.
    _lincongrad_cache : np.ndarray
        Pre-calculated gradients of linear constraints.
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
        self.supports._read_only = True

        # The user places optimizer-specific settings in here.
        self.opt_settings = OrderedDict()

        self.result = None
        self._grad_cache = None
        self._con_cache = None
        self._con_idx = {}
        self._obj_and_nlcons = None
        self._dvlist = None
        self._lincongrad_cache = None
        self.fail = False
        self.iter_count = 0
        self._exc_info = None

        self.cite = CITATIONS

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('optimizer', 'SLSQP', values=_all_optimizers,
                             desc='Name of optimizer to use')
        self.options.declare('tol', 1.0e-6, lower=0.0,
                             desc='Tolerance for termination. For detailed '
                             'control, use solver-specific options.')
        self.options.declare('maxiter', 200, lower=0,
                             desc='Maximum number of iterations.')
        self.options.declare('disp', True, types=bool,
                             desc='Set to False to prevent printing of Scipy convergence messages')

    def _get_name(self):
        """
        Get name of current optimizer.

        Returns
        -------
        str
            The name of the current optimizer.
        """
        return "ScipyOptimize_" + self.options['optimizer']

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

        self.supports._read_only = False
        self.supports['gradients'] = opt in _gradient_optimizers
        self.supports['inequality_constraints'] = opt in _constraint_optimizers
        self.supports['two_sided_constraints'] = opt in _constraint_optimizers
        self.supports['equality_constraints'] = opt in _eq_constraint_optimizers
        self.supports._read_only = True

        # Raises error if multiple objectives are not supported, but more objectives were defined.
        if not self.supports['multiple_objectives'] and len(self._objs) > 1:
            msg = '{} currently does not support multiple objectives.'
            raise RuntimeError(msg.format(self.msginfo))

        # Since COBYLA does not support bounds, we
        #   need to add to the _cons metadata for any bounds that
        #   need to be translated into a constraint
        if opt == 'COBYLA':
            for name, meta in self._designvars.items():
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
                    d['total_adder'] = None
                    d['total_scaler'] = None
                    d['size'] = meta['size']
                    d['global_size'] = meta['global_size']
                    d['distributed'] = meta['distributed']
                    d['linear'] = True
                    d['ivc_source'] = meta['ivc_source']
                    self._cons[name] = d

    def run(self):
        """
        Optimize the problem using selected Scipy optimizer.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        problem = self._problem()
        opt = self.options['optimizer']
        model = problem.model
        self.iter_count = 0
        self._total_jac = None

        self._check_for_missing_objective()

        # Initial Run
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            model.run_solve_nonlinear()
            self.iter_count += 1

        self._con_cache = self.get_constraint_values()
        desvar_vals = self.get_design_var_values()
        self._dvlist = list(self._designvars)

        # maxiter and disp get passed into scipy with all the other options.
        if 'maxiter' not in self.opt_settings:  # lets you override the value in options
            self.opt_settings['maxiter'] = self.options['maxiter']
        self.opt_settings['disp'] = self.options['disp']

        # Size Problem
        nparam = 0
        for param in self._designvars.values():
            size = param['global_size'] if param['distributed'] else param['size']
            nparam += size
        x_init = np.empty(nparam)

        # Initial Design Vars
        i = 0
        use_bounds = (opt in _bounds_optimizers)
        if use_bounds:
            bounds = []
        else:
            bounds = None

        for name, meta in self._designvars.items():
            size = meta['global_size'] if meta['distributed'] else meta['size']
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

        if use_bounds and (opt in _supports_new_style) and _use_new_style:
            # For 'trust-constr' it is better to use the new type bounds, because it seems to work
            # better (for the current examples in the tests) with the "keep_feasible" option
            try:
                from scipy.optimize import Bounds
                from scipy.optimize._constraints import old_bound_to_new
            except ImportError:
                msg = ('The "trust-constr" optimizer is supported for SciPy 1.1.0 and above. '
                       'The installed version is {}')
                raise ImportError(msg.format(scipy_version))

            # Convert "old-style" bounds to "new_style" bounds
            lower, upper = old_bound_to_new(bounds)  # tuple, tuple
            keep_feasible = self.opt_settings.get('keep_feasible_bounds', True)
            bounds = Bounds(lb=lower, ub=upper, keep_feasible=keep_feasible)

        # Constraints
        constraints = []
        i = 1  # start at 1 since row 0 is the objective.  Constraints start at row 1.
        lin_i = 0  # counter for linear constraint jacobian
        lincons = []  # list of linear constraints
        self._obj_and_nlcons = list(self._objs)

        if opt in _constraint_optimizers:
            for name, meta in self._cons.items():
                size = meta['global_size'] if meta['distributed'] else meta['size']
                upper = meta['upper']
                lower = meta['lower']
                equals = meta['equals']
                if opt in _gradient_optimizers and 'linear' in meta and meta['linear']:
                    lincons.append(name)
                    self._con_idx[name] = lin_i
                    lin_i += size
                else:
                    self._obj_and_nlcons.append(name)
                    self._con_idx[name] = i
                    i += size

                # In scipy constraint optimizers take constraints in two separate formats

                # Type of constraints is list of NonlinearConstraint
                if opt in _supports_new_style and _use_new_style:
                    try:
                        from scipy.optimize import NonlinearConstraint
                    except ImportError:
                        msg = ('The "trust-constr" optimizer is supported for SciPy 1.1.0 and'
                               'above. The installed version is {}')
                        raise ImportError(msg.format(scipy_version))

                    if equals is not None:
                        lb = ub = equals
                    else:
                        lb = lower
                        ub = upper
                    # Loop over every index separately,
                    # because scipy calls each constraint by index.
                    for j in range(size):
                        # Double-sided constraints are accepted by the algorithm
                        args = [name, False, j]
                        # TODO linear constraint if meta['linear']
                        # TODO add option for Hessian
                        con = NonlinearConstraint(
                            fun=signature_extender(weak_method_wrapper(self, '_con_val_func'),
                                                   args),
                            lb=lb, ub=ub,
                            jac=signature_extender(weak_method_wrapper(self, '_congradfunc'), args))
                        constraints.append(con)
                else:  # Type of constraints is list of dict
                    # Loop over every index separately,
                    # because scipy calls each constraint by index.
                    for j in range(size):
                        con_dict = {}
                        if meta['equals'] is not None:
                            con_dict['type'] = 'eq'
                        else:
                            con_dict['type'] = 'ineq'
                        con_dict['fun'] = weak_method_wrapper(self, '_confunc')
                        if opt in _constraint_grad_optimizers:
                            con_dict['jac'] = weak_method_wrapper(self, '_congradfunc')
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
                            dcon_dict['fun'] = weak_method_wrapper(self, '_confunc')
                            if opt in _constraint_grad_optimizers:
                                dcon_dict['jac'] = weak_method_wrapper(self, '_congradfunc')
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

        # Hessian calculation method for optimizers, which require it
        if opt in _hessian_optimizers:
            if 'hess' in self.opt_settings:
                hess = self.opt_settings.pop('hess')
            else:
                # Defaults to BFGS, if not in opt_settings
                from scipy.optimize import BFGS
                hess = BFGS()
        else:
            hess = None

        # compute dynamic simul deriv coloring if option is set
        if coloring_mod._use_total_sparsity:
            if ((self._coloring_info['coloring'] is None and self._coloring_info['dynamic'])):
                coloring_mod.dynamic_total_coloring(self, run_model=False,
                                                    fname=self._get_total_coloring_fname())

                # if the improvement wasn't large enough, turn coloring off
                info = self._coloring_info
                if info['coloring'] is not None:
                    pct = info['coloring']._solves_info()[-1]
                    if info['min_improve_pct'] > pct:
                        info['coloring'] = info['static'] = None
                        simple_warning("%s: Coloring was deactivated.  Improvement of %.1f%% was "
                                       "less than min allowed (%.1f%%)." %
                                       (self.msginfo, pct, info['min_improve_pct']))

        # optimize
        try:
            if opt in _optimizers:
                result = minimize(self._objfunc, x_init,
                                  # args=(),
                                  method=opt,
                                  jac=jac,
                                  hess=hess,
                                  # hessp=None,
                                  bounds=bounds,
                                  constraints=constraints,
                                  tol=self.options['tol'],
                                  # callback=None,
                                  options=self.opt_settings)
            elif opt == 'basinhopping':
                from scipy.optimize import basinhopping

                def fun(x):
                    return self._objfunc(x), jac(x)

                if 'minimizer_kwargs' not in self.opt_settings:
                    self.opt_settings['minimizer_kwargs'] = {"method": "L-BFGS-B", "jac": True}
                self.opt_settings.pop('maxiter')  # It does not have this argument

                def accept_test(f_new, x_new, f_old, x_old):
                    # Used to implement bounds besides the original functionality
                    if bounds is not None:
                        bound_check = all([b[0] <= xi <= b[1] for xi, b in zip(x_new, bounds)])
                        user_test = self.opt_settings.pop('accept_test', None)  # callable
                        # has to satisfy both the bounds and the acceptance test defined by the
                        # user
                        if user_test is not None:
                            test_res = user_test(f_new, x_new, f_old, x_old)
                            if test_res == 'force accept':
                                return test_res
                            else:  # result is boolean
                                return bound_check and test_res
                        else:  # no user acceptance test, check only the bounds
                            return bound_check
                    else:
                        return True

                result = basinhopping(fun, x_init,
                                      accept_test=accept_test,
                                      **self.opt_settings)
            elif opt == 'dual_annealing':
                from scipy.optimize import dual_annealing
                self.opt_settings.pop('disp')  # It does not have this argument
                # There is no "options" param, so "opt_settings" can be used to set the (many)
                # keyword arguments
                result = dual_annealing(self._objfunc,
                                        bounds=bounds,
                                        **self.opt_settings)
            elif opt == 'differential_evolution':
                from scipy.optimize import differential_evolution
                # There is no "options" param, so "opt_settings" can be used to set the (many)
                # keyword arguments
                result = differential_evolution(self._objfunc,
                                                bounds=bounds,
                                                **self.opt_settings)
            elif opt == 'shgo':
                from scipy.optimize import shgo
                kwargs = dict()
                for param in ('minimizer_kwargs', 'sampling_method ', 'n', 'iters'):
                    if param in self.opt_settings:
                        kwargs[param] = self.opt_settings[param]
                # Set the Jacobian and the Hessian to the value calculated in OpenMDAO
                if 'minimizer_kwargs' not in kwargs or kwargs['minimizer_kwargs'] is None:
                    kwargs['minimizer_kwargs'] = {}
                kwargs['minimizer_kwargs'].setdefault('jac', jac)
                kwargs['minimizer_kwargs'].setdefault('hess', hess)
                # Objective function tolerance
                self.opt_settings['f_tol'] = self.options['tol']
                result = shgo(self._objfunc,
                              bounds=bounds,
                              constraints=constraints,
                              options=self.opt_settings,
                              **kwargs)
            else:
                msg = 'Optimizer "{}" is not implemented yet. Choose from: {}'
                raise NotImplementedError(msg.format(opt, _all_optimizers))

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

        if hasattr(result, 'success'):
            self.fail = False if result.success else True
            if self.fail:
                print('Optimization FAILED.')
                print(result.message)
                print('-' * 35)

            elif self.options['disp']:
                print('Optimization Complete')
                print('-' * 35)
        else:
            self.fail = True  # It is not known, so the worst option is assumed
            print('Optimization Complete (success not known)')
            print(result.message)
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
        model = self._problem().model

        try:

            # Pass in new parameters
            i = 0
            if MPI:
                model.comm.Bcast(x_new, root=0)
            for name, meta in self._designvars.items():
                size = meta['size']
                self.set_design_var(name, x_new[i:i + size])
                i += size

            with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
                self.iter_count += 1
                model.run_solve_nonlinear()

            # Get the objective function evaluations
            for obj in self.get_objective_values().values():
                f_new = obj
                break

            self._con_cache = self.get_constraint_values()

        except Exception as msg:
            self._exc_info = msg
            return 0

        # print("Functions calculated")
        # print('   xnew', x_new)
        # print('   fnew', f_new)

        return f_new

    def _con_val_func(self, x_new, name, dbl, idx):
        """
        Return the value of the constraint function requested in args.

        The lower or upper bound is **not** subtracted from the value. Used for optimizers,
        which take the bounds of the constraints (e.g. trust-constr)

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
        return self._con_cache[name][idx]

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
            return cons[name][idx] - equals

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
            self._exc_info = msg
            return np.array([[]])

        # print("Gradients calculated for objective")
        # print('   xnew', x_new)
        # print('   grad', grad[0, :])

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
        # print('   xnew', x_new)
        # print('   grad', name, 'idx', idx, grad[grad_idx, :])

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
        raise exc


def signature_extender(fcn, extra_args):
    """
    Closure function, which appends extra arguments to the original function call.

    The first argument is the design vector. The possible extra arguments from the callback
    of :func:`scipy.optimize.minimize` are not passed to the function.

    Some algorithms take a sequence of :class:`~scipy.optimize.NonlinearConstraint` as input
    for the constraints. For this class it is not possible to pass additional arguments.
    With this function the signature will be correct for both scipy and the driver.

    Parameters
    ----------
    fcn : callable
        Function, which takes the design vector as the first argument.
    extra_args : tuple or list
        Extra arguments for the function

    Returns
    -------
    callable
        The function with the signature expected by the driver.
    """
    def closure(x, *args):
        return fcn(x, *extra_args)

    return closure
