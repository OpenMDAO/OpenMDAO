"""
OpenMDAO Wrapper for the NLopt package of optimizers.

"""


import sys
from collections import OrderedDict
from distutils.version import LooseVersion

import numpy as np
import nlopt

import openmdao
import openmdao.utils.coloring as coloring_mod
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.general_utils import simple_warning
from openmdao.utils.class_util import weak_method_wrapper
from openmdao.utils.mpi import MPI


# All optimizers in NLopt and their corresponding package name
optimizer_methods = { 'GN_DIRECT' : nlopt.GN_DIRECT,
                      'GN_DIRECT_L' : nlopt.GN_DIRECT_L,
                      # 'GN_DIRECT_L_RAND' : nlopt.GN_DIRECT_L_RAND,
                      'GN_DIRECT_L_NOSCAL' : nlopt.GN_DIRECT_L_NOSCAL,
                      # 'GN_DIRECT_L_RAND_NOSCAL' : nlopt.GN_DIRECT_L_RAND_NOSCAL,
                      'GN_ORIG_DIRECT' : nlopt.GN_ORIG_DIRECT,
                      'GN_ORIG_DIRECT_L' : nlopt.GN_ORIG_DIRECT_L,
                      'GN_CRS2_LM' : nlopt.GN_CRS2_LM,
                      'G_MLSL_LDS' : nlopt.G_MLSL_LDS,
                      'G_MLSL' : nlopt.G_MLSL,
                      'GD_STOGO' : nlopt.GD_STOGO,
                      'GD_STOGO_RAND' : nlopt.GD_STOGO_RAND,
                      'GN_AGS' : nlopt.GN_AGS,
                      'GN_ISRES' : nlopt.GN_ISRES,
                      'GN_ESCH' : nlopt.GN_ESCH,
                      'LN_COBYLA' : nlopt.LN_COBYLA,
                      'LN_BOBYQA' : nlopt.LN_BOBYQA,
                      'LN_NEWUOA' : nlopt.LN_NEWUOA,
                      'LN_NEWUOA_BOUND' : nlopt.LN_NEWUOA_BOUND,
                      'LN_PRAXIS' : nlopt.LN_PRAXIS,
                      'LN_NELDERMEAD' : nlopt.LN_NELDERMEAD,
                      'LN_SBPLX' : nlopt.LN_SBPLX,
                      'LD_MMA' : nlopt.LD_MMA,
                      'LD_CCSAQ' : nlopt.LD_CCSAQ,
                      'LD_SLSQP' : nlopt.LD_SLSQP,
                      'LD_TNEWTON_PRECOND_RESTART' : nlopt.LD_TNEWTON_PRECOND_RESTART,
                      'LD_TNEWTON_PRECOND' : nlopt.LD_TNEWTON_PRECOND,
                      'LD_TNEWTON_RESTART' : nlopt.LD_TNEWTON_RESTART,
                      'LD_TNEWTON' : nlopt.LD_TNEWTON,
                      'LD_VAR2' : nlopt.LD_VAR2,
                      'LD_VAR1' : nlopt.LD_VAR1,
                      'AUGLAG' : nlopt.AUGLAG,
                      'AUGLAG_EQ' : nlopt.AUGLAG_EQ,
                      }
                
_optimizers = set(optimizer_methods)

# Define subsets of optimizers that support different functions
_gradient_optimizers = {'LD_MMA', 'LD_SLSQP', 'LD_LBFGS',
                        'LD_TNEWTON_PRECOND_RESTART', 'LD_TNEWTON_PRECOND',
                        'LD_TNEWTON_RESTART', 'LD_TNEWTON',
                        'LD_VAR2', 'LD_VAR1', 'AUGLAG',
                        'AUGLAG_EQ', 'GD_STOGO', 'GD_STOGO_RAND'}
_bounds_optimizers = _optimizers
_constraint_optimizers = {'LD_SLSQP', 'LN_COBYLA', 'LD_MMA', 'LD_CCSAQ',
                          'GN_ORIG_DIRECT',
                          'GN_ORIG_DIRECT_L', 'GN_ISRES'}  # These are all tested
_constraint_grad_optimizers = _gradient_optimizers & _constraint_optimizers
_eq_constraint_optimizers = {'LD_SLSQP', 'LN_COBYLA', 'GN_ISRES'}
_global_optimizers = {'GN_DIRECT',
                      'GN_DIRECT_L',
                      'GN_DIRECT_L_RAND',
                      'GN_DIRECT_L_NOSCAL',
                      'GN_DIRECT_L_RAND_NOSCAL',
                      'GN_ORIG_DIRECT',
                      'GN_ORIG_DIRECT_L',
                      'GN_CRS2_LM',
                      'G_MLSL_LDS',
                      'G_MLSL',
                      'GD_STOGO',
                      'GD_STOGO_RAND',
                      'GN_AGS',
                      'GN_ISRES',
                      'GN_ESCH',}

CITATIONS = """
@article{johnson_nlopt
 author = {Johnson, Steven G.},
 title = "{The NLopt nonlinear-optimization package}",
 url = {http://github.com/stevengj/nlopt},
 }
"""


class NLoptDriver(Driver):
    """

    NLoptDriver supports the following:
        equality_constraints
        inequality_constraints

    Attributes
    ----------
    fail : bool
        Flag that indicates failure of most recent optimization.
    iter_count : int
        Counter for function evaluations.
    result : OptimizeResult
        Result returned from NLopt.optimize call.
    opt_settings : dict
        Dictionary of solver-specific options. See the NLopt.optimize documentation.
    _con_cache : dict
        Cached result of constraint evaluations because NLopt asks for them in a separate function.
    _con_idx : dict
        Used for constraint bookkeeping in the presence of 2-sided constraints.
    _grad_cache : OrderedDict
        Cached result of nonlinear constraint derivatives because NLopt asks for them in a separate
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
        Initialize the NLoptDriver.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        super(NLoptDriver, self).__init__(**kwargs)

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['two_sided_constraints'] = True
        self.supports['linear_constraints'] = True
        self.supports['simultaneous_derivatives'] = True

        # What we don't support
        self.supports['multiple_objectives'] = False
        self.supports['active_set'] = False
        self.supports['integer_design_vars'] = False  # TODO : add support for integer variables
        self.supports._read_only = True

        # The user places optimizer-specific settings in here.
        self.opt_settings = OrderedDict()  # TODO : add opt_settings for each optimizer

        self.result = None
        self._grad_cache = None
        self._con_cache = None
        self._con_idx = {}
        self._obj_and_nlcons = None
        self._dvlist = None
        self._lincongrad_cache = None
        self.fail = False  # TODO : add actual failure testing for the driver
        self.iter_count = 0
        self._exc_info = None

        self.cite = CITATIONS

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('optimizer', 'LD_SLSQP', values=_optimizers,
                             desc='Name of optimizer to use')
        self.options.declare('tol', 1.0e-6, lower=0.0,
                             desc='Tolerance for termination. For detailed '
                             'control, use solver-specific options.')
        self.options.declare('maxiter', 200, lower=0,
                             desc='Maximum number of iterations.')

    def _get_name(self):
        """
        Get name of current optimizer.

        Returns
        -------
        str
            The name of the current optimizer.
        """
        return "NLopt" + self.options['optimizer']

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer
        """
        super(NLoptDriver, self)._setup_driver(problem)
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

    def run(self):
        """
        Optimize the problem using selected NLopt optimizer.

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

        # Size Problem
        nparam = 0
        for param in self._designvars.values():
            nparam += param['size']
        x_init = np.empty(nparam)
        
        # Initialize the NLopt problem with the method and number of design vars
        opt_prob = nlopt.opt(optimizer_methods[opt], int(nparam))

        # Initial Design Vars
        i = 0
        use_bounds = (opt in _bounds_optimizers)
        if use_bounds:
            bounds = []
        else:
            bounds = None

        # Loop through all OpenMDAO design variables and process their bounds
        for name, meta in self._designvars.items():
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

        # TODO : refactor this so the call to old_bound_to_new isn't needed.
        # Actually add the bounds to the optimization problem.
        if bounds is not None:
            from scipy.optimize._constraints import old_bound_to_new
            lower, upper = old_bound_to_new(bounds)  # tuple, tuple
            opt_prob.set_lower_bounds(lower)
            opt_prob.set_upper_bounds(upper)

        # Constraints
        constraints = []
        i = 1  # start at 1 since row 0 is the objective.  Constraints start at row 1.
        lin_i = 0  # counter for linear constraint jacobian
        lincons = []  # list of linear constraints
        self._obj_and_nlcons = list(self._objs)
        
        # Process and add constraints to the optimization problem.
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
                
                # Loop over every index separately,
                # because it's easier to defined each
                # constraint by index.
                for j in range(size):
                                    
                    # Equality constraints are added as two inequality constraints
                    if equals is not None:
                        args = [name, False, j]
                        opt_prob.add_equality_constraint(signature_extender(weak_method_wrapper(self, '_confunc'), args))
                        
                    else:
                        # Double-sided constraints are accepted by the algorithm
                        args = [name, False, j]
                        opt_prob.add_inequality_constraint(signature_extender(weak_method_wrapper(self, '_confunc'), args))
                        
                        if isinstance(upper, np.ndarray):
                            upper = upper[j]

                        if isinstance(lower, np.ndarray):
                            lower = lower[j]
                        
                        dblcon = (upper < openmdao.INF_BOUND) and (lower > -openmdao.INF_BOUND)

                        # Add extra constraint if double-sided
                        if dblcon:
                            args = [name, True, j]
                            opt_prob.add_inequality_constraint(signature_extender(weak_method_wrapper(self, '_confunc'), args))
                            
            # precalculate gradients of linear constraints
            if lincons:
                self._lincongrad_cache = self._compute_totals(of=lincons, wrt=self._dvlist,
                                                              return_format='array')
            else:
                self._lincongrad_cache = None

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

        # Finalize the optimization problem setup and actually perform optimization
        try:
            if opt in _optimizers:
                opt_prob.set_min_objective(self._objfunc)
                opt_prob.set_ftol_rel(self.options['tol'])
                opt_prob.set_maxeval(self.options['maxiter'])
                xopt = opt_prob.optimize(x_init)
                # TODO : ensure that the optimal result is the
                # last result run by the model. Unclear if this is always
                # the case for all algos in NLopt
                self.result = result = opt_prob.last_optimize_result()
                
            else:
                msg = 'Optimizer "{}" is not implemented yet. Choose from: {}'
                raise NotImplementedError(msg.format(opt, _optimizers))
                
        # If an exception was swallowed in one of our callbacks, we want to raise it
        except Exception as msg:
            if self._exc_info is not None:
                self._reraise()
            else:
                raise
                
        if self._exc_info is not None:
            self._reraise()
                
        self.fail = False

        return self.fail

    def _objfunc(self, x_new, grad):
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
            for name, meta in self._designvars.items():
                size = meta['size']
                self.set_design_var(name, x_new[i:i + size])
                i += size

            with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
                self.iter_count += 1
                
                # This is the actual model evaluation for OpenMDAO
                model.run_solve_nonlinear()

            # TODO : remove obfuscation caused by this; not needed
            # Get the objective function evaluations
            for obj in self.get_objective_values().values():
                f_new = obj
                break

            self._con_cache = self.get_constraint_values()

        except Exception as msg:
            self._exc_info = msg
            obj = 0
            
        try:
            if grad.size > 0:
                self._grad_cache = self._compute_totals(of=self._obj_and_nlcons, wrt=self._dvlist,
                                            return_format='array')
                grad[:] = self._grad_cache[0, :]

        except Exception as msg:
            self._exc_info = msg
            
        return float(f_new)

    def _confunc(self, x_new, grad, name, dbl, idx):
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
        
        if meta['linear']:
            grad_cache = self._lincongrad_cache
        else:
            grad_cache = self._grad_cache
        
        grad_idx = self._con_idx[name] + idx
        
        # Equality constraints
        equals = meta['equals']
        if equals is not None:
            if isinstance(equals, np.ndarray):
                equals = equals[idx]
            if grad.size > 0:
                grad[:] = grad_cache[grad_idx, :]
            return cons[name][idx] - equals

        # Note, NLopt defines constraints to be satisfied when negative,
        # which is the same as OpenMDAO.
        upper = meta['upper']
        if isinstance(upper, np.ndarray):
            upper = upper[idx]

        lower = meta['lower']
        if isinstance(lower, np.ndarray):
            lower = lower[idx]
            
        if dbl or (lower <= -openmdao.INF_BOUND):
            if grad.size > 0:
                grad[:] = grad_cache[grad_idx, :]
            return cons[name][idx] - upper
        else:
            if grad.size > 0:
                grad[:] = -grad_cache[grad_idx, :]
            return lower - cons[name][idx]

    def _reraise(self):
        """
        Reraise any exception encountered when NLopt calls back into our method.
        """
        exc = self._exc_info
        raise exc


def signature_extender(fcn, extra_args):
    """
    Closure function, which appends extra arguments to the original function call.

    The first argument is the design vector and the second is the gradient vector.
    The possible extra arguments from the callback
    of :func:`NLopt.optimize` are not passed to the function.

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
    def closure(x, grad, *args):
        return fcn(x, grad, *extra_args)

    return closure
