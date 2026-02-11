"""
OpenMDAO Wrapper for the ModOpt optimization library.

ModOpt is a modular optimization framework providing interfaces to various
gradient-based and gradient-free optimization algorithms.

Available Optimizers
--------------------
Gradient-Based:
    - SLSQP: Sequential Least Squares Programming (supports all constraint types)
    - PySLSQP: Pure Python implementation of SLSQP
    - BFGS: Broyden-Fletcher-Goldfarb-Shanno (unconstrained)
    - LBFGSB: Limited-memory BFGS with bounds
    - TrustConstr: Trust-region constrained algorithm
    - SNOPT: Sparse Nonlinear Optimizer (requires license)
    - IPOPT: Interior Point Optimizer (requires separate installation)
    - SQP: Sequential Quadratic Programming
    - InteriorPoint: Interior point method

Gradient-Free:
    - COBYLA: Constrained Optimization BY Linear Approximation
    - COBYQA: Constrained Optimization BY Quadratic Approximation
    - NelderMead: Nelder-Mead simplex algorithm (unconstrained)

Notes
-----
- SLSQP is the default optimizer and supports gradients, bounds, and all constraint types
- SNOPT and IPOPT offer high performance but require separate installation/licenses
- Linear constraints are handled efficiently by pre-computing their Jacobians
- Gradient-free methods (COBYLA, COBYQA, NelderMead) are useful when derivatives are unavailable

See the ModOpt documentation at https://modopt.readthedocs.io for detailed information
on algorithm-specific options and capabilities.
"""
import numpy as np
from openmdao.core.driver import Driver, RecordingDebugging
try:
    import modopt as mo
except ImportError:
    mo = None

# Performant algorithms from ModOpt that support gradients
_gradient_optimizers = {
    'SLSQP', 'PySLSQP', 'BFGS', 'LBFGSB', 'TrustConstr',
    'SNOPT', 'IPOPT', 'SQP', 'InteriorPoint'
}

# Algorithms that support constraints (inequality and/or equality)
_constraint_optimizers = {
    'SLSQP', 'PySLSQP', 'COBYLA', 'TrustConstr', 'COBYQA',
    'SNOPT', 'IPOPT', 'SQP', 'InteriorPoint'
}

# Algorithms that support equality constraints
_eq_constraint_optimizers = {
    'SLSQP', 'PySLSQP', 'TrustConstr', 'SNOPT', 'IPOPT',
    'SQP', 'InteriorPoint', 'NewtonLagrange'
}

# Algorithms that support bounds
_bounds_optimizers = {
    'SLSQP', 'PySLSQP', 'LBFGSB', 'TrustConstr', 'COBYLA',
    'COBYQA', 'SNOPT', 'IPOPT', 'SQP', 'InteriorPoint'
}

# Gradient-based algorithms that also support constraints
_constraint_grad_optimizers = _gradient_optimizers & _constraint_optimizers

# All available optimizers (performant algorithms commonly used)
_all_optimizers = {
    'SLSQP', 'PySLSQP', 'COBYLA', 'BFGS', 'LBFGSB', 'NelderMead',
    'COBYQA', 'TrustConstr', 'SNOPT', 'IPOPT',
    'SQP', 'InteriorPoint'
}

CITATIONS = """
@misc{modopt,
 author = {ModOpt Development Team},
 title = "{ModOpt: A Modular Optimization Framework for Multidisciplinary Design Optimization}",
 howpublished = {\\url{https://github.com/LSDOlab/modopt}},
 year = {2023},
 note = {Software package}
}
"""


class ModOptProblem(mo.Problem):
    """
    ModOpt Problem that delegates objective and constraint evaluation to an OpenMDAO driver.

    This class wraps an OpenMDAO problem as a ModOpt optimization problem, translating
    between ModOpt's interface and OpenMDAO's driver interface.

    Parameters
    ----------
    driver : ModOptDriver
        The OpenMDAO driver managing the optimization.
    x : ndarray
        Initial values for design variables.
    bounds : list of lists or None
        Two-element list containing [lower_bounds, upper_bounds] for design variables,
        or None if no bounds.
    l_con_jac : ndarray or None
        Pre-computed Jacobian for linear constraints, or None if no linear constraints.
    l_con_bounds : list of lists
        Two-element list containing [lower_bounds, upper_bounds] for linear constraints.
    nl_con_bounds : list of lists
        Two-element list containing [lower_bounds, upper_bounds] for nonlinear constraints.

    Attributes
    ----------
    driver : ModOptDriver
        Reference to the OpenMDAO driver.
    x : ndarray
        Initial design variable values.
    bounds : list or None
        Design variable bounds.
    l_con_jac : ndarray or None
        Linear constraint Jacobian.
    l_con_bounds : list
        Linear constraint bounds.
    nl_con_bounds : list
        Nonlinear constraint bounds.
    _con_cache : dict or None
        Cached constraint values to avoid redundant evaluations.
    """

    def __init__(self, driver, x_init, bounds, l_con_jac, l_con_bounds, nl_con_bounds):
        """
        Initialize the ModOptProblem.

        Parameters
        ----------
        driver : ModOptDriver
            The OpenMDAO driver managing the optimization.
        x_init : ndarray
            Initial values for design variables.
        bounds : list of lists or None
            Design variable bounds.
        l_con_jac : ndarray or None
            Pre-computed Jacobian for linear constraints.
        l_con_bounds : list of lists
            Linear constraint bounds.
        nl_con_bounds : list of lists
            Nonlinear constraint bounds.
        """
        self.driver = driver
        self.x_init = x_init
        self.bounds = bounds
        self.l_con_jac = l_con_jac
        self.l_con_bounds = l_con_bounds
        self.nl_con_bounds = nl_con_bounds
        self._con_cache = None
        super().__init__()

    def initialize(self):
        self.problem_name = 'modopt_problem'

    def setup(self):
        """
        Add design variables, constraints, and objective to the ModOpt Problem.
        """
        if self.bounds is not None:
            self.add_design_variables(
                name='x',
                shape=self.x_init.shape,
                lower=np.array(self.bounds[0]),
                upper=np.array(self.bounds[1]),
                vals=self.x_init,
            )
        else:
            self.add_design_variables(
                name='x',
                shape=self.x_init.shape,
                vals=self.x_init,
            )

        self.add_objective(name='f')

        if len(self.l_con_bounds[0]) > 0:
            self.add_constraints(
                name='l_con',
                shape=(len(self.l_con_bounds[0]),),
                lower=np.array(self.l_con_bounds[0]),
                upper=np.array(self.l_con_bounds[1]),
            )
        if len(self.nl_con_bounds[0]) > 0:
            self.add_constraints(
                name='nl_con',
                shape=(len(self.nl_con_bounds[0]),),
                lower=np.array(self.nl_con_bounds[0]),
                upper=np.array(self.nl_con_bounds[1]),
            )

    def setup_derivatives(self):
        """
        Declare objective and constraint gradients to the ModOpt problem.

        This method informs ModOpt about which derivatives are available.
        For linear constraints, the Jacobian is provided directly. For nonlinear
        constraints, derivatives are computed on-demand.
        """
        # TODO: Pull sparsity info here
        self.declare_objective_gradient(wrt='x')

        if len(self.l_con_bounds[0]) > 0 and self.l_con_jac is not None:
            self.declare_constraint_jacobian(of='l_con', wrt='x',
                                             vals=self.l_con_jac)
        if len(self.nl_con_bounds[0]) > 0:
            self.declare_constraint_jacobian(of='nl_con', wrt='x')

    def compute_objective(self, dvs, obj):
        """
        Evaluate the objective function at the given design point.

        This method updates the OpenMDAO model with new design variables,
        runs the model, and retrieves the objective value. Constraint values
        are also cached for efficiency.

        Parameters
        ----------
        dvs : dict
            Dictionary with the current design variable names and values.
        obj : dict
            Dictionary to store the computed objective value.
        """
        model = self.driver._problem().model
        self._update_desvar_values(dvs['x'])

        with RecordingDebugging(self.driver._get_name(), self.driver.iter_count, self.driver):
            self.driver.iter_count += 1
            with model._relevance.nonlinear_active('iter'):
                self.driver._run_solve_nonlinear()

        # Get the objective function evaluations
        f_new = next(iter(self.driver.get_objective_values().values()))

        self._con_cache = self.driver.get_constraint_values()

        obj['f'] = f_new

    def compute_constraints(self, dvs, cons):
        """
        Compute constraint values.

        Uses cached constraint values if available, otherwise evaluates model.

        Parameters
        ----------
        dvs : dict
            Dictionary with the current design variable names and values.
        cons : dict
            Dictionary to store the computed constraint values.
        """
        # Use cached constraint values from compute_objective if available
        if self._con_cache is None:
            self._update_desvar_values(dvs['x'])
            self.driver._run_solve_nonlinear()
            vals = self.driver.get_constraint_values()
        else:
            vals = self._con_cache
            self._con_cache = None  # Clear cache after use

        # Separate linear and nonlinear constraints
        l_cons = []
        nl_cons = []
        for name, meta in self.driver._cons.items():
            if meta.get('linear'):
                l_cons.append(vals[name].flatten())
            else:
                nl_cons.append(vals[name].flatten())

        if l_cons:
            cons['l_con'] = np.concatenate(l_cons)
        if nl_cons:
            cons['nl_con'] = np.concatenate(nl_cons)

    def compute_objective_gradient(self, dvs, grad):
        """
        Compute the gradient of the objective function.

        Parameters
        ----------
        dvs : dict
            Dictionary with the current design variable names and values.

        Returns
        -------
        ndarray
            Gradient of the objective with respect to design variables.
        """
        self._update_desvar_values(dvs['x'])
        totals = self.driver._problem().compute_totals(
            of=list(self.driver._objs),
            wrt=list(self.driver._designvars),
            return_format='array'
        )
        grad['x'] = totals[0]

    def compute_constraint_jacobian(self, dvs, jac):
        """
        Compute the Jacobian of nonlinear constraints.

        Linear constraint Jacobians are pre-computed and provided during setup.
        Only nonlinear constraint Jacobians are computed here.

        Parameters
        ----------
        dvs : dict
            Dictionary with the current design variable names and values.
        jac : dict
            Dictionary to store the computed constraint Jacobians.
        """
        self._update_desvar_values(dvs['x'])

        # Separate linear and nonlinear constraints
        nl_cons = [name for name, meta in self.driver._cons.items() if not meta.get('linear')]

        if nl_cons:
            totals = self.driver._problem().compute_totals(
                of=nl_cons,
                wrt=list(self.driver._designvars),
                return_format='array'
            )
            jac['nl_con', 'x'] = totals

        # Linear constraint jacobian already provided in setup_derivatives

    def _update_desvar_values(self, x):
        """
        Update OpenMDAO design variables from ModOpt's design vector.

        Parameters
        ----------
        x : ndarray
            Flat array of design variable values from ModOpt.
        """
        idx = 0
        for name, meta in self.driver._designvars.items():
            size = meta['global_size'] if meta['distributed'] else meta['size']
            self.driver.set_design_var(name, x[idx : idx + size])
            idx += size


class ModOptDriver(Driver):
    """
    Driver wrapper for the ModOpt optimization library.

    ModOpt provides interfaces to various gradient-based and gradient-free optimization
    algorithms including SLSQP, IPOPT, SNOPT, COBYLA, and others.

    Inequality and equality constraints are supported by several optimizers.
    Refer to the ModOpt documentation for algorithm-specific capabilities.

    ModOptDriver supports the following:
        - equality_constraints
        - inequality_constraints
        - two_sided_constraints
        - linear_constraints (with pre-computed Jacobians for efficiency)

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.

    Attributes
    ----------
    fail : bool
        Flag that indicates failure of most recent optimization.
    iter_count : int
        Counter for function evaluations.
    opt_settings : dict
        Dictionary of optimizer-specific options. See the ModOpt documentation
        for algorithm-specific settings.
    _check_jac : bool
        Used internally to control when to perform singular checks on computed total derivs.
    _con_cache : dict
        Cached result of constraint evaluations.
    _dvlist : list
        Copy of _designvars keys.
    _lincongrad_cache : np.ndarray or None
        Pre-calculated gradients of linear constraints.
    _desvar_array_cache : np.ndarray
        Cached array for setting design variables.
    """

    def __init__(self, **kwargs):
        """
        Initialize the ModOptDriver.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        super().__init__(**kwargs)

        if mo is None:
            raise RuntimeError('ModOptDriver is not available, modopt is not'
                               ' installed.')

        # What we support
        self.supports['optimization'] = True
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['two_sided_constraints'] = True
        self.supports['linear_constraints'] = True
        self.supports['simultaneous_derivatives'] = True

        # What we don't support
        self.supports['multiple_objectives'] = False
        self.supports['active_set'] = False
        self.supports['integer_design_vars'] = False
        self.supports['distributed_design_vars'] = False
        self.supports._read_only = True

        # The user places optimizer-specific settings in here.
        self.opt_settings = {}

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
        self.options.declare('disp', default=True, types=(int, bool),
                             desc='Controls the verbosity of the optimization output.')
        self.options.declare('singular_jac_behavior', default='warn',
                             values=['error', 'warn', 'ignore'],
                             desc='Defines behavior of a zero row/col check after first call to '
                             'compute_totals: '
                             'error - raise an error. '
                             'warn - raise a warning. '
                             "ignore - don't perform check.")
        self.options.declare('singular_jac_tol', default=1e-16,
                             desc='Tolerance for zero row/column check.')

    def _get_name(self):
        """
        Get name of current optimizer.

        Returns
        -------
        str
            The name of the current optimizer.
        """
        return f"ModOpt_{self.options['optimizer']}"

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This method sets optimizer-specific support flags based on the selected algorithm's
        capabilities and validates the problem formulation. This is the final step of setup.

        Parameters
        ----------
        problem : Problem
            Pointer to the containing problem.
        """
        super()._setup_driver(problem)
        opt = self.options['optimizer']

        self.supports._read_only = False
        self.supports['gradients'] = opt in _gradient_optimizers
        self.supports['inequality_constraints'] = opt in _constraint_optimizers
        self.supports['two_sided_constraints'] = opt in _constraint_optimizers
        self.supports['equality_constraints'] = opt in _eq_constraint_optimizers
        self.supports._read_only = True
        self._check_jac = self.options['singular_jac_behavior'] in ['error', 'warn']

        # Raises error if multiple objectives are not supported, but more objectives were defined.
        if not self.supports['multiple_objectives'] and len(self._objs) > 1:
            msg = '{} currently does not support multiple objectives.'
            raise RuntimeError(msg.format(self.msginfo))


    def run(self):
        """
        Optimize the problem using the selected ModOpt optimizer.

        This method performs the following steps:
        1. Performs an initial model evaluation
        2. Extracts and formats design variables, bounds, and constraints
        3. Pre-computes linear constraint Jacobians for efficiency
        4. Creates a ModOptProblem wrapper around the OpenMDAO problem
        5. Instantiates and runs the selected optimizer
        6. Extracts optimal design variables and updates the model
        7. Performs a final evaluation at the optimal point

        The optimization process uses ModOpt's Problem API, which handles callbacks
        for objective and constraint evaluations as well as derivative computations.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False if successful.
        """
        self.result.reset()
        prob = self._problem()
        opt = self.options['optimizer']
        model = prob.model
        self.iter_count = 0
        self._total_jac = None
        self._total_jac_linear = None
        self._desvar_array_cache = None

        self._check_for_missing_objective()
        self._check_for_invalid_desvar_values()

        # Initial Run
        with RecordingDebugging(self._get_name(), self.iter_count, self):
            with model._relevance.nonlinear_active('iter'):
                self._run_solve_nonlinear()
            self.iter_count += 1

        self._con_cache = self.get_constraint_values()
        desvar_vals = self.get_design_var_values()
        self._dvlist = list(self._designvars)

        # maxiter gets passed into ModOpt with all the other options.
        if 'maxiter' not in self.opt_settings:  # lets you override the value in options
            self.opt_settings['maxiter'] = self.options['maxiter']

        # Size Problem
        ndesvar = 0
        for meta in self._designvars.values():
            size = meta['global_size'] if meta['distributed'] else meta['size']
            ndesvar += size
        x_init = np.empty(ndesvar)

        if ndesvar == 0:
            raise RuntimeError('Problem has no design variables.')

        # Initial design vars
        idx = 0
        use_bounds = (opt in _bounds_optimizers)
        if use_bounds:
            bounds = [[], []]
        else:
            bounds = None

        for name, meta in self._designvars.items():
            size = meta['global_size'] if meta['distributed'] else meta['size']
            x_init[idx : idx + size] = desvar_vals[name]
            idx += size

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

                    # Use 1.E16 here in case we've scaled the bounds
                    if p_low <= -1.0E16:
                        p_low = None
                    if p_high >= 1.0E16:
                        p_high = None

                    bounds[0].append(p_low)
                    bounds[1].append(p_high)


        # Initialize constraint-related variables
        lincongrad = None
        nl_con_bounds = [[], []]
        l_con_bounds = [[], []]

        if opt in _constraint_optimizers:
            # get list of linear constraints and precalculate gradients for them (if any)
            if opt in _constraint_grad_optimizers:
                lincons = [name for name, meta in self._cons.items() if meta.get('linear')]
            else:
                lincons = []

            if lincons:
                lincongrad = self._lincongrad_cache = \
                    self._compute_totals(of=lincons, wrt=self._dvlist, return_format='array')
            else:
                self._lincongrad_cache = None

            for name, meta in self._cons.items():
                if meta['indices'] is not None:
                    meta['size'] = size = meta['indices'].indexed_src_size
                else:
                    size = meta['global_size'] if meta['distributed'] else meta['size']

                if meta['linear']:
                    if meta['equals'] is not None:
                        l_con_bounds[0].append(meta['equals'])
                        l_con_bounds[1].append(meta['equals'])
                    else:
                        if meta['lower'] is not None:
                            l_con_bounds[0].append(meta['lower'])
                        if meta['upper'] is not None:
                            l_con_bounds[1].append(meta['upper'])
                else:
                    if meta['equals'] is not None:
                        nl_con_bounds[0].append(meta['equals'])
                        nl_con_bounds[1].append(meta['equals'])
                    else:
                        if meta['lower'] is not None:
                            nl_con_bounds[0].append(meta['lower'])
                        if meta['upper'] is not None:
                            nl_con_bounds[1].append(meta['upper'])

        # optimize
        try:
            if prob.comm.rank != 0:
                self.opt_settings['disp'] = False

            # --- Build modOpt Problem ---
            mo_prob = ModOptProblem(
                driver=self,
                x_init=x_init,
                bounds=bounds,
                l_con_jac=lincongrad,
                l_con_bounds=l_con_bounds,
                nl_con_bounds=nl_con_bounds,
            )

            # --- Optimize ---
            optimizer_cls = getattr(mo, opt)
            optimizer = optimizer_cls(
                problem=mo_prob,
                solver_options=self.opt_settings,
            )
            result = optimizer.solve()

            # Extract optimal design variables from result
            if hasattr(result, 'x'):
                x_opt = result.x
            elif 'x' in result:
                x_opt = result['x']
            elif hasattr(result, 'dvs') and 'x' in result.dvs:
                x_opt = result.dvs['x']
            else:
                # Fallback: get from problem
                x_opt = mo_prob.dvs['x'].get_data()

            # Update design variables in OpenMDAO model
            idx = 0
            for name, meta in self._designvars.items():
                size = meta['global_size'] if meta['distributed'] else meta['size']
                self.set_design_var(name, x_opt[idx : idx + size])
                idx += size

            # Final model evaluation at optimal point
            with RecordingDebugging(self._get_name(), self.iter_count, self):
                with model._relevance.nonlinear_active('iter'):
                    self._run_solve_nonlinear()

            if self.options['disp']:
                if prob.comm.rank == 0:
                    print('Optimization Complete')
                    print('-' * 35)

        # If an exception was swallowed in one of our callbacks, we want to raise it
        # rather than the cryptic message from ModOpt.
        except Exception as msg:
            if self._exc_info is None:
                raise

        if self._exc_info is not None:
            self._reraise()

        return None


import openmdao.api as om
class Paraboloid(om.ExplicitComponent):
    """
    Evaluates the mixed-integer equation f(x,y,w) = (x-3)^2 + xy + (y+4)^2 - 3 + w.
    """

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)
        self.add_input('z', val=0.0)
        self.add_output('f_xy', val=0.0)

    def setup_partials(self):
        self.declare_partials(of='f_xy', wrt='y', method='fd')
        self.declare_partials(of='f_xy', wrt='x', method='fd')
        self.declare_partials(of='f_xy', wrt='z', method='fd')

    # def compute_partials(self, inputs, partials, discrete_inputs):
    #     partials['f_xy', 'x'] = 2 * (inputs['x'] - 3) + inputs['y']
    #     partials['f_xy', 'y'] = 2 * (inputs['y'] + 4) + inputs['x']
    #     partials['f_xy', 'z'] = 0.0

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = ((x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0)


if __name__ == '__main__':
    # build the model
    prob = om.Problem()
    prob.model.add_subsystem('parab', Paraboloid(), promotes_inputs=['x', 'y', 'z'])

    # define the component whose output will be constrained
    prob.model.add_subsystem('const', om.ExecComp('g = x + y + z'), promotes_inputs=['x', 'y', 'z'])

    # Design variables 'x', 'y', and 'z' span components (connected due to our
    # promotion), so we need to provide a common initial value for them. For
    # these variables we have to use "set_input_defaults" because the variable
    # connects to both "const" and "parab", but the initial values from "const"
    # and "parab" are both different. So before we are able to use prob.set_val()
    # for these variables we first have to call "set_input_defaults". Note
    # that we would also have to do this and specify the units of the common
    # initial value if "const" and "parab" had different units for them so that
    # way it knows which units to use if "set_val" is ever used. I'm pretty
    # sure that if a promoted variable name does not span multiple components
    # (like "w") then this doesn't do anything and have to set whatever initial
    # value I want with set_val.
    prob.model.set_input_defaults('x', 3.0)
    prob.model.set_input_defaults('y', -4.0)
    prob.model.set_input_defaults('z', 0.0)

    # setup the optimization
    prob.driver = ModOptDriver()
    prob.driver.options['optimizer'] = 'PySLSQP'
    prob.driver.options['tol'] = 1e-12
    prob.driver.options['maxiter'] = 1000
    # Optimizer specific settings
    prob.driver.opt_settings = {
        'iprint': 0,
    }
    prob.model.add_design_var('x', lower=-50, upper=50)
    prob.model.add_design_var('y', lower=-50, upper=50)
    prob.model.add_design_var('z', lower=-5, upper=5)
    prob.model.add_objective('parab.f_xy')

    # to add the constraint to the model
    prob.model.add_constraint('const.g', lower=5., upper=10.)

    prob.setup()

    prob.run_driver()

    print(prob.get_val('parab.f_xy'))
    print(prob.get_val('x'))
    print(prob.get_val('y'))
