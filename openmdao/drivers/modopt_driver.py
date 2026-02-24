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
    - OpenSQP: A sequential quadratic programming optimizer built into ModOpt

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
import json
from collections import OrderedDict
from openmdao.core.driver import Driver, RecordingDebugging, filter_by_meta
try:
    import modopt as mo
except ImportError:
    mo = None

# TODO: Test and verify MPI compatibility
# TODO: Relevance?
# TODO: Default optimizer file output locations and allow user to define a location
# TODO: SNOPT with the pyoptsparse driver has to use internal FD, not the openmdao FD?
#        - Assume the modopt wrapper already has this sorted out and I don't have to worry about it
# TODO: Look at error catches and other things in the pyoptsparse driver in the obj and grad methods
#       What do we need and what do we not need???

# Gradient-based algorithms from ModOpt
_gradient_optimizers = {
    'SLSQP', 'PySLSQP', 'BFGS', 'LBFGSB', 'TrustConstr',
    'SNOPT', 'IPOPT', 'OpenSQP',
}

# Algorithms that support constraints (inequality and/or equality)
_constraint_optimizers = {
    'SLSQP', 'PySLSQP', 'COBYLA', 'TrustConstr', 'COBYQA',
    'SNOPT', 'IPOPT', 'OpenSQP',
}

# Algorithms that support equality constraints
_eq_constraint_optimizers = {
    'SLSQP', 'PySLSQP', 'TrustConstr', 'SNOPT', 'IPOPT',
    'COBYQA', 'OpenSQP',
}

# Algorithms that support bounds
_bounds_optimizers = {
    'SLSQP', 'PySLSQP', 'LBFGSB', 'TrustConstr', 'COBYLA',
    'COBYQA', 'SNOPT', 'IPOPT', 'OpenSQP',
}

# Gradient-based algorithms that also support constraints (intersection of both sets)
_constraint_grad_optimizers = _gradient_optimizers & _constraint_optimizers

# Optimizers that use solver_options argument (different API from others)
_solver_options_optimizers = {
    'SLSQP', 'PySLSQP', 'COBYLA', 'BFGS', 'LBFGSB', 'NelderMead', 'COBYQA',
    'TrustConstr', 'SNOPT', 'IPOPT', 'ConvexQPSolvers',
}

# All available optimizers (excluding CVXOPT and ConvexQPSolvers which require Hessian)
_all_optimizers = {
    'SLSQP', 'PySLSQP', 'COBYLA', 'BFGS', 'LBFGSB', 'NelderMead',
    'COBYQA', 'TrustConstr', 'OpenSQP', 'SNOPT', 'IPOPT', 'CVXOPT',
    'ConvexQPSolvers',
}

CITATIONS = """
@misc{modopt,
 author = {Joshy, Anugrah J. and Hwang, John T.},
 title = "{modOpt: A Modular development environment and library for optimization algorithms}",
 journal = "{Advances in Engineering Software},
 volume = {213},
 month = feb,
 year = {2026},
 articleno = {104084},
 doi = {10.1016/j.advengsoft.2025.104084}
 howpublished = {\\url{https://github.com/LSDOlab/modopt}},
 year = {2026},
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
    x_info : OrderedDict
        Dictionary with design variable names as keys and dictionaries containing
        'init', 'lower', and 'upper' values as the values.
    lin_con_jac : dict or None
        Pre-computed Jacobian for linear constraints in dictionary format with
        (constraint_name, design_var_name) tuples as keys, or None if no linear constraints.
    lin_con_bounds : dict
        Dictionary with linear constraint names as keys and dictionaries containing
        'lower', 'upper', and 'size' as values.
    nl_con_bounds : dict
        Dictionary with nonlinear constraint names as keys and dictionaries containing
        'lower', 'upper', and 'size' as values.
    nl_con_jac_sparsity : dict
        Sparsity pattern for nonlinear constraint Jacobian with (constraint_name, design_var_name)
        tuples as keys and dictionaries containing 'rows' and 'cols' arrays as values.

    Attributes
    ----------
    driver : ModOptDriver
        Reference to the OpenMDAO driver.
    x_info : OrderedDict
        Design variable metadata including initial values and bounds.
    lin_con_jac : dict or None
        Pre-computed linear constraint Jacobian.
    lin_con_bounds : dict
        Linear constraint bounds.
    nl_con_bounds : dict
        Nonlinear constraint bounds.
    nl_con_jac_sparsity : dict
        Sparsity pattern for nonlinear constraint Jacobian.
    obj_name : str
        Name of the objective function.
    _con_cache : dict or None
        Cached constraint values to avoid redundant evaluations.
    """

    def __init__(self, driver, x_info, lin_con_jac, lin_con_bounds,
                 nl_con_bounds, nl_con_jac_sparsity):
        """
        Initialize the ModOptProblem.

        Parameters
        ----------
        driver : ModOptDriver
            The OpenMDAO driver managing the optimization.
        x_info : OrderedDict
            Dictionary with design variable names as keys and dictionaries containing
            'init', 'lower', and 'upper' values as the values.
        lin_con_jac : dict or None
            Pre-computed Jacobian for linear constraints in dictionary format with
            (constraint_name, design_var_name) tuples as keys.
        lin_con_bounds : dict
            Dictionary with linear constraint names as keys and dictionaries containing
            'lower', 'upper', and 'size' as values.
        nl_con_bounds : dict
            Dictionary with nonlinear constraint names as keys and dictionaries containing
            'lower', 'upper', and 'size' as values.
        nl_con_jac_sparsity : dict
            Sparsity pattern for nonlinear constraint Jacobian with (constraint_name,
            design_var_name) tuples as keys and dictionaries containing 'rows' and 'cols'
            arrays defining the sparse structure.
        """
        self.driver = driver
        self.x_info = x_info
        self.lin_con_jac = lin_con_jac
        self.lin_con_bounds = lin_con_bounds
        self.nl_con_bounds = nl_con_bounds
        self.nl_con_jac_sparsity = nl_con_jac_sparsity
        # ModOpt does not support multiple objectives
        self.obj_name = list(self.driver._objs)[0]
        self._con_cache = None
        super().__init__()

    def initialize(self):
        """
        Initialize the problem name.

        This is called by the ModOpt Problem base class.
        """
        self.problem_name = 'modopt_problem'

    def setup(self):
        """
        Add design variables, constraints, and objective to the ModOpt Problem.
        """
        for name, info in self.x_info.items():
            shape = info['init'].shape if isinstance(info['init'], np.ndarray) else (1,)
            self.add_design_variables(
                name=name,
                shape=shape,
                lower=info['lower'],
                upper=info['upper'],
                vals=info['init'],
            )

        self.add_objective(name=self.obj_name)

        for name, info in self.lin_con_bounds.items():
            self.add_constraints(
                name=name,
                shape=(info['size'],),
                lower=info['lower'],
                upper=info['upper'],
            )

        for name, info in self.nl_con_bounds.items():
            self.add_constraints(
                name=name,
                shape=(info['size'],),
                lower=info['lower'],
                upper=info['upper'],
            )

    def setup_derivatives(self):
        """
        Declare objective and constraint gradients to the ModOpt problem.

        This method informs ModOpt about which derivatives are available.
        For linear constraints, the Jacobian is provided directly. For nonlinear
        constraints, derivatives are computed on-demand, with sparsity information
        if available from OpenMDAO.
        """
        for des_var in self.x_info.keys():
            # Objective gradient doesnt seem to support sparsity declaration
            self.declare_objective_gradient(wrt=des_var)

            # Set fixed linear jacobians
            for lin_con in self.lin_con_bounds.keys():
                self.declare_constraint_jacobian(
                    of=lin_con,
                    wrt=des_var,
                    vals=self.lin_con_jac[lin_con, des_var]
                )

            # Declare nonlinear constraint Jacobian with sparsity if available
            for nl_con in self.nl_con_bounds.keys():
                self.declare_constraint_jacobian(
                    of=nl_con,
                    wrt=des_var,
                    rows=self.nl_con_jac_sparsity[nl_con, des_var]['rows'],
                    cols=self.nl_con_jac_sparsity[nl_con, des_var]['cols'],
                )

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
        self._update_desvar_values(dvs)

        with RecordingDebugging(self.driver._get_name(), self.driver.iter_count, self.driver):
            self.driver.iter_count += 1
            with model._relevance.nonlinear_active('iter'):
                self.driver._run_solve_nonlinear()

        # Get the objective function evaluations
        f_new = next(iter(self.driver.get_objective_values().values()))

        self._con_cache = self.driver.get_constraint_values()

        obj[self.obj_name] = f_new

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
            self._update_desvar_values(dvs)
            self.driver._run_solve_nonlinear()
            vals = self.driver.get_constraint_values()
        else:
            vals = self._con_cache
            self._con_cache = None  # Clear cache after use

        for name in list(self.lin_con_bounds) + list(self.nl_con_bounds):
            cons[name] = vals[name].flatten()

    def compute_objective_gradient(self, dvs, grad):
        """
        Compute the gradient of the objective function.

        Parameters
        ----------
        dvs : dict
            Dictionary with the current design variable names and values.
        grad : dict
            Dictionary to store the gradient values. Keys are design variable names,
            values are gradient arrays with respect to the objective.
        """
        self._update_desvar_values(dvs)

        totals = self.driver._problem().compute_totals(
            of=[self.obj_name],
            wrt=list(self.x_info),
        )
        for des_var in self.x_info.keys():
            grad[des_var] = totals[self.obj_name, des_var]

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
        self._update_desvar_values(dvs)

        # Only need derivatives for the nonlinear constraints
        if self.nl_con_bounds:
            totals = self.driver._problem().compute_totals(
                of=list(self.nl_con_bounds),
                wrt=list(self.x_info),
            )
            for des_var in self.x_info.keys():
                for nl_con in self.nl_con_bounds.keys():
                    jac[nl_con, des_var] = totals[nl_con, des_var]

    def _update_desvar_values(self, dvs):
        """
        Update OpenMDAO design variables from ModOpt's design vector.

        Parameters
        ----------
        dvs : dict
            Dictionary with the current design variable names and values.
        """
        for name in self.x_info.keys():
            self.driver.set_design_var(name, dvs[name])


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
        self.supports['linear_only_designvars'] = True
        self.supports['total_jac_sparsity'] = True

        # What we don't support
        self.supports['multiple_objectives'] = False
        self.supports['active_set'] = False
        self.supports['integer_design_vars'] = False
        self.supports['distributed_design_vars'] = False
        self.supports._read_only = True

        # The user places optimizer-specific settings in here.
        self.opt_settings = {}

        self._total_jac_sparsity = None

        self.cite = CITATIONS

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('optimizer', 'SLSQP', values=_all_optimizers,
                             desc='Name of optimizer to use')
        self.options.declare('maxiter', 200, lower=0,
                             desc='Maximum number of iterations.')
        self.options.declare('disp', default=True, types=(int, bool),
                             desc='Controls optimizer output verbosity. Can be bool (True/False) '
                                  'or int for fine control (0=quiet, higher=more verbose). '
                                  'Automatically maps to optimizer-specific settings. If '
                                  'optimizer specific verbosity settings are provided in the '
                                  '"opt_settings" attribute, then this option will be ignored '
                                  'and those settings will be used instead.')
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
        Get the name of this driver.

        Returns
        -------
        str
            Driver name in the format 'ModOpt_<optimizer_name>'.
        """
        return f"ModOpt_{self.options['optimizer']}"

    def _setup_verbosity(self, opt):
        """
        Configure optimizer-specific verbosity settings based on the driver's disp option.

        Different optimizers use different option names and scales for controlling output
        verbosity. This method translates the driver's 'disp' option to the appropriate
        optimizer-specific settings.

        Parameters
        ----------
        opt : str
            Name of the optimizer being used.
        """
        disp = self.options['disp']

        # Skip if user has already specified verbosity in opt_settings
        verbosity_keys = {
            'disp', 'iprint', 'print_level', 'isumm', 'print_file',
            'print_frequency', 'verbose', 'verbosity', 'major print level',
            'minor print level'
        }
        if any(key.lower() in [k.lower() for k in self.opt_settings.keys()]
               for key in verbosity_keys):
            return

        # Map disp option to optimizer-specific settings
        if opt == 'IPOPT':
            # IPOPT uses print_level (0-12 scale)
            if isinstance(disp, int):
                self.opt_settings['print_level'] = min(max(disp, 0), 12)
            else:
                self.opt_settings['print_level'] = 5 if disp else 0

        elif opt == 'SNOPT':
            # SNOPT uses Major/Minor print levels
            if isinstance(disp, int):
                level = min(max(disp, 0), 10)
                self.opt_settings['Major print level'] = level
                self.opt_settings['Minor print level'] = level
            else:
                self.opt_settings['Major print level'] = 1 if disp else 0
                self.opt_settings['Minor print level'] = 0

        elif opt == 'TrustConstr':
            # TrustConstr uses verbose (0-3 scale)
            if isinstance(disp, int):
                self.opt_settings['verbose'] = min(max(disp, 0), 3)
            else:
                self.opt_settings['verbose'] = 1 if disp else 0

        elif opt in {'OpenSQP', 'PySLSQP'}:
            # OpenSQP uses iprint
            if isinstance(disp, int):
                self.opt_settings['iprint'] = disp
            else:
                self.opt_settings['iprint'] = 1 if disp else 0

        elif opt in {'LBFGSB'}:
            # LBFGSB uses iprint, but with different settings
            if isinstance(disp, int):
                self.opt_settings['iprint'] = disp
            else:
                self.opt_settings['iprint'] = 0 if disp else -1

        else:
            # Most SciPy-based optimizers (SLSQP, COBYLA, COBYQA, BFGS, NelderMead)
            # use 'disp' as boolean
            if isinstance(disp, int):
                self.opt_settings['disp'] = disp > 0
            else:
                self.opt_settings['disp'] = disp

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This method configures optimizer-specific support flags based on the selected
        algorithm's capabilities, validates the problem formulation, and sets up sparsity
        structures. Called during problem setup.

        Parameters
        ----------
        problem : Problem
            The OpenMDAO Problem being optimized.
        """
        super()._setup_driver(problem)
        opt = self.options['optimizer']

        # Update support flags based on optimizer capabilities
        self.supports._read_only = False
        self.supports['gradients'] = opt in _gradient_optimizers
        self.supports['inequality_constraints'] = opt in _constraint_optimizers
        self.supports['two_sided_constraints'] = opt in _constraint_optimizers
        self.supports['equality_constraints'] = opt in _eq_constraint_optimizers
        self.supports._read_only = True
        self._check_jac = self.options['singular_jac_behavior'] in ['error', 'warn']

        # Validate problem formulation
        if not self.supports['multiple_objectives'] and len(self._objs) > 1:
            msg = '{} currently does not support multiple objectives.'
            raise RuntimeError(msg.format(self.msginfo))

        # CVXOPT and ConvexQPSolvers require Hessian information which is not yet supported
        if opt.lower() in ["convexqpsolvers", 'cvxopt']:
            msg = ('{} currently does not support CVXOPT and ConvexQPSolvers '
                   'due to the requirement of Hessian information.')
            raise RuntimeError(msg.format(self.msginfo))

        self._setup_tot_jac_sparsity()

    def run(self):
        """
        Optimize the problem using the selected ModOpt optimizer.

        The optimization process performs the following steps:
        1. Initial model evaluation
        2. Extract and format design variables, bounds, and constraints
        3. Pre-compute linear constraint Jacobians for efficiency
        4. Create ModOptProblem wrapper around the OpenMDAO problem
        5. Instantiate and run the selected optimizer
        6. Extract optimal design variables and update the model
        7. Perform final evaluation at the optimal point

        The optimization uses ModOpt's Problem API, which handles callbacks for
        objective and constraint evaluations as well as derivative computations.

        Returns
        -------
        bool
            Success flag; True if optimization succeeded, False if it failed.
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

        # Perform initial model evaluation
        with RecordingDebugging(self._get_name(), self.iter_count, self):
            with model._relevance.nonlinear_active('iter'):
                self._run_solve_nonlinear()
            self.iter_count += 1

        self._con_cache = self.get_constraint_values()
        desvar_vals = self.get_design_var_values()

        # Configure optimizer-specific verbosity settings
        self._setup_verbosity(opt)

        # Set maxiter for optimizer (unless already specified in opt_settings)
        if 'maxiter' not in self.opt_settings:
            if opt == 'IPOPT':
                # IPOPT uses 'max_iter' instead of 'maxiter'
                self.opt_settings['max_iter'] = self.options['maxiter']
            else:
                self.opt_settings['maxiter'] = self.options['maxiter']

        # Determine total number of design variables
        ndesvar = 0
        for name, meta in self._designvars.items():
            size = meta['global_size'] if meta['distributed'] else meta['size']
            ndesvar += size

        if ndesvar == 0:
            raise RuntimeError('Problem has no design variables.')

        # Collect design variable information (initial values and bounds)
        x_info = OrderedDict()
        use_bounds = (opt in _bounds_optimizers)
        for name, meta in self._designvars.items():
            x_info[name] = {}
            x_info[name]['init'] = desvar_vals[name]

            if use_bounds:
                x_info[name]['lower'] = meta['lower']
                x_info[name]['upper'] = meta['upper']
            else:
                x_info[name]['lower'] = None
                x_info[name]['upper'] = None

        # Collect constraint information
        lincongrad = None
        nl_con_bounds = dict()
        lin_con_bounds = dict()
        nl_con_jac_sparsity = dict()
        if opt in _constraint_optimizers:
            # Identify linear constraints and pre-compute their Jacobians if optimizer
            # uses gradients
            if opt in _constraint_grad_optimizers:
                lincons = [name for name, meta in self._cons.items() if meta.get('linear')]
            else:
                lincons = []

            if lincons:
                lincongrad = self._lincongrad_cache = \
                    self._compute_totals(of=lincons, wrt=list(self._designvars))
            else:
                self._lincongrad_cache = None

            # Process constraints and organize into linear and nonlinear categories
            for name, meta in self._cons.items():
                if meta['indices'] is not None:
                    meta['size'] = size = meta['indices'].indexed_src_size
                else:
                    size = meta['global_size'] if meta['distributed'] else meta['size']

                # Separate linear and nonlinear constraints
                if meta['linear']:
                    if meta['equals'] is not None:
                        lin_con_bounds[name] = {
                            'lower': meta['equals'],
                            'upper': meta['equals'],
                            'size': size
                        }
                    else:
                        lin_con_bounds[name] = {
                            'lower': meta['lower'],
                            'upper': meta['upper'],
                            'size': size
                        }
                else:
                    if meta['equals'] is not None:
                        nl_con_bounds[name] = {
                            'lower': meta['equals'],
                            'upper': meta['equals'],
                            'size': size
                        }
                    else:
                        nl_con_bounds[name] = {
                            'lower': meta['lower'],
                            'upper': meta['upper'],
                            'size': size
                        }

                    # Initialize sparsity structure for nonlinear constraint Jacobians
                    # TODO: populate from OpenMDAO's actual sparsity information
                    for x_name in self._designvars.keys():
                        nl_con_jac_sparsity[name, x_name] = {}
                        nl_con_jac_sparsity[name, x_name]['rows'] = None
                        nl_con_jac_sparsity[name, x_name]['cols'] = None

        # Run optimization
        try:
            # Build ModOpt Problem wrapper
            mo_prob = ModOptProblem(
                driver=self,
                x_info=x_info,
                lin_con_jac=lincongrad,
                lin_con_bounds=lin_con_bounds,
                nl_con_bounds=nl_con_bounds,
                nl_con_jac_sparsity=nl_con_jac_sparsity,
            )

            # Instantiate and run optimizer
            optimizer_cls = getattr(mo, opt)
            if opt in _solver_options_optimizers:
                optimizer = optimizer_cls(
                    problem=mo_prob,
                    solver_options=self.opt_settings,
                )
            else:
                optimizer = optimizer_cls(
                    problem=mo_prob,
                    **self.opt_settings,
                )
            result = optimizer.solve()

            # Extract optimal design variables and success flag from optimizer result
            # Different optimizers return results in different formats
            if hasattr(result, 'x'):
                x_opt = result.x
                success = result.success
            elif 'x' in result:
                x_opt = result['x']
                if opt == 'IPOPT':
                    # IPOPT success status is not consistently available through ModOpt's
                    # interface due to how CasADi's nlpsol wrapper handles the solver object
                    # TODO: Request improvement in ModOpt to expose IPOPT status
                    print(f'{"-" * 40}\n IPOPT does not return success status in '
                          f'a consistent, easily readable way, so defaulting to '
                          f'success=True. \n{"-" * 40}\n')
                    success = True
                else:
                    success = result['success']
            else:
                # Fallback for optimizers with non-standard result format
                x_opt = mo_prob.dvs['x'].get_data()
                success = True

            # Update OpenMDAO design variables with optimal values
            idx = 0
            for name in mo_prob.x_info.keys():
                meta = self._designvars[name]
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

        except Exception:
            # If an exception occurred in one of our callbacks, re-raise it with
            # the original context rather than ModOpt's generic exception message
            if self._exc_info is None:
                raise

        if self._exc_info is not None:
            self._reraise()

        return success

    def _setup_tot_jac_sparsity(self, coloring=None):
        """
        Set up total Jacobian sub-Jacobian sparsity pattern.

        This method extracts sparsity information from coloring objects or user-specified
        sparsity patterns and stores it for use during optimization.

        Parameters
        ----------
        coloring : Coloring or None
            Coloring object containing sparsity information, or None to use
            user-specified _total_jac_sparsity.
        """
        total_sparsity = None
        self._con_subjacs = {'rows': [], 'cols': []}
        coloring = coloring if coloring is not None else self._get_static_coloring()

        # Extract sparsity from coloring or user-specified sparsity
        if coloring is not None:
            total_sparsity = coloring.get_subjac_sparsity()
            if self._total_jac_sparsity is not None:
                raise RuntimeError("Total jac sparsity was set in both _total_coloring"
                                   " and _setup_tot_jac_sparsity.")
        elif self._total_jac_sparsity is not None:
            # Load sparsity from file if provided as string path
            if isinstance(self._total_jac_sparsity, str):
                with open(self._total_jac_sparsity, 'r') as f:
                    self._total_jac_sparsity = json.load(f)
            total_sparsity = self._total_jac_sparsity

        if total_sparsity is None:
            return

        use_approx = self._problem().model._owns_approx_of is not None

        nl_dvs = self._get_nl_dvs()

        # Build sparsity structure for nonlinear constraints only
        # (linear constraints have pre-computed Jacobians)
        for con, conmeta in filter_by_meta(self._cons.items(), 'linear', exclude=True):
            self._con_subjacs[con] = {}
            consrc = conmeta['source']
            for dv, dvmeta in nl_dvs.items():
                if use_approx:
                    dvsrc = dvmeta['source']
                    rows, cols, shape = total_sparsity[consrc][dvsrc]
                else:
                    rows, cols, shape = total_sparsity[con][dv]
                self._con_subjacs[con][dv] = {
                    'coo': [rows, cols, np.zeros(rows.size)],
                    'shape': shape,
                }


import openmdao.api as om


class Paraboloid(om.ExplicitComponent):
    """
    Evaluates a paraboloid function for testing.

    Computes f(x,y) = (x[0] + x[2] - 3)^2 + x[0]*y + (y + 4)^2 - 3

    where x is a 3-element array.
    """

    def setup(self):
        self.add_input('x', val=np.zeros(3))
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

        outputs['f_xy'] = ((x[0] + x[2] - 3.0)**2 + x[0] * y + (y + 4.0)**2 - 3.0)


class Con(om.ExplicitComponent):
    """
    Simple constraint component for testing.

    Computes g = x[0] + x[2] + y + z where x is a 3-element array.
    """

    def setup(self):
        self.add_input('x', val=np.zeros(3))
        self.add_input('y', val=0.0)
        self.add_input('z', val=0.0)
        self.add_output('g', val=0.0)

    def setup_partials(self):
        self.declare_partials(of='g', wrt='y', method='fd')
        self.declare_partials(of='g', wrt='x', method='fd', rows=[0, 0], cols=[0, 2])
        self.declare_partials(of='g', wrt='z', method='fd')

    def compute(self, inputs, outputs):
        outputs['g'] = inputs['x'][0] + inputs['x'][2] + inputs['y'] + inputs['z']


if __name__ == '__main__':

    # # build the model
    # prob = om.Problem()
    # prob.model.add_subsystem('parab', Paraboloid(), promotes_inputs=['x', 'y', 'z'])

    # # define the component whose output will be constrained
    # prob.model.add_subsystem('const', Con(), promotes_inputs=['x', 'y', 'z'])

    # # Design variables 'x', 'y', and 'z' span components (connected due to our
    # # promotion), so we need to provide a common initial value for them. For
    # # these variables we have to use "set_input_defaults" because the variable
    # # connects to both "const" and "parab", but the initial values from "const"
    # # and "parab" are both different. So before we are able to use prob.set_val()
    # # for these variables we first have to call "set_input_defaults". Note
    # # that we would also have to do this and specify the units of the common
    # # initial value if "const" and "parab" had different units for them so that
    # # way it knows which units to use if "set_val" is ever used. I'm pretty
    # # sure that if a promoted variable name does not span multiple components
    # # (like "w") then this doesn't do anything and have to set whatever initial
    # # value I want with set_val.
    # prob.model.set_input_defaults('x', np.array([3.0, 3.0, 3.0]))
    # prob.model.set_input_defaults('y', -4.0)
    # prob.model.set_input_defaults('z', 0.0)

    # # setup the optimization
    # prob.driver = ModOptDriver()
    # prob.driver.options['optimizer'] = 'SLSQP'
    # prob.driver.options['maxiter'] = 1000
    # # Optimizer specific settings
    # prob.driver.opt_settings = {
    #     # 'opt_tol': 1e-6,
    # }
    # prob.model.add_design_var('x', lower=-50, upper=50)
    # prob.model.add_design_var('y', lower=-50, upper=50)
    # prob.model.add_design_var('z', lower=-5, upper=5)
    # prob.model.add_objective('parab.f_xy')

    # # to add the constraint to the model
    # prob.model.add_constraint('const.g', lower=0., upper=10.)
    # # prob.model.add_constraint('y', equals=10.)

    # prob.driver.declare_coloring()
    # prob.setup()

    # prob.run_driver()

    # print(prob.get_val('parab.f_xy'))
    # print(prob.get_val('x'))
    # print(prob.get_val('y'))



    # p = om.Problem()

    # exec = om.ExecComp(['y = a*x**2',
    #                     'z = a + x**2'],
    #                     a={'shape': (1,)},
    #                     y={'shape': (101,)},
    #                     x={'shape': (101,)},
    #                     z={'shape': (101,)})

    # p.model.add_subsystem('exec', exec)

    # p.model.add_design_var('exec.a', lower=-1000, upper=1000)
    # p.model.add_objective('exec.y', index=50)
    # p.model.add_constraint('exec.z', indices=[0], equals=25)
    # p.model.add_constraint('exec.z', indices=[-1], lower=20, alias="ALIAS_TEST")

    # p.driver = ModOptDriver()
    # p.driver.options['optimizer'] = "SLSQP"

    # p.driver.declare_coloring(show_summary=True, show_sparsity=True)

    # p.setup(mode='rev')

    # p.set_val('exec.x', np.linspace(-10, 10, 101))

    # p.run_model()
    # p.run_driver()



    class DynamicPartialsComp(om.ExplicitComponent):
        """
        Component with dynamic partial derivative coloring for testing.

        Computes g = arctan(y / x) element-wise for arrays x and y.

        Parameters
        ----------
        size : int
            Size of the input and output arrays.

        Attributes
        ----------
        num_computes : int
            Counter for the number of times compute has been called.
        """

        def __init__(self, size):
            super().__init__()
            self.size = size
            self.num_computes = 0

        def setup(self):
            self.add_input('y', np.ones(self.size))
            self.add_input('x', np.ones(self.size))
            self.add_output('g', np.ones(self.size))

            self.declare_partials('*', '*', method='cs')

            # turn on dynamic partial coloring
            self.declare_coloring(wrt='*', method='cs', perturb_size=1e-5, num_full_jacs=1, tol=1e-20,
                                show_summary=True, show_sparsity=True)

        def compute(self, inputs, outputs):
            outputs['g'] = np.arctan(inputs['y'] / inputs['x'])
            self.num_computes += 1


    SIZE = 10

    p = om.Problem()
    model = p.model

    # DynamicPartialsComp is set up to do dynamic partial coloring
    arctan_yox = model.add_subsystem('arctan_yox', DynamicPartialsComp(SIZE), promotes_inputs=['x', 'y'])

    model.add_subsystem('circle', om.ExecComp('area=pi*r**2'), promotes_inputs=['r'])

    model.add_subsystem('r_con', om.ExecComp('g=x**2 + y**2 - r', has_diag_partials=True,
                                            g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)),
                        promotes_inputs=['x', 'y', 'r'])

    thetas = np.linspace(0, np.pi/4, SIZE)
    model.add_subsystem('theta_con', om.ExecComp('g = x - theta', has_diag_partials=True,
                                                g=np.ones(SIZE), x=np.ones(SIZE),
                                                theta=thetas))
    model.add_subsystem('delta_theta_con', om.ExecComp('g = even - odd', has_diag_partials=True,
                                                        g=np.ones(SIZE//2), even=np.ones(SIZE//2),
                                                        odd=np.ones(SIZE//2)))

    model.add_subsystem('l_conx', om.ExecComp('g=x-1', has_diag_partials=True, g=np.ones(SIZE), x=np.ones(SIZE)),
                        promotes_inputs=['x'])

    IND = np.arange(SIZE, dtype=int)
    ODD_IND = IND[1::2]  # all odd indices
    EVEN_IND = IND[0::2]  # all even indices

    model.connect('arctan_yox.g', 'theta_con.x')
    model.connect('arctan_yox.g', 'delta_theta_con.even', src_indices=EVEN_IND)
    model.connect('arctan_yox.g', 'delta_theta_con.odd', src_indices=ODD_IND)

    p.driver = ModOptDriver()
    p.driver.options['optimizer'] = "SLSQP"

    #####################################
    # set up dynamic total coloring here
    p.driver.declare_coloring(show_summary=True, show_sparsity=True)
    #####################################

    model.add_design_var('x')
    model.add_design_var('y')
    model.add_design_var('r', lower=.5, upper=10)

    # nonlinear constraints
    model.add_constraint('r_con.g', equals=0)

    model.add_constraint('theta_con.g', lower=-1e-5, upper=1e-5, indices=EVEN_IND)
    model.add_constraint('delta_theta_con.g', lower=-1e-5, upper=1e-5)

    # this constrains x[0] to be 1 (see definition of l_conx)
    model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,])

    # linear constraint
    model.add_constraint('y', equals=0, indices=[0,], linear=True)

    model.add_objective('circle.area', ref=-1)

    p.setup(mode='fwd')

    # the following were randomly generated using np.random.random(10)*2-1 to randomly
    # disperse them within a unit circle centered at the origin.
    p.set_val('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                            0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
    p.set_val('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                            -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
    p.set_val('r', .7)

    p.run_driver()