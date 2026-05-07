"""
OpenMDAO Wrapper for the modOpt optimization library.

modOpt is a modular optimization framework providing interfaces to various
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
    - OpenSQP: A sequential quadratic programming optimizer built into modOpt

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

See the modOpt documentation at https://modopt.readthedocs.io for detailed information
on algorithm-specific options and capabilities.
"""
import sys
import numpy as np
import json
from collections import OrderedDict

from openmdao.core.constants import _DEFAULT_REPORTS_DIR, _ReprClass
from openmdao.core.driver import Driver, RecordingDebugging, filter_by_meta
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.mpi import MPI
from openmdao.core.group import Group

try:
    # modopt.core.visualization calls matplotlib.use('TkAgg') at import time.
    # Save and restore the backend so modopt does not permanently change it,
    # which would break OpenMDAO visualization tools in headless environments.
    try:
        import matplotlib as _mpl
        _mpl_backend = _mpl.get_backend()
    except ImportError:
        _mpl_backend = None

    import modopt as mo
    problem = mo.Problem

    if _mpl_backend is not None:
        try:
            import matplotlib as _mpl
            _mpl.use(_mpl_backend)
        except Exception:
            pass
except ImportError:
    mo = None
    problem = object
except Exception as err:
    mo = err
    problem = object


# Gradient-based algorithms from modOpt
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
@article{modopt,
 author = {Joshy, Anugrah J. and Hwang, John T.},
 title = "{modOpt: A Modular development environment and library for optimization algorithms}",
 journal = {Advances in Engineering Software},
 volume = {213},
 month = feb,
 year = {2026},
 articleno = {104084},
 doi = {10.1016/j.advengsoft.2025.104084}
}
"""


class modOptProblem(problem):
    """
    modOpt Problem that delegates objective and constraint evaluation to an OpenMDAO driver.

    This class wraps an OpenMDAO problem as a modOpt optimization problem, translating
    between modOpt's interface and OpenMDAO's driver interface.

    Parameters
    ----------
    driver : modOptDriver
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
    driver : modOptDriver
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
    all_nl_relevant_dvs : set
        Set of all design variable names that are relevant to any nonlinear constraint.
    obj_name : str
        Name of the objective function.
    _con_cache : dict or None
        Cached constraint values to avoid redundant evaluations.
    _all_constraint_names : list of str
        Ordered list of all constraint names (linear followed by nonlinear).
    """

    def __init__(self, driver, x_info, lin_con_jac, lin_con_bounds,
                 nl_con_bounds, nl_con_jac_sparsity, all_nl_relevant_dvs):
        """
        Initialize the modOptProblem.

        Parameters
        ----------
        driver : modOptDriver
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
        all_nl_relevant_dvs : set
            Set of all design variable names that are relevant to any nonlinear constraint.
        """
        self.driver = driver
        self.x_info = x_info
        self.lin_con_jac = lin_con_jac
        self.lin_con_bounds = lin_con_bounds
        self.nl_con_bounds = nl_con_bounds
        self.nl_con_jac_sparsity = nl_con_jac_sparsity
        self.all_nl_relevant_dvs = all_nl_relevant_dvs
        # modOpt does not support multiple objectives
        self.obj_name = list(self.driver._objs)[0]
        self._con_cache = None
        self._all_constraint_names = list(self.lin_con_bounds) + list(self.nl_con_bounds)
        super().__init__()

    def initialize(self):
        """
        Set the modOpt problem name.

        Called by the modOpt Problem base class during initialization.
        """
        self.problem_name = 'modOpt_problem'

    def setup(self):
        """
        Add design variables, constraints, and objective to the modOpt Problem.
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
        Declare objective and constraint gradients to the modOpt problem.

        This method informs modOpt about which derivatives are available.
        For linear constraints, the Jacobian is provided directly. For nonlinear
        constraints, derivatives are computed on-demand, with sparsity information
        if available from OpenMDAO. Only relevant Jacobians (based on OpenMDAO's
        relevance analysis) are declared.
        """
        for des_var in self.x_info.keys():
            # Objective gradient doesnt seem to support sparsity declaration
            self.declare_objective_gradient(wrt=des_var)

            # Set fixed linear jacobians - only for relevant design variable/constraint pairs
            for lin_con in self.lin_con_bounds.keys():
                # Only declare if this pair exists (i.e., is relevant)
                if (lin_con, des_var) in self.lin_con_jac:
                    self.declare_constraint_jacobian(
                        of=lin_con,
                        wrt=des_var,
                        vals=self.lin_con_jac[lin_con, des_var]
                    )

            # Declare nonlinear constraint Jacobian with sparsity if available
            # Only for relevant design variable/constraint pairs
            for nl_con in self.nl_con_bounds.keys():
                # Only declare if this pair exists (i.e., is relevant)
                if (nl_con, des_var) in self.nl_con_jac_sparsity:
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
        dvs : <array_manager.core.native_formats.vector.Vector>
            Vector with the current design variable names and values.
        obj : dict
            Dictionary to store the computed objective value.
        """
        model = self.driver._problem().model

        try:
            self._update_desvar_values(dvs)
            self._run_model()

            # Get the objective function evaluations
            f_new = next(iter(self.driver.get_objective_values().values()))
            self._con_cache = self.driver.get_constraint_values()

            # Broadcast objective and constraint values from rank 0 so all ranks'
            # optimizers see identical values and follow the same code path.
            # Without this, rank-specific model outputs can cause ranks to diverge
            # within scipy's optimizer loop, leading to mismatched MPI collective
            # calls and a deadlock.
            if MPI:
                comm = self.driver._problem().model.comm
                f_arr = np.array([f_new])
                comm.Bcast(f_arr, root=0)
                f_new = f_arr[0]

            obj[self.obj_name] = f_new

        except Exception:
            # Clean up solver print stack and store exception for re-raising later
            self._handle_callback_exception(model)
            obj[self.obj_name] = np.nan

    def compute_constraints(self, dvs, cons):
        """
        Compute constraint values.

        Uses cached constraint values if available, otherwise evaluates model.
        Always calls _update_desvar_values to keep all MPI ranks synchronized.

        Parameters
        ----------
        dvs : <array_manager.core.native_formats.vector.Vector>
            Vector with the current design variable names and values.
        cons : dict
            Dictionary to store the computed constraint values.
        """
        model = self.driver._problem().model

        try:
            self._update_desvar_values(dvs)

            # Use cached constraint values from compute_objective if available
            if self._con_cache is None:
                self._run_model()
                vals = self.driver.get_constraint_values()
            else:
                vals = self._con_cache
                self._con_cache = None  # Clear cache after use

            # Broadcast non-distributed constraint values from rank 0 so all
            # ranks' optimizers see identical values. Applies to both cached and
            # freshly computed values. Distributed constraints are skipped since
            # each rank holds a different valid local portion.
            if MPI:
                comm = self.driver._problem().model.comm
                for name, meta in self.driver._cons.items():
                    if name in vals and not meta.get('distributed', False):
                        comm.Bcast(vals[name], root=0)

            for name in self._all_constraint_names:
                cons[name] = vals[name].flatten()

        except Exception:
            # Clean up solver print stack and store exception for re-raising later
            self._handle_callback_exception(model)
            for name in self._all_constraint_names:
                size = self._get_constraint_size(name)
                cons[name] = np.full(size, np.nan)

    def compute_objective_gradient(self, dvs, grad):
        """
        Compute the gradient of the objective function.

        Parameters
        ----------
        dvs : <array_manager.core.native_formats.vector.Vector>
            Vector with the current design variable names and values.
        grad : dict
            Dictionary to store the gradient values. Keys are design variable names,
            values are gradient arrays with respect to the objective.
        """
        model = self.driver._problem().model

        try:
            totals = self.driver._problem().compute_totals(
                of=[self.obj_name],
                wrt=list(self.x_info),
            )

            # First time through, check for zero row/col.
            if self.driver._check_obj_grad and self.driver._total_jac is not None:
                for subsys in model.system_iter(include_self=True, recurse=True, typ=Group):
                    if subsys._has_approx:
                        break
                else:
                    raise_error = self.driver.options['singular_jac_behavior'] == 'error'
                    self.driver._total_jac.check_total_jac(raise_error=raise_error,
                                                           tol=self.driver.options['singular_jac_tol'])
                self.driver._check_obj_grad = False

            for des_var in self.x_info.keys():
                grad[des_var] = totals[self.obj_name, des_var]

        except Exception:
            # Clean up solver print stack and store exception for re-raising later
            self._handle_callback_exception(model)
            for des_var, info in self.x_info.items():
                grad[des_var] = np.full_like(info['init'], np.nan)

    def compute_constraint_jacobian(self, dvs, jac):
        """
        Compute the Jacobian of nonlinear constraints.

        Linear constraint Jacobians are pre-computed and provided during setup.
        Only nonlinear constraint Jacobians are computed here. Only computes Jacobians
        for relevant design variable/constraint pairs based on relevance analysis.

        Parameters
        ----------
        dvs : <array_manager.core.native_formats.vector.Vector>
            Vector with the current design variable names and values.
        jac : dict
            Dictionary to store the computed constraint Jacobians.
        """
        model = self.driver._problem().model

        try:
            # Only need derivatives for the nonlinear constraints
            if self.nl_con_bounds and self.all_nl_relevant_dvs:
                totals = self.driver._problem().compute_totals(
                    of=list(self.nl_con_bounds),
                    wrt=list(self.all_nl_relevant_dvs),
                )

                # First time through, check for zero row/col.
                if self.driver._check_nl_jac and self.driver._total_jac is not None:
                    for subsys in model.system_iter(include_self=True, recurse=True, typ=Group):
                        if subsys._has_approx:
                            break
                    else:
                        raise_error = self.driver.options['singular_jac_behavior'] == 'error'
                        self.driver._total_jac.check_total_jac(raise_error=raise_error,
                                                            tol=self.driver.options['singular_jac_tol'])
                    self.driver._check_nl_jac = False

                # Extract and store Jacobians for relevant (constraint, design_var) pairs
                for (nl_con, des_var), info in self.nl_con_jac_sparsity.items():
                    rows = info['rows']
                    cols = info['cols']

                    if rows is not None and cols is not None:
                        jac[nl_con, des_var] = totals[nl_con, des_var][rows, cols]
                    else:
                        jac[nl_con, des_var] = totals[nl_con, des_var]

        except Exception:
            # Clean up solver print stack and store exception for re-raising later
            self._handle_callback_exception(model)
            for (nl_con, des_var), info in self.nl_con_jac_sparsity.items():
                rows = info['rows']
                cols = info['cols']

                if rows is not None and cols is not None:
                    jac[nl_con, des_var] = np.full(len(rows), np.nan)
                else:
                    con_size = self.nl_con_bounds[nl_con]['size']
                    dv_size = len(self.x_info[des_var]['init'])
                    jac[nl_con, des_var] = np.full((con_size, dv_size), np.nan)

    def _update_desvar_values(self, dvs):
        """
        Update OpenMDAO design variables from modOpt's design vector.

        Sets design variable values in OpenMDAO from modOpt's design vector.

        Parameters
        ----------
        dvs : <array_manager.core.native_formats.vector.Vector>
            Vector with the current design variable names and values.
        """
        # dvs isn't a dictionary so we can't set _vectors['design_var'] directly
        dv_vec = self.driver._vectors['design_var']
        x_new = np.concatenate([np.asarray(dvs[name]).flatten() for name in self.x_info.keys()])
        if MPI:
            self.driver._problem().model.comm.Bcast(x_new, root=0)
        dv_vec.set_data(x_new, driver_scaling=True)
        self.driver._set_design_vars(list(self.x_info.keys()), driver_scaling=True)

    def _get_constraint_size(self, name):
        """
        Get the size of a constraint by name.

        Parameters
        ----------
        name : str
            Constraint name.

        Returns
        -------
        int
            Size of the constraint.
        """
        if name in self.lin_con_bounds:
            return self.lin_con_bounds[name]['size']
        return self.nl_con_bounds[name]['size']

    def _run_model(self):
        """
        Execute the OpenMDAO model with proper recording and relevance handling.

        On the first iteration relevance filtering is inactive so the full model
        is evaluated. On subsequent iterations relevance filtering is active for
        efficiency.
        """
        model = self.driver._problem().model
        with RecordingDebugging(self.driver._get_name(), self.driver.iter_count, self.driver):
            self.driver.iter_count += 1
            with model._relevance.nonlinear_active('iter', active=self.driver._model_ran):
                self.driver._run_solve_nonlinear()
                self.driver._model_ran = True

    def _handle_callback_exception(self, model):
        """
        Handle exceptions for modOpt callbacks.

        Clears solver print stack and stores exception info for re-raising after
        optimization.

        Parameters
        ----------
        model : System
            The model to clear iprint on.
        """
        model._clear_iprint()
        if self.driver._exc_info is None:
            self.driver._exc_info = sys.exc_info()


class modOptDriver(Driver):
    """
    Driver wrapper for the modOpt optimization library.

    modOpt provides interfaces to various gradient-based and gradient-free optimization
    algorithms including SLSQP, IPOPT, SNOPT, COBYLA, and others.

    Inequality and equality constraints are supported by several optimizers.
    Refer to the modOpt documentation for algorithm-specific capabilities.

    modOptDriver supports the following:
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
        Dictionary of optimizer-specific options. See the modOpt documentation
        for algorithm-specific settings.
    _check_obj_grad : bool
        Used internally to control when to perform singular checks on computed
        objective gradient.
    _check_nl_jac : bool
        Used internally to control when to perform singular checks on computed
        nonlinear constraint jacobian.
    _con_cache : dict
        Cached result of constraint evaluations.
    _lincongrad_cache : np.ndarray or None
        Pre-calculated gradients of linear constraints.
    _model_ran : bool
        Flag indicating whether the model has been evaluated at least once,
        used to activate relevance filtering on subsequent iterations.
    _total_jac_sparsity : dict or None
        User-specified total Jacobian sparsity pattern.
    _mo_prob : <modOpt Problem object>
        The modOpt problem object that is built and fed to the Optimizer.
    """

    def __init__(self, **kwargs):
        """
        Initialize the modOptDriver.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        if mo is None:
            raise RuntimeError('modOptDriver is not available, modOpt is not'
                               ' installed.')

        if isinstance(mo, Exception):
            # there is some other issue with the modOpt installation
            raise mo

        super().__init__(**kwargs)

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

        self._check_obj_grad = False
        self._check_nl_jac = False
        self._total_jac_sparsity = None
        self._model_ran = False
        self._mo_prob = None

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
        self.options.declare('turn_off_outputs', default=False, types=bool,
                             desc='If True, prevents modOpt from generating any output files.')
        self.options.declare('output_dir', types=(str, _ReprClass), default=_DEFAULT_REPORTS_DIR,
                             allow_none=True,
                             desc='The directory to store all the output files generated '
                                  'from the optimization.')

    def _get_name(self):
        """
        Get the name of this driver.

        Returns
        -------
        str
            Driver name in the format 'modOpt_<optimizer_name>'.
        """
        return f"modOpt_{self.options['optimizer']}"

    def _setup_verbosity(self, opt):
        """
        Configure optimizer-specific verbosity settings based on the driver's disp option.

        Different optimizers use different option names and scales for controlling output
        verbosity. This method translates the driver's 'disp' option to the appropriate
        optimizer-specific settings if user hasn't set them already.

        Parameters
        ----------
        opt : str
            Name of the optimizer being used.
        """
        disp = self.options['disp']
        opt_keys_lower = [k.lower() for k in self.opt_settings]

        # Map disp option to optimizer-specific settings, skipping if the user
        # has already specified the relevant key in opt_settings. Optimizer
        # specific display setting by the user takes precedence over "disp" option.
        # CVXOPT and ConvexQPSolvers are rejected at runtime due to Hessian requirements
        if opt == 'IPOPT':
            if 'print_level' not in opt_keys_lower:
                if isinstance(disp, int):
                    self.opt_settings['print_level'] = min(max(disp, 0), 12)
                else:
                    self.opt_settings['print_level'] = 5 if disp else 0

        elif opt == 'SNOPT':
            if 'major print level' not in opt_keys_lower \
                    and 'minor print level' not in opt_keys_lower:
                if isinstance(disp, int):
                    level = min(max(disp, 0), 10)
                    self.opt_settings['Major print level'] = level
                    self.opt_settings['Minor print level'] = level
                else:
                    self.opt_settings['Major print level'] = 1 if disp else 0
                    self.opt_settings['Minor print level'] = 0

        elif opt == 'TrustConstr':
            if 'verbose' not in opt_keys_lower:
                if isinstance(disp, int):
                    self.opt_settings['verbose'] = min(max(disp, 0), 3)
                else:
                    self.opt_settings['verbose'] = 1 if disp else 0

        elif opt == 'PySLSQP':
            if 'iprint' not in opt_keys_lower:
                if isinstance(disp, int):
                    self.opt_settings['iprint'] = disp
                else:
                    self.opt_settings['iprint'] = 1 if disp else 0

        elif opt == 'LBFGSB':
            if 'iprint' not in opt_keys_lower:
                if isinstance(disp, int):
                    self.opt_settings['iprint'] = disp
                else:
                    self.opt_settings['iprint'] = 0 if disp else -1

        elif opt == 'OpenSQP':
            if 'verbosity' not in opt_keys_lower:
                if isinstance(disp, int):
                    self.opt_settings['verbosity'] = disp
                else:
                    self.opt_settings['verbosity'] = 1 if disp else 0

        else:
            # Most SciPy-based optimizers (SLSQP, COBYLA, COBYQA, BFGS, NelderMead)
            # use 'disp' as boolean
            if 'disp' not in opt_keys_lower:
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

        # Validate problem formulation
        if not self.supports['multiple_objectives'] and len(self._objs) > 1:
            msg = '{} currently does not support multiple objectives.'
            raise RuntimeError(msg.format(self.msginfo))

        # CVXOPT and ConvexQPSolvers require Hessian information which is not yet supported
        if opt.lower() in ["convexqpsolvers", 'cvxopt']:
            msg = ('{} currently does not support CVXOPT and ConvexQPSolvers '
                   'due to the requirement of Hessian information.')
            raise RuntimeError(msg.format(self.msginfo))

        self._model_ran = False
        self._setup_tot_jac_sparsity()

    def run(self):
        """
        Optimize the problem using the selected modOpt optimizer.

        The optimization process performs the following steps:
        1. Initial model evaluation
        2. Extract and format design variables, bounds, and constraints
        3. Pre-compute linear constraint Jacobians for efficiency
        4. Create modOptProblem wrapper around the OpenMDAO problem
        5. Instantiate and run the selected optimizer
        6. Extract optimal design variables and update the model
        7. Perform final evaluation at the optimal point

        The optimization uses modOpt's Problem API, which handles callbacks for
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

        self._check_for_missing_objective()
        self._check_for_invalid_desvar_values()
        self._check_obj_grad = self.options['singular_jac_behavior'] in ['error', 'warn']
        self._check_nl_jac = self.options['singular_jac_behavior'] in ['error', 'warn']

        # Perform initial model evaluation
        with RecordingDebugging(self._get_name(), self.iter_count, self):
            self._run_solve_nonlinear()
            model_ran = True
            self.iter_count += 1

        self._model_ran = model_ran
        self._coloring_info.run_model = not model_ran

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
        lower_dv, upper_dv, _ = self._autoscaler.get_bounds_scaling('design_var')
        use_bounds = (opt in _bounds_optimizers)
        for name, meta in self._designvars.items():
            x_info[name] = {}
            x_info[name]['init'] = desvar_vals[name]

            if use_bounds:
                x_info[name]['lower'] = lower_dv[name]
                x_info[name]['upper'] = upper_dv[name]
            else:
                x_info[name]['lower'] = None
                x_info[name]['upper'] = None

        # compute dynamic simul deriv coloring
        prob.get_total_coloring(self._coloring_info, run_model=not model_ran)

        # Check for constraints that don't depend on any design variables
        # relevance._no_dv_responses contains outputs that are not affected by any design variables
        relevance = model._relevance
        bad_resps = [n for n in relevance._no_dv_responses if n in self._cons]
        bad_cons = [n for n, m in self._cons.items() if m['source'] in bad_resps]

        if bad_cons:
            issue_warning(f"Constraint(s) {sorted(bad_cons)} do not depend on any design "
                          "variables and were not added to the optimization.")
            for name in bad_cons:
                del self._cons[name]
                del self._responses[name]

        # Collect constraint information
        lincongrad = None
        nl_con_bounds = dict()
        lin_con_bounds = dict()
        nl_con_jac_sparsity = dict()
        all_nl_relevant_dvs = set()
        if opt in _constraint_optimizers:
            # Identify linear constraints and pre-compute their Jacobians if optimizer
            # uses gradients
            if opt in _constraint_grad_optimizers:
                lincons = [name for name, meta in self._cons.items() if meta.get('linear')]
            else:
                lincons = []

            # Use relevance to determine which design variables affect linear constraints
            # Relevance works by setting a "seed" (the output we care about) and then checking
            # which inputs (design variables) are "relevant" (affect that output).
            # This is like reverse-mode AD: start from output, trace back to find affecting inputs.
            if lincons:
                # Collect all design variables that affect any linear constraint
                all_relevant_dvs = set()
                for name, meta in self._cons.items():
                    if meta.get('linear'):
                        # Set this constraint as a reverse seed to find relevant design variables
                        # meta['source'] is the absolute path of the constraint output variable
                        with relevance.seeds_active(rev_seeds=(meta['source'],)):
                            # Check each design variable to see if it affects this constraint
                            # dv_meta['source'] is the absolute path of the design variable
                            for dv_name, dv_meta in self._designvars.items():
                                if relevance.is_relevant(dv_meta['source']):
                                    all_relevant_dvs.add(dv_name)

                # Compute Jacobians only for relevant design variables
                if all_relevant_dvs:
                    lincongrad = self._lincongrad_cache = \
                        self._compute_totals(of=lincons, wrt=list(all_relevant_dvs))
                else:
                    lincongrad = {}
                    self._lincongrad_cache = None
            else:
                self._lincongrad_cache = None

            # Process constraints and organize into linear and nonlinear categories
            lower_con, upper_con, equals_con = self._autoscaler.get_bounds_scaling('constraint')
            for name, meta in self._cons.items():
                if meta['indices'] is not None:
                    meta['size'] = size = meta['indices'].indexed_src_size
                else:
                    size = meta['global_size'] if meta['distributed'] else meta['size']

                # Separate linear and nonlinear constraints
                if meta['linear']:
                    if meta['equals'] is not None:
                        lin_con_bounds[name] = {
                            'lower': equals_con[name],
                            'upper': equals_con[name],
                            'size': size
                        }
                    else:
                        lin_con_bounds[name] = {
                            'lower': lower_con[name],
                            'upper': upper_con[name],
                            'size': size
                        }
                else:
                    if meta['equals'] is not None:
                        nl_con_bounds[name] = {
                            'lower': equals_con[name],
                            'upper': equals_con[name],
                            'size': size
                        }
                    else:
                        nl_con_bounds[name] = {
                            'lower': lower_con[name],
                            'upper': upper_con[name],
                            'size': size
                        }

                    # Initialize sparsity structure for nonlinear constraint Jacobians
                    # Use relevance to only include design variables that affect this constraint.
                    # For each nonlinear constraint, set it as a reverse seed and check which
                    # design variables are relevant (i.e., affect this constraint).
                    with relevance.seeds_active(rev_seeds=(meta['source'],)):
                        for x_name, x_meta in self._designvars.items():
                            # Only declare Jacobian if this design variable affects this constraint
                            if relevance.is_relevant(x_meta['source']):
                                nl_con_jac_sparsity[name, x_name] = {}

                                # Populate sparsity information from OpenMDAO's coloring
                                # Sparsity (rows/cols) describes which specific elements are nonzero
                                if name in self._con_subjacs and x_name in self._con_subjacs[name]:
                                    # Extract rows/cols from COO format sparsity data
                                    rows, cols, _ = self._con_subjacs[name][x_name]['coo']
                                    nl_con_jac_sparsity[name, x_name]['rows'] = rows
                                    nl_con_jac_sparsity[name, x_name]['cols'] = cols
                                else:
                                    # No sparsity info available - use dense (None means dense
                                    # to modOpt)
                                    nl_con_jac_sparsity[name, x_name]['rows'] = None
                                    nl_con_jac_sparsity[name, x_name]['cols'] = None

                                # Track all design variables relevant to any nonlinear constraint
                                all_nl_relevant_dvs.add(x_name)

        # Run optimization
        try:
            # Build modOpt Problem wrapper
            self._mo_prob = modOptProblem(
                driver=self,
                x_info=x_info,
                lin_con_jac=lincongrad,
                lin_con_bounds=lin_con_bounds,
                nl_con_bounds=nl_con_bounds,
                nl_con_jac_sparsity=nl_con_jac_sparsity,
                all_nl_relevant_dvs=all_nl_relevant_dvs,
            )

            # Resolve output directory. modOpt raises ValueError if out_dir is set
            # while turn_off_outputs=True, so pass None in that case.
            if self.options['turn_off_outputs']:
                out_dir = None
            elif self.options['output_dir'] in (None, _DEFAULT_REPORTS_DIR):
                out_dir = str(self._problem().get_outputs_dir(mkdir=True))
            else:
                out_dir = self.options['output_dir']

            # Instantiate and run optimizer
            optimizer_cls = getattr(mo, opt)
            if opt in _solver_options_optimizers:
                optimizer = optimizer_cls(
                    problem=self._mo_prob,
                    turn_off_outputs=self.options['turn_off_outputs'],
                    out_dir=out_dir,
                    solver_options=self.opt_settings,
                )
            else:
                optimizer = optimizer_cls(
                    problem=self._mo_prob,
                    turn_off_outputs=self.options['turn_off_outputs'],
                    out_dir=out_dir,
                    **self.opt_settings,
                )
            result = optimizer.solve()

            # Ensure all MPI ranks have finished the optimizer before proceeding.
            if MPI:
                prob.model.comm.Barrier()

            # Extract optimal design variables and success flag from optimizer result
            # Different optimizers return results in different formats
            if hasattr(result, 'x'):
                x_opt = result.x
                self.fail = not result.success
            elif 'x' in result:
                x_opt = result['x']
                self.fail = not result['success']
            else:
                # Fallback for optimizers with non-standard result format
                x_opt = self._mo_prob.dvs['x'].get_data()
                self.fail = False
                print(f'{"-" * 40}\n {opt} does not return success status in '
                        f'a consistent, easily readable way, so defaulting to '
                        f'self.fail=False. \n{"-" * 40}\n')

            # Update OpenMDAO design variables with optimal values
            self._vectors['design_var'].set_data(x_opt, driver_scaling=True)
            self._set_design_vars(desvar_names=list(x_info.keys()), driver_scaling=True)

            # Final model evaluation at optimal point
            with RecordingDebugging(self._get_name(), self.iter_count, self):
                self._run_solve_nonlinear()
                self._model_ran = model_ran
            self.iter_count += 1

            if self.options['disp']:
                if prob.comm.rank == 0:
                    print('Optimization Complete')
                    print('-' * 35)

        except Exception:
            # If an exception occurred in one of our callbacks, re-raise it with
            # the original context rather than modOpt's generic exception message
            if self._exc_info is None:
                raise

        if self._exc_info is not None:
            self._reraise()

        return self.fail

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
        self._con_subjacs = {}
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
