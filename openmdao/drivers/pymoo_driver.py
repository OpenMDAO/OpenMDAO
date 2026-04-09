"""
OpenMDAO Wrapper for the pymoo optimization library.

pymoo offers state of the art single- and multi-objective optimization algorithms
and many more features related to multi-objective optimization such as
visualization and decision making.

Available Optimizers
--------------------
Single-Objective:
    - GA: Genetic Algorithm
    - DE: Differential Evolution
    - BRKGA: Biased Random Key Genetic Algorithm
    - NelderMead: Nelder-Mead simplex algorithm
    - PatternSearch: Pattern search algorithm
    - CMAES: Covariance Matrix Adaptation Evolution Strategy
    - ES: Evolution Strategy
    - SRES: Stochastic Ranking Evolution Strategy
    - ISRES: Improved Stochastic Ranking Evolution Strategy
    - NRBO: Newton-Raphson Based Optimizer

Multi-Objective:
    - NSGA2: Non-dominated Sorting Genetic Algorithm II
    - RNSGA2: Reference Point Based NSGA-II
    - PINSGA2: Pareto-Improving NSGA-II
    - NSGA3: Non-dominated Sorting Genetic Algorithm III
    - UNSGA3: Unified NSGA-III
    - RNSGA3: Reference Point Based NSGA-III
    - MOEAD: Multi-Objective Evolutionary Algorithm Based on Decomposition
    - AGEMOEA: Adaptive Geometry Estimation based MOEA
    - CTAEA: Constrained Two-Archive Evolutionary Algorithm
    - SMSEMOA: S-Metric Selection EMOA
    - RVEA: Reference Vector Guided Evolutionary Algorithm
    - CMOPSO: Constrained Multi-Objective Particle Swarm Optimization
    - MOPSO_CD: Multi-Objective Particle Swarm Optimization with Crowding Distance

Notes
-----
Algorithm-specific hyperparameters (e.g. population size, mutation and crossover
operators) are passed via the ``alg_settings`` dict, which is unpacked into the
algorithm constructor.

Run-level settings accepted by pymoo's ``algorithm.setup()`` (e.g. ``seed``,
``verbose``, ``termination``, ``callback``, ``save_history``) are passed via the
``run_settings`` dict, which is unpacked into ``pymoo.optimize.minimize()``.

For multi-objective optimizations the Pareto front is stored on the driver in
``driver.pareto['X']`` and ``driver.pareto['F']`` after ``run_driver()`` completes.

For additional processing, the pymoo results object can be accessed at the
``pymoo``_results attribute on the driver.

See the pymoo documentation at https://pymoo.org/index.html for detailed information
on algorithm-specific options and capabilities.
"""
import sys
import numpy as np
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.om_warnings import issue_warning
from openmdao.core.constants import INF_BOUND
try:
    import pymoo as pm
    problem = pm.core.problem.ElementwiseProblem
except ImportError:
    pm = None
    problem = object
except Exception as err:
    pm = err
    problem = object


# Algorithms that support constraints
_constraint_optimizers = {'GA', 'DE', 'BRKGA', 'NelderMead', 'PatternSearch',
                          'ES', 'SRES', 'ISRES', 'NRBO', 'NSGA2', 'RNSGA2',
                          'PINSGA2', 'NSGA3', 'UNSGA3', 'RNSGA3', 'CTAEA',
                          'SMSEMOA', 'CMOPSO', 'MOPSO_CD'}

# Algorithms that only support a single objective
_single_obj_optimizers = {'GA', 'DE', 'BRKGA', 'NelderMead', 'PatternSearch',
                          'CMAES', 'ES', 'SRES', 'ISRES', 'NRBO'}

# Algorithms that support multiple objectives
_multi_obj_optimizers = {'NSGA2', 'RNSGA2', 'PINSGA2', 'NSGA3', 'UNSGA3', 'RNSGA3',
                         'MOEAD', 'AGEMOEA', 'CTAEA', 'SMSEMOA', 'RVEA', 'CMOPSO',
                         'MOPSO_CD'}

# All available optimizers
_all_optimizers = _single_obj_optimizers | _multi_obj_optimizers

CITATIONS = """
@ARTICLE{pymoo,
    author={J. {Blank} and K. {Deb}},
    journal={IEEE Access},
    title={pymoo: Multi-Objective Optimization in Python},
    year={2020},
    volume={8},
    number={},
    pages={89497-89509},
}
"""


class pymooProblem(problem):
    """
    Pymoo ElementwiseProblem that delegates function evaluation to an OpenMDAO driver.

    Wraps an OpenMDAO problem as a pymoo optimization problem, translating between
    pymoo's interface and OpenMDAO's driver interface. Inequality constraints are
    converted to the pymoo convention (g <= 0) and equality constraints to (h == 0).

    Parameters
    ----------
    driver : pymooDriver
        The OpenMDAO driver managing the optimization.
    x_info : dict
        Design variable metadata with keys 'vars', 'indices', 'lower', 'upper'.
    ieq_con_info : dict
        Inequality constraint metadata with keys 'vars', 'indices', 'is_upper', 'bound'.
    eq_con_info : dict
        Equality constraint metadata with keys 'vars', 'indices', 'equals'.
    obj_info : dict
        Objective metadata with keys 'vars', 'indices', 'size'.

    Attributes
    ----------
    driver : pymooDriver
        Reference to the OpenMDAO driver.
    x_info : dict
        Design variable metadata.
    ieq_con_info : dict
        Inequality constraint metadata.
    eq_con_info : dict
        Equality constraint metadata.
    obj_info : dict
        Objective metadata.
    """

    def __init__(self, driver, x_info, ieq_con_info, eq_con_info, obj_info):
        """
        Initialize the pymooProblem.

        Parameters
        ----------
        driver : pymooDriver
            The OpenMDAO driver managing the optimization.
        x_info : dict
            Design variable metadata with keys 'vars', 'indices', 'lower', 'upper'.
        ieq_con_info : dict
            Inequality constraint metadata with keys 'vars', 'indices', 'is_upper', 'bound'.
        eq_con_info : dict
            Equality constraint metadata with keys 'vars', 'indices', 'equals'.
        obj_info : dict
            Objective metadata with keys 'vars', 'indices', 'size'.
        """
        self.driver = driver
        self.x_info = x_info
        self.ieq_con_info = ieq_con_info
        self.eq_con_info = eq_con_info
        self.obj_info = obj_info

        super().__init__(
            # Integer value representing the number of design variables.
            n_var=len(x_info['upper']),
            # Integer value representing the number of objectives.
            n_obj=obj_info['size'],
            # Integer value representing the number of inequality constraints.
            n_ieq_constr=len(ieq_con_info['bound']),
            # Integer value representing the number of equality constraints.
            n_eq_constr=len(eq_con_info['equals']),
            # Float or np.ndarray of length n_var representing the lower bounds of
            # the design variables.
            xl=x_info['lower'],
            # Float or np.ndarray of length n_var representing the upper bounds of
            # the design variables.
            xu=x_info['upper'],
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate objectives and constraints at the given design point.

        Updates OpenMDAO design variables, runs the model, and populates pymoo's
        output dictionary with objective values in ``out['F']``, inequality
        constraint values in ``out['G']``, and equality constraint values in
        ``out['H']``. On failure, all outputs are set to NaN.

        Parameters
        ----------
        x : np.ndarray
            Current design variable values as a flat array.
        out : dict
            Pymoo output dictionary to populate with F, G, and H values.
        *args : list
            Unused positional arguments passed by pymoo.
        **kwargs : dict
            Unused keyword arguments passed by pymoo.
        """
        model = self.driver._problem().model

        try:
            self._update_desvar_values(x)
            self._run_model()

            # Get the objective function evaluations
            obj_vals = self.driver.get_objective_values()
            obj_zip = zip(self.obj_info['vars'], self.obj_info['indices'])
            for name, indices, in obj_zip:
                out['F'][indices] = obj_vals[name].flatten()

            # Get the constraint evaluations. In pymoo all inequality constraints
            # must be <= 0.0 and all equality constraints must be == 0.0
            con_vals = self.driver.get_constraint_values()
            ieq_zip = zip(self.ieq_con_info['vars'], self.ieq_con_info['indices'],
                          self.ieq_con_info['is_upper'])
            eq_zip = zip(self.eq_con_info['vars'], self.eq_con_info['indices'])
            for name, indices, is_upper in ieq_zip:
                bound = self.ieq_con_info['bound'][indices]
                if is_upper:
                    out['G'][indices] = bound - con_vals[name].flatten()
                else:
                    out['G'][indices] = con_vals[name].flatten() - bound
            for name, indices in eq_zip:
                equals = self.eq_con_info['equals'][indices]
                out['H'][indices] = equals - con_vals[name].flatten()

        except Exception:
            # Clean up solver print stack and store exception for re-raising later
            self._handle_callback_exception(model)
            out['F'] = np.full(len(out['F']), np.nan)
            out['G'] = np.full(len(out['G']), np.nan)
            out['H'] = np.full(len(out['H']), np.nan)

    def _update_desvar_values(self, x):
        """
        Update OpenMDAO design variables from pymoo's design vector.

        Parameters
        ----------
        x : np.ndarray
            Flat array of current design variable values.
        """
        for name, indices in zip(self.x_info['vars'], self.x_info['indices']):
            self.driver.set_design_var(name, x[indices])

    def _run_model(self):
        """
        Execute the OpenMDAO model with proper recording and relevance handling.

        Only evaluates the full model on the first iteration (sets _model_ran flag).
        Subsequent iterations use relevance filtering for efficiency.
        """
        model = self.driver._problem().model
        with RecordingDebugging(self.driver._get_name(), self.driver.iter_count, self.driver):
            self.driver.iter_count += 1
            with model._relevance.nonlinear_active('iter', active=self.driver._model_ran):
                self.driver._run_solve_nonlinear()
                self.driver._model_ran = True

    def _handle_callback_exception(self, model):
        """
        Handle exceptions raised during pymoo evaluation callbacks.

        Clears the solver print stack and stores exception info for re-raising
        after the optimization loop exits.

        Parameters
        ----------
        model : System
            The model to clear iprint on.
        """
        model._clear_iprint()
        if self.driver._exc_info is None:
            self.driver._exc_info = sys.exc_info()


class pymooDriver(Driver):
    """
    Driver wrapper for the pymoo optimization library.

    Supports both single- and multi-objective gradient-free optimization using
    evolutionary and swarm-based algorithms. For single-objective problems the
    model is set to the optimal point after ``run_driver()`` completes. For
    multi-objective problems the full Pareto front is stored in ``driver.pareto``.

    pymooDriver supports the following:
        equality_constraints (algorithm-dependent)
        inequality_constraints (algorithm-dependent)
        two_sided_constraints (algorithm-dependent)
        linear_constraints (algorithm-dependent)
        multiple_objectives (algorithm-dependent)

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.

    Attributes
    ----------
    alg_settings : dict
        Algorithm-specific hyperparameters passed to the algorithm constructor
        (e.g. ``pop_size``, mutation and crossover operators).
    run_settings : dict
        Run-level settings passed to ``pymoo.optimize.minimize()`` and forwarded
        to ``algorithm.setup()`` (e.g. ``seed``, ``verbose``, ``termination``).
    pymoo_results : pymoo.core.result.Result
        The result object returned by ``pymoo.optimize.minimize()`` after the
        optimization completes.
    pareto : dict
        Pareto front results for multi-objective optimizations. Contains keys
        'X' (design variable array, shape (n_solutions, n_vars)) and 'F'
        (objective array, shape (n_solutions, n_objs)). Populated only when a
        multi-objective optimizer is used.
    alg_class : type
        The pymoo algorithm class resolved from the 'optimizer' option.
    _model_ran : bool
        Flag indicating whether the model has been run at least once, used to
        control relevance filtering on subsequent evaluations.
    _moo_prob : pymooProblem or None
        The pymoo problem wrapper built during ``run()``.
    """

    def __init__(self, **kwargs):
        """
        Initialize the pymooDriver.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        if pm is None:
            raise RuntimeError('pymooDriver is not available, pymoo is not'
                               ' installed.')

        if isinstance(pm, Exception):
            # there is some other issue with the pymoo installation
            raise pm

        super().__init__(**kwargs)

        # What we support
        self.supports['optimization'] = True
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['two_sided_constraints'] = True
        self.supports['linear_constraints'] = True
        self.supports['linear_only_designvars'] = True
        self.supports['multiple_objectives'] = True

        # What we don't support
        self.supports['active_set'] = False
        self.supports['integer_design_vars'] = False
        self.supports['distributed_design_vars'] = False
        self.supports['gradients'] = False
        self.supports._read_only = True

        # The user places algorithm-specific settings in here that are passed
        # into the algorithm instantiation.
        self.alg_settings = {}
        # The user places algorithm-specific settings in here that are passed
        # into the algorithm setup.
        self.run_settings = {}

        self._check_obj_grad = False
        self._check_nl_jac = False
        self._total_jac_sparsity = None
        self._model_ran = False
        self._moo_prob = None
        self.alg_class = None
        self.pareto = {'X': None, 'F': None}

        self.cite = CITATIONS

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('optimizer', 'GA', values=_all_optimizers,
                             desc='Name of optimizer to use')
        self.options.declare('disp', default=True, types=(int, bool),
                             desc='Controls optimizer output verbosity. Not used '
                                  'if "verbose" is manually set in "self.run_settings".')

    def _get_name(self):
        """
        Get the name of this driver.

        Returns
        -------
        str
            Driver name in the format 'pymoo_<optimizer_name>'.
        """
        return f"pymoo_{self.options['optimizer']}"

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        Configures support flags based on the selected algorithm's capabilities,
        validates the problem formulation, and resolves the algorithm class.
        Called during problem setup.

        Parameters
        ----------
        problem : Problem
            The OpenMDAO Problem being optimized.
        """
        super()._setup_driver(problem)
        opt = self.options['optimizer']

        # Update support flags based on optimizer capabilities
        self.supports._read_only = False
        self.supports['inequality_constraints'] = opt in _constraint_optimizers
        self.supports['two_sided_constraints'] = opt in _constraint_optimizers
        self.supports['equality_constraints'] = opt in _constraint_optimizers
        self.supports['linear_constraints'] = opt in _constraint_optimizers
        self.supports['multiple_objectives'] = opt in _multi_obj_optimizers
        self.supports._read_only = True

        # Validate problem formulation
        if not self.supports['multiple_objectives'] and len(self._objs) > 1:
            msg = 'The {} algorithm in {} currently does not support multiple objectives.'
            raise RuntimeError(msg.format(opt, self.msginfo))

        self._model_ran = False
        self.alg_class = self.get_algorithm(opt)

    def run(self):
        """
        Optimize the problem using the selected pymoo optimizer.

        Returns
        -------
        bool
            True if the optimization found a feasible solution, False otherwise.
        """
        success = False
        self.result.reset()
        prob = self._problem()
        opt = self.options['optimizer']
        model = prob.model
        self.iter_count = 0
        self._desvar_array_cache = None

        self._check_for_missing_objective()
        self._check_for_invalid_desvar_values()

        # Perform initial model evaluation
        with RecordingDebugging(self._get_name(), self.iter_count, self):
            self._run_solve_nonlinear()
            model_ran = True
            self.iter_count += 1
        self._model_ran = model_ran
        self._con_cache = self.get_constraint_values()

        # Determine total number of design variables for error and x_info initialization
        ndesvar = 0
        for name, meta in self._designvars.items():
            size = meta['global_size'] if meta['distributed'] else meta['size']
            ndesvar += size

        if ndesvar == 0:
            raise RuntimeError('Problem has no design variables.')

        # Collect design variable information (initial values and bounds)
        x_info = {'vars': [], 'upper': np.full(ndesvar, 1e30),
                  'lower': np.full(ndesvar, -1e30), 'indices': []}
        current_idx = 0
        for name, meta in self._designvars.items():
            x_info['vars'].append(name)

            size = meta['global_size'] if meta['distributed'] else meta['size']
            current_indices = list(range(current_idx, current_idx + size))
            x_info['indices'].append(current_indices)
            x_info['lower'][current_indices] = meta['lower']
            x_info['upper'][current_indices] = meta['upper']
            current_idx += size

        # Check for constraints that don't depend on any design variables
        # relevance._no_dv_responses contains outputs that are not affected by any design variables
        relevance = model._relevance
        bad_resps = [n for n in relevance._no_dv_responses if n in self._cons]
        bad_cons = [n for n, m in self._cons.items() if m['source'] in bad_resps]
        if bad_cons:
            issue_warning(f'Constraint(s) {sorted(bad_cons)} do not depend on any design '
                          'variables and were not added to the optimization.')
            for name in bad_cons:
                del self._cons[name]
                del self._responses[name]

        # Determine total number of constraints
        neqcons = 0
        nieqcons = 0
        if opt in _constraint_optimizers:
            for name, meta in self._cons.items():
                if meta['indices'] is not None:
                    meta['size'] = size = meta['indices'].indexed_src_size
                else:
                    size = meta['global_size'] if meta['distributed'] else meta['size']

                if meta['equals'] is not None:
                    neqcons += size
                else:
                    if np.any(meta['upper'] < INF_BOUND):
                        nieqcons += size
                    if np.any(meta['lower'] > -INF_BOUND):
                        nieqcons += size

        # Collect constraint information
        ieq_con_info = {'vars': [], 'indices': [], 'is_upper': [], 'bound': np.empty(nieqcons)}
        eq_con_info = {'vars': [], 'indices': [], 'equals': np.empty(neqcons)}
        current_eq_idx = 0
        current_ieq_idx = 0
        if opt in _constraint_optimizers:

            for name, meta in self._cons.items():
                if meta['indices'] is not None:
                    size = meta['indices'].indexed_src_size
                else:
                    size = meta['global_size'] if meta['distributed'] else meta['size']

                # Separate equality and inequality constraints
                if meta['equals'] is not None:
                    current_eq_indices = list(range(current_eq_idx, current_eq_idx + size))
                    eq_con_info['vars'].append(name)
                    eq_con_info['indices'].append(current_eq_indices)
                    eq_con_info['equals'][current_eq_indices] = meta['equals']
                    current_eq_idx += size

                else:
                    # Need to log upper and lower as separate inequality constraints
                    if np.any(meta['upper'] < INF_BOUND):
                        current_ieq_indices = list(range(current_ieq_idx, current_ieq_idx + size))
                        ieq_con_info['vars'].append(name)
                        ieq_con_info['indices'].append(current_ieq_indices)
                        ieq_con_info['is_upper'].append(True)
                        ieq_con_info['bound'][current_ieq_indices] = meta['upper']
                        current_ieq_idx += size

                    if np.any(meta['lower'] > -INF_BOUND):
                        current_ieq_indices = list(range(current_ieq_idx, current_ieq_idx + size))
                        ieq_con_info['vars'].append(name)
                        ieq_con_info['indices'].append(current_ieq_indices)
                        ieq_con_info['is_upper'].append(False)
                        ieq_con_info['bound'][current_ieq_indices] = meta['lower']
                        current_ieq_idx += size

        # Collect objective information
        obj_info = {'vars': [], 'indices': [], 'size': 1}
        current_idx = 0
        for name, meta in self._objs.items():
            if meta['indices'] is not None:
                meta['size'] = size = meta['indices'].indexed_src_size
            else:
                size = meta['global_size'] if meta['distributed'] else meta['size']

            current_indices = list(range(current_idx, current_idx + size))
            obj_info['vars'].append(name)
            obj_info['indices'].append(current_indices)
            current_idx += size
        obj_info['size'] = current_idx

        # Run optimization
        try:
            # Build modOpt Problem wrapper
            self._moo_prob = pymooProblem(
                driver=self,
                x_info=x_info,
                ieq_con_info=ieq_con_info,
                eq_con_info=eq_con_info,
                obj_info=obj_info,
            )

            # Instantiate and run optimizer
            optimizer = self.alg_class(**self.alg_settings)

            run_settings = {**self.run_settings}
            if "verbose" not in self.run_settings:
                run_settings['verbose'] = self.options['disp']
            self.pymoo_results = pm.optimize.minimize(
                problem=self._moo_prob,
                algorithm=optimizer,
                **run_settings,
            )

            # Extract optimal design variables and success flag from optimizer result
            # Different optimizers return results in different formats
            x_opt = self.pymoo_results.X
            F_opt = self.pymoo_results.F
            success = self.pymoo_results.X is not None
            if opt in _single_obj_optimizers:
                if success:
                    # Update OpenMDAO design variables with optimal values
                    for name, indices in zip(x_info['vars'], x_info['indices']):
                        self.set_design_var(name, x_opt[indices])

                    # Final model evaluation at optimal point
                    with RecordingDebugging(self._get_name(), self.iter_count, self):
                        self._run_solve_nonlinear()
                        self._model_ran = model_ran
                    self.iter_count += 1

            # For pareto frontiers, just leave the model in whatever its last state was
            else:
                self.pareto['X'] = x_opt
                self.pareto['F'] = F_opt

            if run_settings['verbose']:
                if prob.comm.rank == 0:
                    print('Optimization Complete')
                    print('-' * 35)

        except Exception:
            # If an exception occurred in one of our callbacks, re-raise it with
            # the original traceback rather than pymoo's generic exception message
            if self._exc_info is None:
                raise

        if self._exc_info is not None:
            self._reraise()

        return success

    def get_algorithm(self, alg_name):
        """
        Return the pymoo algorithm class for the given algorithm name.

        Parameters
        ----------
        alg_name : str
            Name of the algorithm, must be a member of ``_all_optimizers``.

        Returns
        -------
        type
            The pymoo algorithm class corresponding to ``alg_name``.
        """
        if alg_name in _single_obj_optimizers:
            from pymoo.algorithms.soo import nonconvex as alg_lib
        else:
            from pymoo.algorithms import moo as alg_lib

        # The script where the algorithm classes are located are all just a
        # lowercase of the algorithm name, except for the specifically called
        # out algorithms
        non_default_mapping = {
            'NelderMead': 'nelder',
            'PatternSearch': 'pattern',
            'AGEMOEA': 'age',
            'SMSEMOA': 'sms',
        }
        if alg_name not in non_default_mapping:
            alg_class = getattr(getattr(alg_lib, alg_name.lower()), alg_name)
        else:
            alg_class = getattr(getattr(alg_lib, non_default_mapping[alg_name]), alg_name)

        return alg_class
