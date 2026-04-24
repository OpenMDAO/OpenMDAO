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
    - DIRECT: DIRECT (Dividing RECTangles) deterministic global optimizer
    - G3PCX: Generalized Generation Gap with Parent-Centric Crossover (no constraint support)
    - NicheGA: Niching Genetic Algorithm
    - PSO: Particle Swarm Optimization (no constraint support)
    - EPPSO: Extended and Parallelised PSO (no constraint support)
    - RandomSearch: Random Search (no constraint support)
    - Optuna: Optuna-based optimizer (requires the ``optuna`` package)
    - MixedVariableGA: Genetic Algorithm with support for discrete (integer) and
      mixed integer design variables

Multi-Objective:
    - NSGA2: Non-dominated Sorting Genetic Algorithm II
    - RNSGA2: Reference Point Based NSGA-II
    - PINSGA2: Pareto-Improving NSGA-II
    - NSGA3: Non-dominated Sorting Genetic Algorithm III
    - UNSGA3: Unified NSGA-III
    - RNSGA3: Reference Point Based NSGA-III
    - MOEAD: Multi-Objective Evolutionary Algorithm Based on Decomposition
    - AGEMOEA: Adaptive Geometry Estimation based MOEA
    - AGEMOEA2: Improved Adaptive Geometry Estimation based MOEA (no constraint support)
    - CTAEA: Constrained Two-Archive Evolutionary Algorithm
    - SMSEMOA: S-Metric Selection EMOA
    - RVEA: Reference Vector Guided Evolutionary Algorithm
    - CMOPSO: Constrained Multi-Objective Particle Swarm Optimization
    - MOPSO_CD: Multi-Objective Particle Swarm Optimization with Crowding Distance
    - DNSGA2: Dynamic NSGA-II (unconstrained only)
    - KGB: KGB-DMOEA (unconstrained only)
    - SPEA2: Strength Pareto Evolutionary Algorithm 2

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

Population-level MPI parallelism is available by setting ``run_parallel=True``.
Ranks are divided into groups of ``procs_per_model`` (default 1), where each group
cooperates on a single model evaluation. MPI parallelism is only beneficial when
individual model evaluations are computationally expensive.

For additional processing, the pymoo results object can be accessed at the
``pymoo``_results attribute on the driver.

See the pymoo documentation at https://pymoo.org/index.html for detailed information
on algorithm-specific options and capabilities.
"""
import sys
import importlib
import numpy as np
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.core.constants import INF_BOUND
from openmdao.utils.mpi import MPI
try:
    import pymoo
    from pymoo.core.problem import ElementwiseProblem as problem
    from pymoo.core.variable import Real, Integer
    from pymoo.optimize import minimize
except ImportError:
    pm = None
    problem = object
    minimize = None
    Real = None
    Integer = None
except Exception as err:
    pm = err
    problem = object
    minimize = None
    Real = None
    Integer = None


# Algorithms that support constraints.
# Not all algorithms are explicilty mentioned in the documentation as supporting
# constraints or not, so I had to make an educated guess based on what was in the
# algorithm.
_constraint_optimizers = {'GA', 'DE', 'BRKGA', 'NelderMead', 'PatternSearch',
                          'ES', 'SRES', 'ISRES', 'NRBO', 'DIRECT', 'NicheGA',
                          'Optuna', 'NSGA2', 'RNSGA2', 'PINSGA2', 'NSGA3',
                          'UNSGA3', 'RNSGA3', 'CTAEA', 'SMSEMOA', 'CMOPSO',
                          'MOPSO_CD', 'SPEA2', 'MixedVariableGA'}

# Algorithms that only support a single objective
_single_obj_optimizers = {'GA', 'DE', 'BRKGA', 'NelderMead', 'PatternSearch',
                          'CMAES', 'ES', 'SRES', 'ISRES', 'NRBO', 'DIRECT',
                          'G3PCX', 'NicheGA', 'PSO', 'EPPSO', 'RandomSearch',
                          'Optuna', 'MixedVariableGA'}

# Algorithms that support discrete (integer) and mixed integer design variables
_mixed_var_optimizers = {'MixedVariableGA'}

# Algorithms that support multiple objectives
_multi_obj_optimizers = {'NSGA2', 'RNSGA2', 'PINSGA2', 'NSGA3', 'UNSGA3', 'RNSGA3',
                         'MOEAD', 'AGEMOEA', 'AGEMOEA2', 'CTAEA', 'SMSEMOA', 'RVEA',
                         'CMOPSO', 'MOPSO_CD', 'DNSGA2', 'KGB', 'SPEA2'}

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


class MPIElementwiseRunner:
    """
    Elementwise evaluation runner that distributes population members across MPI ranks.

    **Background**

    pymoo's ``ElementwiseProblem._evaluate`` evaluates one individual at a time.
    The elementwise runner is what calls ``_evaluate`` repeatedly to cover the whole
    population. The default runner, ``LoopedElementwiseEvaluation``, does this
    sequentially on a single process::

        return [f(x) for x in X]

    This runner replaces that loop with a parallel pattern using MPI.

    **How it works**

    The total MPI ranks are divided into *groups* of ``procs_per_model`` ranks each.
    With 8 total ranks and ``procs_per_model=2``, there are 4 groups:

    - Group 0: ranks 0 and 4
    - Group 1: ranks 1 and 5
    - Group 2: ranks 2 and 6
    - Group 3: ranks 3 and 7

    The group a rank belongs to is called its *color*: ``color = rank % n_groups``.
    All ranks in the same group share a model sub-communicator (set up in
    ``_setup_comm`` before ``Problem.setup()`` runs) so that models with parallel
    components (e.g. ``ParallelGroup``) receive the right communicator.

    At each generation, individuals in the population are distributed round-robin
    across groups. With 100 individuals and 4 groups:

    - Group 0 (ranks 0, 4) evaluates individuals 0, 4, 8, ..., 96
    - Group 1 (ranks 1, 5) evaluates individuals 1, 5, 9, ..., 97
    - Group 2 (ranks 2, 6) evaluates individuals 2, 6, 10, ..., 98
    - Group 3 (ranks 3, 7) evaluates individuals 3, 7, 11, ..., 99

    All ranks in a group call ``f(x)`` for the same individual, cooperating through
    the model sub-communicator. After evaluation, only the root rank of each group
    (i.e. ``rank < n_groups``) contributes its results to the ``allgather``, which
    broadcasts the complete population results to all ranks. pymoo then runs its
    selection/crossover/mutation identically on all ranks and moves to the next
    generation.

    When ``procs_per_model=1`` (the default), every rank is its own group — this
    reduces to simple round-robin across all ranks with no sub-communicator overhead.

    Parameters
    ----------
    comm : MPI.Comm
        The full problem-level communicator (not the model sub-communicator).
    procs_per_model : int
        Number of MPI ranks that cooperate on a single model evaluation.

    Attributes
    ----------
    comm : MPI.Comm
        The full problem-level communicator.
    n_groups : int
        Number of parallel evaluation groups (``comm.size // procs_per_model``).
    color : int
        The group index this rank belongs to (``comm.rank % n_groups``).
    """

    def __init__(self, comm, procs_per_model=1):
        """
        Initialize the MPIElementwiseRunner.

        Parameters
        ----------
        comm : MPI.Comm
            The full problem-level communicator (not the model sub-communicator).
        procs_per_model : int
            Number of MPI ranks that cooperate on a single model evaluation.
        """
        self.comm = comm
        self.n_groups = comm.size // procs_per_model
        # color identifies which evaluation group this rank belongs to.
        # Ranks with the same color share a model sub-communicator and always
        # evaluate the same individual together.
        self.color = comm.rank % self.n_groups

    def __call__(self, f, X):
        """
        Evaluate each individual in X, distributing work across MPI groups.

        ``f`` is pymoo's ``ElementwiseEvaluationFunction``, which calls
        ``pymooProblem._evaluate(x, out)`` for a single individual and returns
        the populated ``out`` dict containing 'F', 'G', and 'H' values.

        Parameters
        ----------
        f : callable
            Pymoo's per-individual evaluation function. Calling ``f(x)`` sets the
            design variables on the local model, runs it, and returns a dict with
            keys 'F' (objectives), 'G' (inequality constraints), 'H' (equality
            constraints).
        X : list
            List of individual design points for the current population.

        Returns
        -------
        list
            Evaluated output dicts in the same order as X, assembled from all groups.
        """
        n = len(X)

        # Round-robin by group: all ranks in the same group evaluate the same
        # individuals together via their shared model sub-communicator.
        local_indices = list(range(self.color, n, self.n_groups))
        local_results = [(i, f(X[i])) for i in local_indices]

        # Only the root rank of each group (rank < n_groups) contributes to the
        # allgather. Since every rank in a group evaluated the same individuals,
        # the non-root ranks would produce duplicate results. allgather (not gather)
        # is used so that ALL ranks end up with the full population — pymoo needs
        # to run its selection/crossover/mutation on every rank.
        allgather_input = local_results if self.comm.rank < self.n_groups else []
        all_results = self.comm.allgather(allgather_input)

        # Reconstruct results in original population order.
        ordered = [None] * n
        for rank_results in all_results:
            for i, result in rank_results:
                ordered[i] = result

        return ordered


class pymooProblem(problem):
    """
    Pymoo ElementwiseProblem that delegates function evaluation to an OpenMDAO driver.

    Wraps an OpenMDAO problem as a pymoo optimization problem, translating between
    pymoo's interface and OpenMDAO's driver interface. Inequality constraints are
    converted to the pymoo convention (g <= 0) and equality constraints to (h == 0).
    When ``MixedVariableGA`` is selected or discrete (integer) design variables
    are present, pymoo's ``vars`` dict interface is used so that pymoo samples
    and mutates variable types correctly.

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
    runner : callable or None
        Pymoo elementwise runner. Pass an ``MPIElementwiseRunner`` instance for
        population-level MPI parallelism. If None, uses pymoo's default sequential
        ``LoopedElementwiseEvaluation``.

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
    fail : bool
        Flag set to True if an exception occurred during evaluation.
    """

    def __init__(self, driver, x_info, ieq_con_info, eq_con_info, obj_info, runner=None):
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
        runner : callable or None
            Pymoo elementwise runner. Pass an ``MPIElementwiseRunner`` instance for
            population-level MPI parallelism. If None, uses pymoo's default sequential
            ``LoopedElementwiseEvaluation``.
        """
        self.driver = driver
        self.x_info = x_info
        self.ieq_con_info = ieq_con_info
        self.eq_con_info = eq_con_info
        self.obj_info = obj_info
        self.fail = False

        n_obj = obj_info['size']
        n_ieq_constr = len(ieq_con_info['bound'])
        n_eq_constr = len(eq_con_info['equals'])

        use_vars_dict = (driver._designvars_discrete or
                         driver.options['optimizer'] in _mixed_var_optimizers)
        if use_vars_dict:
            # Build a vars dict so pymoo samples integers and reals correctly.
            # Required for MixedVariableGA regardless of whether discrete variables
            # are present. Each scalar element of each design variable gets its own
            # entry using the key '{om_name}__{i}'.
            vars_dict = {}
            for name, indices in zip(x_info['vars'], x_info['indices']):
                lower = x_info['lower'][indices]
                upper = x_info['upper'][indices]
                for i, (lb, ub) in enumerate(zip(lower, upper)):
                    key = f'{name}__{i}'
                    if name in driver._designvars_discrete:
                        vars_dict[key] = Integer(bounds=(int(lb), int(ub)))
                    else:
                        vars_dict[key] = Real(bounds=(lb, ub))

            runner_kwargs = {'elementwise_runner': runner} if runner is not None else {}
            super().__init__(
                vars=vars_dict,
                n_obj=n_obj,
                n_ieq_constr=n_ieq_constr,
                n_eq_constr=n_eq_constr,
                **runner_kwargs
            )
        else:
            runner_kwargs = {'elementwise_runner': runner} if runner is not None else {}
            super().__init__(
                n_var=len(x_info['upper']),
                n_obj=n_obj,
                n_ieq_constr=n_ieq_constr,
                n_eq_constr=n_eq_constr,
                xl=x_info['lower'],
                xu=x_info['upper'],
                **runner_kwargs,
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
        x : np.ndarray or dict
            Current design variable values. A dict keyed by ``'{om_name}__{i}'``
            when using ``MixedVariableGA``, or a flat array for all other optimizers.
        out : dict
            Pymoo output dictionary to populate with F, G, and H values.
        *args : list
            Unused positional arguments passed by pymoo.
        **kwargs : dict
            Unused keyword arguments passed by pymoo.
        """
        model = self.driver._problem().model

        # Start empty and need to be populated
        out['F'] = np.empty(self.obj_info['size'])
        out['G'] = np.empty(len(self.ieq_con_info['bound']))
        out['H'] = np.empty(len(self.eq_con_info['equals']))

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
                    out['G'][indices] = con_vals[name].flatten() - bound
                else:
                    out['G'][indices] = bound - con_vals[name].flatten()
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
        x : np.ndarray or dict
            Current design variable values. A dict keyed by ``'{om_name}__{i}'``
            when using ``MixedVariableGA`` (including continuous-only problems),
            or a flat array for all other optimizers.
        """
        if isinstance(x, dict):
            for name, indices in zip(self.x_info['vars'], self.x_info['indices']):
                vals = np.array([x[f'{name}__{i}'] for i in range(len(indices))])
                self.driver.set_design_var(name, vals)
        else:
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

    Discrete (integer) and mixed integer design variables are supported via the
    ``MixedVariableGA`` optimizer, which uses pymoo's mixed-variable-aware sampling
    and mating operators. All other optimizers require continuous design variables only.

    Population-level MPI parallelism is enabled by setting ``run_parallel=True``.
    Ranks are divided into groups of ``procs_per_model`` (default 1). Each group
    cooperates on one model evaluation, enabling models with parallel components
    (e.g. ``ParallelGroup``) to each receive their own sub-communicator. With 8
    ranks and ``procs_per_model=2``, 4 individuals are evaluated simultaneously.

    pymooDriver supports the following:
        equality_constraints (algorithm-dependent)
        inequality_constraints (algorithm-dependent)
        two_sided_constraints (algorithm-dependent)
        linear_constraints (algorithm-dependent)
        multiple_objectives (algorithm-dependent)
        integer_design_vars (MixedVariableGA only)

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
    _problem_comm : MPI.Comm or None
        The full problem-level communicator across all ranks. Stored in
        ``_setup_comm`` before ``Problem.setup()`` runs. Used by the MPI runner
        to coordinate population distribution across all ranks.
    """

    def __init__(self, **kwargs):
        """
        Initialize the pymooDriver.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        if pymoo is None:
            raise RuntimeError('pymooDriver is not available, pymoo is not'
                               ' installed.')

        if isinstance(pymoo, Exception):
            # there is some other issue with the pymoo installation
            raise pymoo

        super().__init__(**kwargs)

        # What we support
        self.supports['optimization'] = True
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['two_sided_constraints'] = True
        self.supports['linear_constraints'] = True
        self.supports['linear_only_designvars'] = True
        self.supports['multiple_objectives'] = True
        self.supports['integer_design_vars'] = True

        # What we don't support
        self.supports['active_set'] = False
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

        # Full communicator across all ranks, stored in _setup_comm before
        # Problem.setup() runs. Used by the MPI runner to coordinate population
        # distribution.
        self._problem_comm = None

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
        self.options.declare('run_parallel', types=bool, default=False,
                             desc='Set to True to execute the points in a generation in parallel.')
        self.options.declare('procs_per_model', default=1, lower=1,
                             desc='Number of processors to give each model under MPI.')

    def _setup_comm(self, comm):
        """
        Split the communicator into model sub-communicators for parallel evaluation.

        OpenMDAO calls this method during ``Problem.setup()`` **before** the model
        is set up. By returning a sub-communicator here, we ensure the model —
        including any ``ParallelGroup`` components — is initialized with the
        correct sub-communicator rather than the full one.

        **How the split works**

        With ``N`` total ranks and ``procs_per_model=P``, there are ``N/P`` groups.
        Each group gets its own sub-communicator by splitting on a *color* value::

            n_groups = N // P
            color    = rank % n_groups

        For example, with 8 ranks and ``procs_per_model=2`` (n_groups=4):

        - color 0 → ranks 0, 4  → model sub-comm for group 0
        - color 1 → ranks 1, 5  → model sub-comm for group 1
        - color 2 → ranks 2, 6  → model sub-comm for group 2
        - color 3 → ranks 3, 7  → model sub-comm for group 3

        The returned sub-communicator is what the model uses internally. The full
        communicator is stored as ``_problem_comm`` for use by the runner when
        distributing the population.

        Parameters
        ----------
        comm : MPI.Comm or None
            The full communicator for the Problem.

        Returns
        -------
        MPI.Comm or None
            The sub-communicator for the model on this rank. When not running in
            parallel, returns ``comm`` unchanged.
        """
        self._problem_comm = comm

        if not MPI:
            if self.options['run_parallel']:
                raise RuntimeError(
                    f'{self.msginfo}: run_parallel=True requires MPI but MPI is not available.'
                )
            if self.options['procs_per_model'] != 1:
                raise RuntimeError(
                    f'{self.msginfo}: procs_per_model != 1 requires MPI but MPI is not available.'
                )

        if MPI and self.options['run_parallel']:
            procs_per_model = self.options['procs_per_model']
            full_size = comm.size
            n_groups = full_size // procs_per_model

            if full_size != n_groups * procs_per_model:
                raise RuntimeError(
                    f'{self.msginfo}: Total number of processors ({full_size}) is not '
                    f'evenly divisible by procs_per_model ({procs_per_model}). '
                    f'Provide a number of processors that is a multiple of '
                    f'{procs_per_model}.'
                )

            color = comm.rank % n_groups
            model_comm = comm.Split(color)

            return model_comm

        return comm

    def _setup_recording(self):
        """
        Set up case recording, restricting which ranks write records under MPI.

        When running in parallel, only the root rank of each model group records
        to avoid duplicate case entries. When not running in parallel, only rank 0
        records.
        """
        if MPI:
            run_parallel = self.options['run_parallel']
            procs_per_model = self.options['procs_per_model']

            for recorder in self._rec_mgr:
                if run_parallel:
                    if procs_per_model == 1:
                        recorder.record_on_process = True
                    else:
                        n_groups = self._problem_comm.size // procs_per_model
                        if self._problem_comm.rank < n_groups:
                            recorder.record_on_process = True

                elif self._problem_comm.rank == 0:
                    recorder.record_on_process = True

        super()._setup_recording()

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
        self.supports['integer_design_vars'] = opt in _mixed_var_optimizers
        self.supports._read_only = True

        # Validate problem formulation
        if not self.supports['multiple_objectives'] and len(self._objs) > 1:
            msg = 'The {} algorithm in {} currently does not support multiple objectives.'
            raise RuntimeError(msg.format(opt, self.msginfo))

        if self._designvars_discrete and opt not in _mixed_var_optimizers:
            raise RuntimeError(
                f'{self.msginfo}: Optimizer {opt!r} does not support discrete design '
                f'variables. Use MixedVariableGA for mixed integer problems.'
            )

        self._model_ran = False
        self.alg_class = self.get_algorithm(opt)

    def run(self):
        """
        Optimize the problem using the selected pymoo optimizer.

        Returns
        -------
        bool
            Failure flag; True if the optimization failed to find a feasible
            solution, False if successful.
        """
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
        desvar_vals = self.get_design_var_values()

        # Determine total number of design variables for error and x_info initialization
        # For descrete variables, the size key will show as zero so need to check manually
        ndesvar = 0
        for name, meta in self._designvars.items():
            if name in self._designvars_discrete:
                val = desvar_vals[name]
                ndesvar += 1 if np.ndim(val) == 0 else len(val)
            else:
                ndesvar += meta['global_size'] if meta['distributed'] else meta['size']

        if ndesvar == 0:
            raise RuntimeError('Problem has no design variables.')

        # Collect design variable information (initial values and bounds)
        x_info = {'vars': [], 'upper': np.full(ndesvar, 1e30),
                  'lower': np.full(ndesvar, -1e30), 'indices': []}
        current_idx = 0
        for name, meta in self._designvars.items():
            x_info['vars'].append(name)

            if name in self._designvars_discrete:
                val = desvar_vals[name]
                size = 1 if np.ndim(val) == 0 else len(val)
            else:
                size = meta['global_size'] if meta['distributed'] else meta['size']
            current_indices = list(range(current_idx, current_idx + size))
            x_info['indices'].append(current_indices)
            x_info['lower'][current_indices] = meta['lower']
            x_info['upper'][current_indices] = meta['upper']
            current_idx += size

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

        # _problem_comm is the full communicator; the model may be running on a
        # sub-communicator if procs_per_model > 1.
        if MPI and self.options['run_parallel']:
            runner = MPIElementwiseRunner(self._problem_comm,
                                          self.options['procs_per_model'])
        else:
            runner = None  # use pymoo's default LoopedElementwiseEvaluation

        # Run optimization
        try:
            # Build pymoo problem wrapper
            self._moo_prob = pymooProblem(
                driver=self,
                x_info=x_info,
                ieq_con_info=ieq_con_info,
                eq_con_info=eq_con_info,
                obj_info=obj_info,
                runner=runner,
            )

            # Build algorithm settings. For CMAES, redirect its output files
            # (written by the underlying cma package) to the problem's outputs
            # directory unless the user has already specified a custom prefix
            # via alg_settings. To disable file output entirely, set
            # alg_settings['verb_log'] = 0.
            alg_settings = dict(self.alg_settings)
            if opt == 'CMAES' and 'verb_filenameprefix' not in alg_settings:
                outputs_dir = prob.get_outputs_dir(mkdir=True)
                alg_settings['verb_filenameprefix'] = str(outputs_dir / 'cmaes_')

            # Instantiate and run optimizer
            optimizer = self.alg_class(**alg_settings)

            # Make sure only the main rank prints from pymoo
            run_settings = {**self.run_settings}
            if prob.comm.rank == 0:
                if "verbose" not in self.run_settings:
                    run_settings['verbose'] = self.options['disp']
            else:
                run_settings['verbose'] = False

            self.pymoo_results = minimize(
                problem=self._moo_prob,
                algorithm=optimizer,
                **run_settings,
            )

            # Extract optimal design variables and success flag from optimizer result
            # Pymoo sets result.X to None when no feasible solution is found.
            # Feasibility is determined per-individual using cv_eps=0.0 (exact),
            # though equality constraint violations up to 1e-4 are already absorbed
            # into the CV calculation by pymoo's default cv_eq config.
            x_opt = self.pymoo_results.X
            F_opt = self.pymoo_results.F
            self.fail = x_opt is None
            if opt in _single_obj_optimizers:
                if x_opt is not None:
                    # Update OpenMDAO design variables with optimal values
                    if isinstance(x_opt, dict):
                        for name, indices in zip(x_info['vars'], x_info['indices']):
                            vals = np.array([x_opt[f'{name}__{i}'] for i in range(len(indices))])
                            self.set_design_var(name, vals)
                    else:
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

        return self.fail

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
        if alg_name in _mixed_var_optimizers:
            module = importlib.import_module('pymoo.core.mixed')
            return getattr(module, alg_name)

        if alg_name in _single_obj_optimizers:
            base = 'pymoo.algorithms.soo.nonconvex'
        else:
            base = 'pymoo.algorithms.moo'

        # The script where the algorithm classes are located are all just a
        # lowercase of the algorithm name, except for the specifically called
        # out algorithms
        non_default_mapping = {
            'NelderMead': 'nelder',
            'PatternSearch': 'pattern',
            'AGEMOEA': 'age',
            'AGEMOEA2': 'age2',
            'SMSEMOA': 'sms',
            'NicheGA': 'ga_niching',
            'EPPSO': 'pso_ep',
            'RandomSearch': 'random_search',
        }
        module_name = non_default_mapping.get(alg_name, alg_name.lower())
        module = importlib.import_module(f'{base}.{module_name}')
        return getattr(module, alg_name)
