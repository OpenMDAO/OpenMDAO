"""Base class for genetic algorithm drivers in OpenMDAO."""

import numpy as np

from openmdao.core.constants import INF_BOUND
from openmdao.core.optimization_driver_base import OptimizationDriverBase
from openmdao.utils.mpi import MPI


class GeneticAlgorithmDriverBase(OptimizationDriverBase):
    """
    Base class for genetic algorithm-based optimization drivers.

    This class provides common functionality for genetic algorithm drivers including
    MPI communicator setup, case recording configuration, and constraint penalty
    function handling. All genetic algorithm drivers (SimpleGADriver, DifferentialEvolutionDriver)
    inherit from this class.

    Genetic algorithm drivers that inherit from GeneticAlgorithmDriverBase can support
    parallel population evaluation through the 'run_parallel' and 'procs_per_model' options.

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.

    Attributes
    ----------
    _problem_comm : MPI.Comm or None
        The MPI communicator for the Problem.
    _concurrent_pop_size : int
        Number of points to run concurrently when model is a parallel one.
    _concurrent_color : int
        Color of current rank when running a parallel model.
    """

    def __init__(self, **kwargs):
        """
        Initialize the genetic algorithm driver.
        """
        super().__init__(**kwargs)

        # Support for Parallel models.
        self._concurrent_pop_size = 0
        self._concurrent_color = 0
        self._problem_comm = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('run_parallel', types=bool, default=False,
                             desc='Set to True to execute the points in a generation in parallel.')
        self.options.declare('procs_per_model', default=1, lower=1,
                             desc='Number of processors to give each model under MPI.')
        self.options.declare('penalty_parameter', default=10., lower=0.,
                             desc='Penalty function parameter.')
        self.options.declare('penalty_exponent', default=1.,
                             desc='Penalty function exponent.')
        self.options.declare('multi_obj_weights', default={}, types=(dict),
                             desc='Weights of objectives for multi-objective optimization. '
                             'Weights are specified as a dictionary with the absolute names '
                             'of the objectives. The same weights for all objectives are assumed, '
                             'if not given.')
        self.options.declare('multi_obj_exponent', default=1., lower=0.,
                             desc='Multi-objective weighting exponent.')

    def _setup_comm(self, comm):
        """
        Perform any driver-specific setup of communicators for the model.

        Here, we generate the model communicators for parallel population evaluation.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The communicator for the Problem.

        Returns
        -------
        MPI.Comm or <FakeComm> or None
            The communicator for the Problem model.
        """
        self._problem_comm = comm

        procs_per_model = self.options['procs_per_model']
        if MPI and self.options['run_parallel']:

            full_size = comm.size
            size = full_size // procs_per_model
            if full_size != size * procs_per_model:
                raise RuntimeError("The total number of processors is not evenly divisible by the "
                                   "specified number of processors per model.\n Provide a "
                                   "number of processors that is a multiple of %d, or "
                                   "specify a number of processors per model that divides "
                                   "into %d." % (procs_per_model, full_size))
            color = comm.rank % size
            model_comm = comm.Split(color)

            # Everything we need to figure out which case to run.
            self._concurrent_pop_size = size
            self._concurrent_color = color

            return model_comm

        self._concurrent_pop_size = 0
        self._concurrent_color = 0
        return comm

    def _setup_recording(self):
        """
        Set up case recording for parallel population evaluation.

        When run_parallel is True, recording is configured so that only
        the root processes for each concurrent model evaluation record cases.
        """
        if MPI:
            run_parallel = self.options['run_parallel']
            procs_per_model = self.options['procs_per_model']

            for recorder in self._rec_mgr:
                if run_parallel:
                    # write cases only on procs up to the number of parallel models
                    # (i.e. on the root procs for the cases)
                    if procs_per_model == 1:
                        recorder.record_on_process = True
                    else:
                        size = self._problem_comm.size // procs_per_model
                        if self._problem_comm.rank < size:
                            recorder.record_on_process = True

                elif self._problem_comm.rank == 0:
                    # if not running cases in parallel, then just record on proc 0
                    recorder.record_on_process = True

        super()._setup_recording()

    def _validate_finite_bounds(self):
        """
        Check that all design variables and constraints have finite bounds.

        Genetic algorithm drivers require finite bounds for all design variables
        and constraints. This method validates that no bounds are None or exceed
        INF_BOUND.

        Raises
        ------
        ValueError
            If any design variable or constraint has invalid bounds.
        """
        # check design vars for invalid bounds
        for name, meta in self._designvars.items():
            lower, upper = meta['lower'], meta['upper']
            for param in (lower, upper):
                if param is None or np.all(np.abs(param) >= INF_BOUND):
                    msg = (f"Invalid bounds for design variable '{name}'. When using "
                           f"{self.__class__.__name__}, values for both 'lower' and 'upper' "
                           f"must be specified between +/-INF_BOUND ({INF_BOUND}), "
                           f"but they are: lower={lower}, upper={upper}.")
                    raise ValueError(msg)

        # check constraints for invalid bounds
        for name, meta in self._cons.items():
            equals, lower, upper = meta['equals'], meta['lower'], meta['upper']
            if ((equals is None or np.all(np.abs(equals) >= INF_BOUND)) and
               (lower is None or np.all(np.abs(lower) >= INF_BOUND)) and
               (upper is None or np.all(np.abs(upper) >= INF_BOUND))):
                msg = (f"Invalid bounds for constraint '{name}'. "
                       f"When using {self.__class__.__name__}, the value for 'equals', "
                       f"'lower' or 'upper' must be specified between +/-INF_BOUND "
                       f"({INF_BOUND}), but they are: "
                       f"equals={equals}, lower={lower}, upper={upper}.")
                raise ValueError(msg)

    def _get_multi_objective_weights(self, objs):
        """
        Get normalized weights for multi-objective optimization.

        Parameters
        ----------
        objs : dict
            Dictionary of objective values from get_objective_values().

        Returns
        -------
        obj_weights : dict
            Dictionary mapping objective names to weights.
        sum_weights : float
            Sum of all weights (for normalization).
        obj_exponent : float
            Multi-objective exponent from options.
        """
        obj_exponent = self.options['multi_obj_exponent']
        if self.options['multi_obj_weights']:  # not empty
            obj_weights = self.options['multi_obj_weights']
        else:
            # Same weight for all objectives, if not specified
            obj_weights = {name: 1. for name in objs.keys()}
        sum_weights = sum(obj_weights.values())
        return obj_weights, sum_weights, obj_exponent

    def _compute_weighted_objective(self, obj_values, obj_weights, sum_weights, obj_exponent):
        """
        Compute weighted sum of multiple objectives.

        Implements the formula:
            f = (Σ(w_k * f_k / size_k) / Σ(w_k))^a

        where w_k are weights, f_k are objective values, size_k are sizes,
        and a is the exponent.

        Parameters
        ----------
        obj_values : dict
            Dictionary of objective values.
        obj_weights : dict
            Dictionary of objective weights.
        sum_weights : float
            Sum of all weights.
        obj_exponent : float
            Exponent for weighted sum.

        Returns
        -------
        float
            Weighted objective value.

        Raises
        ------
        KeyError
            If a weight key doesn't match an objective name.
        """
        weighted_objectives = np.array([])
        for name, val in obj_values.items():
            try:
                weighted_obj = val * obj_weights[name] / val.size
            except KeyError:
                msg = ('Name "{}" in "multi_obj_weights" option '
                       'is not an absolute name of an objective.')
                raise KeyError(msg.format(name))
            weighted_objectives = np.hstack((weighted_objectives, weighted_obj))

        return sum(weighted_objectives / sum_weights)**obj_exponent

    def _compute_constraint_violations(self, constraint_values):
        """
        Compute constraint violations for penalty function.

        For inequality constraints: violation = max(0, g) for upper bounds
                                                = max(0, -g) for lower bounds
        For equality constraints: violation = |h|

        Parameters
        ----------
        constraint_values : dict
            Dictionary of constraint values from get_constraint_values().

        Returns
        -------
        ndarray
            Array of all constraint violations.
        """
        almost_inf = INF_BOUND
        constraint_violations = np.array([])

        for name, val in constraint_values.items():
            con = self._cons[name]

            # Check which bounds are actually specified
            has_lower = (con['lower'] is not None) and np.any(con['lower'] > -almost_inf)
            has_upper = (con['upper'] is not None) and np.any(con['upper'] < almost_inf)
            has_equals = (con['equals'] is not None) and np.any(np.abs(con['equals']) < almost_inf)

            if has_lower:
                lb_diff = val - con['lower']
                lb_violation = np.array([0. if d >= 0 else abs(d) for d in lb_diff])

            if has_upper:
                ub_diff = val - con['upper']
                ub_violation = np.array([0. if d <= 0 else abs(d) for d in ub_diff])

            # Determine which violation to use
            if has_lower and not has_upper:
                violation = lb_violation
            elif has_upper and not has_lower:
                violation = ub_violation
            elif has_upper and has_lower:
                violation = np.maximum(lb_violation, ub_violation)
            elif has_equals and not (has_lower or has_upper):
                diff = val - con['equals']
                violation = np.absolute(diff)
            else:
                violation = np.array([0.])

            constraint_violations = np.hstack((constraint_violations, violation))

        return constraint_violations

    def _apply_penalty_to_objective(self, obj, constraint_violations):
        """
        Apply penalty function to objective for constraint violations.

        Implements: f_penalized = f + p * Σ(violation^κ)

        where p is the penalty parameter and κ is the penalty exponent.

        Parameters
        ----------
        obj : float or ndarray
            Objective value(s).
        constraint_violations : ndarray
            Array of constraint violations.

        Returns
        -------
        float or ndarray
            Penalized objective value.
        """
        penalty = self.options['penalty_parameter']
        exponent = self.options['penalty_exponent']

        if penalty == 0:
            return obj
        else:
            return obj + penalty * sum(np.power(constraint_violations, exponent))
