"""
Driver for a differential evolution genetic algorithm.

TODO: add better references than: https://en.wikipedia.org/wiki/Differential_evolution

Most of this driver (except execute_ga) is based on SimpleGA, so the following may still apply:

The following reference is only for the penalty function:
Smith, A. E., Coit, D. W. (1995) Penalty functions. In: Handbook of Evolutionary Computation, 97(1).

The following reference is only for weighted sum multi-objective optimization:
Sobieszczanski-Sobieski, J., Morris, A. J., van Tooren, M. J. L. (2015)
Multidisciplinary Design Optimization Supported by Knowledge Based Engineering.
John Wiley & Sons, Ltd.
"""
import os
import copy

import numpy as np

import openmdao
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.concurrent import concurrent_eval
from openmdao.utils.mpi import MPI
from openmdao.core.analysis_error import AnalysisError


class DifferentialEvolutionDriver(Driver):
    """
    Driver for a differential evolution genetic algorithm.

    This algorithm requires that inputs are floating point numbers.

    Attributes
    ----------
    _concurrent_pop_size : int
        Number of points to run concurrently when model is a parallel one.
    _concurrent_color : int
        Color of current rank when running a parallel model.
    _desvar_idx : dict
        Keeps track of the indices for each desvar, since DifferentialEvolution sees an array of
        design variables.
    _ga : <DifferentialEvolution>
        Main genetic algorithm lies here.
    _randomstate : np.random.RandomState, int
         Random state (or seed-number) which controls the seed and random draws.
    """

    def __init__(self, **kwargs):
        """
        Initialize the DifferentialEvolutionDriver driver.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        super().__init__(**kwargs)

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['multiple_objectives'] = True

        # What we don't support yet
        self.supports['integer_design_vars'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['linear_constraints'] = False
        self.supports['simultaneous_derivatives'] = False
        self.supports['active_set'] = False
        self.supports['distributed_design_vars'] = False
        self.supports._read_only = True

        self._desvar_idx = {}
        self._ga = None

        # random state can be set for predictability during testing
        if 'DifferentialEvolutionDriver_seed' in os.environ:
            self._randomstate = int(os.environ['DifferentialEvolutionDriver_seed'])
        else:
            self._randomstate = None

        # Support for Parallel models.
        self._concurrent_pop_size = 0
        self._concurrent_color = 0

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('max_gen', default=100,
                             desc='Number of generations before termination.')
        self.options.declare('pop_size', default=0,
                             desc='Number of points in the GA. Set to 0 and it will be computed '
                             'as 20 times the total number of inputs.')
        self.options.declare('run_parallel', types=bool, default=False,
                             desc='Set to True to execute the points in a generation in parallel.')
        self.options.declare('procs_per_model', default=1, lower=1,
                             desc='Number of processors to give each model under MPI.')
        self.options.declare('penalty_parameter', default=10., lower=0.,
                             desc='Penalty function parameter.')
        self.options.declare('penalty_exponent', default=1.,
                             desc='Penalty function exponent.')
        self.options.declare('Pc', default=0.9, lower=0., upper=1.,
                             desc='Crossover probability.')
        self.options.declare('F', default=0.9, lower=0., upper=1., allow_none=True,
                             desc='Differential rate.')
        self.options.declare('multi_obj_weights', default={}, types=(dict),
                             desc='Weights of objectives for multi-objective optimization.'
                             'Weights are specified as a dictionary with the absolute names'
                             'of the objectives. The same weights for all objectives are assumed, '
                             'if not given.')
        self.options.declare('multi_obj_exponent', default=1., lower=0.,
                             desc='Multi-objective weighting exponent.')

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super()._setup_driver(problem)

        model_mpi = None
        comm = problem.comm
        if self._concurrent_pop_size > 0:
            model_mpi = (self._concurrent_pop_size, self._concurrent_color)
        elif not self.options['run_parallel']:
            comm = None

        self._ga = DifferentialEvolution(self.objective_callback, comm=comm, model_mpi=model_mpi)

    def _setup_comm(self, comm):
        """
        Perform any driver-specific setup of communicators for the model.

        Here, we generate the model communicators.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The communicator for the Problem.

        Returns
        -------
        MPI.Comm or <FakeComm> or None
            The communicator for the Problem model.
        """
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

    def _get_name(self):
        """
        Get name of current Driver.

        Returns
        -------
        str
            Name of current Driver.
        """
        return "DifferentialEvolution"

    def run(self):
        """
        Execute the genetic algorithm.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        model = self._problem().model
        ga = self._ga

        pop_size = self.options['pop_size']
        max_gen = self.options['max_gen']
        F = self.options['F']
        Pc = self.options['Pc']

        self._check_for_missing_objective()

        # Size design variables.
        desvars = self._designvars
        desvar_vals = self.get_design_var_values()

        count = 0
        for name, meta in desvars.items():
            size = meta['size']
            self._desvar_idx[name] = (count, count + size)
            count += size

        lower_bound = np.empty((count, ))
        upper_bound = np.empty((count, ))
        x0 = np.empty(count)

        # Figure out bounds vectors and initial design vars
        for name, meta in desvars.items():
            i, j = self._desvar_idx[name]
            lower_bound[i:j] = meta['lower']
            upper_bound[i:j] = meta['upper']
            x0[i:j] = desvar_vals[name]

        abs2prom = model._var_abs2prom['output']

        # Automatic population size.
        if pop_size == 0:
            pop_size = 20 * count

        desvar_new, obj, nfit = ga.execute_ga(x0, lower_bound, upper_bound,
                                              pop_size, max_gen,
                                              self._randomstate, F, Pc)

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        for name in desvars:
            i, j = self._desvar_idx[name]
            val = desvar_new[i:j]
            self.set_design_var(name, val)

        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            model.run_solve_nonlinear()
            rec.abs = 0.0
            rec.rel = 0.0
        self.iter_count += 1

        return False

    def objective_callback(self, x, icase):
        r"""
        Evaluate problem objective at the requested point.

        In case of multi-objective optimization, a simple weighted sum method is used:

        .. math::

           f = (\sum_{k=1}^{N_f} w_k \cdot f_k)^a

        where :math:`N_f` is the number of objectives and :math:`a>0` is an exponential
        weight. Choosing :math:`a=1` is equivalent to the conventional weighted sum method.

        The weights given in the options are normalized, so:

        .. math::

            \sum_{k=1}^{N_f} w_k = 1

        If one of the objectives :math:`f_k` is not a scalar, its elements will have the same
        weights, and it will be normed with length of the vector.

        Takes into account constraints with a penalty function.

        All constraints are converted to the form of :math:`g_i(x) \leq 0` for
        inequality constraints and :math:`h_i(x) = 0` for equality constraints.
        The constraint vector for inequality constraints is the following:

        .. math::

           g = [g_1, g_2  \dots g_N], g_i \in R^{N_{g_i}}

           h = [h_1, h_2  \dots h_N], h_i \in R^{N_{h_i}}

        The number of all constraints:

        .. math::

           N_g = \sum_{i=1}^N N_{g_i},  N_h = \sum_{i=1}^N N_{h_i}

        The fitness function is constructed with the penalty parameter :math:`p`
        and the exponent :math:`\kappa`:

        .. math::

           \Phi(x) = f(x) + p \cdot \sum_{k=1}^{N^g}(\delta_k \cdot g_k)^{\kappa}
           + p \cdot \sum_{k=1}^{N^h}|h_k|^{\kappa}

        where :math:`\delta_k = 0` if :math:`g_k` is satisfied, 1 otherwise

        .. note::

            The values of :math:`\kappa` and :math:`p` can be defined as driver options.

        Parameters
        ----------
        x : ndarray
            Value of design variables.
        icase : int
            Case number, used for identification when run in parallel.

        Returns
        -------
        float
            Objective value
        bool
            Success flag, True if successful
        int
            Case number, used for identification when run in parallel.
        """
        model = self._problem().model
        success = 1

        objs = self.get_objective_values()
        nr_objectives = len(objs)

        # Single objective, if there is only one objective, which has only one element
        if nr_objectives > 1:
            is_single_objective = False
        else:
            for obj in objs.items():
                is_single_objective = len(obj) == 1
                break

        obj_exponent = self.options['multi_obj_exponent']
        if self.options['multi_obj_weights']:  # not empty
            obj_weights = self.options['multi_obj_weights']
        else:
            # Same weight for all objectives, if not specified
            obj_weights = {name: 1. for name in objs.keys()}
        sum_weights = sum(obj_weights.values())

        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

        # a very large number, but smaller than the result of nan_to_num in Numpy
        almost_inf = openmdao.INF_BOUND

        # Execute the model
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            self.iter_count += 1
            try:
                model.run_solve_nonlinear()

            # Tell the optimizer that this is a bad point.
            except AnalysisError:
                model._clear_iprint()
                success = 0

            obj_values = self.get_objective_values()
            if is_single_objective:  # Single objective optimization
                for i in obj_values.values():
                    obj = i  # First and only key in the dict
            else:  # Multi-objective optimization with weighted sums
                weighted_objectives = np.array([])
                for name, val in obj_values.items():
                    # element-wise multiplication with scalar
                    # takes the average, if an objective is a vector
                    try:
                        weighted_obj = val * obj_weights[name] / val.size
                    except KeyError:
                        msg = ('Name "{}" in "multi_obj_weights" option '
                               'is not an absolute name of an objective.')
                        raise KeyError(msg.format(name))
                    weighted_objectives = np.hstack((weighted_objectives, weighted_obj))

                obj = sum(weighted_objectives / sum_weights)**obj_exponent

            # Parameters of the penalty method
            penalty = self.options['penalty_parameter']
            exponent = self.options['penalty_exponent']

            if penalty == 0:
                fun = obj
            else:
                constraint_violations = np.array([])
                for name, val in self.get_constraint_values().items():
                    con = self._cons[name]
                    # The not used fields will either None or a very large number
                    if (con['lower'] is not None) and np.any(con['lower'] > -almost_inf):
                        diff = val - con['lower']
                        violation = np.array([0. if d >= 0 else abs(d) for d in diff])
                    elif (con['upper'] is not None) and np.any(con['upper'] < almost_inf):
                        diff = val - con['upper']
                        violation = np.array([0. if d <= 0 else abs(d) for d in diff])
                    elif (con['equals'] is not None) and np.any(np.abs(con['equals']) < almost_inf):
                        diff = val - con['equals']
                        violation = np.absolute(diff)
                    constraint_violations = np.hstack((constraint_violations, violation))
                fun = obj + penalty * sum(np.power(constraint_violations, exponent))
            # Record after getting obj to assure they have
            # been gathered in MPI.
            rec.abs = 0.0
            rec.rel = 0.0

        return fun, success, icase


class DifferentialEvolution(object):
    """
    Differential Evolution Genetic Algorithm.

    TODO : add better references than: https://en.wikipedia.org/wiki/Differential_evolution

    Attributes
    ----------
    comm : MPI communicator or None
        The MPI communicator that will be used objective evaluation for each generation.
    lchrom : int
        Chromosome length.
    model_mpi : None or tuple
        If the model in objfun is also parallel, then this will contain a tuple with the the
        total number of population points to evaluate concurrently, and the color of the point
        to evaluate on this rank.
    npop : int
        Population size.
    objfun : function
        Objective function callback.
    """

    def __init__(self, objfun, comm=None, model_mpi=None):
        """
        Initialize genetic algorithm object.

        Parameters
        ----------
        objfun : function
            Objective callback function.
        comm : MPI communicator or None
            The MPI communicator that will be used objective evaluation for each generation.
        model_mpi : None or tuple
            If the model in objfun is also parallel, then this will contain a tuple with the the
            total number of population points to evaluate concurrently, and the color of the point
            to evaluate on this rank.
        """
        self.objfun = objfun
        self.comm = comm

        self.lchrom = 0
        self.npop = 0
        self.model_mpi = model_mpi

    def execute_ga(self, x0, vlb, vub, pop_size, max_gen, random_state, F=0.5, Pc=0.5):
        """
        Perform the genetic algorithm.

        Parameters
        ----------
        x0 : ndarray
            Initial design values
        vlb : ndarray
            Lower bounds array.
        vub : ndarray
            Upper bounds array.
        pop_size : int
            Number of points in the population.
        max_gen : int
            Number of generations to run the GA.
        random_state : np.random.RandomState, int
            Random state (or seed-number) which controls the seed and random draws.
        F : float
            Differential rate
        Pc : float
            Crossover rate

        Returns
        -------
        ndarray
            Best design point
        float
            Objective value at best design point.
        int
            Number of successful function evaluations.
        """
        comm = self.comm
        xopt = copy.deepcopy(vlb)
        fopt = np.inf
        self.lchrom = len(x0)

        if np.mod(pop_size, 2) == 1:
            pop_size += 1
        self.npop = int(pop_size)

        # use different seeds in different MPI processes
        seed = random_state + self.comm.Get_rank() if self.comm else 0
        rng = np.random.default_rng(seed)

        # create random initial population, scaled to bounds
        population = rng.random([self.npop, self.lchrom]) * (vub - vlb) + vlb  # scale to bounds
        fitness = np.ones(self.npop) * np.inf  # initialize fitness to infinitely bad

        # Main Loop
        nfit = 0
        for generation in range(max_gen + 1):
            # Evaluate fitness of points in this generation
            if comm is not None:  # Parallel
                # Since GA is random, ranks generate different new populations, so just take one
                # and use it on all.
                population = comm.bcast(population, root=0)

                cases = [((item, ii), None) for ii, item in enumerate(population)]

                # Pad the cases with some dummy cases to make the cases divisible amongst the procs.
                # TODO: Add a load balancing option to this driver.
                extra = len(cases) % comm.size
                if extra > 0:
                    for j in range(comm.size - extra):
                        cases.append(cases[-1])

                results = concurrent_eval(self.objfun, cases, comm,
                                          allgather=True, model_mpi=self.model_mpi)

                fitness[:] = np.inf
                for result in results:
                    returns, traceback = result

                    if returns:
                        val, success, ii = returns
                        if success:
                            fitness[ii] = val
                            nfit += 1
                    else:
                        # Print the traceback if it fails
                        print('A case failed:')
                        print(traceback)
            else:  # Serial
                for ii in range(self.npop):
                    fitness[ii], success, _ = self.objfun(population[ii], 0)
                    nfit += 1

            # Find best performing point in this generation.
            min_fit = np.min(fitness)
            min_index = np.argmin(fitness)
            min_gen = population[min_index]

            # Update overall best.
            if min_fit < fopt:
                fopt = min_fit
                xopt = min_gen

            if generation == max_gen:  # finished
                break

            # Selection: new generation members replace parents, if better (implied elitism)
            if generation == 0:
                parentPop = copy.deepcopy(population)
                parentFitness = copy.deepcopy(fitness)
            else:
                for ii in range(self.npop):
                    if fitness[ii] < parentFitness[ii]:  # if child is better, else parent unchanged
                        parentPop[ii] = population[ii]
                        parentFitness[ii] = fitness[ii]

            # Evolve new generation.
            population = np.zeros(np.shape(parentPop))
            fitness = np.ones(self.npop) * np.inf

            for ii in range(self.npop):
                # randomly select 3 different population members other than the current choice
                a, b, c = ii, ii, ii
                while a == ii:
                    a = rng.integers(0, self.npop)
                while b == ii or b == a:
                    b = rng.integers(0, self.npop)
                while c == ii or c == a or c == b:
                    c = rng.integers(0, self.npop)

                # randomly select chromosome index for forced crossover
                r = rng.integers(0, self.lchrom)

                # crossover and mutation
                population[ii] = parentPop[ii]  # start the same as parent
                # clip mutant so that it cannot be outside the bounds
                mutant = np.clip(parentPop[a] + F * (parentPop[b] - parentPop[c]), vlb, vub)
                # sometimes replace parent's feature with mutant's
                rr = rng.random(self.lchrom)
                idx = np.where(rr < Pc)
                population[ii][idx] = mutant[idx]
                population[ii][r] = mutant[r]  # always replace at least one with mutant's

        return xopt, fopt, nfit
