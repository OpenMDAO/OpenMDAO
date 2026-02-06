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
from importlib.util import find_spec
import os
import copy

import numpy as np

from openmdao.core.driver import RecordingDebugging
from openmdao.drivers.genetic_algorithm_driver_base import GeneticAlgorithmDriverBase
from openmdao.utils.concurrent_utils import concurrent_eval


class DifferentialEvolutionDriver(GeneticAlgorithmDriverBase):
    """
    Driver for a differential evolution genetic algorithm.

    This algorithm requires that inputs are floating point numbers.

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
    _desvar_idx : dict
        Keeps track of the indices for each desvar, since DifferentialEvolution sees an array of
        design variables.
    _ga : <DifferentialEvolution>
        Main genetic algorithm lies here.
    _nfit : int
         Number of successful function evaluations.
    _randomstate : int
        Seed-number which controls the random draws.
    """

    def __init__(self, **kwargs):
        """
        Initialize the DifferentialEvolutionDriver driver.
        """
        if find_spec('pyDOE3') is None:
            raise RuntimeError(f"{self.__class__.__name__} requires the 'pyDOE3' package, "
                               "which can be installed with one of the following commands:\n"
                               "    pip install openmdao[doe]\n"
                               "    pip install pyDOE3")

        super().__init__(**kwargs)

        # What we support
        self.supports['optimization'] = True
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
        self._nfit = 0

        # random state can be set for predictability during testing
        if 'DifferentialEvolutionDriver_seed' in os.environ:
            self._randomstate = int(os.environ['DifferentialEvolutionDriver_seed'])
        else:
            self._randomstate = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('max_gen', default=100,
                             desc='Number of generations before termination.')
        self.options.declare('pop_size', default=0,
                             desc='Number of points in the GA. Set to 0 and it will be computed '
                             'as 20 times the total number of inputs.')
        self.options.declare('Pc', default=0.9, lower=0., upper=1.,
                             desc='Crossover probability.')
        self.options.declare('F', default=0.9, lower=0., upper=1., allow_none=True,
                             desc='Differential rate.')

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

        # check design vars and constraints for invalid bounds
        self._validate_finite_bounds()

        model_mpi = None
        comm = problem.comm
        if self._concurrent_pop_size > 0:
            model_mpi = (self._concurrent_pop_size, self._concurrent_color)
        elif not self.options['run_parallel']:
            comm = None

        self._ga = DifferentialEvolution(self.objective_callback, comm=comm, model_mpi=model_mpi)

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
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        self.result.reset()
        ga = self._ga

        pop_size = self.options['pop_size']
        max_gen = self.options['max_gen']
        F = self.options['F']
        Pc = self.options['Pc']

        self._check_for_missing_objective()
        self._check_for_invalid_desvar_values()

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

        # Automatic population size.
        if pop_size == 0:
            pop_size = 20 * count

        desvar_new, obj, self._nfit = ga.execute_ga(x0, lower_bound, upper_bound,
                                                    pop_size, max_gen,
                                                    self._randomstate, F, Pc)

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        for name in desvars:
            i, j = self._desvar_idx[name]
            val = desvar_new[i:j]
            self.set_design_var(name, val)

        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            self._run_solve_nonlinear()
            rec.abs = 0.0
            rec.rel = 0.0
        self.iter_count += 1

        return False



class DifferentialEvolution(object):
    """
    Differential Evolution Genetic Algorithm.

    TODO : add better references than: https://en.wikipedia.org/wiki/Differential_evolution

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
    _lhs : function
        The lazily-imported pyDOE3 latin hypercube sampling function.
    """

    def __init__(self, objfun, comm=None, model_mpi=None):
        """
        Initialize genetic algorithm object.
        """
        try:
            from pyDOE3 import lhs
            self._lhs = lhs
        except ImportError:
            raise RuntimeError(f"{self.__class__.__name__} requires the 'pyDOE3' package, "
                               "which can be installed with one of the following commands:\n"
                               "    pip install openmdao[doe]\n"
                               "    pip install pyDOE3")

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
            Initial design values.
        vlb : ndarray
            Lower bounds array.
        vub : ndarray
            Upper bounds array.
        pop_size : int
            Number of points in the population.
        max_gen : int
            Number of generations to run the GA.
        random_state : int
            Seed-number which controls the random draws.
        F : float
            Differential rate.
        Pc : float
            Crossover rate.

        Returns
        -------
        ndarray
            Best design point.
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

        if self.comm is not None and self.comm.size > 1:
            if random_state is None:
                # if no random_state is given, generate one on rank 0 and broadcast it to all
                # ranks.  Because we add the rank to the starting random state, no ranks will
                # have the same seed.
                rng = np.random.default_rng()
                random_state = rng.integers(2**31)  # Must be less than 2^32-1
                if self.comm.rank == 0:
                    self.comm.bcast(random_state, root=0)
                else:
                    random_state = self.comm.bcast(None, root=0)

            # add rank to ensure different seed in each MPI process
            seed = random_state + self.comm.rank
        else:
            seed = random_state

        rng = np.random.default_rng(seed)

        # create LHS initial population (scaled to bounds) + user initial condition
        population = self._lhs(self.lchrom, self.npop - 1, criterion='center', random_state=seed)
        population = population * (vub - vlb) + vlb  # scale to bounds
        population = np.vstack((population, x0))
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
