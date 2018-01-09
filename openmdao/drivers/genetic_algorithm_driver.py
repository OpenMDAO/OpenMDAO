"""
Driver for a simple genetic algorithm.

This is the Simple Genetic Algorithm implementation based on 2009 AAE550: MDO Lecture notes of
Prof. William A. Crossley.

This basic GA algorithm is compartmentalized into the GeneticAlgorithm class so that it can be
used in more complicated driver.
"""
import copy

from six import iteritems
from six.moves import range

import numpy as np
from pyDOE import lhs

from openmdao.core.driver import Driver
from openmdao.recorders.recording_iteration_stack import Recording


class SimpleGADriver(Driver):
    """
    Driver for a simple genetic algorithm.

    Options
    -------
    options['elitism'] :  bool(True)
        If True, replace worst performing point with best from previous generation each iteration.
    options['max_gen'] :  int(300)
        Number of generations before termination.
    options['pop_size'] :  int(25)
        Number of points in the GA.

    Attributes
    ----------
    problem : <Problem>
        Pointer to the containing problem.
    supports : <OptionsDictionary>
        Provides a consistant way for drivers to declare what features they support.
    _cons : dict
        Contains all constraint info.
    _designvars : dict
        Contains all design variable info.
    _desvar_idx : dict
        Keeps track of the indices for each desvar, since GeneticAlgorithm seess an array of
        design variables.
    _ga : <GeneticAlgorithm>
        Main genetic algorithm lies here.
    _objs : dict
        Contains all objective info.
    _quantities : list
        Contains the objectives plus nonlinear constraints.
    _responses : dict
        Contains all response info.
    """

    def __init__(self):
        """
        Initialize the SimpleGADriver driver.
        """
        super(SimpleGADriver, self).__init__()

        # What we support
        self.supports['integer_design_vars'] = True

        # What we don't support yet
        self.supports['inequality_constraints'] = False
        self.supports['equality_constraints'] = False
        self.supports['multiple_objectives'] = False
        self.supports['two_sided_constraints'] = False
        self.supports['linear_constraints'] = False
        self.supports['simultaneous_derivatives'] = False
        self.supports['active_set'] = False

        # User Options
        self.options.declare('bits', default={}, types=(dict),
                             desc='Number of bits of resolution. Default is an empty dict, where '
                             'every unspecified variable is assumed to be integer, and the number '
                             'of bits is calculated automatically. If you have a continuous var, '
                             'you should set a bits value as a key in this dictionary.')
        self.options.declare('elitism', default=True,
                             desc='If True, replace worst performing point with best from previous'
                             ' generation each iteration.')
        self.options.declare('max_gen', default=300,
                             desc='Number of generations before termination.')
        self.options.declare('pop_size', default=25,
                             desc='Number of points in the GA.')

        self._ga = GeneticAlgorithm(self.objective_callback)

        self._desvar_idx = {}

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super(SimpleGADriver, self)._setup_driver(problem)

        if len(self._objs) > 1:
            msg = 'SimpleGADriver currently does not support multiple objectives.'
            raise RuntimeError(msg)

        if len(self._cons) > 0:
            msg = 'SimpleGADriver currently does not support constraints.'
            raise RuntimeError(msg)

    def run(self):
        """
        Excute the genetic algorithm.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        model = self._problem.model
        ga = self._ga

        # Size design variables.
        desvars = self._designvars
        count = 0
        for name, meta in iteritems(desvars):
            size = meta['size']
            self._desvar_idx[name] = (count, count+size)
            count += size

        lower_bound = np.empty((count, ))
        upper_bound = np.empty((count, ))

        # Figure out bounds vectors.
        for name, meta in iteritems(desvars):
            i, j = self._desvar_idx[name]
            lower_bound[i:j] = meta['lower']
            upper_bound[i:j] = meta['upper']

        ga.elite = self.options['elitism']
        pop_size = self.options['pop_size']
        max_gen = self.options['max_gen']
        user_bits = self.options['bits']

        # Bits of resolution
        bits = np.ceil(np.log2(upper_bound - lower_bound + 1)).astype(int)
        prom2abs = model._var_allprocs_prom2abs_list['output']

        for name, val in iteritems(user_bits):
            try:
                i, j = self._desvar_idx[name]
            except KeyError:
                abs_name = prom2abs[name][0]
                i, j = self._desvar_idx[abs_name]

            bits[i:j] = val

        desvar_new, obj, nfit = ga.execute_ga(lower_bound, upper_bound, bits, pop_size, max_gen)

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        for name in desvars:
            i, j = self._desvar_idx[name]
            val = desvar_new[i:j]
            self.set_design_var(name, val)

        with Recording('SimpleGA', self.iter_count, self) as rec:
            model._solve_nonlinear()
            rec.abs = 0.0
            rec.rel = 0.0
        self.iter_count += 1

    def objective_callback(self, x):
        """
        Evaluate problem objective at the requested point.

        Parameters
        ----------
        x : ndarray
            Value of design variables.

        Returns
        -------
        float
            Objective value
        bool
            Success flag, True if successful
        """
        model = self._problem.model
        success = 1

        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

        # Execute the model
        with Recording('SimpleGA', self.iter_count, self) as rec:
            self.iter_count += 1
            try:
                model._solve_nonlinear()

            # Tell the optimizer that this is a bad point.
            except AnalysisError:
                model._clear_iprint()
                success = 0

            for name, val in iteritems(self.get_objective_values()):
                obj = val
                break

            # Record after getting obj to assure they have
            # been gathered in MPI.
            rec.abs = 0.0
            rec.rel = 0.0

        # print("Functions calculated")
        # print(x)
        # print(obj)
        return obj, success


class GeneticAlgorithm():
    """
    Simple Genetic Algorithm.

    This is the Simple Genetic Algorithm implementation based on 2009 AAE550: MDO Lecture notes of
    Prof. William A. Crossley. It can be used standalone or as part of the OpenMDAO Driver.
    """

    def __init__(self, objfun, comm=None):
        """
        Initialize genetic algorithm object.

        Parameters
        ----------
        objfun : function
            Objective callback function.

        comm : MPI communicator or None
            The MPI communicator that will be used objective evaluation for each generation.
        """
        self.objfun = objfun
        self.comm = None

        self.lchrom = 0
        self.npop = 0
        self.elite = True

    def execute_ga(self, vlb, vub, bits, pop_size, max_gen):
        """
        Perform the genetic algorithm.

        Parameters
        ----------
        vlb : ndarray
            Lower bounds array.
        vub : ndarray
            Upper bounds array.
        bits : ndarray
            Number of bits to encode the design space for each element of the design vector.
        pop_size : int
            Number of points in the population.
        max_gen : int
            Number of generations to run the GA.

        Returns
        -------
        ndarray
            Best design point
        float
            Objective value at best design point.
        int
            Number of successful function evaluations.
        """
        xopt = copy.deepcopy(vlb)
        fopt = np.inf
        self.lchrom = int(np.sum(bits))

        if np.mod(pop_size, 2) == 1:
            pop_size += 1
        self.npop = int(pop_size)
        fitness = np.zeros((self.npop, ))

        Pc = 0.5
        Pm = (self.lchrom + 1.0) / (2.0 * pop_size * np.sum(bits))
        elite = self.elite

        # TODO: from an user-supplied intial population
        # new_gen, lchrom = encode(x0, vlb, vub, bits)
        new_gen = np.round(lhs(self.lchrom, self.npop, criterion='center'))

        # Main Loop
        nfit = 0
        for generation in range(max_gen + 1):
            old_gen = copy.deepcopy(new_gen)
            x_pop = self.decode(old_gen, vlb, vub, bits)

            # Evaluate points in this generation.
            for ii in range(self.npop):
                x = x_pop[ii]
                fitness[ii], success = self.objfun(x)
                if success:
                    nfit += 1
                else:
                    fitness[ii] = np.inf

            # Elitism means replace worst performing point with best from previous generation.
            if elite and generation > 0:
                max_index = np.argmax(fitness)
                old_gen[max_index] = min_gen
                x_pop[max_index] = min_x
                fitness[max_index] = min_fit

            # Find best performing point in this generation.
            min_fit = np.min(fitness)
            min_index = np.argmin(fitness)
            min_gen = old_gen[min_index]
            min_x = x_pop[min_index]

            if min_fit < fopt:
                fopt = min_fit
                xopt = min_x

            # Evolve new generation.
            new_gen = self.tournament(old_gen, fitness)
            new_gen = self.crossover(new_gen, Pc)
            new_gen = self.mutate(new_gen, Pm)

        return xopt, fopt, nfit

    def tournament(self, old_gen, fitness):
        """
        Apply tournament selection and keep the best points.

        Parameters
        ----------
        old_gen : ndarray
            Points in current generation

        fitness : ndarray
            Objective value of each point.

        Returns
        -------
        ndarray
            New generation with best points.
        """
        new_gen = []
        idx = np.array(range(0, self.npop - 1, 2))
        for j in range(2):
            old_gen, i_shuffled = self.shuffle(old_gen)
            fitness = fitness[i_shuffled]

            # Each point competes with its neighbor; save the best.
            i_min = np.argmin(np.array([[fitness[idx]], [fitness[idx + 1]]]), axis=0)
            selected = i_min + idx
            new_gen.append(old_gen[selected])

        return np.concatenate(np.array(new_gen), axis=1).reshape(old_gen.shape)

    def crossover(self, old_gen, Pc):
        """
        Apply crossover to the current generation.

        Crossover flips two adjacent genes.

        Parameters
        ----------
        old_gen : ndarray
            Points in current generation

        Pc : float
            Probability of crossover.

        Returns
        -------
        ndarray
            Current generation with crossovers applied.
        """
        new_gen = copy.deepcopy(old_gen)
        num_sites = self.npop // 2
        sites = np.random.rand(num_sites, self.lchrom)
        idx, idy = np.where(sites < Pc)
        for ii, jj in zip(idx, idy):
            i = 2 * ii
            j = i + 1
            new_gen[i][jj] = old_gen[j][jj]
            new_gen[j][jj] = old_gen[i][jj]
        return new_gen

    def mutate(self, current_gen, Pm):
        """
        Apply mutations to the current generation.

        A mutation flips the state of the gene from 0 to 1 or 1 to 0.

        Parameters
        ----------
        current_gen : ndarray
            Points in current generation

        Pm : float
            Probability of mutation.

        Returns
        -------
        ndarray
            Current generation with mutations applied.
        """
        temp = np.random.rand(self.npop, self.lchrom)
        idx, idy = np.where(temp < Pm)
        for ii, jj in zip(idx, idy):
            current_gen[ii][jj] = 1 - current_gen[ii][jj]
        return current_gen

    def shuffle(self, old_gen):
        """
        Shuffle (reorder) the points in the population.

        Used in tournament selection.

        Parameters
        ----------
        old_gen : ndarray
            Old population.

        Returns
        -------
        ndarray
            New shuffled population.
        ndarray(dtype=np.int)
            Index array that maps the shuffle from old to new.
        """
        temp = np.random.rand(self.npop)
        index = np.argsort(temp)
        return old_gen[index], index

    def decode(self, gen, vlb, vub, bits):
        """
        Decode from binary array to real value array.

        Parameters
        ----------
        gen : ndarray
            Population of points, encoded.
        vlb : ndarray
            Lower bound array.
        vub : ndarray
            Upper bound array.
        bits : ndarray
            Number of bits for decoding.

        Returns
        -------
        ndarray
            Decoded design variable values.
        """
        num_desvar = len(bits)
        interval = (vub - vlb) / (2**bits - 1)
        x = np.empty((self.npop, num_desvar))
        sbit = 0
        ebit = 0
        for jj in range(num_desvar):
            exponents = 2**np.array(range(bits[jj] - 1, -1, -1))
            ebit += bits[jj]
            fact = exponents * (gen[:, sbit:ebit])
            x[:, jj] = np.einsum('ij->i', fact) * interval[jj] + vlb[jj]
            sbit = ebit
        return x

    def encode(self, x, vlb, vub, bits):
        """
        Encode array of real values to array of binary arrays.

        Parameters
        ----------
        x : ndarray
            Design variable values.
        vlb : ndarray
            Lower bound array.
        vub : ndarray
            Upper bound array.
        bits : int
            Number of bits for decoding.

        Returns
        -------
        ndarray
            Population of points, encoded.
        """
        # TODO : We need this method if we ever start with user defined initial sampling points.
        pass

