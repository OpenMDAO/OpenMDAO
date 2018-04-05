"""
Latin hypercube case generator for Design-of-Experiments Driver.
"""
from random import shuffle, randint, seed

from six import iteritems, itervalues
from six.moves import range, zip

import numpy as np

from six import iteritems, itervalues

from collections import OrderedDict

from openmdao.drivers.doe_driver import DOEGenerator


class LatinHypercubeGenerator(DOEGenerator):
    """
    DOE case generator implementing the Latin hypercube method.

    Attributes
    ----------
    _num_samples : int
        The number of samples to run.
    _seed : int or None
        Random seed.
    """

    def __init__(self, num_samples=1, seed=None):
        """
        Initialize the LatinHypercubeGenerator.

        Parameters
        ----------
        num_samples : int, optional
            The number of samples to run. Defaults to 1.

        seed : int or None, optional
            Random seed. Defaults to None.
        """
        super(LatinHypercubeGenerator, self).__init__()

        self._num_samples = num_samples
        self._seed = seed

    def __call__(self, design_vars):
        """
        Generate case.

        Parameters
        ----------
        design_vars : dict
            Dictionary of design variables for which to generate values.

        Yields
        ------
        dict
            Dictionary of input values for the case.
        """
        # Add up sizes
        self.num_design_vars = sum(meta['size'] for meta in itervalues(design_vars))

        if self._seed is not None:
            seed(self._seed)
            np.random.seed(self._seed)

        # Generate an LHC of the proper size
        rand_lhc = self._get_lhc()

        # Map LHC to buckets
        buckets = OrderedDict()
        j = 0

        for (name, meta) in iteritems(design_vars):
            buckets[name] = []

            nval = meta['size']

            for k in range(nval):
                lower = meta['lower']
                upper = meta['upper']
                if isinstance(lower, np.ndarray):
                    lower = lower[k]
                if isinstance(upper, np.ndarray):
                    upper = upper[k]

                design_var_buckets = self._get_buckets(lower, upper)

                buckets[name].append([
                    design_var_buckets[rand_lhc[i, j]] for i in range(self._num_samples)
                ])
                j += 1

        # Return random values in given buckets
        for i in range(self._num_samples):
            sample = {}
            for key, bounds in iteritems(buckets):
                vals = [
                    np.random.uniform(bounds[k][i][0], bounds[k][i][1])
                        for k in range(design_vars[key]['size'])
                ]
                sample[key] = np.array(vals)
            yield sample

    def _get_lhc(self):
        """
        Generate a Latin hypercube based on number of samples and design variables.

        Returns
        -------
        _LHC_Individual
            The randomly generated Latin hypercube.
        """
        rand_lhc = _rand_latin_hypercube(self._num_samples, self.num_design_vars)
        return rand_lhc.astype(int)

    def _get_buckets(self, low, high):
        """
        Determine the distribution of samples.

        Parameters
        ----------
        low : int
            The low value for the range.

        high : int
            The high value for the range.

        Returns
        -------
        list
            The upper and lower bounds for each bucket.
        """
        bucket_walls = np.linspace(low, high, self._num_samples + 1)
        return list(zip(bucket_walls[0:-1], bucket_walls[1:]))


CITATIONS = """
@journal {Morris:1995:SOU,
        title = {Exploratory Design for Computer Experiments},
        booktitle = {Journal of Statistical Planing and Inference},
        year = {1995},
        author = {Morris, M., and Mitchell, T}
}
"""


class OptimizedLatinHypercubeGenerator(LatinHypercubeGenerator):
    """
    DOE case generator implementing Morris-Mitchell method for Optimized Latin hypercube.

    Attributes
    ----------
    _population : int
        The population size.
    _generations : int
        The number of generations.
    _norm_method : int
        The method for calculating the norm.
    _qs : list
        List of qs to try for Phi_q optimization.
    """

    def __init__(self, num_samples=1, seed=None, population=20, generations=2, norm_method=1):
        """
        Initialize the OptimizedLatinHypercubeGenerator.

        Parameters
        ----------
        num_samples : int, optional
            The number of samples to run. Defaults to 1.
        seed : int or None, optional
            Random seed. Defaults to None.
        population : int, optional
            The population size. Defaults to 20.
        generations : int, optional
            The number of generations. Defaults to 2.
        norm_method : int, optional
            The method for calculating the norm. Defaults to 1.
            @see https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
        """
        super(OptimizedLatinHypercubeGenerator, self).__init__(num_samples=num_samples, seed=seed)
        self._qs = [1, 2, 5, 10, 20, 50, 100]  # List of qs to try for Phi_q optimization
        self._population = population
        self._generations = generations
        self._norm_method = norm_method
        # self.cite = CITATIONS

    def _get_lhc(self):
        """
        Generate an Optimized Latin hypercube.

        Returns
        -------
        _LHC_Individual
            The optimized Latin hypercube.
        """
        rand_lhc = _rand_latin_hypercube(self._num_samples, self.num_design_vars)

        # optimize our LHC before returning it
        best_lhc = _LHC_Individual(rand_lhc, q=1, p=self._norm_method)
        for q in self._qs:
            lhc_start = _LHC_Individual(rand_lhc, q, self._norm_method)
            lhc_opt = _mmlhs(lhc_start, self._population, self._generations)
            if lhc_opt.mmphi() < best_lhc.mmphi():
                best_lhc = lhc_opt

        return best_lhc.doe.astype(int)


class _LHC_Individual(object):
    """
    An instance of a Latin hypercube that can be perturbed to create variations.

    Attributes
    ----------
    _q : int, optional
        q.
    _p : int, optional
        The method for calculating the norm.
    _doe : numpy.array
        The initial Latin hypercube (set of points).
    _phi : float
        The Morris-Mitchell sampling criterion for this Latin hypercube.
    """

    def __init__(self, doe, q=2, p=1):
        """
        Initialize the _LHC_Individual.

        Parameters
        ----------
        doe : numpy.array
            The initial Latin hypercube (set of points).
        q : int, optional
            q. Defaults to 2.
        p : int, optional
            The method for calculating the norm. Defaults to 1.
            @see https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
        """
        self._q = q
        self._p = p
        self._doe = doe
        self._phi = None

    @property
    def doe(self):
        """
        Provide 'doe' property.

        Returns
        -------
        tuple
            The Latin hypercube.
        """
        return self._doe

    @property
    def shape(self):
        """
        Provide 'shape' property.

        Returns
        -------
        tuple
            The shape of the Latin hypercube (rows, cols).
        """
        return self._doe.shape

    def mmphi(self):
        """
        Provide Morris-Mitchell sampling criterion property for this Latin hypercube.

        Returns
        -------
        float
            The Morris-Mitchell sampling criterion for this Latin hypercube.
        """
        if self._phi is None:
            distdict = {}

            # Calculate the norm between each pair of points in the DOE
            arr = self._doe
            n, m = arr.shape
            for i in range(1, n):
                nrm = np.linalg.norm(arr[i] - arr[:i], ord=self._p, axis=1)
                for j in range(0, i):
                    nrmj = nrm[j]
                    if nrmj in distdict:
                        distdict[nrmj] += 1
                    else:
                        distdict[nrmj] = 1

            size = len(distdict)

            distinct_d = np.fromiter(distdict, dtype=float, count=size)

            # Mutltiplicity array with a count of how many pairs of points
            # have a given distance
            J = np.fromiter(itervalues(distdict), dtype=int, count=size)

            self._phi = sum(J * (distinct_d ** (-self._q))) ** (1.0 / self._q)

        return self._phi

    def perturb(self, mutation_count):
        """
        Generate a new Latin hypercube by perturbing this one.

        Interchanges pairs of randomly chosen elements within randomly chosen
        columns of a DOE a number of times.

        Parameters
        ----------
        mutation_count : int
            The number of mutations to make.

        Returns
        -------
        ndarray
            The perturbed Latin hypercube.
        """
        doe = self._doe
        new_doe = doe.copy()
        n, k = doe.shape
        for count in range(mutation_count):
            col = randint(0, k - 1)

            # Choosing two distinct random points
            el1 = randint(0, n - 1)
            el2 = randint(0, n - 1)
            while el1 == el2:
                el2 = randint(0, n - 1)

            new_doe[el1, col] = doe[el2, col]
            new_doe[el2, col] = doe[el1, col]

        return _LHC_Individual(new_doe, self._q, self._p)

    def __iter__(self):
        for row in self._doe:
            yield row

    def __repr__(self):
        return repr(self._doe)

    def __str__(self):
        return str(self._doe)

    def __getitem__(self, *args):
        return self._doe.__getitem__(*args)


def _rand_latin_hypercube(n, k):
    """
    Calculate random Latin hypercube set of n points in k dimensions within [0,n-1]^k hypercube.

    Parameters
    ----------
    n : int
        The number of points in each dimension.

    k : int
        The number of  dimensions.

    Returns
    -------
    ndarray
        The randomized Latin hypercube.
    """
    arr = np.zeros((n, k))
    row = list(range(0, n))
    for i in range(k):
        shuffle(row)
        arr[:, i] = row
    return arr


def _is_latin_hypercube(lh):
    """
    Determine if the given numpy array is a Latin hypercube.

    Parameters
    ----------
    lh : ndarray
        A set of points that may represent a Latin hypercube.

    Returns
    -------
    bool
        True is the data represents a Latin hypercube
    """
    n, k = lh.shape
    for j in range(k):
        col = lh[:, j]
        colset = set(col)
        if len(colset) < len(col):
            return False  # something was duplicated
    return True


def _mmlhs(x_start, population, generations):
    """
    Evolutionary search for most space filling Latin hypercube.

    Parameters
    ----------
    x_start : _LHC_Individual
        The initial Latin hypercube.
    population : int
        The population size.
    generations : int
        The number of generations.

    Returns
    -------
    ndarray
        a new LatinHypercube instance with an optimized set of points.
    """
    x_best = x_start
    phi_best = x_start.mmphi()
    n = x_start.shape[1]

    level_off = np.floor(0.85 * generations)
    for it in range(generations):
        if it < level_off and level_off > 1.:
            mutations = int(round(1 + (0.5 * n - 1) * (level_off - it) / (level_off - 1)))
        else:
            mutations = 1

        x_improved = x_best
        phi_improved = phi_best

        for offspring in range(population):
            x_try = x_best.perturb(mutations)
            phi_try = x_try.mmphi()

            if phi_try < phi_improved:
                x_improved = x_try
                phi_improved = phi_try

        if phi_improved < phi_best:
            phi_best = phi_improved
            x_best = x_improved

    return x_best
