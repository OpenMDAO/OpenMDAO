"""
Case generators for Design-of-Experiments Driver using pyDOE.
"""

from six import iteritems, itervalues
from six.moves import range, zip

import numpy as np

import itertools
from six import iteritems, itervalues

from collections import OrderedDict

from openmdao.drivers.doe_driver import DOEGenerator
import pyDOE


class _Generator(DOEGenerator):
    """
    Base class for DOE case generators implementing the pyDOE factorial methods.

    Attributes
    ----------
    _levels : int
        The number of evenly spaced levels between each design variable
        lower and upper bound.
    _supported_methods : list
        supported pyDOE function names.
    _method : string
        the pyDOE function to use.
    """

    _supported_methods = ['fullfact', 'pbdesign', 'bbdesign']

    def __init__(self, method, levels=2):
        """
        Initialize the FullFactorialGenerator.

        Parameters
        ----------
        levels : int, optional
            The number of evenly spaced levels between each design variable
            lower and upper bound. Defaults to 2.
        """
        super(_Generator, self).__init__()

        if method not in self._supported_methods:
            raise ValueError("Invalid method '%s' specified for %s. "
                             "Method must be one of %s." %
                             (method, self.__class__.__name__,
                              self._supported_methods))

        self._method = method
        self._levels = levels

    def __call__(self, design_vars):
        """
        Generate case.

        Parameters
        ----------
        design_vars : dict
            Dictionary of design variables for which to generate values.

        Yields
        ------
        list
            list of name, value tuples for the design variables.
        """
        names = design_vars.keys()

        size = sum([meta['size'] for name, meta in iteritems(design_vars)])

        # generate indices
        if self._method is 'fullfact':
            doe = pyDOE.fullfact([self._levels]*size)
        elif self._method is 'pbdesign':
            doe = pyDOE.pbdesign(size)
            doe[doe < 0] = 0  # replace -1 with zero
        elif self._method is 'bbdesign':
            if size < 3:
                raise RuntimeError("Total size of design variables is %d,"
                                   "but must be at least 3 when using %s. " %
                                   (size, self.__class__.__name__))
            doe = pyDOE.bbdesign(size, center=self._center)
            doe = doe + 1  # replace [-1, 0, 1] with [0, 1, 2]
        else:
            raise RuntimeError("Invalid method '%s' specified for %s. "
                               "Method must be one of %s." %
                               (self._method, self.__class__.__name__,
                                self._supported_methods))

        # generate values for each level for each design variable
        # over the range of that varable's lower to upper bound

        # rows = vars (# rows/var = var size), cols = levels
        values = np.zeros((size, self._levels))

        row = 0
        for name, meta in iteritems(design_vars):
            size = meta['size']

            for k in range(size):
                lower = meta['lower']
                if isinstance(lower, np.ndarray):
                    lower = lower[k]

                upper = meta['upper']
                if isinstance(upper, np.ndarray):
                    upper = upper[k]

                values[row][:] = np.linspace(lower, upper, num=self._levels)
                row += 1

        # yield values for doe generated indices
        for idxs in doe.astype('int'):
            retval = []
            var = row = 0
            for name, meta in iteritems(design_vars):
                size = meta['size']
                val = np.zeros(size)
                for k in range(size):
                    idx = idxs[var+k]
                    val[k] = values[row+k][idx]
                retval.append((name, val))
                var += 1
                row += size

            yield retval


class FullFactorialGenerator(_Generator):
    """
    DOE case generator implementing the Full Factorial method.
    """

    def __init__(self, levels=2):
        """
        Initialize the FullFactorialGenerator.

        Parameters
        ----------
        levels : int, optional
            The number of evenly spaced levels between each design variable
            lower and upper bound. Defaults to 2.
        """
        super(FullFactorialGenerator, self).__init__('fullfact', levels)


class PlackettBurmanGenerator(_Generator):
    """
    DOE case generator implementing the Plackett-Burman method.
    """

    def __init__(self):
        """
        Initialize the PlackettBurmanGenerator.
        """
        super(PlackettBurmanGenerator, self).__init__('pbdesign', 2)


class BoxBehnkenGenerator(_Generator):
    """
    DOE case generator implementing the Box-Behnken method.

    Attributes
    ----------
    _center : int
        The number of center points to include.
    """

    def __init__(self, center=1):
        """
        Initialize the BoxBehnkenGenerator.

        Parameters
        ----------
        center : int, optional
            The number of center points to include (default = 1).
        """
        super(BoxBehnkenGenerator, self).__init__('bbdesign', 3)

        self._center = center


class LatinHypercubeGenerator(DOEGenerator):
    """
    DOE case generators implementing Latin hypercube method via pyDOE.

    Attributes
    ----------
    _samples : int
        The number of evenly spaced levels between each design variable
        lower and upper bound.
    _supported_criterion : list
        supported pyDOE criterion names.
    _criterion : string
        the pyDOE criterion to use.
    _iterations : int
        The number of iterations to use for maximin and correlations algorithms.
    _seed : int or None
        Random seed.
    """

    _supported_criterion = [
        "center", "c",
        "maximin", "m",
        "centermaximin", "cm",
        "correlation", "corr",
        None
    ]

    def __init__(self, samples=None, criterion=None, iterations=5, seed=None):
        """
        Initialize the LatinHypercubeGenerator.

        See: https://pythonhosted.org/pyDOE/randomized.html

        Parameters
        ----------
        samples : int, optional
            The number of samples to generate for each factor (Defaults to n)
        criterion : str, optional
            Allowable values are "center" or "c", "maximin" or "m",
            "centermaximin" or "cm", and "correlation" or "corr". If no value
            given, the design is simply randomized.
        iterations : int, optional
            The number of iterations in the maximin and correlations algorithms
            (Defaults to 5).
        seed : int, optional
            Random seed to use if design is randomized. Defaults to None.
        """
        super(LatinHypercubeGenerator, self).__init__()

        if criterion not in self._supported_criterion:
            raise ValueError("Invalid criterion '%s' specified for %s. "
                             "Must be one of %s." %
                             (criterion, self.__class__.__name__,
                             self._supported_criterion))

        self._samples = samples
        self._criterion = criterion
        self._iterations = iterations
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
        list
            list of name, value tuples for the design variables.
        """
        if self._seed is not None:
            np.random.seed(self._seed)

        size = sum([meta['size'] for name, meta in iteritems(design_vars)])

        if self._samples is None:
            self._samples = size

        # generate design
        lhd = pyDOE.lhs(size, samples=self._samples,
                        criterion=self._criterion,
                        iterations=self._iterations)

        # generate values for each level for each design variable
        # over the range of that varable's lower to upper bound
        # rows = vars (# rows/var = var size), cols = levels
        values = np.zeros((size, self._samples))

        # yield desvar values for lhd samples
        for row in lhd:
            retval = []
            col = 0
            var = 0
            for name, meta in iteritems(design_vars):
                size = meta['size']
                val = np.zeros(size)
                for k in range(size):
                    sample = row[col+k]

                    lower = meta['lower']
                    if isinstance(lower, np.ndarray):
                        lower = lower[k]

                    upper = meta['upper']
                    if isinstance(upper, np.ndarray):
                        upper = upper[k]

                    val[k] = lower + sample*(upper-lower)

                retval.append((name, val))
                var += 1
                col += size

            yield retval
