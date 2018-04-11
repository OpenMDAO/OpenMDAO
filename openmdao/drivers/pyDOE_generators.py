"""
Factorial Case generators for Design-of-Experiments Driver using pyDOE.
"""

from six import iteritems, itervalues
from six.moves import range, zip

import numpy as np

import itertools
from six import iteritems, itervalues

from collections import OrderedDict

from openmdao.drivers.doe_driver import DOEGenerator
import pyDOE

_methods = ['fullfact', 'ff2n', 'fractfract', 'pbdesign', 'bbdesign', 'ccdesign', 'lhs']


class _FactorialGenerator(DOEGenerator):
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

    _supported_methods = ['fullfact', 'pbdesign']

    def __init__(self, method, levels):
        """
        Initialize the FullFactorialGenerator.

        Parameters
        ----------
        levels : int, optional
            The number of evenly spaced levels between each design variable
            lower and upper bound. Defaults to 2.
        """
        super(_FactorialGenerator, self).__init__()

        if method not in self._supported_methods:
            raise RuntimeError('Invalid method specified for generator: %s. '
                               'Method must be one of %s.' %
                               (method, _supported_methods))

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
            ff = pyDOE.fullfact([self._levels]*size)
        elif self._method is 'pbdesign':
            ff = pyDOE.pbdesign(size)
            ff[ff < 0] = 0  # replace -1 with zero
        else:
            raise RuntimeError("Invalid method for _FactorialGenerator.")

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

        # yield values for ff generated indices
        for idxs in ff.astype('int'):
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


class FullFactorialGenerator(_FactorialGenerator):
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


class PlackettBurmanGenerator(_FactorialGenerator):
    """
    DOE case generator implementing the Plackett-Burman method.
    """

    def __init__(self):
        """
        Initialize the PlackettBurmanGenerator.
        """
        super(PlackettBurmanGenerator, self).__init__('pbdesign', 2)


class LatinHypercubeGenerator(DOEGenerator):
    """
    Base class for DOE case generators implementing the pyDOE factorial methods.

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
            raise RuntimeError("Invalid criterion '%s' specified for %s. "
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
        print('criterion:', self._criterion)
        lhd = pyDOE.lhs(size, samples=self._samples,
                        criterion=self._criterion,
                        iterations=self._iterations)
        print(lhd)

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
