"""
Case generators for Design-of-Experiments Driver.
"""

import numpy as np

from six import iteritems
from six.moves import range

import json

from openmdao.drivers.doe_driver import DOEGenerator
import pyDOE2


class JSONGenerator(DOEGenerator):
    """
    DOE case generator that reads cases from JSON data.

    Attributes
    ----------
    _num_samples : int
        The number of samples in the DOE.
    _data : list
        List of list of name, value tuples for the design variables.
    """

    def __init__(self, data):
        """
        Initialize the JSONGenerator.

        Parameters
        ----------
        data : list of list of name, value tuples for the design variables
               or string encoded JSON version of that data.
        """
        super(JSONGenerator, self).__init__()

        if isinstance(data, list):
            self._data = data
        elif isinstance(data, str):
            try:
                self._data = json.loads(data)
            except err:
                self._data = None
        else:
            self._data = None

        if not isinstance(self._data, list):
            raise RuntimeError("%s was not provided valid DOE case data." %
                               self.__class__.__name__)

        self._num_samples = len(self._data)

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
        for case in self._data:
            if not isinstance(case, list):
                msg = "Invalid DOE case found, expecting a list of name/value pairs:\n%s"
                raise RuntimeError(msg % str(case))

            invalid_desvars = []
            for tup in case:
                if not isinstance(tup, list) or len(tup) != 2:
                    msg = "Invalid DOE case found, expecting a list of name/value pairs:\n%s"
                    raise RuntimeError(msg % str(case))

                if tup[0] not in design_vars:
                    invalid_desvars.append(tup[0])

            if invalid_desvars:
                if len(invalid_desvars) > 1:
                    msg = "Invalid DOE case found, %s are not valid design variables:\n%s"
                    raise RuntimeError(msg % (str(invalid_desvars), str(case)))
                else:
                    msg = "Invalid DOE case found, '%s' is not a valid design variable:\n%s"
                    raise RuntimeError(msg % (str(invalid_desvars[0]), str(case)))

            yield case


class UniformGenerator(DOEGenerator):
    """
    DOE case generator implementing the Uniform method.

    Attributes
    ----------
    _num_samples : int
        The number of samples in the DOE.
    _seed : int or None
        Random seed.
    """

    def __init__(self, num_samples=1, seed=None):
        """
        Initialize the UniformGenerator.

        Parameters
        ----------
        num_samples : int, optional
            The number of samples to run. Defaults to 1.

        seed : int or None, optional
            Seed for randon number generator.
        """
        super(UniformGenerator, self).__init__()

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
        list
            list of name, value tuples for the design variables.
        """
        if self._seed is not None:
            np.random.seed(self._seed)

        for i in range(self._num_samples):
            sample = []

            for (name, meta) in iteritems(design_vars):
                values = []

                for k in range(meta['size']):
                    lower = meta['lower']
                    if isinstance(lower, np.ndarray):
                        lower = lower[k]

                    upper = meta['upper']
                    if isinstance(upper, np.ndarray):
                        upper = upper[k]

                    values.append(np.random.uniform(lower, upper))

                sample.append((name, np.array(values)))

            yield sample


class _pyDOE_Generator(DOEGenerator):
    """
    Base class for DOE case generators implementing methods from pyDOE2.

    Attributes
    ----------
    _levels : int
        The number of evenly spaced levels between each design variable
        lower and upper bound.
    _num_samples : int
        The number of samples in the DOE.
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
        super(_pyDOE_Generator, self).__init__()
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
        size = sum([meta['size'] for name, meta in iteritems(design_vars)])

        doe = self._generate_design(size)

        self._num_samples = len(doe)

        # generate values for each level for each design variable
        # over the range of that varable's lower to upper bound

        # rows = vars (# rows/var = var size), cols = levels
        values = np.empty((size, self._levels))

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
                val = np.empty(size)
                for k in range(size):
                    idx = idxs[var + k]
                    val[k] = values[row + k][idx]
                retval.append((name, val))
                var += 1
                row += size

            yield retval

    def _generate_design(self, size):
        """
        Generate DOE design.

        Parameters
        ----------
        size : int
            The number of factors for the design.

        Returns
        -------
        ndarray
            The design matrix as a size x levels array of indices.
        """
        pass


class FullFactorialGenerator(_pyDOE_Generator):
    """
    DOE case generator implementing the Full Factorial method.
    """

    def _generate_design(self, size):
        """
        Generate a full factorial DOE design.

        Parameters
        ----------
        size : int
            The number of factors for the design.

        Returns
        -------
        ndarray
            The design matrix as a size x levels array of indices.
        """
        return pyDOE2.fullfact([self._levels] * size)


class PlackettBurmanGenerator(_pyDOE_Generator):
    """
    DOE case generator implementing the Plackett-Burman method.
    """

    def __init__(self):
        """
        Initialize the PlackettBurmanGenerator.
        """
        super(PlackettBurmanGenerator, self).__init__(levels=2)

    def _generate_design(self, size):
        """
        Generate a Plackett-Burman DOE design.

        Parameters
        ----------
        size : int
            The number of factors for the design.

        Returns
        -------
        ndarray
            The design matrix as a size x levels array of indices.
        """
        doe = pyDOE2.pbdesign(size)

        doe[doe < 0] = 0  # replace -1 with zero

        return doe


class BoxBehnkenGenerator(_pyDOE_Generator):
    """
    DOE case generator implementing the Box-Behnken method.

    Attributes
    ----------
    _center : int
        The number of center points to include.
    """

    def __init__(self, center=None):
        """
        Initialize the BoxBehnkenGenerator.

        Parameters
        ----------
        center : int, optional
            The number of center points to include (default = None).
        """
        super(BoxBehnkenGenerator, self).__init__(levels=3)
        self._center = center

    def _generate_design(self, size):
        """
        Generate a Box-Behnken DOE design.

        Parameters
        ----------
        size : int
            The number of factors for the design.

        Returns
        -------
        ndarray
            The design matrix as a size x levels array of indices.
        """
        if size < 3:
            raise RuntimeError("Total size of design variables is %d,"
                               "but must be at least 3 when using %s. " %
                               (size, self.__class__.__name__))

        doe = pyDOE2.bbdesign(size, center=self._center)

        return doe + 1  # replace [-1, 0, 1] with [0, 1, 2]


class LatinHypercubeGenerator(DOEGenerator):
    """
    DOE case generator implementing Latin hypercube method via pyDOE2.

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
    _num_samples : int
        The number of samples in the DOE.
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
        doe = pyDOE2.lhs(size, samples=self._samples,
                         criterion=self._criterion,
                         iterations=self._iterations,
                         random_state=self._seed)

        self._num_samples = len(doe)

        # yield desvar values for doe samples
        for row in doe:
            retval = []
            col = 0
            var = 0
            for name, meta in iteritems(design_vars):
                size = meta['size']
                val = np.empty(size)
                for k in range(size):
                    sample = row[col + k]

                    lower = meta['lower']
                    if isinstance(lower, np.ndarray):
                        lower = lower[k]

                    upper = meta['upper']
                    if isinstance(upper, np.ndarray):
                        upper = upper[k]

                    val[k] = lower + sample * (upper - lower)

                retval.append((name, val))
                var += 1
                col += size

            yield retval
