"""
Case generators for Design-of-Experiments Driver.
"""

import numpy as np

import os.path
import csv
import re

import pyDOE2

from openmdao.utils.name_maps import prom_name2abs_name

_LEVELS = 2  # default number of levels for pyDOE generators


class DOEGenerator(object):
    """
    Base class for a callable object that generates cases for a DOEDriver.
    """

    def __call__(self, design_vars, model=None):
        """
        Generate case.

        Parameters
        ----------
        design_vars : OrderedDict
            Dictionary of design variables for which to generate values.

        model : Group
            The model containing the design variables (used by some subclasses).

        Returns
        -------
        list
            list of name, value tuples for the design variables.
        """
        return []


class ListGenerator(DOEGenerator):
    """
    DOE case generator that reads cases from a provided list of DOE cases.

    This DOE case generator will accept an existing data set in the form of
    a list of DOE cases, each of which consists of a collection of name/value
    pairs specifying values for design variables.

    Attributes
    ----------
    _data : list
        List of collections of name, value pairs for the design variables.
    """

    def __init__(self, data=[]):
        """
        Initialize the ListGenerator.

        Parameters
        ----------
        data : list
            list of collections of name, value pairs for the design variables
        """
        super(ListGenerator, self).__init__()

        if not isinstance(data, list):
            msg = "Invalid DOE case data, expected a list but got a {}."
            raise RuntimeError(msg.format(data.__class__.__name__))

        self._data = data

    def __call__(self, design_vars, model=None):
        """
        Generate case.

        Parameters
        ----------
        design_vars : OrderedDict
            Dictionary of design variables for which to generate values.

        model : Group
            The model containing the design variables.

        Yields
        ------
        list
            list of name, value tuples for the design variables.
        """
        for case in self._data:
            if not isinstance(case, list):
                msg = "Invalid DOE case found, expecting a list of name/value pairs:\n%s"
                raise RuntimeError(msg % str(case))

            name_map = {}

            for tup in case:
                if type(tup) not in (tuple, list) or len(tup) != 2:
                    msg = "Invalid DOE case found, expecting a list of name/value pairs:\n%s"
                    raise RuntimeError(msg % str(case))

                name = tup[0]
                if name in design_vars:
                    name_map[name] = name
                elif model:
                    abs_name = prom_name2abs_name(model, name, 'output')
                    if abs_name in design_vars:
                        name_map[name] = abs_name

            # any names not found in name_map are invalid design vars
            invalid_desvars = [name for name, _ in case if name not in name_map]
            if invalid_desvars:
                if len(invalid_desvars) > 1:
                    msg = "Invalid DOE case found, %s are not valid design variables:\n%s"
                    raise RuntimeError(msg % (str(invalid_desvars), str(case)))
                else:
                    msg = "Invalid DOE case found, '%s' is not a valid design variable:\n%s"
                    raise RuntimeError(msg % (str(invalid_desvars[0]), str(case)))

            yield [(name_map[name], val) for name, val in case]


class CSVGenerator(DOEGenerator):
    """
    DOE case generator that reads cases from a CSV file.

    This DOE case generator will accept an existing data set in the form of
    a CSV file containing DOE cases. The CSV file should have one column per
    design variable and the header row should have the names of the design
    variables.

    Attributes
    ----------
    _filename : str
           the name of the file from which to read cases
    """

    def __init__(self, filename):
        """
        Initialize the CSVGenerator.

        Parameters
        ----------
        filename : str
               the name of the file from which to read cases
        """
        super(CSVGenerator, self).__init__()

        if not isinstance(filename, str):
            raise RuntimeError("'%s' is not a valid file name." % str(filename))

        if not os.path.isfile(filename):
            raise RuntimeError("File not found: %s" % filename)

        self._filename = filename

    def __call__(self, design_vars, model=None):
        """
        Generate case.

        Parameters
        ----------
        design_vars : OrderedDict
            Dictionary of design variables for which to generate values.

        model : Group
            The model containing the design variables.

        Yields
        ------
        list
            list of name, value tuples for the design variables.
        """
        name_map = {}

        with open(self._filename, 'r') as f:
            # map header names to absolute names if necessary
            names = re.sub(' ', '', f.readline()).strip().split(',')
            for name in names:
                if name in design_vars:
                    name_map[name] = name
                elif model:
                    abs_name = prom_name2abs_name(model, name, 'output')
                    if abs_name in design_vars:
                        name_map[name] = abs_name

            # any names not found in name_map are invalid design vars
            invalid_desvars = [name for name in names if name not in name_map]
            if invalid_desvars:
                if len(invalid_desvars) > 1:
                    msg = "Invalid DOE case file, %s are not valid design variables."
                    raise RuntimeError(msg % str(invalid_desvars))
                else:
                    msg = "Invalid DOE case file, '%s' is not a valid design variable."
                    raise RuntimeError(msg % str(invalid_desvars[0]))

        # read cases from file, parse values into numpy arrays
        with open(self._filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                case = [(name_map[name.strip()],
                         np.fromstring(re.sub(r'[\[\]]', '', row[name]), sep=' '))
                        for name in reader.fieldnames]
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
            Seed for random number generator.
        """
        super(UniformGenerator, self).__init__()

        self._num_samples = num_samples
        self._seed = seed

    def __call__(self, design_vars, model=None):
        """
        Generate case.

        Parameters
        ----------
        design_vars : OrderedDict
            Dictionary of design variables for which to generate values.

        model : Group
            The model containing the design variables (not used).

        Yields
        ------
        list
            list of name, value tuples for the design variables.
        """
        if self._seed is not None:
            np.random.seed(self._seed)

        for _ in range(self._num_samples):
            sample = []

            for name, meta in design_vars.items():
                size = meta['size']

                lower = meta['lower']
                if not isinstance(lower, np.ndarray):
                    lower = lower * np.ones(size)

                upper = meta['upper']
                if not isinstance(upper, np.ndarray):
                    upper = upper * np.ones(size)

                sample.append((name, np.random.uniform(lower, upper)))

            yield sample


class _pyDOE_Generator(DOEGenerator):
    """
    Base class for DOE case generators implementing methods from pyDOE2.

    Attributes
    ----------
    _levels : int or dict(str, int)
        The number of evenly spaced levels between each design variable
        lower and upper bound.
    """

    def __init__(self, levels=_LEVELS):
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
        self._level_lst = None

    def __call__(self, design_vars, model=None):
        """
        Generate case.

        Parameters
        ----------
        design_vars : OrderedDict
            Dictionary of design variables for which to generate values.

        model : Group
            The model containing the design variables (not used).

        Yields
        ------
        list
            list of name, value tuples for the design variables.
        """
        size = sum([meta['size'] for name, meta in design_vars.items()])

        doe = self._generate_design(size)

        # generate values for each level for each design variable
        # over the range of that variable's lower to upper bound

        # rows = vars (# rows/var = var size), cols = levels
        values = np.empty((size, self._levels))

        row = 0
        for name, meta in design_vars.items():
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
            row = 0
            for name, meta in design_vars.items():
                size = meta['size']
                val = np.empty(size)
                for k in range(size):
                    idx = idxs[row + k]
                    val[k] = values[row + k][idx]
                retval.append((name, val))
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
    _criterion : string
        the pyDOE criterion to use.
    _iterations : int
        The number of iterations to use for maximin and correlations algorithms.
    _seed : int or None
        Random seed.
    """

    # supported pyDOE criterion names.
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

    def __call__(self, design_vars, model=None):
        """
        Generate case.

        Parameters
        ----------
        design_vars : OrderedDict
            Dictionary of design variables for which to generate values.

        model : Group
            The model containing the design variables (not used).

        Yields
        ------
        list
            list of name, value tuples for the design variables.
        """
        if self._seed is not None:
            np.random.seed(self._seed)

        size = sum([meta['size'] for meta in design_vars.values()])

        if self._samples is None:
            self._samples = size

        # generate design
        doe = pyDOE2.lhs(size, samples=self._samples,
                         criterion=self._criterion,
                         iterations=self._iterations,
                         random_state=self._seed)

        # yield desvar values for doe samples
        for row in doe:
            retval = []
            col = 0
            for name, meta in design_vars.items():
                size = meta['size']
                sample = row[col:col + size]

                lower = meta['lower']
                if not isinstance(lower, np.ndarray):
                    lower = lower * np.ones(size)

                upper = meta['upper']
                if not isinstance(upper, np.ndarray):
                    upper = upper * np.ones(size)

                val = lower + sample * (upper - lower)

                retval.append((name, val))
                col += size

            yield retval
