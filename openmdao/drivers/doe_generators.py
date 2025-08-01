"""
Case generators for Design-of-Experiments Driver.
"""
import csv
import os.path
import re
from collections import OrderedDict

import numpy as np

try:
    import pyDOE3
except ImportError:
    pyDOE3 = None


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

    Parameters
    ----------
    data : list
        List of collections of name, value pairs for the design variables.

    Attributes
    ----------
    _data : list
        List of collections of name, value pairs for the design variables.
    """

    def __init__(self, data=[]):
        """
        Initialize the ListGenerator.
        """
        super().__init__()

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
                msg = "Invalid DOE case found, expecting a list of name/value pairs:\n{}"
                raise RuntimeError(msg.format(case))

            name_map = {}

            for tup in case:
                if not isinstance(tup, (tuple, list)) or len(tup) != 2:
                    msg = "Invalid DOE case found, expecting a list of name/value pairs:\n{}"
                    raise RuntimeError(msg.format(case))

                name = tup[0]
                if name in design_vars:
                    name_map[name] = name
                elif model:
                    abs_name = model._resolver.any2abs(name, 'output')
                    if abs_name in design_vars:
                        name_map[name] = abs_name

            # any names not found in name_map are invalid design vars
            invalid_desvars = [name for name, _ in case if name not in name_map]
            if invalid_desvars:
                if len(invalid_desvars) > 1:
                    msg = "Invalid DOE case found, {} are not valid design variables:\n{}"
                    raise RuntimeError(msg.format(invalid_desvars, case))
                else:
                    msg = "Invalid DOE case found, '{}' is not a valid design variable:\n{}"
                    raise RuntimeError(msg.format(invalid_desvars[0], case))

            yield [(name_map[name], val) for name, val in case]


class CSVGenerator(DOEGenerator):
    """
    DOE case generator that reads cases from a CSV file.

    This DOE case generator will accept an existing data set in the form of
    a CSV file containing DOE cases. The CSV file should have one column per
    design variable and the header row should have the names of the design
    variables.

    Parameters
    ----------
    filename : str
        The name of the file from which to read cases.

    Attributes
    ----------
    _filename : str
           the name of the file from which to read cases
    """

    def __init__(self, filename):
        """
        Initialize the CSVGenerator.
        """
        super().__init__()

        if not isinstance(filename, str):
            raise RuntimeError("'{}' is not a valid file name.".format(filename))

        if not os.path.isfile(filename):
            raise RuntimeError("File not found: {}".format(filename))

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
                    abs_name = model._resolver.any2abs(name, 'output')
                    if abs_name in design_vars:
                        name_map[name] = abs_name

            # any names not found in name_map are invalid design vars
            invalid_desvars = [name for name in names if name not in name_map]
            if invalid_desvars:
                if len(invalid_desvars) > 1:
                    msg = "Invalid DOE case file, {} are not valid design variables."
                    raise RuntimeError(msg.format(invalid_desvars))
                else:
                    msg = "Invalid DOE case file, '{}' is not a valid design variable."
                    raise RuntimeError(msg.format(invalid_desvars[0]))

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

    Parameters
    ----------
    num_samples : int, optional
        The number of samples to run. Defaults to 1.
    seed : int or None, optional
        Seed for random number generator.

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
        """
        super().__init__()

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
    Base class for DOE case generators implementing methods from pyDOE3.

    Parameters
    ----------
    levels : int or dict, optional
        The number of evenly spaced levels between each design variable
        lower and upper bound.  Dictionary input is supported by Full Factorial or
        Generalized Subset Design.
        Defaults to 2.

    Attributes
    ----------
    _levels : int or dict(str, int)
        The number of evenly spaced levels between each design variable
        lower and upper bound. Dictionary input is supported by Full Factorial or
        Generalized Subset Design.
    """

    def __init__(self, levels=_LEVELS):
        """
        Initialize the _pyDOE_Generator.
        """
        if pyDOE3 is None:
            raise RuntimeError(f"{self.__class__.__name__} requires the 'pyDOE3' package, "
                               "which can be installed with one of the following commands:\n"
                               "    pip install openmdao[doe]\n"
                               "    pip install pyDOE3")

        super().__init__()
        self._levels = levels
        self._sizes = None

    def _get_dv_levels(self, name):
        """
        Get the number of levels of a design variable.

        If the name is not given, it looks for a "default" key in the dictionary. If this is also
        missing, it uses the default number of levels (2).

        Parameters
        ----------
        name : str
            Design variable name

        Returns
        -------
            int
        """
        levels = self._levels
        if isinstance(levels, int):
            return levels
        else:
            return levels.get(name, levels.get("default", _LEVELS))

    def _get_all_levels(self):
        """Return the levels of all factors."""
        sizes = self._sizes
        if isinstance(self._levels, int):  # All have the same number of levels
            return [self._levels] * sum(self._sizes.values())
        elif isinstance(self._levels, dict):  # Different DVs have different number of levels
            return sum([v * [self._get_dv_levels(k)] for k, v in sizes.items()], [])
        else:
            raise ValueError(f"Levels should be an int or dictionary, not '{type(self._levels)}'")

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
        self._sizes = OrderedDict([(name, _get_size(meta))
                                   for name, meta in design_vars.items()])
        size = sum(self._sizes.values())
        doe = self._generate_design(size).astype('int')

        # Maximum number of levels, or the default if the maximum is smaller than the default.
        # This is to ensure that the array will be big enough even if some keys are missing
        # from levels (defaulted).
        levels_max = self._levels if isinstance(self._levels, int) else \
            max(max(self._levels.values()), _LEVELS)

        # Generate values for each level for each design variable
        # over the range of that variable's lower to upper bound

        # rows = vars (# rows/var = var size), cols = levels
        values = np.empty((size, levels_max))  # Initialize array for the largest number of levels
        values[:] = np.nan  # and fill with NaNs.

        row = 0
        for name, meta in design_vars.items():
            size = _get_size(meta)

            for k in range(size):
                lower = meta['lower']
                if isinstance(lower, np.ndarray):
                    lower = lower[k]

                upper = meta['upper']
                if isinstance(upper, np.ndarray):
                    upper = upper[k]

                levels = self._get_dv_levels(name)
                values[row, 0:levels] = np.linspace(lower, upper, num=levels)

                row += 1

        # yield values for doe generated indices
        for idxs in doe:
            retval = []
            row = 0
            for name, meta in design_vars.items():
                size_i = _get_size(meta)
                val = np.empty(size_i)
                for k in range(size_i):
                    idx = idxs[row + k]
                    val[k] = values[row + k][idx]
                retval.append((name, val))
                row += size_i
            yield retval

    def _generate_design(self, size):
        """
        Generate DOE design.

        Parameters
        ----------
        size : int
            The total size (sum of sizes) of all factors for the design.

        Returns
        -------
        ndarray
            The design matrix as a size x levels array of indices.
        """
        pass


class FullFactorialGenerator(_pyDOE_Generator):
    """
    DOE case generator implementing the Full Factorial method.

    Parameters
    ----------
    levels : int or dict, optional
        The number of evenly spaced levels between each design variable
        lower and upper bound.  Dictionary input is supported by Full Factorial or
        Generalized Subset Design.
        Defaults to 2.
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
        return pyDOE3.fullfact(self._get_all_levels())


class GeneralizedSubsetGenerator(_pyDOE_Generator):
    """
    DOE case generator implementing the General Subset Design Factorial method.

    Parameters
    ----------
    levels : int or dict
        The number of evenly spaced levels between each design variable
        lower and upper bound. Defaults to 2.
    reduction : int
        Reduction factor (bigger than 1). Larger `reduction` means fewer
        experiments in the design and more possible complementary designs.
    n : int, optional
        Number of complementary GSD-designs. The complementary
        designs are balanced analogous to fold-over in two-level fractional
        factorial designs.
        Defaults to 1.

    Attributes
    ----------
    _reduction : int
        Reduction factor (bigger than 1). Larger `reduction` means fewer
        experiments in the design and more possible complementary designs.
    _n : int, optional
        Number of complementary GSD-designs. The complementary
        designs are balanced analogous to fold-over in two-level fractional
        factorial designs.
        Defaults to 1.
    """

    def __init__(self, levels, reduction, n=1):
        """
        Initialize the GeneralizedSubsetGenerator.
        """
        super().__init__(levels=levels)
        self._reduction = reduction
        self._n = n

    def _generate_design(self, size):
        """
        Generate a general subset DOE design.

        Parameters
        ----------
        size : int
            The number of factors for the design.

        Returns
        -------
        ndarray
            The design matrix as a size x levels array of indices.
        """
        return pyDOE3.gsd(levels=self._get_all_levels(), reduction=self._reduction, n=self._n)


class PlackettBurmanGenerator(_pyDOE_Generator):
    """
    DOE case generator implementing the Plackett-Burman method.
    """

    def __init__(self):
        """
        Initialize the PlackettBurmanGenerator.
        """
        super().__init__(levels=2)

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
        doe = pyDOE3.pbdesign(size)

        doe[doe < 0] = 0  # replace -1 with zero

        return doe


class BoxBehnkenGenerator(_pyDOE_Generator):
    """
    DOE case generator implementing the Box-Behnken method.

    Parameters
    ----------
    center : int, optional
        The number of center points to include (default = None).

    Attributes
    ----------
    _center : int
        The number of center points to include.
    """

    def __init__(self, center=None):
        """
        Initialize the BoxBehnkenGenerator.
        """
        super().__init__(levels=3)
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

        doe = pyDOE3.bbdesign(size, center=self._center)

        return doe + 1  # replace [-1, 0, 1] with [0, 1, 2]


class LatinHypercubeGenerator(DOEGenerator):
    """
    DOE case generator implementing Latin hypercube method via pyDOE3.

    Parameters
    ----------
    samples : int, optional
        The number of samples to generate for each factor (Defaults to n).
    criterion : str, optional
        Allowable values are "center" or "c", "maximin" or "m",
        "centermaximin" or "cm", and "correlation" or "corr". If no value
        given, the design is simply randomized.
    iterations : int, optional
        The number of iterations in the maximin and correlations algorithms
        (Defaults to 5).
    seed : int, optional
        Random seed to use if design is randomized. Defaults to None.

    Attributes
    ----------
    _samples : int
        The number of evenly spaced levels between each design variable
        lower and upper bound.
    _criterion : str
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

        See : https://pythonhosted.org/pyDOE/randomized.html
        """
        super().__init__()

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
        doe = pyDOE3.lhs(size, samples=self._samples,
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


def _get_size(dct):
    # Returns global size of the variable if it is distributed, size otherwise.
    return dct['global_size'] if dct['distributed'] else dct['size']
