"""
Uniform and DOE generators for Analysis Driver.
"""
import numpy as np

from openmdao.drivers.analysis_generator import AnalysisGenerator

try:
    import pyDOE3
except ImportError:
    pyDOE3 = None


_LEVELS = 2  # default number of levels for pyDOE generators


class UniformGenerator(AnalysisGenerator):
    """
    DOE case generator implementing the Uniform method.

    Parameters
    ----------
    var_dict : dict
        A dictionary mapping a variable name to 'upper' and 'lower' values to be assumed,
        as well as optional units and indices specifications.
    num_samples : int, optional
        The number of samples to run. Defaults to 1.
    seed : int or None, optional
        Seed for random number generator.
    """

    def __init__(self, var_dict, num_samples=1, seed=None):
        """
        Initialize the UniformGenerator.
        """
        if seed is not None:
            np.random.seed(seed)

        samples = {}

        for _ in range(num_samples):
            for name, meta in var_dict.items():
                try:
                    lower = meta['lower']
                except KeyError:
                    raise RuntimeError("UniformGenerator requires 'lower' and 'upper' values for "
                                       "each sample variable, but the 'lower' value is missing "
                                       f"for '{name}'.")

                try:
                    upper = meta['upper']
                except KeyError:
                    raise RuntimeError("UniformGenerator requires 'lower' and 'upper' values for "
                                       "each sample variable, but the 'upper' value is missing "
                                       f"for '{name}'.")

                samples[name] = {'vals': np.random.uniform(lower, upper, num_samples),
                                 'units': meta.get('units'),
                                 'indices': meta.get('indices')}

        super().__init__(samples)

    def __next__(self):
        """
        Provide a dictionary of the next point to be analyzed.

        The key of each entry is the promoted path of var whose values are to be set.
        The associated value is the values to set (required), units (options),
        and indices (optional).

        Raises
        ------
        StopIteration
            When all analysis var_dict have been exhausted.

        Returns
        -------
        dict
            A dictionary containing the promoted paths of variables to
            be set by the AnalysisDriver
        """
        i = self._run_count
        d = {}
        try:
            for name in self._var_dict:
                d[name] = {'val': self._var_dict[name]['vals'][i],
                           'units': self._var_dict[name].get('units', None),
                           'indices': self._var_dict[name].get('indices', None)}
            self._run_count += 1
            return d
        except IndexError:
            self._run_count = 0
            raise StopIteration("All samples have been exhausted for UniformGenerator.")


class _pyDOE3_Generator(AnalysisGenerator):
    """
    Base class for DOE generators implementing methods from pyDOE3.

    Parameters
    ----------
    var_dict : dict
        A dictionary whose keys are promoted paths of variables to be set, and whose
        values are the arguments to `set_val` and may include additional metadata such
        as size, global_size, val, lower, and upper.  Both 'lower' and 'upper' keys
        are required to define the range of each factor.
    levels : int or dict, optional
        The number of evenly spaced levels between each factor
        lower and upper bound.  Dictionary input is supported by Full Factorial or
        Generalized Subset Design.
        Defaults to 2.

    Attributes
    ----------
    _levels : int or dict(str, int)
        The number of evenly spaced levels between each factor
        lower and upper bound. Dictionary input is supported by Full Factorial or
        Generalized Subset Design.
    """

    def __init__(self, var_dict, levels=_LEVELS):
        """
        Initialize the _pyDOE3_Generator.
        """
        if pyDOE3 is None:
            raise RuntimeError(f"{self.__class__.__name__} requires the 'pyDOE3' package, "
                               "which can be installed with one of the following commands:\n"
                               "    pip install openmdao[doe]\n"
                               "    pip install pyDOE3")

        self._levels = levels
        self._sizes = None
        super().__init__(var_dict)

    def _get_levels(self, name):
        """
        Get the number of levels of a factor.

        If the name is not given, it looks for a "default" key in the dictionary. If this is also
        missing, it uses the default number of levels (2).

        Parameters
        ----------
        name : str
            Name of factor for which to get the number of levels.

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
            return sum([v * [self._get_levels(k)] for k, v in sizes.items()], [])
        else:
            raise ValueError(f"Levels should be an int or dictionary, not '{type(self._levels)}'")

    def _setup(self):
        """
        Generate the DOE and instantiate the internal Iterator.
        """
        super()._setup()

        factors = self._var_dict

        self._sizes = dict([(name, _get_size(name, meta)) for name, meta in factors.items()])
        size = sum(self._sizes.values())
        doe = self._generate_design(size).astype('int')

        # Maximum number of levels, or the default if the maximum is smaller than the default.
        # This is to ensure that the array will be big enough even if some keys are missing
        # from levels (defaulted).
        levels_max = self._levels if isinstance(self._levels, int) else \
            max(max(self._levels.values()), _LEVELS)

        # Generate values for each level for each factor
        # over the range of that variable's lower to upper bound

        # rows = vars (# rows/var = var size), cols = levels
        values = np.empty((size, levels_max))  # Initialize array for the largest number of levels
        values[:] = np.nan  # and fill with NaNs.

        row = 0
        for name, meta in factors.items():
            size = _get_size(name, meta)

            try:
                for k in range(size):
                    lower = meta['lower']
                    if isinstance(lower, np.ndarray):
                        lower = lower[k]

                    upper = meta['upper']
                    if isinstance(upper, np.ndarray):
                        upper = upper[k]

                    levels = self._get_levels(name)
                    values[row, 0:levels] = np.linspace(lower, upper, num=levels)

                    row += 1
            except KeyError:
                raise RuntimeError(f"Unable to determine levels for factor '{name}'. "
                                   "Factors dictionary must contain both 'lower' and 'upper' keys.")

        # construct iterator for doe values
        retvals = []
        for idxs in doe:
            retval = []
            row = 0
            for name, meta in factors.items():
                size_i = _get_size(name, meta)
                val = np.empty(size_i)
                for k in range(size_i):
                    idx = idxs[row + k]
                    val[k] = values[row + k][idx]
                retval.append(val)
                row += size_i

            retvals.append(retval)

        self._iter = iter(retvals)

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


class FullFactorialGenerator(_pyDOE3_Generator):
    """
    DOE case generator implementing the Full Factorial method.

    Parameters
    ----------
    var_dict : dict
        A dictionary whose keys are promoted paths of variables to be set, and whose
        values are the arguments to `set_val` and may include additional metadata such
        as size, global_size, val, lower, and upper.  Both 'lower' and 'upper' keys
        are required to define the range of each factor.
    levels : int or dict, optional
        The number of evenly spaced levels between each factor
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


class GeneralizedSubsetGenerator(_pyDOE3_Generator):
    """
    DOE case generator implementing the General Subset Design Factorial method.

    Parameters
    ----------
    var_dict : dict
        A dictionary whose keys are promoted paths of variables to be set, and whose
        values are the arguments to `set_val` and may include additional metadata such
        as size, global_size, val, lower, and upper.  Both 'lower' and 'upper' keys
        are required to define the range of each factor.
    levels : int or dict
        The number of evenly spaced levels between each factor
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

    def __init__(self, var_dict, levels, reduction, n=1):
        """
        Initialize the GeneralizedSubsetGenerator.
        """
        self._reduction = reduction
        self._n = n
        super().__init__(var_dict, levels=levels)

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


class PlackettBurmanGenerator(_pyDOE3_Generator):
    """
    DOE case generator implementing the Plackett-Burman method.

    Parameters
    ----------
    var_dict : dict
        A dictionary whose keys are promoted paths of variables to be set, and whose
        values are the arguments to `set_val` and may include additional metadata such
        as size, global_size, val, lower, and upper.  Both 'lower' and 'upper' keys
        are required to define the range of each factor.
    """

    def __init__(self, var_dict):
        """
        Initialize the PlackettBurmanGenerator.
        """
        super().__init__(var_dict, levels=2)

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


class BoxBehnkenGenerator(_pyDOE3_Generator):
    """
    DOE case generator implementing the Box-Behnken method.

    Parameters
    ----------
    var_dict : dict
        A dictionary whose keys are promoted paths of variables to be set, and whose
        values are the arguments to `set_val`.
    center : int, optional
        The number of center points to include (default = None).

    Attributes
    ----------
    _center : int
        The number of center points to include.
    """

    def __init__(self, var_dict, center=None):
        """
        Initialize the BoxBehnkenGenerator.
        """
        self._center = center
        super().__init__(var_dict, levels=3)

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
            raise RuntimeError("Total size of factors is %d,"
                               "but must be at least 3 when using %s. " %
                               (size, self.__class__.__name__))

        doe = pyDOE3.bbdesign(size, center=self._center)

        return doe + 1  # replace [-1, 0, 1] with [0, 1, 2]


class LatinHypercubeGenerator(AnalysisGenerator):
    """
    DOE case generator implementing Latin hypercube method via pyDOE3.

    Parameters
    ----------
    var_dict : dict
        A dictionary whose keys are promoted paths of variables to be set, and whose
        values are the arguments to `set_val`.
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
        The number of evenly spaced levels between each factor
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

    def __init__(self, var_dict, samples=None, criterion=None, iterations=5, seed=None):
        """
        Initialize the LatinHypercubeGenerator.

        See : https://pythonhosted.org/pyDOE/randomized.html
        """
        if criterion not in self._supported_criterion:
            raise ValueError("Invalid criterion '%s' specified for %s. "
                             "Must be one of %s." %
                             (criterion, self.__class__.__name__,
                              self._supported_criterion))

        self._samples = samples
        self._criterion = criterion
        self._iterations = iterations
        self._seed = seed

        super().__init__(var_dict)

    def _setup(self):
        """
        Generate the DOE and instantiate the internal Iterator.
        """
        if self._seed is not None:
            np.random.seed(self._seed)

        factors = self._var_dict

        size = sum([_get_size(name, meta) for name, meta in factors.items()])

        if self._samples is None:
            self._samples = size

        # generate design
        doe = pyDOE3.lhs(size, samples=self._samples,
                         criterion=self._criterion,
                         iterations=self._iterations,
                         random_state=self._seed)

        # construct iterator for doe values
        # rows = vars (# rows/var = var size), cols = levels
        retvals = []
        for row in doe:
            retval = []
            col = 0
            for name, meta in factors.items():
                size = _get_size(name, meta)
                sample = row[col:col + size]

                lower = meta['lower']
                if not isinstance(lower, np.ndarray):
                    lower = lower * np.ones(size)

                upper = meta['upper']
                if not isinstance(upper, np.ndarray):
                    upper = upper * np.ones(size)

                val = lower + sample * (upper - lower)

                retval.append(val)
                col += size

            retvals.append(retval)

        self._iter = iter(retvals)


def _get_size(name, dct):
    """
    Get the size of a variable from its metadata dictionary.

    Parameters
    ----------
    name : str
        The name of the variable for which to determine the size.
    dct : dict
        Dictionary containing metadata for the variable, must include 'upper', and 'lower' keys.

    Returns
    -------
    int
        The size of the variable as determined from the lower and upper bounds of the range.
        Note that both 'lower' and 'upper' must be present in the dictionary and have the same size.

    Raises
    ------
    ValueError
        The size of the specified lower bound does not match the size of the upper bound.
    RuntimeError
        The required metadata was not found in the dictionary to determine the size.
    """
    try:
        lower_size = np.size(dct['lower'])
        upper_size = np.size(dct['upper'])
        if lower_size != upper_size:
            raise ValueError(f"Size mismatch for factor '{name}': 'lower' bound size "
                             f"({lower_size}) does not match 'upper' bound size ({upper_size}).")
        return lower_size
    except KeyError:
        raise RuntimeError(f"Unable to determine levels for factor '{name}'. "
                           "Factors dictionary must contain both 'lower' and 'upper' keys.")
