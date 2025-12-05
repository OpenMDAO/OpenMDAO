"""
pyDOE3 sample generators for Analysis Driver.
"""
import numpy as np

from openmdao.drivers.analysis_generator import AnalysisGenerator
from openmdao.drivers.sampling.sampling_util import _get_size


_LEVELS = 2  # default number of levels for pyDOE generators


class _pyDOE_AnalysisGenerator(AnalysisGenerator):
    """
    Base class for Analysis generators implementing methods from pyDOE3.

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
        Initialize the _pyDOE_AnalysisGenerator.
        """
        self._levels = levels
        self._sizes = sizes = {}

        for name, meta in var_dict.items():
            sizes[name] = _get_size(name, meta)

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
        """
        Return the levels of all factors.
        """
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
            size = self._sizes[name]

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


class FullFactorialGenerator(_pyDOE_AnalysisGenerator):
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

    Attributes
    ----------
    _fullfact : function
        A lazily imported pyDOE full factorial function.
    """

    def __init__(self, var_dict, levels=_LEVELS):
        """
        Initialize the FullFactorialGenerator.
        """
        try:
            from pyDOE3 import fullfact
            self._fullfact = fullfact
        except ImportError:
            raise RuntimeError(f"{self.__class__.__name__} requires the 'pyDOE3' package, "
                               "which can be installed with one of the following commands:\n"
                               "    pip install openmdao[doe]\n"
                               "    pip install pyDOE3")

        super().__init__(var_dict=var_dict, levels=levels)

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
        return self._fullfact(self._get_all_levels())


class GeneralizedSubsetGenerator(_pyDOE_AnalysisGenerator):
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
    _gsd : function
        The pyDOE3 generalized subset function, lazily imported.
    """

    def __init__(self, var_dict, levels, reduction, n=1):
        """
        Initialize the GeneralizedSubsetGenerator.
        """
        self._reduction = reduction
        self._n = n

        try:
            from pyDOE3 import gsd
            self._gsd = gsd
        except ImportError:
            raise RuntimeError(f"{self.__class__.__name__} requires the 'pyDOE3' package, "
                               "which can be installed with one of the following commands:\n"
                               "    pip install openmdao[doe]\n"
                               "    pip install pyDOE3")

        super().__init__(var_dict, levels=levels)

    def _generate_design(self, size):
        """
        Generate a general subset DOE design.

        Parameters
        ----------
        size : int
            The total size (sum of sizes) of all factors for the design.

        Returns
        -------
        ndarray
            The design matrix as a size x levels array of indices.
        """
        return self._gsd(levels=self._get_all_levels(), reduction=self._reduction, n=self._n)


class PlackettBurmanGenerator(_pyDOE_AnalysisGenerator):
    """
    DOE case generator implementing the Plackett-Burman method.

    Parameters
    ----------
    var_dict : dict
        A dictionary whose keys are promoted paths of variables to be set, and whose
        values are the arguments to `set_val` and may include additional metadata such
        as size, global_size, val, lower, and upper.  Both 'lower' and 'upper' keys
        are required to define the range of each factor.

    Attributes
    ----------
    _pbdesign : function
        The pyDOE3 Plackett-Burman function, lazily imported.
    """

    def __init__(self, var_dict):
        """
        Initialize the PlackettBurmanGenerator.
        """
        try:
            from pyDOE3 import pbdesign
            self._pbdesign = pbdesign
        except ImportError:
            raise RuntimeError(f"{self.__class__.__name__} requires the 'pyDOE3' package, "
                               "which can be installed with one of the following commands:\n"
                               "    pip install openmdao[doe]\n"
                               "    pip install pyDOE3")

        super().__init__(var_dict, levels=2)

    def _generate_design(self, size):
        """
        Generate a Plackett-Burman DOE design.

        Parameters
        ----------
        size : int
            The total size (sum of sizes) of all factors for the design.

        Returns
        -------
        ndarray
            The design matrix as a size x levels array of indices.
        """
        doe = self._pbdesign(size)

        doe[doe < 0] = 0  # replace -1 with zero

        return doe


class BoxBehnkenGenerator(_pyDOE_AnalysisGenerator):
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
    _bbdesign : function
        The pyDOE3 Box-Behnken function, lazily imported.
    """

    def __init__(self, var_dict, center=None):
        """
        Initialize the BoxBehnkenGenerator.
        """
        self._center = center
        try:
            from pyDOE3 import bbdesign
            self._bbdesign = bbdesign
        except ImportError:
            raise RuntimeError(f"{self.__class__.__name__} requires the 'pyDOE3' package, "
                               "which can be installed with one of the following commands:\n"
                               "    pip install openmdao[doe]\n"
                               "    pip install pyDOE3")

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

        doe = self._bbdesign(size, center=self._center)

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
    _lhs : function
        The pyDOE3 latin hypercube sampling function, lazily imported.
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

        try:
            from pyDOE3 import lhs
            self._lhs = lhs
        except ImportError:
            raise RuntimeError(f"{self.__class__.__name__} requires the 'pyDOE3' package, "
                               "which can be installed with one of the following commands:\n"
                               "    pip install openmdao[doe]\n"
                               "    pip install pyDOE3")

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
        doe = self._lhs(size, samples=self._samples,
                        criterion=self._criterion,
                        iterations=self._iterations,
                        random_state=self._seed)

        # construct iterator for doe values
        # rows = vars (# rows/var = var size), cols = levels
        def generator():
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

                yield retval

        self._iter = iter(generator())
