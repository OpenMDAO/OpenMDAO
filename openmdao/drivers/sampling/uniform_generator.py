"""
Uniform generator for Analysis Driver.
"""
import numpy as np

from openmdao.drivers.analysis_generator import AnalysisGenerator
from openmdao.drivers.sampling.sampling_util import _get_size


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

    Attributes
    ----------
    _num_samples : int
        The number of samples in the DOE.
    _seed : int or None
        Random seed.
    _sizes : dict
        A dictionary mapping variable names to their sizes, determined from the 'lower' and 'upper'
        bounds in the var_dict.
    """

    def __init__(self, var_dict, num_samples=1, seed=None):
        """
        Initialize the UniformGenerator.
        """
        self._num_samples = num_samples
        self._seed = seed
        self._sizes = sizes = {}

        for name, meta in var_dict.items():
            sizes[name] = _get_size(name, meta)

        super().__init__(var_dict)

    def _setup(self):
        """
        Set up the iterator which provides each case.

        Raises
        ------
        ValueError
            Raised if the length of var_dict for each case are not all the same size.
        """
        if self._seed is not None:
            np.random.seed(self._seed)

        self._iter = iter(range(self._num_samples))

        super()._setup()

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
        next(self._iter)

        sizes = self._sizes

        d = {}
        for name, meta in self._var_dict.items():
            d[name] = {
                'val': np.random.uniform(meta['lower'], meta['upper'], sizes[name]),
                'units': meta.get('units', None),
                'indices': meta.get('indices', None)
            }
        self._run_count += 1
        return d
