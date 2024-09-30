"""
Provide generators for use with AnalysisDriver.

These generators are pythonic, lazy generators which, when provided with a dictionary
of variables and values to be tested, produce some set of sample values to be evaluated.

"""

from collections.abc import Iterator
import itertools


class AnalysisGenerator(Iterator):
    """
    Provide a generator which provides case data for AnalysisDriver.

    Parameters
    ----------
    var_dict : dict
        A dictionary whose keys are promoted paths of variables to be set, and whose
        keys are the arguments to `set_val`.

    Attributes
    ----------
    _iter : Iterator
        The underlying iterator for variable values.
    _run_count : int
        A running count of the samples obtained from the iterator.
    _var_dict : dict
        An internal copy of the var_dict used to create the generator.
    """

    def __init__(self, var_dict):
        """
        Instantiate the base class for AnalysisGenerators.

        Parameters
        ----------
        var_dict : dict
            A dictionary mapping a variable name to values to be assumed, as well as optional
            units and indices specifications.
        """
        super().__init__()
        self._run_count = 0
        self._var_dict = var_dict
        self._iter = None

        self._setup()

    def _setup(self):
        """
        Reset the run counter and instantiate the internal Iterator.

        Subclasses of AnalysisGenerator should override this method
        to define self._iter.
        """
        self._run_count = 0

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
        d = {}
        vals = next(self._iter)

        for i, name in enumerate(self._var_dict.keys()):
            d[name] = {'val': vals[i],
                       'units': self._var_dict[name].get('units', None),
                       'indices': self._var_dict[name].get('indices', None)}
        self._run_count += 1
        return d


class ZipGenerator(AnalysisGenerator):
    """
    A generator which provides case data for AnalysisDriver by zipping values of each factor.

    Parameters
    ----------
    var_dict : dict
        A dictionary which maps promoted path names of variables to be
        set in each itearation with their values to be assumed (required),
        units (optional), and indices (optional).
    """

    def _setup(self):
        """
        Set up the iterator which provides each case.

        Raises
        ------
        ValueError
            Raised if the length of var_dict for each case are not all the same size.
        """
        super()._setup()
        sampler = (c['val'] for c in self._var_dict.values())
        _lens = {name: len(meta['val']) for name, meta in self._var_dict.items()}
        if len(set([_l for _l in _lens.values()])) != 1:
            raise ValueError('ZipGenerator requires that val '
                             f'for all var_dict have the same length:\n{_lens}')
        sampler = (c['val'] for c in self._var_dict.values())
        self._iter = zip(*sampler)


class ProductGenerator(AnalysisGenerator):
    """
    A generator which provides full-factorial case data for AnalysisDriver.

    Parameters
    ----------
    var_dict : dict
        A dictionary which maps promoted path names of variables to be
        set in each itearation with their values to be assumed (required),
        units (optional), and indices (optional).
    """

    def _setup(self):
        """
        Set up the iterator which provides each case.

        Raises
        ------
        ValueError
            Raised if the length of var_dict for each case are not all the same size.
        """
        super()._setup()
        sampler = (c['val'] for c in self._var_dict.values())
        self._iter = itertools.product(*sampler)
