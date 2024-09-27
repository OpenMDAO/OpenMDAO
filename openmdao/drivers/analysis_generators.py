from collections.abc import Iterator
import itertools


class AnalysisGenerator(Iterator):
    """
    Provide a generator which provides case data for AnalysisDriver.

    Attributes
    ----------
    _iter : Iterator
        The underlying iterator for variable values.
    _run_count : int
        A running count of the samples obtained from the iterator.
    _var_dict : dict
        An internal copy of the var_dict used to create the generator.

    Parameters
    ----------
    var_dict : dict
        A dictionary whose keys are promoted paths of variables to be set, and whose
        keys are the arguments to `set_val`.
    """
    def __init__(self, var_dict):
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
    """

    def _setup(self):
        """
        Setup the iterator which provides each case.

        Parameters
        ----------
        var_dict : dict
            A dictionary which maps promoted path names of variables to be
            set in each itearation with their values to be assumed (required),
            units (optional), and indices (optional).

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
    """

    def _setup(self):
        """
        Setup the iterator which provides each case.

        Parameters
        ----------
        var_dict : dict
            A dictionary which maps promoted path names of variables to be
            set in each itearation with their values to be assumed (required),
            units (optional), and indices (optional).

        Raises
        ------
        ValueError
            Raised if the length of var_dict for each case are not all the same size.
        """
        super()._setup()
        sampler = (c['val'] for c in self._var_dict.values())
        self._iter = itertools.product(*sampler)
