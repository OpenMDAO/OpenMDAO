from collections.abc import Iterator
import itertools


class AnalysisGenerator(Iterator):
    """
    Provide a generator which provides case data for AnalysisDriver.

    Parameter
    ---------
    cases : dict
        A dictionary whose keys are promoted paths of variables to be set, and whose
        keys are the arguments to `set_val`.
    """
    
    def __init__(self, cases=None):
        super().__init__()
        self._run_count = 0
        self._vars = {}
        self._iter = None

        if cases is not None:
            self._setup(cases)

    def _setup(self, cases):
        """
        
        """
        self._vars = cases.copy()
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
            When all analysis cases have been exhausted.

        Returns
        -------
        dict
            A dictionary containing the promoted paths of variables to 
            be set by the AnalysisDriver

        """
        d = {}
        vals = next(self._iter)

        for i, name in enumerate(self._vars.keys()):
            d[name] = {'val': vals[i],
                       'units': self._vars[name].get('units', None),
                       'indices': self._vars[name].get('indices', None)}
        self._run_count += 1
        return d


class ZipGenerator(AnalysisGenerator):
    """
    A generator which provides case data for AnalysisDriver by zipping values of each factor.
    """
     
    def _setup(self, cases):
        """
        Setup the iterator which provides each case.

        Parameters
        ----------
        cases : dict
            A dictionary which maps promoted path names of variables to be
            set in each itearation with their values to be assumed (required),
            units (optional), and indices (optional).

        Raises
        ------
        ValueError
            Raised if the length of samples for each case are not all the same size.
        """
        super()._setup(cases)
        _case_vals = (c['val'] for c in cases.values())
        _lens = (len(_c) for _c in _case_vals)
        if len(set(_lens)) != 1:
            raise ValueError('zip sampler requires that val '
                             f'for all cases have the same length\n{_lens}')
        _case_vals = (c['val'] for c in cases.values())
        self._iter = zip(*_case_vals)


class ProductGenerator(AnalysisGenerator):
    """
    A generator which provides full-factorial case data for AnalysisDriver.
    """

    def _setup(self, cases):
        """
        Setup the iterator which provides each case.

        Parameters
        ----------
        cases : dict
            A dictionary which maps promoted path names of variables to be
            set in each itearation with their values to be assumed (required),
            units (optional), and indices (optional).

        Raises
        ------
        ValueError
            Raised if the length of samples for each case are not all the same size.
        """
        super()._setup(cases)
        _case_vals = (c['val'] for c in cases.values())
        self._iter = itertools.product(*_case_vals)


if __name__ == '__main__':
    cases = {'x': dict(val=[2, 3, 4], units='m'),
             'y': dict(val=[0.0, 0.5, 1.0], units='km'),
             'z': dict(val=3*['a'], units='kg')}

    g = ProductGenerator(cases)
    
    for design_point in g:
        print(design_point)

    g = ZipGenerator(cases)
    
    for design_point in g:
        print(design_point)

    
