"""
Provide generators for use with AnalysisDriver.

These generators are pythonic, lazy generators which, when provided with a dictionary
of variables and values to be tested, produce some set of sample values to be evaluated.

"""

from collections.abc import Iterator
import csv
import itertools


class AnalysisGenerator(Iterator):
    """
    Provide a generator which provides case data for AnalysisDriver.

    Parameters
    ----------
    var_dict : dict
        A dictionary whose keys are promoted paths of variables to be set, and whose
        values are the arguments to `set_val`.

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

    def _get_sampled_vars(self):
        """
        Return the set of variable names whose value are provided by this generator.
        """
        return self._var_dict.keys()

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


class CSVGenerator(AnalysisGenerator):
    """
    A generator which provides cases for AnalysisDriver by pulling rows from a CSV file.

    Parameters
    ----------
    filename : str
        The filename for the CSV file containing the variable data.
    has_units : bool
        If True, the second line of the CSV contains the units of each variable.
    has_indices : bool
        If True, the line after units (if present) contains the indices being set.

    Attributes
    ----------
    _filename : str
        The filename of the CSV file providing the samples.
    _has_units : bool
        True if the CSV file contains a row of the units for each variable.
    _has_indices : bool
        True if the CSV file contains a row of indices being provided for each variable.
        If units are present, indices will be on the line following units.
    _csv_file : file
        The file object for the CSV file.
    _csv_reader : DictReader
        The reader object for the CSV file.
    _var_names : set of str
        The set of variable names provided by this CSVGenerator.
    _ret_val : dict
        The dict which is returned by each call to __next__.
    """

    def __init__(self, filename, has_units=False, has_indices=False):
        """
        Instantiate CSVGenerator.

        Parameters
        ----------
        filename : str
            The filename for the CSV file containing the variable data.
        has_units : bool
            If True, the second line of the CSV contains the units of each variable.
        has_indices : bool
            If True, the line after units (if present) contains the indices being set.
        """
        self._filename = filename
        self._has_units = has_units
        self._has_indices = has_indices

        self._csv_file = open(self._filename, 'r')
        self._csv_reader = csv.DictReader(self._csv_file)

        self._var_names = set(self._csv_reader.fieldnames)

        self._ret_val = {var: {'units': None, 'indices': None}
                         for var in self._csv_reader.fieldnames}

        if self._has_units:
            var_units_dict = next(self._csv_reader)
            for var, units in var_units_dict.items():
                self._ret_val[var]['units'] = None if not units else units

        if self._has_indices:
            var_idxs_dict = next(self._csv_reader)
            for var, idxs in var_idxs_dict.items():
                idxs = eval(idxs, {'__builtins__': {}})  # nosec: scope limited
                self._ret_val[var]['indices'] = idxs

    def _get_sampled_vars(self):
        return self._var_names

    def __next__(self):
        """
        Provide the data from the next row of the csv file.
        """
        try:
            var_val_dict = next(self._csv_reader)
            for var, val in var_val_dict.items():
                self._ret_val[var]['val'] = val
            return self._ret_val
        except StopIteration:
            # Close the file and propagate the exception
            self._csv_file.close()
            raise

    def __del__(self):
        """
        Ensure the file is closed if we don't exhaust the iterator.
        """
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()


class SequenceGenerator:
    """
    A generator which provides samples from python lists or tuples.

    Internally this generator converts the list or tuple to a deque and then consumes it
    as it iterates over it.

    Parameters
    ----------
    container : container
        A python container, excluding strings, bytes, or bytearray.

    Attributes
    ----------
    _sampled_vars : list(str)
        A list of the variables in the model being sampled.
    _iter : Iterator
        The internal iterator over the users case data.

    Raises
    ------
    StopIteration
        When given list or tuple is exhausted.
    """

    def __init__(self, container):
        """
        Instantiate a SequenceGenerator with the given container of samples.
        """
        self._sampled_vars = [k for k in list(container)[0].keys()]
        self._iter = iter(container)

    def __iter__(self):
        """
        Provide the python iterator for this instance.
        """
        return self

    def __next__(self):
        """
        Provide the next values for the variables in the generator.
        """
        return next(self._iter)

    def _get_sampled_vars(self):
        """
        Return the set of variable names whose value are provided by this generator.
        """
        return self._sampled_vars
