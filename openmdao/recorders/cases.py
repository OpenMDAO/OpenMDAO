"""Define a BaseCases class for all CaseReader implementations."""
from abc import ABCMeta, abstractmethod


class BaseCases(object):
    """
    Abstract base class of all CaseReader implementations.

    Attributes
    ----------
    filename : str
        The name of the file from which the recorded cases are to be loaded.
    num_cases : int
        The number of cases contained in the recorded file.
    _case_keys : tuple
        Case string identifiers available in this CaseReader.
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    _abs2meta : dict
        Dictionary mapping absolute variable names to variable metadata.
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    _cases : dict
        Dictionary mapping iteration coordinates to cases that have already been loaded.
    """

    __metaclass__ = ABCMeta

    def __init__(self, filename, abs2prom, abs2meta, prom2abs):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The name of the recording file from which to instantiate the case reader.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        abs2meta : dict
            Dictionary mapping absolute variable names to variable metadata.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        """
        self._case_keys = ()
        self.num_cases = 0
        self.filename = filename
        self._abs2prom = abs2prom
        self._abs2meta = abs2meta
        self._prom2abs = prom2abs
        self._cases = {}

    @abstractmethod
    def get_case(self, case_id, scaled=False):
        """
        Get cases.

        Parameters
        ----------
        case_id : str or int
            If int, the index of the case to be read in the case iterations.
            If given as a string, it is the identifier of the case.
        scaled : bool
            If True, return the scaled values.

        Returns
        -------
        Case : object
            The case from the recorded file with the given identifier or index.

        """
        pass

    def list_cases(self):
        """
        Return a tuple of the case string identifiers available in this instance of the CaseReader.

        Returns
        -------
        _case_keys : tuple
            The case string identifiers.
        """
        return self._case_keys

    def get_iteration_coordinate(self, case_id):
        """
        Return the iteration coordinate.

        Parameters
        ----------
        case_id : int
            The case number that we want the iteration coordinate for.

        Returns
        -------
        iteration_coordinate : str
            The iteration coordinate.
        """
        if isinstance(case_id, int):
            # If case_id is an integer, assume the user
            # wants a case as an index
            iteration_coordinate = self._case_keys[case_id]  # handles negative indices for example
        else:
            # Otherwise assume we were given the case string identifier
            iteration_coordinate = case_id

        return iteration_coordinate
