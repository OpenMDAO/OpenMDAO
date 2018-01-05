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
    """

    __metaclass__ = ABCMeta

    def __init__(self, filename):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The name of the recording file from which to instantiate the case reader.
        """
        self._case_keys = ()
        self.num_cases = 0
        self.filename = filename

    @abstractmethod
    def get_case(self, case_id):
        """
        Get cases.

        Parameters
        ----------
        case_id : str or int
            If int, the index of the case to be read in the case iterations.
            If given as a string, it is the identifier of the case.

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
