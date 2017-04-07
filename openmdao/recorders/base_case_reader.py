"""
Docstring.
"""
from abc import ABCMeta, abstractmethod


class BaseCaseReader(object):
    """
    Abstract base class of all CaseReader implementations.

    Parameters
    ----------
    filename : str
        The name of the file from which to instantiate the case reader.

    Attributes
    ----------
    format_version : int
        An integer representation of the format version in the recorded file.
    filename : str
        The name of the file from which the recorded cases are to be loaded.
    parameters : dict
        Parameters metadata from the cases.
    unknowns : dict
        Unknowns metadata from the cases.
    num_cases : int
        The number of cases contained in the recorded file.
    """

    __metaclass__ = ABCMeta

    def __init__(self, filename):
        """
        Initialize.
        """
        self.format_version = None
        self.filename = filename
        self.parameters = None
        self.unknowns = None
        self._case_keys = ()
        self.num_cases = 0

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
        Case
            The case from the recorded file with the given identifier or index.

        """
        pass

    def list_cases(self):
        """
        Return a tuple of the case string identifiers available in this instance of the CaseReader.
        """
        return self._case_keys
