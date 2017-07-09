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
        self.driver_cases = None
        self.system_cases = None
        self.solver_cases = None

        self.driver_metadata = {}
        self.system_metadata = {}
        self.solver_metadata = {}

