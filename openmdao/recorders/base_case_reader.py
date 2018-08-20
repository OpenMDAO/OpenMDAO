"""
Base class for all CaseReaders.
"""
from abc import ABCMeta


class BaseCaseReader(object):
    """
    Abstract base class of all CaseReader implementations.

    Attributes
    ----------
    format_version : int
        An integer representation of the format version in the recorded file.
    filename : str
        The name of the file from which the recorded cases are to be loaded.
    driver_cases : list
        The list of driver cases to be loaded.
    system_cases : list
        The list of system cases to be loaded.
    solver_cases : list
        The list of solver cases to be loaded.
    problem_cases : list
        The list of problem cases to be loaded.
    driver_derivative_cases : list
        The list of driver derivative cases to be loaded.
    driver_metadata : dict
        The dictionary of driver metadata to be loaded.
    system_metadata : dict
        The dictionary of system metadata to be loaded.
    solver_metadata : dict
        The dictionary of solver metadata to be loaded..
    """

    __metaclass__ = ABCMeta

    def __init__(self, filename):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The name of the file from which to instantiate the case reader.
        """
        self.format_version = None
        self.filename = filename

        self.driver_cases = None
        self.system_cases = None
        self.solver_cases = None
        self.problem_cases = None

        self.driver_derivative_cases = None

        self.driver_metadata = {}
        self.system_metadata = {}
        self.solver_metadata = {}
