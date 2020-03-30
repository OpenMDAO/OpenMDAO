"""
Base class for all CaseReaders.
"""

from openmdao.utils.assert_utils import warn_deprecation


class BaseCaseReader(object):
    """
    Base class of all CaseReader implementations.

    Attributes
    ----------
    _format_version : int
        The version of the format assumed when loading the file.
    problem_metadata : dict
        Metadata about the problem, including the system hierachy and connections.
    solver_metadata : dict
        The solver options for each solver in the recorded model.
    system_options : dict
        Metadata about each system in the recorded model, including options and scaling factors.
    """

    def __init__(self, filename, pre_load=False):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The path to the file containing the recorded data.
        pre_load : bool
            If True, load all the data into memory during initialization.
        """
        self._format_version = None
        self.problem_metadata = {}
        self.solver_metadata = {}
        self.system_options = {}

    @property
    def system_metadata(self):
        """
        Provide 'system_metadata' property for backwards compatibility.

        Returns
        -------
        dict
            reference to the 'system_options' attribute.
        """
        warn_deprecation("The BaseCaseReader.system_metadata attribute is deprecated. "
                         "Use the BaseCaseReader.system_option attribute instead.")
        return self.system_options

    def get_cases(self, source, recurse=True, flat=False):
        """
        Iterate over the cases.

        Parameters
        ----------
        source : {'problem', 'driver', <system hierarchy location>, <solver hierarchy location>,
            case name}
            Identifies which cases to return.
        recurse : bool, optional
            If True, will enable iterating over all successors in case hierarchy
        flat : bool, optional
            If False and there are child cases, then a nested ordered dictionary
            is returned rather than an iterator.

        Returns
        -------
        list or dict
            The cases identified by source
        """
        pass

    def get_case(self, case_id, recurse=True):
        """
        Get case identified by case_id.

        Parameters
        ----------
        case_id : str or int
            The unique identifier of the case to return or an index into all cases.
        recurse : bool, optional
            If True, will return all successors to the case as well.

        Returns
        -------
        dict
            The case identified by case_id
        """
        pass

    def list_sources(self):
        """
        List of all the different recording sources for which there is recorded data.

        Returns
        -------
        list
            One or more of: 'problem', 'driver', <system hierarchy location>,
                            <solver hierarchy location>
        """
        pass

    def list_source_vars(self, source):
        """
        List of all inputs and outputs recorded by the specified source.

        Parameters
        ----------
        source : {'problem', 'driver', <system hierarchy location>, <solver hierarchy location>}
            Identifies the source for which to return information.

        Returns
        -------
        dict
            {'inputs':[list of keys], 'outputs':[list of keys]}. Does not recurse.
        """
        pass

    def list_cases(self, source=None, recurse=True, flat=True):
        """
        Iterate over Driver, Solver and System cases in order.

        Parameters
        ----------
        source : {'problem', 'driver', <system hierarchy location>, <solver hierarchy location>,
            case name}
            If not None, only cases originating from the specified source or case are returned.
        recurse : bool, optional
            If True, will enable iterating over all successors in case hierarchy.
        flat : bool, optional
            If False and there are child cases, then a nested ordered dictionary
            is returned rather than an iterator.

        Returns
        -------
        iterator or dict
            An iterator or a nested dictionary of identified cases.
        """
        pass
