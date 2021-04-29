"""
Base class for all CaseReaders.
"""

from openmdao.warnings import warn_deprecation
from openmdao.core.constants import _DEFAULT_OUT_STREAM


class BaseCaseReader(object):
    """
    Base class of all CaseReader implementations.

    Attributes
    ----------
    _format_version : int
        The version of the format assumed when loading the file.
    _openmdao_version : str
        The version of OpenMDAO used to generate the case recorder file.
    problem_metadata : dict
        Metadata about the problem, including the system hierachy and connections.
    solver_metadata : dict
        The solver options for each solver in the recorded model.
    _system_options : dict
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
        self._openmdao_version = None
        self.problem_metadata = {}
        self.solver_metadata = {}
        self._system_options = {}

    @property
    def system_options(self):
        """
        Provide '_system_options' property for backwards compatibility.

        Returns
        -------
        dict
            reference to the _system_options attribute.
        """
        warn_deprecation("The system_options attribute is deprecated. "
                         "Use `list_model_options` instead.")
        return self._system_options

    @property
    def system_metadata(self):
        """
        Provide 'system_metadata' property for backwards compatibility.

        Returns
        -------
        dict
            reference to the '_system_options' attribute.
        """
        warn_deprecation("The BaseCaseReader.system_metadata attribute is deprecated. "
                         "Use `list_model_options` instead.")
        return self._system_options

    @property
    def openmdao_version(self):
        """
        Provide the version of OpenMDAO that was used to record this file.

        Returns
        -------
        str
            version of OpenMDAO that was used to record this file.
        """
        return self._openmdao_version

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

    def list_sources(self, out_stream=_DEFAULT_OUT_STREAM):
        """
        List of all the different recording sources for which there is recorded data.

        Parameters
        ----------
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list
            One or more of: `problem`, `driver`, `<system hierarchy location>`,
                            `<solver hierarchy location>`
        """
        pass

    def list_source_vars(self, source, out_stream=_DEFAULT_OUT_STREAM):
        """
        List of all inputs and outputs recorded by the specified source.

        Parameters
        ----------
        source : {'problem', 'driver', <system hierarchy location>, <solver hierarchy location>}
            Identifies the source for which to return information.
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        dict
            {'inputs':[key list], 'outputs':[key list], 'residuals':[key list]}. No recurse.
        """
        pass

    def list_cases(self, source=None, recurse=True, flat=True, out_stream=_DEFAULT_OUT_STREAM):
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
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        iterator or dict
            An iterator or a nested dictionary of identified cases.
        """
        pass

    def list_model_options(self, run_number=0, system=None, out_stream=_DEFAULT_OUT_STREAM):
        """
        List model options for the specified run.

        Parameters
        ----------
        run_number : int
            Run_driver or run_model iteration to inspect
        system : str or None
            Pathname of system (None for all systems)
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        dict
            {system: {key: val}}
        """
        pass

    def list_solver_options(self, run_number=0, solver=None, out_stream=_DEFAULT_OUT_STREAM):
        """
        List solver options for the specified run.

        Parameters
        ----------
        run_number : int
            Run_driver or run_model iteration to inspect
        solver : str or None
            Pathname of solver (None for all solvers)
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        dict
            {solver: {key: val}}
        """
        pass
