"""
Class definition for CaseRecorder, the base class for all recorders.
"""
from openmdao.core.system import System
from openmdao.core.driver import Driver
from openmdao.solvers.solver import Solver
from openmdao.core.problem import Problem
from openmdao.utils.mpi import MPI
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.record_util import check_path


class CaseRecorder(object):
    """
    Base class for all case recorders and is not a functioning case recorder on its own.

    Attributes
    ----------
    _record_viewer_data : bool
        Flag indicating whether to record data needed to generate N2 diagram.
    _counter : int
        A global counter for execution order, used in iteration coordinate.
    _inputs : dict
        System inputs values, post-filtering, to be used by a derived recorder.
    _outputs : dict
        System or Solver output values, post-filtering, to be used by a derived recorder.
    _resids : dict
        System or Solver residual values, post-filtering, to be used by a derived recorder.
    _abs_error : float
        Solver abs_error value, to be used by a derived recorder.
    _rel_error : float
        Solver abs_error value, to be used by a derived recorder.
    _iteration_coordinate : str
        The unique iteration coordinate of where an iteration originates.
    _parallel : bool
        Designates if the current recorder is parallel-recording-capable.
    """

    def __init__(self, record_viewer_data=True):
        """
        Initialize.

        Parameters
        ----------
        record_viewer_data : bool, optional
            If True, record data needed for visualization.
        """
        self._record_viewer_data = record_viewer_data

        # global counter that is used in iteration coordinate
        self._counter = 0

        # For Systems
        self._inputs = None
        self._outputs = None

        # For Solvers
        self._abs_error = 0.0
        self._rel_error = 0.0

        # For Drivers, Systems, and Solvers
        self._iteration_coordinate = None

        # By default, this is False, but it should be set to True
        # if the recorder will record data on each process to avoid
        # unnecessary gathering.
        self._parallel = False

    def startup(self, recording_requester):
        """
        Prepare for a new run and calculate inclusion lists.

        Parameters
        ----------
        recording_requester : object
            Object to which this recorder is attached.
        """
        self._counter = 0

    def record_metadata(self, recording_requester):
        """
        Route the record_metadata call to the proper method.

        Parameters
        ----------
        recording_requester : object
            The object that would like to record its metadata.
        """
        if isinstance(recording_requester, System):
            self.record_metadata_system(recording_requester)
        elif isinstance(recording_requester, Solver):
            self.record_metadata_solver(recording_requester)

    def _get_metadata_system(self, recording_requester):
        # Cannot handle PETScVector yet
        from openmdao.api import PETScVector
        if PETScVector and isinstance(recording_requester._outputs, PETScVector):
            return None, None  # Cannot handle PETScVector yet

        # collect scaling arrays
        scaling_vecs = {}
        for kind, odict in recording_requester._vectors.items():
            scaling_vecs[kind] = scaling = {}
            for vecname, vec in odict.items():
                scaling[vecname] = vec._scaling

        # create a copy of the system's metadata excluding what is in 'options_excludes'
        excludes = recording_requester.recording_options['options_excludes']

        if excludes:
            user_options = OptionsDictionary()
            user_options._all_recordable = recording_requester.options._all_recordable
            for key in recording_requester.options._dict:
                if check_path(key, [], excludes, True):
                    user_options._dict[key] = recording_requester.options._dict[key]
            user_options._read_only = recording_requester.options._read_only

            return scaling_vecs, user_options
        else:
            return scaling_vecs, recording_requester.options

    def record_metadata_system(self, recording_requester):
        """
        Record system metadata.

        Parameters
        ----------
        recording_requester : System
            The System that would like to record its metadata.
        """
        raise NotImplementedError()

    def record_metadata_solver(self, recording_requester):
        """
        Record solver metadata.

        Parameters
        ----------
        recording_requester : Solver
            The Solver that would like to record its metadata.
        """
        raise NotImplementedError()

    def record_iteration(self, recording_requester, data, metadata, **kwargs):
        """
        Route the record_iteration call to the proper method.

        Parameters
        ----------
        recording_requester : object
            System, Solver, Driver in need of recording.
        metadata : dict, optional
            Dictionary containing execution metadata.
        data : dict
            Dictionary containing desvars, objectives, constraints, responses, and System vars.
        **kwargs : keyword args
            Some implementations of record_iteration need additional args.
        """
        if not self._parallel:
            if MPI and MPI.COMM_WORLD.rank > 0:
                raise RuntimeError("Non-parallel recorders should not be recording on ranks > 0")

        self._counter += 1

        self._iteration_coordinate = \
            recording_requester._recording_iter.get_formatted_iteration_coordinate()

        if isinstance(recording_requester, Driver):
            self.record_iteration_driver(recording_requester, data, metadata)
        elif isinstance(recording_requester, System):
            self.record_iteration_system(recording_requester, data, metadata)
        elif isinstance(recording_requester, Solver):
            self.record_iteration_solver(recording_requester, data, metadata)
        elif isinstance(recording_requester, Problem):
            self.record_iteration_problem(recording_requester, data, metadata)
        else:
            raise ValueError("Recorders must be attached to Drivers, Systems, or Solvers.")

    def record_iteration_driver(self, recording_requester, data, metadata):
        """
        Record data and metadata from a Driver.

        Parameters
        ----------
        recording_requester : Driver
            Driver in need of recording.
        data : dict
            Dictionary containing desvars, objectives, constraints, responses, and System vars.
        metadata : dict
            Dictionary containing execution metadata.
        """
        raise NotImplementedError("record_iteration_driver has not been overridden")

    def record_iteration_system(self, recording_requester, data, metadata):
        """
        Record data and metadata from a System.

        Parameters
        ----------
        recording_requester : System
            System in need of recording.
        data : dict
            Dictionary containing inputs, outputs, and residuals.
        metadata : dict
            Dictionary containing execution metadata.
        """
        raise NotImplementedError("record_iteration_system has not been overridden")

    def record_iteration_solver(self, recording_requester, data, metadata):
        """
        Record data and metadata from a Solver.

        Parameters
        ----------
        recording_requester : Solver
            Solver in need of recording.
        data : dict
            Dictionary containing outputs, residuals, and errors.
        metadata : dict
            Dictionary containing execution metadata.
        """
        raise NotImplementedError("record_iteration_solver has not been overridden")

    def record_iteration_problem(self, recording_requester, data, metadata):
        """
        Record data and metadata from a Problem.

        Parameters
        ----------
        recording_requester : Problem
            Problem in need of recording.
        data : dict
            Dictionary containing desvars, objectives, constraints.
        metadata : dict
            Dictionary containing execution metadata.
        """
        raise NotImplementedError("record_iteration_problem has not been overridden")

    def record_derivatives(self, recording_requester, data, metadata, **kwargs):
        """
        Route the record_derivatives call to the proper method.

        Parameters
        ----------
        recording_requester : object
            System, Solver, Driver in need of recording.
        data : dict
            Dictionary containing derivatives keyed by 'of,wrt' to be recorded.
        metadata : dict
            Dictionary containing execution metadata.
        **kwargs : keyword args
            Some implementations of record_derivatives need additional args.
        """
        if not self._parallel:
            if MPI and MPI.COMM_WORLD.rank > 0:
                raise RuntimeError("Non-parallel recorders should not be recording on ranks > 0")

        self._iteration_coordinate = \
            recording_requester._recording_iter.get_formatted_iteration_coordinate()

        self.record_derivatives_driver(recording_requester, data, metadata)

    def record_derivatives_driver(self, recording_requester, data, metadata):
        """
        Record derivatives data from a Driver.

        Parameters
        ----------
        recording_requester : Driver
            Driver in need of recording.
        data : dict
            Dictionary containing derivatives keyed by 'of,wrt' to be recorded.
        metadata : dict
            Dictionary containing execution metadata.
        """
        raise NotImplementedError("record_derivatives_driver has not been overridden")

    def record_viewer_data(self, model_viewer_data):
        """
        Record model viewer data.

        Parameters
        ----------
        model_viewer_data : dict
            Data required to visualize the model.
        """
        raise NotImplementedError("record_viewer_data has not been overridden")

    def shutdown(self):
        """
        Shut down the recorder.
        """
        pass
