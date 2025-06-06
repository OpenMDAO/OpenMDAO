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


# default pickle protocol version for serialization
PICKLE_VER = 4


class CaseRecorder(object):
    """
    Base class for all case recorders and is not a functioning case recorder on its own.

    Parameters
    ----------
    record_viewer_data : bool, optional
        If True, record data needed for visualization.

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
    _abs_error : float
        Solver abs_error value, to be used by a derived recorder.
    _rel_error : float
        Solver abs_error value, to be used by a derived recorder.
    _iteration_coordinate : str
        The unique iteration coordinate of where an iteration originates.
    _parallel : bool
        Flag indicating if this recorder will record on multiple processes.
    _do_gather : bool
        Flag indicating if this recorder will gather data from all ranks in the requestor's comm.
    _record_on_proc : bool or None
        Flag indicating if this recorder will record on the current process (None if unspecified).
    _recording_ranks : list
        List of ranks on which this recorder will record if running under MPI.
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

        # By default, this is False, but it will be set to True if the recorder
        # will record data on multiple processes
        self._parallel = False

        # gather variables from all ranks in the requestor's comm if necessary
        self._do_gather = False

        # Flag indicating if recording will be performed on the current process.
        # If the value is not set to True on any process (the default), then
        # recording will be performed only on rank 0.
        # If the value is set to True on any process, then the _parallel flag
        # will be set and recording will occur on all processes for which the
        # value is True.
        self._record_on_proc = None

        # List of ranks on which this recorder will record if running under MPI.
        # Only used when running under MPI with communicator size greater than one.
        self._recording_ranks = None

    @property
    def record_on_process(self):
        """
        Determine if recording should be performed on this process.
        """
        return self._record_on_proc

    @record_on_process.setter
    def record_on_process(self, record):
        """
        Specify that recording should be performed on this process.

        Parameters
        ----------
        record : bool
            If True, then recording should be performed on this process.
        """
        self._record_on_proc = record

    @property
    def parallel(self):
        """
        Return True if this recorder is recording on multiple processes.
        """
        return self._parallel

    def startup(self, recording_requester, comm=None):
        """
        Prepare for a new run.

        Parameters
        ----------
        recording_requester : object
            Object to which this recorder is attached.
        comm : MPI.Comm or <FakeComm> or None
            The MPI communicator for the recorder (should be the comm for the Problem).
        """
        self._counter = 0

        if MPI and comm and comm.size > 1:
            record_on_ranks = comm.allgather(self._record_on_proc)
            recording_ranks = [rnk for rnk, rec in enumerate(record_on_ranks) if rec]
            if recording_ranks:
                # recording ranks have been specified
                self._recording_ranks = recording_ranks
                self._parallel = len(recording_ranks) > 1
            else:
                # default to just record on rank 0
                self._record_on_proc = comm.rank == 0
                self._recording_ranks = [0]

            self._do_gather = len(recording_ranks) < comm.size

    def _get_metadata_system(self, system):
        """
        Get system metadata.

        Parameters
        ----------
        system : System
            The System for which to record metadata.

        Returns
        -------
        dict
            dictionary of scaling vectors
        OptionsDictionary
            dictionary with recordable options for system
        """
        # Cannot handle PETScVector yet
        from openmdao.api import PETScVector
        if PETScVector and isinstance(system._outputs, PETScVector):
            return None, None  # Cannot handle PETScVector yet

        # collect scaling arrays
        scaling_vecs = {}
        for kind, odict in system._vectors.items():
            scaling_vecs[kind] = scaling = {}
            for vecname, vec in odict.items():
                if vec._scaling:
                    scaler, adder = vec._scaling
                    scaling[vecname] = (adder, scaler)  # keep old order for backwards compatibility
                else:
                    scaling[vecname] = None

        # create a copy of the system's metadata excluding what is in 'options_excludes'
        excludes = system.recording_options['options_excludes']

        if excludes:
            user_options = OptionsDictionary()
            user_options._all_recordable = system.options._all_recordable
            for key in system.options._dict:
                if check_path(key, [], excludes, True):
                    user_options._dict[key] = system.options._dict[key]
            user_options._read_only = system.options._read_only

            return scaling_vecs, user_options
        else:
            return scaling_vecs, system.options

    def record_metadata_system(self, system, run_number=None):
        """
        Record system metadata.

        Parameters
        ----------
        system : System
            The System for which to record metadata.
        run_number : int or None
            Number indicating which run the metadata is associated with.
            None for the first run, 1 for the second, etc.
        """
        raise NotImplementedError()

    def record_metadata_solver(self, solver, run_number=None):
        """
        Record solver metadata.

        Parameters
        ----------
        solver : Solver
            The Solver for which to record metadata.
        run_number : int or None
            Number indicating which run the metadata is associated with.
            None for the first run, 1 for the second, etc.
        """
        raise NotImplementedError()

    def record_iteration(self, recording_requester, data, metadata, **kwargs):
        """
        Route the record_iteration call to the proper method.

        Parameters
        ----------
        recording_requester : object
            System, Solver, Driver in need of recording.
        data : dict
            Dictionary containing desvars, objectives, constraints, responses, and System vars.
        metadata : dict, optional
            Dictionary containing execution metadata.
        **kwargs : keyword args
            Some implementations of record_iteration need additional args.
        """
        if not self._parallel or self._record_on_proc:
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
