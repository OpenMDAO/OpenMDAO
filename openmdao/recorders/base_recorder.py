"""
Class definition for BaseRecorder, the base class for all recorders.
"""
from six import StringIO

from openmdao.core.system import System
from openmdao.core.driver import Driver
from openmdao.solvers.solver import Solver
from openmdao.recorders.recording_iteration_stack import get_formatted_iteration_coordinate
from openmdao.utils.mpi import MPI


class BaseRecorder(object):
    """
    Base class for all case recorders and is not a functioning case recorder on its own.

    Options
    -------
    recording_options['record_metadata'] :  bool(True)
        Tells recorder whether to record variable attribute metadata.
    recording_options['record_outputs'] :  bool(True)
        Tells recorder whether to record the outputs of a System.
    recording_options['record_inputs'] :  bool(False)
        Tells recorder whether to record the inputs of a System.
    recording_options['record_residuals'] :  bool(False)
        Tells recorder whether to record the residuals of a System.
    recording_options['metadata_excludes']: list of strings([])
        Patterns for user-defined metadata to exclude in recording of a System.
    recording_options['record_derivatives'] :  bool(False)
        Tells recorder whether to record the derivatives of a Driver.
    recording_options['record_desvars'] :  bool(True)
        Tells recorder whether to record the desvars of a Driver.
    recording_options['record_responses'] :  bool(False)
        Tells recorder whether to record the responses of a Driver.
    recording_options['record_objectives'] :  bool(False)
        Tells recorder whether to record the objectives of a Driver.
    recording_options['record_constraints'] :  bool(False)
        Tells recorder whether to record the constraints of a Driver.
    recording_options['record_abs_error'] :  bool(True)
        Tells recorder whether to record the absolute error of a Solver.
    recording_options['record_rel_error'] :  bool(True)
        Tells recorder whether to record the relative error of a Solver.
    recording_options['record_solver_derivatives'] :  bool(False)
        Tells recorder whether to record the derivatives of a Solver.
    recording_options['includes'] :  list of strings("*")
        Patterns for variables to include in recording.
    recording_options['excludes'] :  list of strings([])
        Patterns for variables to exclude in recording (processed after includes).

    Attributes
    ----------
    out : StringIO
        Output to the recorder.
    _counter : int
        A global counter for execution order, used in iteration coordinate.
    _filtered_driver : dict
        Filtered subset of driver variables to record, based on includes/excludes.
    _filtered_system_outputs_driver_recording : dict
        Filtered subset of System outputs to record during Driver recording.
    _filtered_solver : dict
        Filtered subset of solver variables to record, based on includes/excludes.
    _filtered_system : dict
        Filtered subset of system variables to record, based on includes/excludes.
    _desvars_values : dict
        Driver desvar values, post-filtering, to be used by a derived recorder.
    _responses_values : dict
        Driver response values, post-filtering, to be used by a derived recorder.
    _objectives_values : dict
        Driver objectives values, post-filtering, to be used by a derived recorder.
    _constraints_values : dict
        Driver constraints values, post-filtering, to be used by a derived recorder.
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

    def __init__(self):
        """
        Initialize.
        """
        self.out = None

        # global counter that is used in iteration coordinate
        self._counter = 0

        # For Systems
        self._inputs = None
        self._outputs = None
        self._resids = None

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
        if isinstance(recording_requester, Driver):
            self.record_metadata_driver(recording_requester)
        elif isinstance(recording_requester, System):
            self.record_metadata_system(recording_requester)
        elif isinstance(recording_requester, Solver):
            self.record_metadata_solver(recording_requester)

    def record_metadata_driver(self, recording_requester):
        """
        Record driver metadata.

        Parameters
        ----------
        recording_requester : Driver
            The Driver that would like to record its metadata.
        """
        raise NotImplementedError()

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
            get_formatted_iteration_coordinate(rank=metadata.get('override_rank'))

        if isinstance(recording_requester, Driver):
            self.record_iteration_driver(recording_requester, data, metadata)
        elif isinstance(recording_requester, System):
            self.record_iteration_system(recording_requester, data, metadata)
        elif isinstance(recording_requester, Solver):
            self.record_iteration_solver(recording_requester, data, metadata)
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

    def close(self):
        """
        Clean up the recorder.
        """
        pass
