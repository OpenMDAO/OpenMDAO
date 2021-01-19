"""
RecordingManager class definition.
"""
import time

from openmdao.utils.general_utils import simple_warning

try:
    from openmdao.utils.mpi import MPI
except ImportError:
    MPI = None


class RecordingManager(object):
    """
    Object that routes function calls to all attached recorders.

    Attributes
    ----------
    _recorders : list of CaseRecorder
        All of the recorders attached to the current object.
    rank : int
        Rank of the iteration coordinate.
    _has_serial_recorders: bool
        True if any of the recorders managed by this object are serial recorders.
    """

    def __init__(self):
        """
        init.
        """
        self._recorders = []
        self._has_serial_recorders = False

        if MPI:
            self.rank = MPI.COMM_WORLD.rank
        else:
            self.rank = 0

    def __getitem__(self, index):
        """
        Get a particular recorder in the manager.

        Parameters
        ----------
        index : int
            an index into _recorders.

        Returns
        -------
        recorder : CaseRecorder
            a recorder from _recorders
        """
        return self._recorders[index]

    def __iter__(self):
        """
        Iterate.

        Returns
        -------
        iter : CaseRecorder
            a recorder from _recorders.
        """
        return iter(self._recorders)

    def append(self, recorder):
        """
        Add a recorder for recording.

        Parameters
        ----------
        recorder : CaseRecorder
           Recorder instance to be added to the manager.
        """
        self._recorders.append(recorder)

    def startup(self, recording_requester):
        """
        Run startup on each recorder in the manager.

        Parameters
        ----------
        recording_requester : object
            The object that needs an iteration of itself recorded.
        """
        # Will only add parallel code for Drivers. Use the old method for System and Solver
        from openmdao.core.driver import Driver
        if not isinstance(recording_requester, Driver):
            for recorder in self._recorders:
                recorder.startup(recording_requester)
            return

        # The remaining code only works for recording of Drivers
        model = recording_requester._problem().model
        if MPI:
            # TODO Eventually, we think we can get rid of this next check. But to be safe,
            #       we are leaving it in there.
            if not model.is_active():
                raise RuntimeError("RecordingManager.startup should never be called when "
                                   "running in parallel on an inactive System")

        for recorder in self._recorders:
            # Each of the recorders determines its self._filtered_* list of vars
            #   to record
            recorder.startup(recording_requester)

            if not recorder._parallel:
                self._has_serial_recorders = True

    def shutdown(self):
        """
        Shut down and remove all recorders.
        """
        for recorder in self._recorders:
            recorder.shutdown()
        self._recorders = []

    def record_iteration(self, recording_requester, data, metadata):
        """
        Call record_iteration on all recorders.

        Parameters
        ----------
        recording_requester : object
            The object that needs an iteration of itself recorded.
        data : dict
            Dictionary containing desvars, objectives, constraints, responses, and System vars.
        metadata : dict
            Metadata for iteration coordinate.
        """
        if not self._recorders:
            return

        if metadata is not None:
            metadata['timestamp'] = time.time()

        for recorder in self._recorders:
            if recorder._parallel or MPI is None or self.rank == 0:
                recorder.record_iteration(recording_requester, data, metadata)

    def record_metadata(self, recording_requester):
        """
        Call record_metadata for all recorders.

        Parameters
        ----------
        recording_requester : object
            The object that needs its metadata recorded.

        """
        for recorder in self._recorders:
            # If the recorder does not support parallel recording
            # we need to make sure we only record on rank 0.
            if recorder._parallel or self.rank == 0:
                recorder.record_metadata(recording_requester)

    def record_derivatives(self, recording_requester, data, metadata):
        """
        Call record_derivatives on all recorders.

        Parameters
        ----------
        recording_requester : object
            The object that needs an iteration of itself recorded.
        data : dict
            Dictionary containing derivatives keyed by 'of,wrt' to be recorded.
        metadata : dict
            Metadata for iteration coordinate.
        """
        if not self._recorders:
            return

        if metadata is not None:
            metadata['timestamp'] = time.time()

        for recorder in self._recorders:
            if recorder._parallel or MPI is None or self.rank == 0:
                recorder.record_derivatives(recording_requester, data, metadata)

    def has_recorders(self):
        """
        Are there any recorders managed by this RecordingManager.

        Returns
        -------
        True/False : bool
            True if RecordingManager is managing at least one recorder
        """
        return True if self._recorders else False

    def _check_parallel(self):
        pset = {bool(r._parallel) for r in self._recorders}

        # check to make sure we don't have mixed parallel/non-parallel, because that
        # currently won't work properly.
        if len(pset) > 1:
            raise RuntimeError("OpenMDAO currently does not support a mixture of parallel "
                               "and non-parallel recorders.")
        return pset.pop()


def _get_all_requesters(problem):
    yield problem
    yield problem.driver
    for system in problem.model.system_iter(include_self=True, recurse=True):
        yield system
        nl = system._nonlinear_solver
        if nl:
            yield nl
            if hasattr(nl, 'linesearch') and nl.linesearch:
                yield nl.linesearch


def _get_all_viewer_data_recorders(problem):
    for req in _get_all_requesters(problem):
        for r in req._rec_mgr._recorders:
            if r._record_viewer_data:
                yield r


def _get_all_recorders(problem):
    for req in _get_all_requesters(problem):
        for r in req._rec_mgr._recorders:
            yield r


def record_viewer_data(problem):
    """
    Record model viewer data for all recorders that have that option enabled.

    We don't want to collect the viewer data if it's not needed though,
    so first we'll find all recorders that need the data (if any) and
    then record it for those recorders.

    Parameters
    ----------
    problem : Problem
        The problem for which model viewer data is to be recorded.
    """
    # get all recorders that need to record the viewer data
    recorders = set(_get_all_viewer_data_recorders(problem))

    # if any recorders were found, get the viewer data and record it
    if recorders:
        from openmdao.visualization.n2_viewer.n2_viewer import _get_viewer_data
        viewer_data = _get_viewer_data(problem)
        viewer_data.pop('abs2prom', None)  # abs2prom already recorded in metadata table
        for recorder in recorders:
            recorder.record_viewer_data(viewer_data)


def record_system_options(problem):
    """
    Record the system options for all systems in the model.

    Parameters
    ----------
    problem : Problem
        The problem for which all its systems' options are to be recorded.
    """
    # get all recorders in the problem
    recorders = set(_get_all_recorders(problem))
    if recorders:
        if problem._system_options_recorded:
            simple_warning("The model is being run again, if the options or scaling of any "
                           "components has changed then only their new values will be recorded.")
        else:
            problem._system_options_recorded = True

        for recorder in recorders:
            for sub in problem.model.system_iter(recurse=True, include_self=True):
                if problem._run_counter >= 1:
                    recorder.record_metadata_system(sub, problem._run_counter)
                else:
                    recorder.record_metadata_system(sub)
