"""
RecordingManager class definition.
"""
import time

try:
    from openmdao.utils.mpi import MPI
except ImportError:
    MPI = None


class RecordingManager(object):
    """
    Object that routes function calls to all attached recorders.

    Attributes
    ----------
    _recorders : list of BaseRecorder
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
        recorder : BaseRecorder
            a recorder from _recorders
        """
        return self._recorders[index]

    def __iter__(self):
        """
        Iterate.

        Returns
        -------
        iter : BaseRecorder
            a recorder from _recorders.
        """
        return iter(self._recorders)

    def append(self, recorder):
        """
        Add a recorder for recording.

        Parameters
        ----------
        recorder : BaseRecorder
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
        model = recording_requester._problem.model
        if MPI:
            # TODO Eventually, we think we can get rid of this next check. But to be safe,
            #       we are leaving it in there.
            if not model.is_active():
                raise RuntimeError(
                    "RecordingManager.startup should never be called when "
                    "running in parallel on an inactive System")

        for recorder in self._recorders:
            # Each of the recorders determines its self._filtered_* list of vars
            #   to record
            recorder.startup(recording_requester)

            if not recorder._parallel:
                self._has_serial_recorders = True

    def close(self):
        """
        Close all recorders in the manager.
        """
        for recorder in self._recorders:
            recorder.close()

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

    def has_recorders(self):
        """
        Are there any recorders managed by this RecordingManager.

        Returns
        -------
        True/False : bool
            True if RecordingManager is managing at least one recorder
        """
        return True if self._recorders else False
