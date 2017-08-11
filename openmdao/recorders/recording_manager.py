"""
RecordingManager class definition.
"""
import time
from openmdao.recorders.recording_iteration_stack import recording_iteration_stack

from openmdao.utils.mpi import MPI


class RecordingManager(object):
    """
    Object that routes function calls to all attached recorders.

    Attributes
    ----------
    _recorders : list of <BaseRecorder>
        All of the recorders attached to the current object.
    rank : int
        Rank of the iteration coordinate.
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
        """
        return self._recorders[index]

    def __iter__(self):
        """
        Iterate.

        Returns
        -------
        iter: <BaseRecorder>
            a recorder from _recorders
        """
        return iter(self._recorders)

    def append(self, recorder):
        """
        Add a recorder for recording.

        Parameters
        ----------
        recorder : <BaseRecorder>
           Recorder instance to be added to the manager.
        """
        self._recorders.append(recorder)

    def startup(self, object_requesting_recording):
        """
        Run startup on each recorder in the manager.
        """
        for recorder in self._recorders:
            recorder.startup(object_requesting_recording)

    def close(self):
        """
        Close all recorders in the manager.
        """
        for recorder in self._recorders:
            recorder.close()

    def record_iteration(self, object_requesting_recording, metadata, **kwargs):
        """
        Call record_iteration on all recorders.

        Parameters
        ----------
        object_requesting_recording : <object>
            The object that needs an iteration of itself recorded.
        metadata : dict
            Metadata for iteration coordinate
        **kwargs :
            Keyword args needed for different versions of record_iteration
        """
        if not self._recorders:
            return

        if metadata is not None:
            metadata['timestamp'] = time.time()

        for recorder in self._recorders:
            recorder.record_iteration(object_requesting_recording, metadata, **kwargs)

    def record_metadata(self, object_requesting_recording):
        """
        Call record_metadata for all recorders.

        Parameters
        ----------
        object_requesting_recording : <object>
            The object that needs its metadata recorded.

        """
        for recorder in self._recorders:
            # If the recorder does not support parallel recording
            # we need to make sure we only record on rank 0.
            if recorder._parallel or self.rank == 0:
                recorder.record_metadata(object_requesting_recording)
