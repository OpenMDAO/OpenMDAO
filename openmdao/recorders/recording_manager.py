"""
RecordingManager class definition.
"""
import time


class RecordingManager(object):
    """
    Object that routes function calls to all attached recorders.
    """

    def __init__(self):
        """
        init.
        """
        self._recorders = []
        self.rank = 0

    def __getitem__(self, index):
        """
        Get stuff.
        """
        return self._recorders[index]

    def __iter__(self):
        """
        Iterate over stuff.

        Returns
        -------
        iter:
            a recorder
        """
        return iter(self._recorders)

    def append(self, recorder):
        """
        Add a recorder for recording.

        Args
        ----
        recorder : `BaseRecorder`
           Recorder instance.
        """
        self._recorders.append(recorder)

    def startup(self):
        """
        Initialization during setup.
        """
        for recorder in self._recorders:
            recorder.startup()

    def close(self):
        """
        Close all recorders.
        """
        for recorder in self._recorders:
            recorder.close()

    def record_iteration(self, object_requesting_recording, metadata):
        """
        Gather variables for non-parallel case recorders and calls record for all recorders.

        Args
        ----
        root : `System`
           System containing variables.

        metadata : dict
            Metadata for iteration coordinate
        """
        if not self._recorders:
            return

        if metadata is not None:
            metadata['timestamp'] = time.time()

        for recorder in self._recorders:
            recorder.record_iteration(object_requesting_recording, metadata)
