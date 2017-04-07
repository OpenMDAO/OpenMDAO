""" RecordingManager class definition. """

import sys
import os
import itertools
import time
import traceback

from six import iteritems


class RecordingManager(object):
    """ Object that routes function calls to all attached recorders. """

    def __init__(self):
        self._recorders = []
        self.rank = 0

    def append(self, recorder):
        """ Add a recorder for recording.

        Args
        ----
        recorder : `BaseRecorder`
           Recorder instance.
        """
        self._recorders.append(recorder)

    def __getitem__(self, index):
        return self._recorders[index]

    def __iter__(self):
        return iter(self._recorders)



    def startup(self):
        """ Initialization during setup.

        Args
        ----
        root : `System`
           System containing variables.
        """

        for recorder in self._recorders:
            recorder.startup()



    def close(self):
        """ Close all recorders. """
        for recorder in self._recorders:
            recorder.close()

    def record_metadata(self, root):
        """ Record metadata for all variables of interest.

        Args
        ----
        root : `System`
           System containing variables.
        """

        for recorder in self._recorders:
            # If the recorder does not support parallel recording
            # we need to make sure we only record on rank 0.
            if recorder._parallel or self.rank == 0:
                if recorder.options['record_metadata']:
                    recorder.record_metadata(root)


    def record_iteration(self, object_requesting_recording, metadata):
        """ Gathers variables for non-parallel case recorders and calls
        record for all recorders.

        Args
        ----
        root : `System`
           System containing variables.

        metadata : dict
            Metadata for iteration coordinate

        dummy : bool, optional
            If True, this is a dummy iteration, so no data will be colllected
            from the model, but collective gather call will still be made.
        """
        if not self._recorders:
            return

        if metadata is not None:
            metadata['timestamp'] = time.time()

        for recorder in self._recorders:
            recorder.record_iteration(object_requesting_recording, metadata)

