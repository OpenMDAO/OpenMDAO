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

    def _gather_vars(self, root, local_vars):
        """Gathers and returns only variables listed in
        `local_vars` from the `root` System.
        """

        # if trace:
        #     debug("gathering vars for recording in %s" % root.pathname)
        all_vars = root.comm.gather(local_vars, root=0)
        # if trace:
        #     debug("DONE gathering rec vars for %s" % root.pathname)

        if root.comm.rank == 0:
            dct = all_vars[-1]
            for d in all_vars[:-1]:
                dct.update(d)
            return dct

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
        pathname = object_requesting_recording.pathname
        if MPI and root.is_active():
            rrank = root.comm.rank
            rowned = root._owning_ranks

        if isinstance(object_requesting_recording,Driver):
            self._record_desvars = self._record_responses = False
            self._record_objectives = self._record_constraints = False


        for recorder in self._recorders:
            recorder.startup(object_requesting_recording)

            if not recorder._parallel:
                self._has_serial_recorders = True

            desvarnames = recorder._filtered[pathname]['desvars']
            responsenames = recorder._filtered[pathname]['responses']
            objectivenames = recorder._filtered[pathname]['objectives']
            constraintnames = recorder._filtered[pathname]['constraints']

            if desvarnames:
                self._record_desvars = True
            if responsenames:
                self._record_responses = True
            if objectivenames:
                self._record_objectives = True
            if constraintnames:
                self._record_constraints = True

            # now localize the lists to only
            # include local vars.  We need to do this after determining
            # if any mpi procs need to record each of params, unknowns,
            # and resids.  If none of them do, we can skip the mpi gather
            # for that group of vars.
            if MPI:
                desvarnames = [n for n in desvarnames if rrank==rowned[n]]
                responsenames = [n for n in responsenames if rrank==rowned[n]]
                objectivenames = [n for n in objectivenames if rrank==rowned[n]]
                constraintnames = [n for n in constraintnames if rrank==rowned[n]]

                # reduce the filter set for any parallel recorders to only
                # those variables that are owned by that rank
                if recorder._parallel:
                    recorder._filtered[pathname]['desvars'] = desvarnames
                    recorder._filtered[pathname]['responses'] = responsenames
                    recorder._filtered[pathname]['objectives'] = objectivenames
                    recorder._filtered[pathname]['constraints'] = constraintnames

            self._vars_to_record['desvarnames'].update(desvarnames)
            self._vars_to_record['responsenames'].update(responsenames)
            self._vars_to_record['objectivenames'].update(objectivenames)
            self._vars_to_record['constraintnames'].update(constraintnames)




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
