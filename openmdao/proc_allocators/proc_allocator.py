"""Define the base ProcAllocator class."""
from __future__ import division
import numpy as np
from six.moves import range

from openmdao.utils.options_dictionary import OptionsDictionary


class ProcAllocator(object):
    """
    Algorithm for allocating processors to a given system's subsystems.

    Attributes
    ----------
    parallel : boolean
        True means the comm is split across subsystems;
        False means the comm is passed to all subsystems.
    """

    def __init__(self, parallel=False):
        """
        Initialize all attributes.

        Parameters
        ----------
        parallel : bool
            If True, split subsystem comm.
        """
        self.parallel = parallel

    def __call__(self, proc_info, nsubs, comm):
        """
        Perform the allocation if parallel.

        Parameters
        ----------
        proc_info : list of (min_procs, max_procs, weight)
            Information used to determine MPI process allocation to subsystems.
        comm : MPI.Comm or <FakeComm>
            communicator of the owning system.

        Returns
        -------
        isubs : [int, ...]
            indices of the owned local subsystems.
        sub_comm : MPI.Comm or <FakeComm>
            communicator to pass to the subsystems.
        sub_proc_range : (int, int)
            The range of processors that the subcomm owns, among those of comm.
        """
        if self.parallel and comm.size > 1:
            # This is a parallel group
            return self._divide_procs(proc_info, comm)
        else:
            # This is a serial group - all procs get all subsystems
            return list(range(nsubs)), comm, (0, comm.size)

    def _divide_procs(self, proc_info, comm):
        """
        Perform the parallel processor allocation.

        Parameters
        ----------
        proc_info : list of (min_procs, max_procs, weight)
            Information used to determine MPI process allocation to subsystems.
        comm : MPI.Comm or <FakeComm>
            communicator of the owning system.

        Returns
        -------
        isubs : [int, ...]
            indices of the owned local subsystems.
        sub_comm : MPI.Comm or <FakeComm>
            communicator to pass to the subsystems.
        """
        pass
