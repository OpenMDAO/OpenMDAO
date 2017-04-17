"""Define the base ProcAllocator class."""
from __future__ import division
import numpy as np
from six.moves import range

from openmdao.utils.generalized_dict import OptionsDictionary


class ProcAllocationError(Exception):
    """
    Exception raised when processor allocation fails.

    Attributes
    ----------
    sub_idx : int
        Index into the parent's _subsystems_allprocs list.
    requested : int
        Number of processes requested by the indexed subsystem.
    remaining : int
        Number of processes available to the indexed subsystem.
    """

    def __init__(self, sub_idx, requested, remaining):
        """
        Initialize all attributes.

        Parameters
        ----------
        sub_idx : int
            Index into the parent's _subsystems_allprocs list.
        requested : int
            Number of processes requested by the indexed subsystem.
        remaining : int
            Number of processes available to the indexed subsystem.
        """
        super(ProcAllocationError, self).__init__('')
        self.sub_idx = sub_idx
        self.requested = requested
        self.remaining = remaining


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

    def __call__(self, req_procs, comm):
        """
        Perform the allocation if parallel.

        Parameters
        ----------
        req_procs : list of (int, int)
            list of min/max usable procs for each subsystem.
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
            return self._divide_procs(req_procs, comm)
        else:
            # This is a serial group - all procs get all subsystems
            return list(range(len(req_procs))), comm, (0, comm.size)

    def _divide_procs(self, req_procs, comm, proc_range):
        """
        Perform the parallel processor allocation.

        Parameters
        ----------
        req_procs : list of (int, int)
            list of min/max usable procs for each subsystem.
        comm : MPI.Comm or <FakeComm>
            communicator of the owning system.
        proc_range : (int, int)
            The range of processors that the comm on this system owns, in the global index space.

        Returns
        -------
        isubs : [int, ...]
            indices of the owned local subsystems.
        sub_comm : MPI.Comm or <FakeComm>
            communicator to pass to the subsystems.
        """
        pass
