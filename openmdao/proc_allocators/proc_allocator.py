"""Define the base ProcAllocator class."""
from __future__ import division
import numpy as np
from six.moves import range

from openmdao.utils.generalized_dict import OptionsDictionary

class ProcAllocationError(Exception):
    def __init__(self, sub_idx, requested, remaining):
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
        """
        # This is a serial group - all procs get all subsystems
        if not self.parallel or comm.size == 1:
            isubs = list(range(len(req_procs)))
            return isubs, comm
        else:  # This is a parallel group
            return self._divide_procs(req_procs, comm)

    def _divide_procs(self, req_procs, comm):
        """
        Perform the parallel processor allocation.

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
        """
        pass
