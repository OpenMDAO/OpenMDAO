"""Define the base ProcAllocator class."""
from __future__ import division
import numpy
from six.moves import range

from openmdao.utils.generalized_dict import GeneralizedDictionary


class ProcAllocator(object):
    """Algorithm for allocating processors to a given system's subsystems.

    Attributes
    ----------
    _parallel : boolean
        True means the comm is split across subsystems;
        False means the comm is passed to all subsystems.
    options : GeneralizedDictionary
        options dictionary.
    """

    def __init__(self, **kwargs):
        """Initialize all attributes.

        Args
        ----
        **kwargs : dict
            Contains options.
        """
        self.options = GeneralizedDictionary(kwargs)
        self.options.declare('parallel', typ=bool, value=False,
                             desc='whether to parallelize between subsystems')

    def __call__(self, nsub, comm, proc_range):
        """Perform the allocation if parallel.

        Args
        ----
        nsub : int
            Number of subsystems.
        comm : MPI.Comm or FakeComm
            communicator of the owning system.
        proc_range : [int, int]
            global processor index range.

        Returns
        -------
        isubs : [int, ...]
            indices of the owned local subsystems.
        sub_comm : MPI.Comm or FakeComm
            communicator to pass to the subsystems.
        sub_proc_range : [int, int]
            global processor index range to pass to the subsystems.
        """
        # This is a serial group - all procs get all subsystems
        if not self.options['parallel'] or comm.size == 1:
            isubs = list(range(nsub))
            sub_comm = comm
            sub_proc_range = [proc_range[0], proc_range[1]]
            return isubs, sub_comm, sub_proc_range
        # This is a parallel group
        else:
            return self._divide_procs(nsub, comm, proc_range)

    def _divide_procs(self, nsub, comm, proc_range):
        """Perform the parallel processor allocation.

        Args
        ----
        nsub : int
            Number of subsystems.
        comm : MPI.Comm or FakeComm
            communicator of the owning system.
        proc_range : [int, int]
            global processor index range.

        Returns
        -------
        isubs : [int, ...]
            indices of the owned local subsystems.
        sub_comm : MPI.Comm or FakeComm
            communicator to pass to the subsystems.
        sub_proc_range : [int, int]
            global processor index range to pass to the subsystems.
        """
        pass
