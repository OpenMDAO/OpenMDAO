"""Define the base ProcAllocator class."""
import numpy as np


class ProcAllocationError(Exception):
    """
    Exception containing subsystem index information for use at higher levels.

    Attributes
    ----------
    msg : str
        The message string.
    sub_inds : list of int
        Indices of subsystems in _subsystems_allprocs in parent.
    """

    def __init__(self, msg, sub_inds=None):
        """
        Initialize all attributes.

        Parameters
        ----------
        msg : str
            The message string.
        sub_inds : list of int
            Indices of subsystems in _subsystems_allprocs in parent.
        """
        super().__init__(msg)
        self.msg = msg
        self.sub_inds = sub_inds


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
        nsubs : int
            Number of subsystems.
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
            nproc = comm.size
            min_procs, max_procs, _ = self._split_proc_info(proc_info, comm)

            if np.any(max_procs < nproc):
                raise ProcAllocationError("too many MPI procs allocated (%d)" % nproc,
                                          np.array(list(range(nsubs)))[max_procs < nproc])
            if np.any(min_procs > nproc):
                raise ProcAllocationError("can't meet min_procs required",
                                          np.array(list(range(nsubs)))[min_procs > nproc])

            # This is a serial group - all procs get all subsystems
            return list(range(nsubs)), comm, (0, comm.size)

    def _split_proc_info(self, proc_info, comm):
        """
        Split proc_info into min_procs, max_procs, and weights.

        Parameters
        ----------
        proc_info : list of (min_procs, max_procs, weight)
            Information used to determine MPI process allocation to subsystems.
        comm : MPI.Comm or <FakeComm>
            communicator of the owning system.

        Returns
        -------
        list of int
            Min procs required for each subsystem.
        list of int
            Max procs required for each subsystem.
        list of float
            Weights for each subsystem.
        """
        nproc = comm.size

        min_procs = np.array([minp for minp, _, _ in proc_info], dtype=int)
        # if max_procs entry is None or > nproc, it just becomes nproc
        max_procs = np.array([nproc if maxp is None or maxp > nproc else
                              maxp for _, maxp, _ in proc_info], dtype=int)
        weights = np.array([weight for _, _, weight in proc_info])

        return min_procs, max_procs, weights

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
