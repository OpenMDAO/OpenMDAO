"""Define the base ProcAllocator class."""
import numpy as np
from openmdao.core.constants import INT_DTYPE


class ProcAllocationError(Exception):
    """
    Exception containing subsystem index information for use at higher levels.

    Parameters
    ----------
    msg : str
        The message string.
    sub_inds : list of int
        Indices of subsystems in _subsystems_allprocs in parent.

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
        """
        super().__init__(msg)
        self.msg = msg
        self.sub_inds = sub_inds


class ProcAllocator(object):
    """
    Algorithm for allocating processors to a given system's subsystems.

    Parameters
    ----------
    parallel : bool
        If True, split subsystem comm.

    Attributes
    ----------
    parallel : bool
        True means the comm is split across subsystems;
        False means the comm is passed to all subsystems.
    """

    def __init__(self, parallel=False):
        """
        Initialize all attributes.
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
            min_procs, max_procs, _, _ = self._split_proc_info(proc_info, comm)

            if np.any(max_procs < nproc):
                raise ProcAllocationError("too many MPI procs allocated (%d)" % nproc,
                                          np.array(list(range(nsubs)))[max_procs < nproc])
            if np.any(min_procs > nproc):
                raise ProcAllocationError("can't meet min_procs required",
                                          np.array(list(range(nsubs)))[min_procs > nproc])

            # This is a serial group - all procs get all subsystems
            return list(range(nsubs)), comm

    def _split_proc_info(self, proc_info, comm):
        """
        Split proc_info into min_procs, max_procs, and weights.

        Parameters
        ----------
        proc_info : list of (min_procs, max_procs, weight, proc_group)
            Information used to determine MPI process allocation to subsystems.
        comm : MPI.Comm or <FakeComm>
            communicator of the owning system.

        Returns
        -------
        ndarray of int
            Min procs required for each subsystem.
        ndarray of int
            Max procs required for each subsystem.
        ndarray of float
            Weights for each subsystem.
        dict of str: list of int
            Proc groups (if any) and their corresponding proc_info indices.
        """
        nproc = comm.size

        min_procs = np.array([minp for minp, _, _, _ in proc_info], dtype=INT_DTYPE)
        # if max_procs entry is None or > nproc, it just becomes nproc
        max_procs = np.array([nproc if maxp is None or maxp > nproc else
                              maxp for _, maxp, _, _ in proc_info], dtype=INT_DTYPE)
        weights = np.array([weight for _, _, weight, _ in proc_info])

        min_sum = np.sum(min_procs)

        if self.parallel and nproc > 1:
            if np.sum(max_procs) < nproc:
                raise ProcAllocationError("too many MPI procs allocated. Comm is size %d but "
                                          "can only use %d." % (nproc, np.sum(max_procs)))
            if min_sum > nproc and np.any(min_procs > 1):
                raise ProcAllocationError("can't meet min_procs required because the sum of the "
                                          "min procs required exceeds the procs allocated and the "
                                          "min procs required is > 1",
                                          np.array(list(range(len(proc_info))))[min_procs > 1])

        gdict = {}
        for i, (_, _, _, g) in enumerate(proc_info):
            if g is None:
                continue
            if g in gdict:
                gdict[g].append(i)
            else:
                gdict[g] = [i]

        for grp, idxs in gdict.items():
            min_match, max_match, wmatch, _ = proc_info[idxs[0]]
            for i in range(1, len(idxs)):
                mn, mx, w, _ = proc_info[idxs[i]]
                if mn != min_match or mx != max_match or w != wmatch:
                    raise ProcAllocationError(f"proc_group '{grp}' members do not all have matching"
                                              " min_procs, max_procs, and/or proc_weights.")

        if gdict:
            # get reduced min_procs, max_procs, and weights based on groups, and compute index map
            mask = np.ones(len(proc_info), dtype=bool)
            gimap = {}  # maps first index of a group to all group indices
            for idxs in gdict.values():
                # keep first index and zero out the rest
                gimap[idxs[0]] = idxs
                for i in range(1, len(idxs)):
                    mask[idxs[i]] = False
            min_procs = min_procs[mask]
            max_procs = max_procs[mask]
            weights = weights[mask]
            reduced_inds = np.arange(len(proc_info), dtype=int)[mask]
            rimap = []
            for i, ridx in enumerate(reduced_inds):
                if ridx in gimap:
                    rimap.append(gimap[ridx])
                else:
                    rimap.append([ridx])
        else:
            rimap = None

        return min_procs, max_procs, weights, rimap

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
