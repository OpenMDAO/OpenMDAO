"""Define the DefaultAllocator class."""
import warnings

import numpy as np

from openmdao.proc_allocators.proc_allocator import ProcAllocator, ProcAllocationError
from openmdao.utils.mpi import MPI


class DefaultAllocator(ProcAllocator):
    """
    Default processor allocator.
    """

    def _divide_procs(self, proc_info, comm):
        """
        Perform the parallel processor allocation.

        Parameters
        ----------
        proc_info : list of (min_procs, max_procs, weight)
            Information used to determine MPI process allocation to subsystems.
        comm : MPI.Comm or <FakeComm>
            communicator of the owning System.

        Returns
        -------
        isubs : [int, ...]
            indices of the owned local subsystems.
        sub_comm : MPI.Comm or <FakeComm>
            communicator to pass to the subsystems.
        sub_proc_range : (int, int)
            The range of processors that the subcomm owns, among those of comm.
        """
        iproc = comm.rank
        nproc = comm.size

        nsubs = len(proc_info)
        min_procs, max_procs, proc_weights = self._split_proc_info(proc_info, comm)
        min_sum = np.sum(min_procs)

        if np.sum(max_procs) < nproc:
            raise ProcAllocationError("too many MPI procs allocated. Comm is size %d but "
                                      "can only use %d." % (nproc, np.sum(max_procs)))
        if min_sum > nproc and np.any(min_procs > 1):
            raise ProcAllocationError("can't meet min_procs required because the sum of the "
                                      "min procs required exceeds the procs allocated and the "
                                      "min procs required is > 1",
                                      np.array(list(range(nsubs)))[min_procs > 1])

        # Define the normalized weights for all subsystems
        proc_weights /= np.sum(proc_weights)

        if min_sum > nproc:
            isubs_list = [[] for ind in range(nproc)]
            proc_load = np.zeros(nproc)

            sub_sort_idxs = np.flipud(np.argsort(proc_weights))
            vals = proc_weights

            # Assign the slowest subsystem to the most free processor
            for isub in sub_sort_idxs:
                min_loads = np.argsort(proc_load)
                for i in range(min_procs[isub]):
                    iproc1 = min_loads[i]
                    isubs_list[iproc1].append(isub)
                    proc_load[iproc1] += vals[isub]

            # Result
            sub_comm = comm.Split(iproc)
            return sorted(isubs_list[iproc]), sub_comm, [comm.rank, comm.rank + sub_comm.size]

        num_procs = min_procs.copy()

        if min_sum < nproc:
            # weighted sums to nproc
            weighted = proc_weights * nproc

            # the number of procs expected beyond the min requested
            weighted_less_min = weighted.astype(int) - min_procs
            weighted_less_min[weighted_less_min < 0] = 0

            if np.sum(weighted_less_min) + min_sum <= nproc:
                # start with min procs then add what's left over using weights
                num_procs += weighted_less_min

        excess_idxs = (max_procs - num_procs) < 0

        # limit all procs to their stated max
        num_procs[excess_idxs] = max_procs[excess_idxs]

        expected_total = np.sum(num_procs)
        extras = nproc - expected_total

        if extras > 0:  # we have some extra procs lying around.
            # give remaining procs such that after each addition we are closest to
            # desired weights
            newsum = expected_total
            eye = np.eye(weighted.size)
            weighted[:] = proc_weights
            for i in range(extras):
                mask = max_procs <= num_procs
                weighted[mask] = 0.0
                weighted *= (1. / np.sum(weighted))
                newsum += 1
                mat = eye + num_procs
                mat *= (1. / newsum)
                mat -= weighted
                # prevent rows associated with the maxed out subsystems from having the
                # smallest norm.
                mat[mask] = 1e99
                # zero out columns for maxed out subsystems
                mat[:, mask] = 0.0
                norm = np.linalg.norm(mat, axis=1)
                # add a proc to a subsystem based on matching closest to desired weights for
                # the remaining 'active' subsystems.
                num_procs[np.argmin(norm)] += 1

        # Compute the coloring
        color = np.zeros(nproc, int)
        start, end = 0, 0
        for isub in range(nsubs):
            end += num_procs[isub]
            color[start:end] = isub
            start += num_procs[isub]

        isub = color[iproc]

        # Result
        isubs = [isub]
        sub_comm = comm.Split(isub)
        start = list(color).index(isub)  # find lowest matching color
        sub_proc_range = [start, start + sub_comm.size]

        return isubs, sub_comm, sub_proc_range
