"""Define the DefaultAllocator class."""
from __future__ import division, print_function

import warnings

import numpy as np
from six.moves import range

from openmdao.proc_allocators.proc_allocator import ProcAllocator
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

        proc_weights = np.array([weight for _, _, weight in proc_info])
        min_procs = np.array([minp for minp, _, _ in proc_info], dtype=int)
        # if max_procs entry is None or > nproc, it just becomes nproc
        max_procs = np.array([nproc if maxp is None or maxp > nproc else
                              maxp for _, maxp, _ in proc_info], dtype=int)

        if np.sum(max_procs) < nproc:
            raise RuntimeError("Too many MPI procs allocated. Comm is size %d but "
                               "can only use %d" % (nproc, np.sum(max_procs)))

        # Define the normalized weights for all subsystems
        proc_weights /= np.sum(proc_weights)

        if nproc >= nsubs:
            if np.sum(min_procs) > nproc:
                raise RuntimeError("can't meet min_procs required")

            num_procs = min_procs.copy()

            if np.sum(num_procs) < nproc:
                # weighted sums to nproc
                weighted = proc_weights * nproc

                # the number of procs expected beyond the min requested
                weighted_less_min = weighted.astype(int) - min_procs
                weighted_less_min[weighted_less_min < 0] = 0

                # start with min procs then add what's left over using weights
                num_procs += weighted_less_min

            excess_idxs = (max_procs - num_procs) < 0

            # limit all procs to their stated max
            num_procs[excess_idxs] = max_procs[excess_idxs]

            expected_total = np.sum(num_procs)
            extras = nproc - expected_total

            if expected_total <= nproc:
                if extras > 0:  # we have some extra procs lying around.
                    # give remaining procs such that after each addition we are closest to
                    # desired weights
                    newsum = expected_total
                    eye = np.eye(weighted.size)
                    for i in range(extras):
                        exmask = max_procs <= num_procs
                        incmask = max_procs > num_procs
                        partial_weights = proc_weights[incmask]
                        partial_weights *= (1. / np.sum(partial_weights))
                        newsum += 1
                        mat = eye + num_procs
                        mat *= (1. / newsum)
                        weighted[incmask] = partial_weights
                        mat -= weighted
                        mat[exmask] = 1e99
                        mat[:, exmask] = 0.0
                        norm = np.linalg.norm(mat, axis=1)
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

        else:  # number of procs < number of subsystems
            isubs_list = [[] for ind in range(nproc)]
            proc_load = np.zeros(nproc)
            weights = proc_weights.copy()

            # Assign the slowest subsystem to the most free processor
            for ind in range(nsubs):
                iproc1 = np.argmin(proc_load)
                isub = np.argmax(weights)
                isubs_list[iproc1].append(isub)
                proc_load[iproc1] += weights[isub]
                weights[isub] = -1.  # mark negative so argmax won't pick it

            # Result
            isubs = isubs_list[iproc]
            sub_comm = comm.Split(iproc)
            sub_proc_range = [comm.rank, comm.rank + sub_comm.size]

        return isubs, sub_comm, sub_proc_range
