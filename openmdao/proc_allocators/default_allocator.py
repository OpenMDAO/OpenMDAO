"""Define the DefaultAllocator class."""
from __future__ import division

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

        # Define the normalized weights for all subsystems
        proc_weights /= np.sum(proc_weights)

        # prod sums to nproc
        prod = proc_weights * nproc

        # scale so smallest weight is 1.0
        expected = prod * (1.0 / prod[np.argmin(prod)])

        min_needed = np.array([minp for minp, _, _ in proc_info], dtype=int)

        if nproc >= nsubs:

            if np.any(prod < 1.):
                # start everybody with their min procs
                num_procs = np.array([minp for minp, _, _ in proc_info], dtype=int)
            else:
                # give everybody what they asked for, except for any fractional parts
                num_procs = np.array(np.trunc(prod), int)

            left = nproc - np.sum(num_procs)

            # give remaining procs to whoever has largest diff between what they want and
            # what they have
            for i in range(left):
                diff = expected - num_procs
                num_procs[np.argmax(diff)] += 1

            print("    num_procs:", num_procs)

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

            print(isubs_list)

            # Result
            isubs = isubs_list[iproc]
            sub_comm = comm.Split(iproc)
            sub_proc_range = [comm.rank, comm.rank + sub_comm.size]

        return isubs, sub_comm, sub_proc_range
