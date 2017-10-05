"""Define the DefaultAllocator class."""
from __future__ import division

import warnings

import numpy as np
from six.moves import range

from openmdao.proc_allocators.proc_allocator import ProcAllocator, ProcAllocationError
from openmdao.utils.mpi import MPI


class DefaultAllocator(ProcAllocator):
    """
    Default processor allocator.
    """

    def _divide_procs(self, proc_weights, nsubs, comm):
        """
        Perform the parallel processor allocation.

        Parameters
        ----------
        proc_weights : list of float
            list of proc weight for each subsystem.
        nsubs : int
            Number of subsystems of the owning System.
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

        if proc_weights is None:
            proc_weights = np.ones(nsubs) / nsubs

        # TODO: improve this algorithm - maybe Fortran/C
        # TODO: maybe combine the 2 cases below
        if nproc >= nsubs:
            # Define the normalized weights for all subsystems
            if len(proc_weights) == nsubs:
                proc_weights = np.atleast_1d(proc_weights)
                proc_weights /= np.sum(proc_weights)
            else:
                raise RuntimeError("length of proc_weights (%d) does not match the number of "
                                   "subsystems (%d)" % (len(proc_weights), nsubs))

            # Next-one-up algorithm to assign procs to subsystems
            num_procs = np.ones(nsubs, int)
            pctg_procs = np.zeros(nsubs)
            for ind in range(nproc - nsubs):
                pctg_procs[:] = num_procs / np.sum(num_procs)
                num_procs[np.argmax(proc_weights - pctg_procs)] += 1

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
        else:
            # TODO: improve this algorithm - maybe Fortran/C
            bool_unused_sub = np.ones(nsubs, bool)
            isubs_list = [[] for ind in range(nproc)]
            proc_load = np.zeros(nproc)
            # Assign the slowest subsystem to the most free processor
            for ind in range(nsubs):
                iproc1 = np.argmin(proc_load)
                isub = np.argmax(proc_weights[bool_unused_sub])

                bool_unused_sub[isub] = False
                isubs_list[iproc1].append(isub)
                proc_load[iproc1] += proc_weights[isub]

            # iproc1 = proc_range[0] + iproc
            # iproc2 = proc_range[0] + iproc + 1
            # Result
            isubs = isubs_list[iproc]
            sub_comm = comm.Split(iproc)
            sub_proc_range = [comm.rank, comm.rank + sub_comm.size]

        # # a 'color' is assigned to each bucket, with
        # # an entry for each processor it will be given
        # # e.g. [0, 1, 1, 1, 1, 2, 2, 3, 3, 3, UND, UND]
        # color = np.full(nproc, MPI.UNDEFINED, dtype=int)
        # comm_sizes = np.empty(nproc, int)
        # start, end = 0, 0
        # for i, b in enumerate(buckets):
        #     num_procs = b[0]
        #     end += num_procs
        #     color[start:end] = i
        #     comm_sizes[start:end] = num_procs
        #     start += num_procs
        #
        # # create a sub-communicator for each color and
        # # get the one assigned to our color/process
        # rank_color = color[iproc]
        # sub_comm = comm.Split(rank_color)
        # sub_proc_range = (np.sum(comm_sizes[:iproc]), np.sum(comm_sizes[:iproc + 1]))
        #
        # isubs = [] if sub_comm == MPI.COMM_NULL else buckets[rank_color][1]

        return isubs, sub_comm, sub_proc_range
