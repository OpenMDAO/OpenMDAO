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

    def _divide_procs(self, req_procs, comm):
        """
        Perform the parallel processor allocation.

        Parameters
        ----------
        req_procs : list of (int, int)
            List of min/max usable procs for each subsystem.
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
        iproc = comm.rank
        nproc = comm.size
        nsub = len(req_procs)

        min_req_procs = [minproc for minproc, _ in req_procs]
        max_req_procs = [maxproc for _, maxproc in req_procs]

        assigned_procs = np.zeros(nsub, dtype=int)

        assigned = 0

        total_req = np.sum(min_req_procs)

        if None in max_req_procs:
            limit = nproc
            max_requested = nproc
        else:
            max_requested = np.sum(max_req_procs)
            limit = min(nproc, max_requested)

        # first, just use simple round robin assignment of requested procs
        # until everybody has what they asked for or we run out
        if total_req:
            if nproc >= total_req:  # we have enough for all subsystems
                while assigned < limit:
                    for i, max_req in enumerate(max_req_procs):
                        if max_req is None or assigned_procs[i] < max_req:
                            assigned_procs[i] += 1
                            assigned += 1
                            if assigned == limit:
                                break

                # create buckets (one sub per bucket) to be consistent in how
                # we split procs below
                buckets = [(n, [i]) for i, n in enumerate(assigned_procs)]

            else:  # we don't have enough, so have to group subsystems
                remaining = nproc
                # sort req procs in descending order
                tups = sorted([(req[0], i) for i, req in enumerate(req_procs)],
                              reverse=True)
                buckets = []
                for i, (req, sub_idx) in enumerate(tups):
                    if remaining >= req:
                        buckets.append([req, [sub_idx]])
                        remaining -= req
                    elif i == 0:
                        # since we sorted in descending order by number of
                        # requested procs, only in the first iteration is there
                        # a chance that we've requested more procs than we have
                        raise ProcAllocationError(sub_idx, req, remaining)
                    else:
                        # we already have at least one in the bucket list that's
                        # big enough, so go through buckets, find all that are
                        # big enough, and add the current sub to the one with
                        # the fewest number of subs already in it. In the event
                        # of a tie, take the bucket with the lowest number of
                        # requested procs.
                        lenlist = sorted([b for b in buckets if b[0] >= req],
                                         key=lambda t: len(t[1]))
                        shortest = len(lenlist[0][1])
                        final = sorted(b for b in lenlist
                                       if len(b[1]) == shortest)
                        final[0][1].append(sub_idx)

                warnings.warn("System requested %d processes to run fully "
                              "in parallel, but it only got %d" %
                              (total_req, nproc))

                # if we have any procs left over, apply them to any sub that
                # can use them
                while remaining > 0:
                    for i, max_req in enumerate(max_req_procs):
                        procs, subs = buckets[i]
                        for sub_idx in subs:
                            if (max_req is None or max_req > procs):
                                # add 1 to procs for this bucket
                                buckets[i][0] += 1
                                remaining -= 1
                                break
                        if remaining == 0:
                            break

                assigned = nproc - remaining

        # a 'color' is assigned to each bucket, with
        # an entry for each processor it will be given
        # e.g. [0, 1, 1, 1, 1, 2, 2, 3, 3, 3, UND, UND]
        color = np.full(nproc, MPI.UNDEFINED, dtype=int)
        comm_sizes = np.empty(nproc, int)
        start, end = 0, 0
        for i, b in enumerate(buckets):
            num_procs = b[0]
            end += num_procs
            color[start:end] = i
            comm_sizes[start:end] = num_procs
            start += num_procs

        # create a sub-communicator for each color and
        # get the one assigned to our color/process
        rank_color = color[iproc]
        sub_comm = comm.Split(rank_color)
        sub_proc_range = (np.sum(comm_sizes[:iproc]), np.sum(comm_sizes[:iproc + 1]))

        isubs = [] if sub_comm == MPI.COMM_NULL else buckets[rank_color][1]

        return isubs, sub_comm, sub_proc_range
