from __future__ import division
import numpy



class ProcAllocator(object):

    def __init__(self, parallel=False, **kwargs):
        self._parallel = parallel
        self._kwargs = kwargs

    def __call__(self, nsub, comm, proc_range):
        """ Assigns subsystems and a sub-comm to the current processor """
        # This is a serial group - all procs get all subsystems
        if not self._parallel or comm.size == 1:
            isubs = range(nsub)
            sub_comm = comm
            sub_proc_range = [proc_range[0], proc_range[1]]
            return isubs, sub_comm, sub_proc_range
        # This is a parallel group
        else:
            return self._divide_procs(nsub, comm, proc_range)



class DefaultProcAllocator(ProcAllocator):

    def _divide_procs(self, nsub, comm, proc_range):
        iproc = comm.rank
        nproc = comm.size

        # TODO: improve this algorithm - maybe Fortran/C
        # TODO: maybe combine the 2 cases below
        if nproc >= nsub:
            # Define the normalized weights for all subsystems
            kwargs = self.kwargs
            if 'weights' in kwargs and len(kwargs['weights']) == nsub:
                weights = kwargs['weights']
                weights = 1.0 * weights / numpy.sum(weights)
            else:
                weights = numpy.ones(nsub) / nsub

            # Next-one-up algorithm to assign procs to subsystems
            num_procs = numpy.ones(nsub, int)
            pctg_procs = numpy.zeros(nsub)
            for ind in xrange(nproc - nsub):
                pctg_procs[:] = 1.0 * num_procs / numpy.sum(num_procs)
                num_procs[numpy.argmax(weights - pctg_procs)] += 1

            # Compute the coloring
            color = numpy.zeros(nproc, int)
            start, end = 0, 0
            for isub in xrange(nsub):
                end += num_procs[isub]
                color[start:end] = isub
                start += num_procs[isub]

            isub = color[iproc]
            iproc1 = proc_range[0] + numpy.sum(num_procs[:isub])
            iproc2 = proc_range[0] + numpy.sum(num_procs[:isub+1])
            # Result
            isubs = [isub]
            sub_comm = comm.Split(isub)
            sub_proc_range = [iproc1, iproc2]
        else:
            # TODO: improve this algorithm - maybe Fortran/C
            bool_unused_sub = numpy.ones(nsub, bool)
            isubs_list = [[] for ind in xrange(nproc)]
            proc_load = numpy.zeros(nproc)
            # Assign the slowest subsystem to the most free processor
            for ind in xrange(nsub):
                iproc = numpy.argmin(proc_load)
                isub = numpy.argmax(weights[bool_unused_sub])

                bool_unused_sub[isub] = False
                isubs_list[iproc].append(isub)
                proc_load[iproc] += weights[isub]

            iproc1 = proc_range[0] + iproc
            iproc2 = proc_range[0] + iproc + 1
            # Result
            isubs = isubs_list[iproc]
            sub_comm = comm.Split(iproc)
            sub_proc_range = [iproc1, iproc2]

        return isubs, sub_comm, sub_proc_range
