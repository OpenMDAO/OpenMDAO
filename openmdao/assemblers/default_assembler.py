"""Define the DefaultAssembler class."""
from __future__ import division
import numpy

from six.moves import range

from openmdao.assemblers.assembler import Assembler


class DefaultAssembler(Assembler):
    """Default <Assembler> implementation."""

    def _compute_transfers(self, nsub_allprocs, var_range,
                           subsystems_myproc, subsystems_inds):
        """Compute the transfer indices.

        Args
        ----
        nsub_allprocs : int
            number of subsystems on all procs.
        var_range : [int, int]
            variable index range for the current system.
        subsystems_myproc : [System, ...]
            list of subsystems on my proc.
        subsystems_inds : [int, ...]
            list of indices of subsystems on this proc among all subsystems.

        Returns
        -------
        xfer_ip_inds : dict of int ndarray[:]
            input indices of global transfer.
        xfer_op_inds : dict of int ndarray[:]
            output indices of global transfer.
        fwd_xfer_ip_inds : [dict of int ndarray[:], ...]
            list of input indices of forward transfers.
        fwd_xfer_op_inds : [dict of int ndarray[:], ...]
            list of output indices of forward transfers.
        rev_xfer_ip_inds : [dict of int ndarray[:], ...]
            list of input indices of reverse transfers.
        rev_xfer_op_inds : [dict of int ndarray[:], ...]
            list of output indices of reverse transfers.
        """
        ip_set_indices = self._variable_set_indices['input']
        op_set_indices = self._variable_set_indices['output']

        ip_ind1, ip_ind2 = var_range['input']
        op_ind1, op_ind2 = var_range['output']
        ip_isub_var = -numpy.ones(ip_ind2 - ip_ind1, int)
        op_isub_var = -numpy.ones(op_ind2 - op_ind1, int)
        for ind, subsys in enumerate(subsystems_myproc):
            isub = subsystems_inds[ind]

            sub_var_range = subsys._variable_allprocs_range
            sub_ip_ind1, sub_ip_ind2 = sub_var_range['input']
            sub_op_ind1, sub_op_ind2 = sub_var_range['output']
            for ip_ind in range(ip_ind1, ip_ind2):
                if sub_ip_ind1 <= ip_ind < sub_ip_ind2:
                    ip_isub_var[ip_ind - ip_ind1] = isub
            for op_ind in range(op_ind1, op_ind2):
                if sub_op_ind1 <= op_ind < sub_op_ind2:
                    op_isub_var[op_ind - op_ind1] = isub

        xfer_ip_inds = {}
        xfer_op_inds = {}
        fwd_xfer_ip_inds = [{} for sub_ind in range(nsub_allprocs)]
        fwd_xfer_op_inds = [{} for sub_ind in range(nsub_allprocs)]
        rev_xfer_ip_inds = [{} for sub_ind in range(nsub_allprocs)]
        rev_xfer_op_inds = [{} for sub_ind in range(nsub_allprocs)]
        for iset in range(len(self._variable_sizes['input'])):
            for jset in range(len(self._variable_sizes['output'])):
                xfer_ip_inds[iset, jset] = []
                xfer_op_inds[iset, jset] = []
                for sub_ind in range(nsub_allprocs):
                    fwd_xfer_ip_inds[sub_ind][iset, jset] = []
                    fwd_xfer_op_inds[sub_ind][iset, jset] = []
                    rev_xfer_ip_inds[sub_ind][iset, jset] = []
                    rev_xfer_op_inds[sub_ind][iset, jset] = []

        ip_ind1, ip_ind2 = var_range['input']
        op_ind1, op_ind2 = var_range['output']
        for ip_ind in range(ip_ind1, ip_ind2):
            op_ind = self._input_src_ids[ip_ind]
            if op_ind1 <= op_ind < op_ind2:

                ip_isub = ip_isub_var[ip_ind - ip_ind1]
                op_isub = op_isub_var[op_ind - op_ind1]

                if ip_isub != -1 and ip_isub != op_isub:
                    ip_iset, ip_ivar_set = ip_set_indices[ip_ind, :]
                    op_iset, op_ivar_set = op_set_indices[op_ind, :]

                    ip_sizes = self._variable_sizes['input'][ip_iset]
                    op_sizes = self._variable_sizes['output'][op_iset]

                    ind1, ind2 = self._src_indices_range[ip_ivar_set, :]
                    inds = self._src_indices[ind1:ind2]

                    output_inds = numpy.zeros(inds.shape[0], int)
                    ind1, ind2 = 0, 0
                    for iproc in range(self._comm.size):
                        ind2 += op_sizes[iproc, op_ivar_set]

                        on_iproc = numpy.logical_and(ind1 <= inds, inds < ind2)
                        offset = -ind1
                        offset += numpy.sum(op_sizes[:iproc, :])
                        offset += numpy.sum(op_sizes[iproc, :op_ivar_set])
                        output_inds[on_iproc] = inds[on_iproc] + offset

                        ind1 += op_sizes[iproc, op_ivar_set]

                    iproc = self._comm.rank

                    ind1 = ind2 = numpy.sum(ip_sizes[:iproc, :])
                    ind1 += numpy.sum(ip_sizes[iproc, :ip_ivar_set])
                    ind2 += numpy.sum(ip_sizes[iproc, :ip_ivar_set + 1])
                    input_inds = numpy.arange(ind1, ind2)

                    xfer_ip_inds[ip_iset, op_iset].append(input_inds)
                    xfer_op_inds[ip_iset, op_iset].append(output_inds)

                    # rev mode wouldn't work for GS with a parallel group
                    if op_isub != -1:
                        key = (ip_iset, op_iset)
                        fwd_xfer_ip_inds[ip_isub][key].append(input_inds)
                        fwd_xfer_op_inds[ip_isub][key].append(output_inds)
                        rev_xfer_ip_inds[op_isub][key].append(input_inds)
                        rev_xfer_op_inds[op_isub][key].append(output_inds)

        def merge(indices_list):
            if len(indices_list) > 0:
                return numpy.concatenate(indices_list)
            else:
                return numpy.array([], int)

        for iset in range(len(self._variable_sizes['input'])):
            for jset in range(len(self._variable_sizes['output'])):
                xfer_ip_inds[iset, jset] = merge(xfer_ip_inds[iset, jset])
                xfer_op_inds[iset, jset] = merge(xfer_op_inds[iset, jset])
                for sub_ind in range(nsub_allprocs):
                    fwd_xfer_ip_inds[sub_ind][iset, jset] = \
                        merge(fwd_xfer_ip_inds[sub_ind][iset, jset])
                    fwd_xfer_op_inds[sub_ind][iset, jset] = \
                        merge(fwd_xfer_op_inds[sub_ind][iset, jset])
                    rev_xfer_ip_inds[sub_ind][iset, jset] = \
                        merge(rev_xfer_ip_inds[sub_ind][iset, jset])
                    rev_xfer_op_inds[sub_ind][iset, jset] = \
                        merge(rev_xfer_op_inds[sub_ind][iset, jset])

        return (xfer_ip_inds, xfer_op_inds, fwd_xfer_ip_inds, fwd_xfer_op_inds,
                rev_xfer_ip_inds, rev_xfer_op_inds)
