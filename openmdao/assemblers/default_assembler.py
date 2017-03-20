"""Define the DefaultAssembler class."""
from __future__ import division
import numpy as np

from six.moves import range

from openmdao.assemblers.assembler import Assembler


class DefaultAssembler(Assembler):
    """
    Default <Assembler> implementation.
    """

    def _compute_transfers(self, nsub_allprocs, var_range,
                           subsystems_myproc, subsystems_inds):
        """
        Compute the transfer indices.

        Parameters
        ----------
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
        xfer_in_inds : dict of int ndarray[:]
            input indices of global transfer.
        xfer_out_inds : dict of int ndarray[:]
            output indices of global transfer.
        fwd_xfer_in_inds : [dict of int ndarray[:], ...]
            list of input indices of forward transfers.
        fwd_xfer_out_inds : [dict of int ndarray[:], ...]
            list of output indices of forward transfers.
        rev_xfer_in_inds : [dict of int ndarray[:], ...]
            list of input indices of reverse transfers.
        rev_xfer_out_inds : [dict of int ndarray[:], ...]
            list of output indices of reverse transfers.
        """
        in_set_indices = self._variable_set_indices['input']
        out_set_indices = self._variable_set_indices['output']

        in_ind1, in_ind2 = var_range['input']
        out_ind1, out_ind2 = var_range['output']
        in_isub_var = -np.ones(in_ind2 - in_ind1, int)
        out_isub_var = -np.ones(out_ind2 - out_ind1, int)
        for ind, subsys in enumerate(subsystems_myproc):
            isub = subsystems_inds[ind]

            sub_var_range = subsys._var_allprocs_idx_range
            sub_in_ind1, sub_in_ind2 = sub_var_range['input']
            sub_out_ind1, sub_out_ind2 = sub_var_range['output']
            for in_ind in range(in_ind1, in_ind2):
                if sub_in_ind1 <= in_ind < sub_in_ind2:
                    in_isub_var[in_ind - in_ind1] = isub
            for out_ind in range(out_ind1, out_ind2):
                if sub_out_ind1 <= out_ind < sub_out_ind2:
                    out_isub_var[out_ind - out_ind1] = isub

        xfer_in_inds = {}
        xfer_out_inds = {}
        fwd_xfer_in_inds = [{} for sub_ind in range(nsub_allprocs)]
        fwd_xfer_out_inds = [{} for sub_ind in range(nsub_allprocs)]
        rev_xfer_in_inds = [{} for sub_ind in range(nsub_allprocs)]
        rev_xfer_out_inds = [{} for sub_ind in range(nsub_allprocs)]
        for iset in range(len(self._variable_sizes['input'])):
            for jset in range(len(self._variable_sizes['output'])):
                xfer_in_inds[iset, jset] = []
                xfer_out_inds[iset, jset] = []
                for sub_ind in range(nsub_allprocs):
                    fwd_xfer_in_inds[sub_ind][iset, jset] = []
                    fwd_xfer_out_inds[sub_ind][iset, jset] = []
                    rev_xfer_in_inds[sub_ind][iset, jset] = []
                    rev_xfer_out_inds[sub_ind][iset, jset] = []

        in_ind1, in_ind2 = var_range['input']
        out_ind1, out_ind2 = var_range['output']
        for in_ind in range(in_ind1, in_ind2):
            iabs = self._var_allprocs_abs_names['input'][in_ind]
            oabs = self._abs_input2src[iabs]
            if oabs is None:
                continue

            out_ind = self._var_allprocs_abs2idx_io[oabs]
            if out_ind1 <= out_ind < out_ind2:

                in_isub = in_isub_var[in_ind - in_ind1]
                out_isub = out_isub_var[out_ind - out_ind1]

                if in_isub != -1 and in_isub != out_isub:
                    in_iset, in_ivar_set = in_set_indices[in_ind, :]
                    out_iset, out_ivar_set = out_set_indices[out_ind, :]

                    in_sizes = self._variable_sizes['input'][in_iset]
                    out_sizes = self._variable_sizes['output'][out_iset]

                    ind1, ind2 = self._src_indices_range[in_ivar_set, :]
                    inds = self._src_indices[ind1:ind2]

                    output_inds = np.zeros(inds.shape[0], int)
                    ind1, ind2 = 0, 0
                    for iproc in range(self._comm.size):
                        ind2 += out_sizes[iproc, out_ivar_set]

                        on_iproc = np.logical_and(ind1 <= inds, inds < ind2)
                        offset = -ind1
                        offset += np.sum(out_sizes[:iproc, :])
                        offset += np.sum(out_sizes[iproc, :out_ivar_set])
                        output_inds[on_iproc] = inds[on_iproc] + offset

                        ind1 += out_sizes[iproc, out_ivar_set]

                    iproc = self._comm.rank

                    ind1 = ind2 = np.sum(in_sizes[:iproc, :])
                    ind1 += np.sum(in_sizes[iproc, :in_ivar_set])
                    ind2 += np.sum(in_sizes[iproc, :in_ivar_set + 1])
                    input_inds = np.arange(ind1, ind2)

                    xfer_in_inds[in_iset, out_iset].append(input_inds)
                    xfer_out_inds[in_iset, out_iset].append(output_inds)

                    # rev mode wouldn't work for GS with a parallel group
                    if out_isub != -1:
                        key = (in_iset, out_iset)
                        fwd_xfer_in_inds[in_isub][key].append(input_inds)
                        fwd_xfer_out_inds[in_isub][key].append(output_inds)
                        rev_xfer_in_inds[out_isub][key].append(input_inds)
                        rev_xfer_out_inds[out_isub][key].append(output_inds)

        def merge(indices_list):
            if len(indices_list) > 0:
                return np.concatenate(indices_list)
            else:
                return np.array([], int)

        for iset in range(len(self._variable_sizes['input'])):
            for jset in range(len(self._variable_sizes['output'])):
                xfer_in_inds[iset, jset] = merge(xfer_in_inds[iset, jset])
                xfer_out_inds[iset, jset] = merge(xfer_out_inds[iset, jset])
                for sub_ind in range(nsub_allprocs):
                    fwd_xfer_in_inds[sub_ind][iset, jset] = \
                        merge(fwd_xfer_in_inds[sub_ind][iset, jset])
                    fwd_xfer_out_inds[sub_ind][iset, jset] = \
                        merge(fwd_xfer_out_inds[sub_ind][iset, jset])
                    rev_xfer_in_inds[sub_ind][iset, jset] = \
                        merge(rev_xfer_in_inds[sub_ind][iset, jset])
                    rev_xfer_out_inds[sub_ind][iset, jset] = \
                        merge(rev_xfer_out_inds[sub_ind][iset, jset])

        return (xfer_in_inds, xfer_out_inds, fwd_xfer_in_inds, fwd_xfer_out_inds,
                rev_xfer_in_inds, rev_xfer_out_inds)
