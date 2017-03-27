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
        in_start, in_end = var_range['input']
        out_start, out_end = var_range['output']
        in_isub_var = -np.ones(in_end - in_start, int)
        out_isub_var = -np.ones(out_end - out_start, int)
        for ind, subsys in enumerate(subsystems_myproc):
            sub_var_range = subsys._var_allprocs_idx_range
            sub_in_start, sub_in_end = sub_var_range['input']
            sub_out_start, sub_out_end = sub_var_range['output']
            isub = subsystems_inds[ind]
            in_isub_var[sub_in_start - in_start:sub_in_end - in_start] = isub
            out_isub_var[sub_out_start - out_start:sub_out_end - out_start] = isub

        xfer_in_inds = {}
        xfer_out_inds = {}
        fwd_xfer_in_inds = [{} for sub_ind in range(nsub_allprocs)]
        fwd_xfer_out_inds = [{} for sub_ind in range(nsub_allprocs)]

        for iset in range(len(self._var_sizes_by_set['input'])):
            for jset in range(len(self._var_sizes_by_set['output'])):
                key = (iset, jset)
                xfer_in_inds[key] = []
                xfer_out_inds[key] = []
                for sub_ind in range(nsub_allprocs):
                    fwd_xfer_in_inds[sub_ind][key] = []
                    fwd_xfer_out_inds[sub_ind][key] = []

        rev = self._mode == 'rev'
        if rev:
            rev_xfer_in_inds = [{} for sub_ind in range(nsub_allprocs)]
            rev_xfer_out_inds = [{} for sub_ind in range(nsub_allprocs)]
            for i, idxs in enumerate(fwd_xfer_out_inds):
                for key in idxs:
                    rev_xfer_in_inds[i][key] = []
                    rev_xfer_out_inds[i][key] = []
        else:
            rev_xfer_out_inds = rev_xfer_in_inds = ()

        in_set_indices = self._var_set_indices['input']
        out_set_indices = self._var_set_indices['output']
        in_abs_vars = self._var_allprocs_abs_names['input']

        rank = self._comm.rank

        for in_ind in range(in_start, in_end):
            in_isub = in_isub_var[in_ind - in_start]
            if in_isub == -1:
                continue

            iabs = in_abs_vars[in_ind]
            oabs = self._abs_input2src[iabs]
            if oabs is None:  # input is not connected
                continue

            out_ind = self._var_allprocs_abs2idx_io[oabs]

            input_inds = None
            has_src_indices = self._var_allprocs_abs2meta_io[iabs]['has_src_indices']

            if out_start <= out_ind < out_end:
                out_isub = out_isub_var[out_ind - out_start]
                if out_isub == -1 or out_isub == in_isub:
                    continue
                in_iset, in_ivar_set = in_set_indices[in_ind, :]
                out_iset, out_ivar_set = out_set_indices[out_ind, :]

                start, end = self._src_indices_range[in_ind, :]
                inds = self._src_indices[start:end]

                out_sizes = self._var_sizes_by_set['output'][out_iset]
                out_offsets = self._var_offsets_by_set['output'][out_iset]
                if has_src_indices:
                    output_inds = np.zeros(inds.size, int)
                    start, end = 0, 0
                    for iproc in range(self._comm.size):
                        end += out_sizes[iproc, out_ivar_set]

                        on_iproc = np.logical_and(start <= inds, inds < end)
                        offset = out_offsets[iproc, out_ivar_set]
                        output_inds[on_iproc] = inds[on_iproc] + offset

                        start += out_sizes[iproc, out_ivar_set]
                else:
                    if out_sizes[rank, out_ivar_set] == 0:
                        # find lowest rank remote owner of output
                        iproc = np.nonzero(out_sizes[:, out_ivar_set])[0][0]
                    else:
                        iproc = rank
                    output_inds = inds + out_offsets[iproc, out_ivar_set]

                in_sizes = self._var_sizes_by_set['input'][in_iset]
                if in_sizes[iproc, in_ivar_set] == 0:
                    input_inds = output_inds = None
                else:
                    in_offsets = self._var_offsets_by_set['input'][in_iset]
                    start = in_offsets[rank, in_ivar_set]
                    end = start + in_sizes[rank, in_ivar_set]
                    input_inds = np.arange(start, end)

                if input_inds is not None:
                    key = (in_iset, out_iset)
                    xfer_in_inds[key].append(input_inds)
                    xfer_out_inds[key].append(output_inds)

                    fwd_xfer_in_inds[in_isub][key].append(input_inds)
                    fwd_xfer_out_inds[in_isub][key].append(output_inds)

                if rev:
                    input_inds, output_inds = self._get_rev_idxs(in_ind, out_ind,
                                                                 in_set_indices,
                                                                 out_set_indices)
                    if input_inds is not None:
                        rev_xfer_in_inds[out_isub][key].append(input_inds)
                        rev_xfer_out_inds[out_isub][key].append(output_inds)

        def merge(indices_list):
            if len(indices_list) > 0:
                return np.concatenate(indices_list)
            else:
                return np.array([], int)

        for key in xfer_in_inds:
            xfer_in_inds[key] = merge(xfer_in_inds[key])
            xfer_out_inds[key] = merge(xfer_out_inds[key])
            for sub_ind in range(nsub_allprocs):
                fwd_xfer_in_inds[sub_ind][key] = \
                    merge(fwd_xfer_in_inds[sub_ind][key])
                fwd_xfer_out_inds[sub_ind][key] = \
                    merge(fwd_xfer_out_inds[sub_ind][key])
                if rev:
                    rev_xfer_in_inds[sub_ind][key] = \
                        merge(rev_xfer_in_inds[sub_ind][key])
                    rev_xfer_out_inds[sub_ind][key] = \
                        merge(rev_xfer_out_inds[sub_ind][key])

        return (xfer_in_inds, xfer_out_inds, fwd_xfer_in_inds, fwd_xfer_out_inds,
                rev_xfer_in_inds, rev_xfer_out_inds)

    def _get_rev_idxs(self, in_ind, out_ind, in_set_indices, out_set_indices):
        input_inds = output_inds = None
        in_iset = out_iset = 0

        iabs = self._var_allprocs_abs_names['input'][in_ind]
        imeta = self._var_allprocs_abs2meta_io[iabs]

        rank = self._comm.rank

        in_iset, in_ivar_set = in_set_indices[in_ind, :]
        out_iset, out_ivar_set = out_set_indices[out_ind, :]

        has_src_indices = imeta['has_src_indices']
        ind1, ind2 = self._src_indices_range[in_ind, :]
        if ind1 == ind2:  # input isn't local
            if has_src_indices:
                # FIXME: In rev mode I think we actually need to retrieve
                # src_indices from other proc so we can use them here...
                return None, None
            else:
                inds = np.arange(np.prod(imeta['shape']))
        else:
            inds = self._src_indices[ind1:ind2]

        output_inds = np.zeros(inds.size, int)

        out_sizes = self._var_sizes_by_set['output'][out_iset]
        out_offsets = self._var_offsets_by_set['output'][out_iset]
        if has_src_indices:
            ind1, ind2 = 0, 0
            for iproc in range(self._comm.size):
                ind2 += out_sizes[iproc, out_ivar_set]

                on_iproc = np.logical_and(ind1 <= inds, inds < ind2)
                offset = out_offsets[iproc, out_ivar_set]
                output_inds[on_iproc] = inds[on_iproc] + offset

                ind1 += out_sizes[iproc, out_ivar_set]
        else:
            if out_sizes[rank, out_ivar_set] == 0:
                return None, None
            output_inds = inds + out_offsets[rank, out_ivar_set]

        in_sizes = self._var_sizes_by_set['input'][in_iset]
        in_offsets = self._var_offsets_by_set['input'][in_iset]
        if in_sizes[rank, in_ivar_set] == 0:
            # find lowest rank owner of input
            iproc = np.nonzero(in_sizes[:, in_ivar_set])[0][0]
        else:
            iproc = rank
        ind1 = in_offsets[iproc, in_ivar_set]
        ind2 = ind1 + in_sizes[iproc, in_ivar_set]
        input_inds = np.arange(ind1, ind2)

        return input_inds, output_inds
