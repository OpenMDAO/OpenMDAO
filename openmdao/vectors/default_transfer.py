"""Define the default Transfer class."""

from itertools import product, chain
from collections import defaultdict

import numpy as np

from openmdao.vectors.vector import INT_DTYPE
from openmdao.vectors.transfer import Transfer
from openmdao.utils.array_utils import convert_neg, _global2local_offsets, _flatten_src_indices
from openmdao.utils.general_utils import _is_slice, _slice_indices
from openmdao.utils.mpi import MPI

_empty_idx_array = np.array([], dtype=INT_DTYPE)


def _merge(indices_list):
    if len(indices_list) > 0:
        return np.concatenate(indices_list)
    else:
        return _empty_idx_array


class DefaultTransfer(Transfer):
    """
    Default NumPy transfer.
    """

    @staticmethod
    def _setup_transfers(group):
        """
        Compute all transfers that are owned by our parent group.

        Parameters
        ----------
        group : <Group>
            Parent group.
        """
        iproc = group.comm.rank
        rev = group._mode == 'rev' or group._mode == 'auto'

        for subsys in group._subgroups_myproc:
            subsys._setup_transfers()

        abs2meta = group._var_abs2meta
        allprocs_abs2meta = group._var_allprocs_abs2meta

        group._transfers = transfers = {}
        vectors = group._vectors
        offsets = _global2local_offsets(group._get_var_offsets())

        vec_names = group._lin_rel_vec_name_list if group._use_derivatives else group._vec_names

        mypathlen = len(group.pathname + '.' if group.pathname else '')
        sub_inds = group._subsystems_inds

        for vec_name in vec_names:
            relvars, _ = group._relevant[vec_name]['@all']
            relvars_in = relvars['input']
            relvars_out = relvars['output']

            # Initialize empty lists for the transfer indices
            nsub_allprocs = len(group._subsystems_allprocs)
            xfer_in = []
            xfer_out = []
            fwd_xfer_in = [[] for s in group._subsystems_allprocs]
            fwd_xfer_out = [[] for s in group._subsystems_allprocs]
            if rev:
                rev_xfer_in = [[] for s in group._subsystems_allprocs]
                rev_xfer_out = [[] for s in group._subsystems_allprocs]

            allprocs_abs2idx = group._var_allprocs_abs2idx[vec_name]
            sizes_in = group._var_sizes[vec_name]['input']
            sizes_out = group._var_sizes[vec_name]['output']
            offsets_in = offsets[vec_name]['input']
            offsets_out = offsets[vec_name]['output']

            # Loop through all connections owned by this group
            for abs_in, abs_out in group._conn_abs_in2out.items():
                if abs_out not in relvars_out or abs_in not in relvars_in:
                    continue

                # Only continue if the input exists on this processor
                if abs_in in abs2meta:

                    # Get meta
                    meta_in = abs2meta[abs_in]
                    meta_out = allprocs_abs2meta[abs_out]

                    idx_in = allprocs_abs2idx[abs_in]
                    idx_out = allprocs_abs2idx[abs_out]

                    # Read in and process src_indices
                    src_indices = meta_in['src_indices']
                    if src_indices is None:
                        pass
                    elif src_indices.ndim == 1:
                        if isinstance(src_indices, tuple) or \
                                     (isinstance(src_indices, np.ndarray) and
                                      src_indices.dtype == object):
                            if _is_slice(src_indices):
                                indices = _slice_indices(src_indices, meta_out['global_size'],
                                                         meta_out['global_shape'])
                                src_indices = convert_neg(indices, meta_out['global_size'])
                        else:
                            src_indices = convert_neg(src_indices, meta_out['global_size'])
                    else:
                        src_indices = _flatten_src_indices(src_indices, meta_in['shape'],
                                                           meta_out['global_shape'],
                                                           meta_out['global_size'])
                        meta_in['src_indices'] = src_indices

                    # 1. Compute the output indices
                    offset = offsets_out[iproc, idx_out]
                    if src_indices is None:
                        output_inds = np.arange(offset, offset + meta_in['size'], dtype=INT_DTYPE)
                    else:
                        output_inds = src_indices + offset

                    # 2. Compute the input indices
                    input_inds = np.arange(offsets_in[iproc, idx_in],
                                           offsets_in[iproc, idx_in] +
                                           sizes_in[iproc, idx_in], dtype=INT_DTYPE)

                    # Now the indices are ready - input_inds, output_inds
                    sub_in = abs_in[mypathlen:].split('.', 1)[0]
                    isub = sub_inds[sub_in]
                    fwd_xfer_in[isub].append(input_inds)
                    fwd_xfer_out[isub].append(output_inds)
                    if rev and abs_out in abs2meta:
                        sub_out = abs_out[mypathlen:].split('.', 1)[0]
                        isub = sub_inds[sub_out]
                        rev_xfer_in[isub].append(input_inds)
                        rev_xfer_out[isub].append(output_inds)

            tot_size = 0
            for isub in range(nsub_allprocs):
                fwd_xfer_in[isub] = _merge(fwd_xfer_in[isub])
                fwd_xfer_out[isub] = _merge(fwd_xfer_out[isub])
                tot_size += fwd_xfer_in[isub].size
                if rev:
                    rev_xfer_in[isub] = _merge(rev_xfer_in[isub])
                    rev_xfer_out[isub] = _merge(rev_xfer_out[isub])

            transfers[vec_name] = {}

            if tot_size > 0:
                xfer_in = np.concatenate(fwd_xfer_in)
                xfer_out = np.concatenate(fwd_xfer_out)

                out_vec = vectors['output'][vec_name]

                xfer_all = DefaultTransfer(vectors['input'][vec_name], out_vec,
                                           xfer_in, xfer_out, group.comm)
            else:
                xfer_all = None
            transfers[vec_name]['fwd', None] = xfer_all
            if rev:
                transfers[vec_name]['rev', None] = xfer_all
            for isub in range(nsub_allprocs):
                if fwd_xfer_in[isub].size > 0:
                    transfers[vec_name]['fwd', isub] = DefaultTransfer(
                        vectors['input'][vec_name], vectors['output'][vec_name],
                        fwd_xfer_in[isub], fwd_xfer_out[isub], group.comm)
                else:
                    transfers[vec_name]['fwd', isub] = None
                if rev:
                    if rev_xfer_out[isub].size > 0:
                        transfers[vec_name]['rev', isub] = DefaultTransfer(
                            vectors['input'][vec_name], vectors['output'][vec_name],
                            rev_xfer_in[isub], rev_xfer_out[isub], group.comm)
                    else:
                        transfers[vec_name]['rev', isub] = None

        if group._use_derivatives:
            transfers['nonlinear'] = transfers['linear']

    @staticmethod
    def _setup_discrete_transfers(group):
        """
        Compute all transfers that are owned by our parent group.

        Parameters
        ----------
        group : <Group>
            Parent group.
        """
        group._discrete_transfers = transfers = defaultdict(list)
        name_offset = len(group.pathname) + 1 if group.pathname else 0

        iproc = group.comm.rank
        owns = group._owning_rank

        for tgt, src in group._conn_discrete_in2out.items():
            src_sys, src_var = src[name_offset:].split('.', 1)
            tgt_sys, tgt_var = tgt[name_offset:].split('.', 1)
            xfer = (src_sys, src_var, tgt_sys, tgt_var)
            transfers[tgt_sys].append(xfer)

        if group.comm.size > 1:
            # collect all xfers for each tgt system
            for tgt, src in group._conn_discrete_in2out.items():
                src_sys, src_var = src[name_offset:].split('.', 1)
                tgt_sys, tgt_var = tgt[name_offset:].split('.', 1)
                xfer = (src_sys, src_var, tgt_sys, tgt_var)
                transfers[tgt_sys].append(xfer)

            total_send = set()
            total_recv = []
            total_xfers = []

            for tgt_sys, xfers in transfers.items():
                send = set()
                recv = []
                for src_sys, src_var, tgt_sys, tgt_var in xfers:
                    if group.pathname:
                        src_abs = '.'.join([group.pathname, src_sys, src_var])
                    else:
                        src_abs = '.'.join([src_sys, src_var])
                    tgt_rel = '.'.join((tgt_sys, tgt_var))
                    src_rel = '.'.join((src_sys, src_var))
                    if iproc == owns[src_abs]:
                        # we own this var, so we'll send it out to others
                        send.add(src_rel)
                    if (tgt_rel in group._var_discrete['input'] and
                            src_rel not in group._var_discrete['output']):
                        # we have the target locally, but not the source, so we need someone
                        # to send it to us.
                        recv.append(src_rel)

                transfers[tgt_sys] = (xfers, send, recv)
                total_xfers.extend(xfers)
                total_send.update(send)
                total_recv.extend(recv)

            transfers[None] = (total_xfers, total_send, total_recv)

            # find out all ranks that need to receive each discrete source var
            allproc_xfers = group.comm.allgather(transfers)
            allprocs_recv = defaultdict(lambda: defaultdict(list))
            for rank, rank_transfers in enumerate(allproc_xfers):
                for tgt_sys, (_, _, recvs) in rank_transfers.items():
                    for recv in recvs:
                        allprocs_recv[tgt_sys][recv].append(rank)

            group._allprocs_discrete_recv = allprocs_recv

            # if we own a src var but it's local for every rank, we don't need to send it to anyone.
            total_send = total_send.intersection(allprocs_recv)

            for tgt_sys in transfers:
                xfers, send, _ = transfers[tgt_sys]
                # update send list to remove any vars that don't have a remote receiver,
                # and get rid of recv list because allprocs_recv has the necessary info.
                transfers[tgt_sys] = (xfers, send.intersection(allprocs_recv[tgt_sys]))

    def _transfer(self, in_vec, out_vec, mode='fwd'):
        """
        Perform transfer.

        Parameters
        ----------
        in_vec : <Vector>
            pointer to the input vector.
        out_vec : <Vector>
            pointer to the output vector.
        mode : str
            'fwd' or 'rev'.

        """
        if mode == 'fwd':
            # this works whether the vecs have multi columns or not due to broadcasting
            in_vec._data[self._in_inds] = out_vec._data[self._out_inds]

        else:  # rev
            if out_vec._ncol == 1:
                out_vec._data[:] += np.bincount(self._out_inds, in_vec._data[self._in_inds],
                                                minlength=out_vec._data.size)
            else:  # matrix-matrix   (bincount only works with 1d arrays)
                np.add.at(out_vec._data, self._out_inds, in_vec._data[self._in_inds])
