"""Define the default Transfer class."""

from collections import defaultdict

import numpy as np

from openmdao.core.constants import INT_DTYPE
from openmdao.vectors.transfer import Transfer
from openmdao.utils.array_utils import _global2local_offsets
from openmdao.utils.mpi import MPI


def _fill(arr, indices_iter):
    """
    Fill the given array with the given list of indices.

    Parameters
    ----------
    arr : ndarray
        Array to be filled.
    indices_iter : iterator of int ndarrays or ranges
        Iterator of ranges/indices to be placed into arr.
    """
    start = end = 0
    for inds in indices_iter:
        end += len(inds)
        arr[start:end] = inds
        start = end


def _setup_index_views(tot_size, in_xfers, out_xfers):
    """
    Create index views for all subsystems and allocate full transfer arrays.

    Parameters
    ----------
    tot_size : int
        Total size of each full array.
    in_xfers : dict
        Mapping of subsystem name to input index arrays.
    out_xfers : dict
        Mapping of subsystem name to output index arrays.
    """
    full_in = np.empty(tot_size, dtype=INT_DTYPE)
    full_out = np.empty(tot_size, dtype=INT_DTYPE)

    start = end = 0
    for sname, ranges in in_xfers.items():
        # input inds are always ranges.  output inds may be ranges or ndarrays.
        rstart = rend = start
        for rng in ranges:
            rend += len(rng)
            full_in[rstart:rend] = rng
            rstart = rend

        end += rend - start
        _fill(full_out[start:end], out_xfers[sname])

        # change subsystem transfer entries to be views of the full transfer arrays
        in_xfers[sname] = full_in[start:end]
        out_xfers[sname] = full_out[start:end]
        start = end

    return full_in, full_out


def _setup_index_arrays(tot_size, in_xfers, out_xfers, vectors):
    """
    Create index arrays for all subsystems.

    Parameters
    ----------
    tot_size : int
        Total size of each full array.
    in_xfers : dict
        Mapping of subsystem name to input index arrays.
    out_xfers : dict
        Mapping of subsystem name to output index arrays.
    vectors : dict
        Dictionary of input and output vectors.

    Returns
    -------
    dict
        Mapping of subsystem name to Transfer object. None key maps to the
        'full' transfer across all subsystems.
    """
    xfer_in, xfer_out = _setup_index_views(tot_size, in_xfers, out_xfers)

    if tot_size > 0:
        xfer_all = DefaultTransfer(vectors['input']['nonlinear'],
                                   vectors['output']['nonlinear'], xfer_in, xfer_out)
    else:
        xfer_all = None

    xfer_dict = {None: xfer_all}

    for sname, inds in in_xfers.items():
        if inds.size > 0:
            xfer_dict[sname] = DefaultTransfer(vectors['input']['nonlinear'],
                                               vectors['output']['nonlinear'],
                                               inds, out_xfers[sname])
        else:
            xfer_dict[sname] = None

    return xfer_dict


class DefaultTransfer(Transfer):
    """
    Default NumPy transfer.

    Parameters
    ----------
    in_vec : <Vector>
        Pointer to the input vector.
    out_vec : <Vector>
        Pointer to the output vector.
    in_inds : int ndarray
        Input indices for the transfer.
    out_inds : int ndarray
        Output indices for the transfer.
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
        rev = group._orig_mode != 'fwd'

        abs2meta = group._var_abs2meta

        group._transfers = transfers = {}
        vectors = group._vectors
        offsets = _global2local_offsets(group._get_var_offsets())
        mypathlen = len(group.pathname + '.' if group.pathname else '')

        # Initialize empty lists for the transfer indices
        fwd_xfer_in = defaultdict(list)
        fwd_xfer_out = defaultdict(list)
        if rev:
            rev_xfer_in = defaultdict(list)
            rev_xfer_out = defaultdict(list)

        allprocs_abs2idx = group._var_allprocs_abs2idx
        sizes_in = group._var_sizes['input']
        offsets_in = offsets['input']
        if sizes_in.size > 0:
            sizes_in = sizes_in[iproc]
            offsets_in = offsets_in[iproc]
        offsets_out = offsets['output']
        if offsets_out.size > 0:
            offsets_out = offsets_out[iproc]

        tot_size = 0

        # Loop through all connections owned by this group
        for abs_in, abs_out in group._conn_abs_in2out.items():
            # This weeds out discrete vars (all vars are local if using this Transfer)
            if abs_in in abs2meta['input']:

                # Get meta
                meta_in = abs2meta['input'][abs_in]

                idx_in = allprocs_abs2idx[abs_in]
                idx_out = allprocs_abs2idx[abs_out]

                # Read in and process src_indices
                src_indices = meta_in['src_indices']
                if src_indices is not None:
                    src_indices = src_indices.shaped_array()

                # 1. Compute the output indices
                offset = offsets_out[idx_out]
                if src_indices is None:
                    output_inds = range(offset, offset + meta_in['size'])
                else:
                    output_inds = src_indices + offset

                # 2. Compute the input indices
                # all input indices can be simple ranges during this part in order to save memory
                input_inds = range(offsets_in[idx_in], offsets_in[idx_in] + sizes_in[idx_in])
                tot_size += sizes_in[idx_in]

                # Now the indices are ready - input_inds, output_inds
                sub_in = abs_in[mypathlen:].split('.', 1)[0]
                fwd_xfer_in[sub_in].append(input_inds)
                fwd_xfer_out[sub_in].append(output_inds)
                if rev and abs_out in abs2meta['output']:
                    sub_out = abs_out[mypathlen:].split('.', 1)[0]
                    rev_xfer_in[sub_out].append(input_inds)
                    rev_xfer_out[sub_out].append(output_inds)

        transfers['fwd'] = _setup_index_arrays(tot_size, fwd_xfer_in, fwd_xfer_out, vectors)
        if rev:
            transfers['rev'] = _setup_index_arrays(tot_size, rev_xfer_in, rev_xfer_out, vectors)

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
            in_vec.set_val(out_vec.asarray()[self._out_inds.flat], self._in_inds)

        else:  # rev
            out_vec.iadd(np.bincount(self._out_inds, in_vec._get_data()[self._in_inds],
                                     minlength=out_vec._data.size))
