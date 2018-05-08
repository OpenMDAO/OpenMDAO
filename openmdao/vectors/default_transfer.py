"""Define the default Transfer class."""

from __future__ import division

from itertools import product, chain
from six import iteritems, itervalues
import numpy as np
from openmdao.vectors.vector import INT_DTYPE
from openmdao.vectors.transfer import Transfer
from openmdao.utils.array_utils import convert_neg

_empty_idx_array = np.array([], dtype=INT_DTYPE)


class DefaultTransfer(Transfer):
    """
    Default NumPy transfer.
    """

    @staticmethod
    def _setup_transfers(group, recurse=True):
        """
        Compute all transfers that are owned by our parent group.

        Parameters
        ----------
        group : <Group>
            Parent group.
        recurse : bool
            Whether to call this method in subsystems.
        """
        group._transfers = {}
        iproc = group.comm.rank
        rev = group._mode == 'rev'

        def merge(indices_list):
            if len(indices_list) > 0:
                return np.concatenate(indices_list)
            else:
                return _empty_idx_array

        if recurse:
            for subsys in group._subgroups_myproc:
                subsys._setup_transfers(recurse)

        # Pre-compute map from abs_names to the index of the containing subsystem
        abs2isub = {}
        for subsys, isub in zip(group._subsystems_myproc, group._subsystems_myproc_inds):
            for type_ in ['input', 'output']:
                for abs_name in subsys._var_allprocs_abs_names[type_]:
                    abs2isub[abs_name] = isub

        abs2meta = group._var_abs2meta
        allprocs_abs2meta = group._var_allprocs_abs2meta

        transfers = group._transfers
        vectors = group._vectors
        for vec_name in group._lin_rel_vec_name_list:
            relvars, _ = group._relevant[vec_name]['@all']
            relvars_in = relvars['input']
            relvars_out = relvars['output']

            # Initialize empty lists for the transfer indices
            nsub_allprocs = len(group._subsystems_allprocs)
            xfer_in = {}
            xfer_out = {}
            fwd_xfer_in = [{} for i in range(nsub_allprocs)]
            fwd_xfer_out = [{} for i in range(nsub_allprocs)]
            if rev:
                rev_xfer_in = [{} for i in range(nsub_allprocs)]
                rev_xfer_out = [{} for i in range(nsub_allprocs)]
            for set_name_in in group._num_var_byset[vec_name]['input']:
                for set_name_out in group._num_var_byset[vec_name]['output']:
                    key = (set_name_in, set_name_out)
                    xfer_in[key] = []
                    xfer_out[key] = []
                    for isub in range(nsub_allprocs):
                        fwd_xfer_in[isub][key] = []
                        fwd_xfer_out[isub][key] = []
                        if rev:
                            rev_xfer_in[isub][key] = []
                            rev_xfer_out[isub][key] = []

            allprocs_abs2idx_byset = group._var_allprocs_abs2idx_byset[vec_name]
            sizes_byset_in = group._var_sizes_byset[vec_name]['input']
            sizes_byset_out = group._var_sizes_byset[vec_name]['output']

            # Loop through all explicit / implicit connections owned by this system
            for abs_in, abs_out in iteritems(group._conn_abs_in2out):
                if abs_out not in relvars_out or abs_in not in relvars_in:
                    continue

                # Only continue if the input exists on this processor
                if abs_in in abs2meta and abs_in in relvars['input']:

                    # Get meta
                    meta_in = abs2meta[abs_in]
                    meta_out = allprocs_abs2meta[abs_out]

                    # Get varset info
                    set_name_in = meta_in['var_set']
                    set_name_out = meta_out['var_set']
                    idx_byset_in = allprocs_abs2idx_byset[abs_in]
                    idx_byset_out = allprocs_abs2idx_byset[abs_out]

                    # Get the sizes (byset) array
                    sizes_in = sizes_byset_in[set_name_in]
                    sizes_out = sizes_byset_out[set_name_out]

                    # Read in and process src_indices
                    shape_in = meta_in['shape']
                    shape_out = meta_out['shape']
                    global_size_out = meta_out['global_size']
                    src_indices = meta_in['src_indices']
                    if src_indices is None:
                        pass
                    elif src_indices.ndim == 1:
                        src_indices = convert_neg(src_indices, global_size_out)
                    else:
                        if len(shape_out) == 1 or shape_in == src_indices.shape:
                            src_indices = src_indices.flatten()
                            src_indices = convert_neg(src_indices, global_size_out)
                        else:
                            # TODO: this duplicates code found
                            # in System._setup_scaling.
                            entries = [list(range(x)) for x in shape_in]
                            cols = np.vstack(src_indices[i] for i in product(*entries))
                            dimidxs = [convert_neg(cols[:, i], shape_out[i])
                                       for i in range(cols.shape[1])]
                            src_indices = np.ravel_multi_index(dimidxs, shape_out)

                    # 1. Compute the output indices
                    if src_indices is None:
                        offset = np.sum(sizes_out[iproc, :idx_byset_out])
                        output_inds = np.arange(offset, offset + meta_in['size'], dtype=INT_DTYPE)
                    else:
                        output_inds = src_indices + np.sum(sizes_out[iproc, :idx_byset_out])

                    # 2. Compute the input indices
                    ind1 = np.sum(sizes_in[iproc, :idx_byset_in])
                    ind2 = ind1 + sizes_in[iproc, idx_byset_in]
                    input_inds = np.arange(ind1, ind2, dtype=INT_DTYPE)

                    # Now the indices are ready - input_inds, output_inds
                    key = (set_name_in, set_name_out)
                    xfer_in[key].append(input_inds)
                    xfer_out[key].append(output_inds)

                    isub = abs2isub[abs_in]
                    fwd_xfer_in[isub][key].append(input_inds)
                    fwd_xfer_out[isub][key].append(output_inds)
                    if rev and abs_out in abs2isub:
                        isub = abs2isub[abs_out]
                        rev_xfer_in[isub][key].append(input_inds)
                        rev_xfer_out[isub][key].append(output_inds)

            for set_name_in in group._num_var_byset[vec_name]['input']:
                for set_name_out in group._num_var_byset[vec_name]['output']:
                    key = (set_name_in, set_name_out)
                    xfer_in[key] = merge(xfer_in[key])
                    xfer_out[key] = merge(xfer_out[key])
                    for isub in range(nsub_allprocs):
                        fwd_xfer_in[isub][key] = merge(fwd_xfer_in[isub][key])
                        fwd_xfer_out[isub][key] = merge(fwd_xfer_out[isub][key])
                        if rev:
                            rev_xfer_in[isub][key] = merge(rev_xfer_in[isub][key])
                            rev_xfer_out[isub][key] = merge(rev_xfer_out[isub][key])

            out_vec = vectors['output'][vec_name]

            transfers[vec_name] = {}
            xfer_all = DefaultTransfer(vectors['input'][vec_name], out_vec,
                                       xfer_in, xfer_out, group.comm)
            transfers[vec_name]['fwd', None] = xfer_all
            if rev:
                transfers[vec_name]['rev', None] = xfer_all
            for isub in range(nsub_allprocs):
                transfers[vec_name]['fwd', isub] = DefaultTransfer(
                    vectors['input'][vec_name], vectors['output'][vec_name],
                    fwd_xfer_in[isub], fwd_xfer_out[isub], group.comm)
                if rev:
                    transfers[vec_name]['rev', isub] = DefaultTransfer(
                        vectors['input'][vec_name], vectors['output'][vec_name],
                        rev_xfer_in[isub], rev_xfer_out[isub], group.comm)

        transfers['nonlinear'] = transfers['linear']

    def _initialize_transfer(self, in_vec, out_vec):
        """
        Set up the transfer; do any necessary pre-computation.

        Optionally implemented by the subclass.

        Parameters
        ----------
        in_vec : <Vector>
            reference to the input vector.
        out_vec : <Vector>
            reference to the output vector.
        """
        in_inds = self._in_inds
        out_inds = self._out_inds

        # filter out any empty transfers
        outs = {}
        ins = {}
        for key in in_inds:
            if len(in_inds[key]) > 0:
                ins[key] = in_inds[key]
                outs[key] = out_inds[key]

        self._in_inds = ins
        self._out_inds = outs

    def transfer(self, in_vec, out_vec, mode='fwd'):
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
        in_inds = self._in_inds
        out_inds = self._out_inds

        if mode == 'fwd':
            do_complex = in_vec._vector_info._under_complex_step and out_vec._alloc_complex

            for key in in_inds:
                in_set_name, out_set_name = key
                # this works whether the vecs have multi columns or not due to broadcasting
                in_vec._data[in_set_name][in_inds[key]] = \
                    out_vec._data[out_set_name][out_inds[key]]

                # Imaginary transfer
                # (for CS, so only need in fwd)
                if do_complex:
                    in_vec._imag_data[in_set_name][in_inds[key]] = \
                        out_vec._imag_data[out_set_name][out_inds[key]]

        else:  # rev
            for key in in_inds:
                in_set_name, out_set_name = key
                np.add.at(
                    out_vec._data[out_set_name], out_inds[key],
                    in_vec._data[in_set_name][in_inds[key]])
