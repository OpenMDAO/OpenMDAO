"""Define the PETSc Transfer class."""
from __future__ import division

import numpy as np
from petsc4py import PETSc
from six import iteritems, itervalues
from itertools import product, chain

from openmdao.vectors.transfer import Transfer
from openmdao.vectors.default_transfer import DefaultTransfer
from openmdao.vectors.vector import INT_DTYPE
from openmdao.utils.mpi import MPI
from openmdao.utils.array_utils import convert_neg

_empty_idx_array = np.array([], dtype=INT_DTYPE)


class PETScTransfer(DefaultTransfer):
    """
    PETSc Transfer implementation for running in parallel.
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
        rev = group._mode == 'rev' or group._mode == 'auto'

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
        myproc = group.comm.rank

        transfers = group._transfers
        vectors = group._vectors
        offsets = group._get_var_offsets()

        vec_names = group._lin_rel_vec_name_list if group._use_derivatives else group._vec_names

        for vec_name in vec_names:
            relvars, _ = group._relevant[vec_name]['@all']

            # Initialize empty lists for the transfer indices
            nsub_allprocs = len(group._subsystems_allprocs)
            xfer_in = []
            xfer_out = []
            fwd_xfer_in = [[] for i in range(nsub_allprocs)]
            fwd_xfer_out = [[] for i in range(nsub_allprocs)]
            if rev:
                rev_xfer_in = [[] for i in range(nsub_allprocs)]
                rev_xfer_out = [[] for i in range(nsub_allprocs)]

            allprocs_abs2idx = group._var_allprocs_abs2idx[vec_name]
            sizes_in = group._var_sizes[vec_name]['input']
            sizes_out = group._var_sizes[vec_name]['output']
            offsets_in = offsets[vec_name]['input']
            offsets_out = offsets[vec_name]['output']

            # Loop through all explicit / implicit connections owned by this system
            for abs_in, abs_out in iteritems(group._conn_abs_in2out):
                if abs_out not in relvars['output']:
                    continue

                # Only continue if the input exists on this processor
                if abs_in in abs2meta and abs_in in relvars['input']:

                    # Get meta
                    meta_in = abs2meta[abs_in]
                    meta_out = allprocs_abs2meta[abs_out]

                    idx_in = allprocs_abs2idx[abs_in]
                    idx_out = allprocs_abs2idx[abs_out]

                    # Read in and process src_indices
                    shape_in = meta_in['shape']
                    shape_out = meta_out['shape']
                    global_shape_out = meta_out['global_shape']
                    global_size_out = meta_out['global_size']
                    src_indices = meta_in['src_indices']
                    if src_indices is None:
                        owner = group._owning_rank[abs_out]
                        # if the input is larger than the output on a single proc, we have
                        # to just loop over the procs in the same way we do when src_indices
                        # is defined.
                        if meta_in['size'] > sizes_out[owner, idx_out]:
                            src_indices = np.arange(meta_in['size'], dtype=INT_DTYPE)
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
                            dimidxs = [convert_neg(cols[:, i], global_shape_out[i])
                                       for i in range(cols.shape[1])]
                            src_indices = np.ravel_multi_index(dimidxs, global_shape_out)

                    # 1. Compute the output indices
                    if src_indices is None:
                        start = 0 if owner == 0 else np.sum(sizes_out[:owner, idx_out])
                        offset = offsets_out[owner, idx_out] - start
                        output_inds = np.arange(offset, offset + meta_in['size'], dtype=INT_DTYPE)
                    else:
                        output_inds = np.zeros(src_indices.size, INT_DTYPE)
                        start = end = 0
                        for iproc in range(group.comm.size):
                            end += sizes_out[iproc, idx_out]

                            # The part of src on iproc
                            on_iproc = np.logical_and(start <= src_indices, src_indices < end)

                            if np.any(on_iproc):
                                # This converts from iproc-then-ivar to ivar-then-iproc ordering
                                # Subtract off part of previous procs
                                # Then add all variables on previous procs
                                # Then all previous variables on this proc
                                # - np.sum(out_sizes[:iproc, idx_out])
                                # + np.sum(out_sizes[:iproc, :])
                                # + np.sum(out_sizes[iproc, :idx_out])
                                # + inds
                                offset = offsets_out[iproc, idx_out] - start
                                output_inds[on_iproc] = src_indices[on_iproc] + offset

                            start = end

                    # 2. Compute the input indices
                    input_inds = np.arange(offsets_in[myproc, idx_in],
                                           offsets_in[myproc, idx_in] +
                                           sizes_in[myproc, idx_in], dtype=INT_DTYPE)

                    # Now the indices are ready - input_inds, output_inds
                    xfer_in.append(input_inds)
                    xfer_out.append(output_inds)

                    isub = abs2isub[abs_in]
                    fwd_xfer_in[isub].append(input_inds)
                    fwd_xfer_out[isub].append(output_inds)
                    if rev and abs_out in abs2isub:
                        isub = abs2isub[abs_out]
                        rev_xfer_in[isub].append(input_inds)
                        rev_xfer_out[isub].append(output_inds)

            xfer_in = merge(xfer_in)
            xfer_out = merge(xfer_out)
            for isub in range(nsub_allprocs):
                fwd_xfer_in[isub] = merge(fwd_xfer_in[isub])
                fwd_xfer_out[isub] = merge(fwd_xfer_out[isub])
                if rev:
                    rev_xfer_in[isub] = merge(rev_xfer_in[isub])
                    rev_xfer_out[isub] = merge(rev_xfer_out[isub])

            out_vec = vectors['output'][vec_name]

            transfers[vec_name] = {}
            xfer_all = PETScTransfer(vectors['input'][vec_name], out_vec,
                                     xfer_in, xfer_out, group.comm)
            transfers[vec_name]['fwd', None] = xfer_all
            if rev:
                transfers[vec_name]['rev', None] = xfer_all
            for isub in range(nsub_allprocs):
                transfers[vec_name]['fwd', isub] = PETScTransfer(
                    vectors['input'][vec_name], vectors['output'][vec_name],
                    fwd_xfer_in[isub], fwd_xfer_out[isub], group.comm)
                if rev:
                    transfers[vec_name]['rev', isub] = PETScTransfer(
                        vectors['input'][vec_name], vectors['output'][vec_name],
                        rev_xfer_in[isub], rev_xfer_out[isub], group.comm)

        if group._use_derivatives:
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
        in_indexset = PETSc.IS().createGeneral(self._in_inds, comm=self._comm)
        out_indexset = PETSc.IS().createGeneral(self._out_inds, comm=self._comm)

        self._transfer = PETSc.Scatter().create(out_vec._petsc, out_indexset, in_vec._petsc,
                                                in_indexset)

        if in_vec._ncol > 1:
            self.transfer = self.multi_transfer

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
        flag = False
        if mode == 'rev':
            flag = True
            in_vec, out_vec = out_vec, in_vec

        in_petsc = in_vec._petsc
        out_petsc = out_vec._petsc

        # For Complex Step, need to disassemble real and imag parts, transfer them separately,
        # then reassemble them.
        if in_vec._under_complex_step and out_vec._alloc_complex:

            # Real
            in_petsc.array = in_vec._data.real
            out_petsc.array = out_vec._data.real
            self._transfer.scatter(out_petsc, in_petsc, addv=flag, mode=flag)

            # Imaginary
            in_petsc_imag = in_vec._imag_petsc
            out_petsc_imag = out_vec._imag_petsc
            in_petsc_imag.array = in_vec._data.imag
            out_petsc_imag.array = out_vec._data.imag
            self._transfer.scatter(out_petsc_imag, in_petsc_imag, addv=flag, mode=flag)

            in_vec._data[:] = in_petsc.array + in_petsc_imag.array * 1j

        else:

            # Anything that has been allocated complex requires an additional step because
            # the petsc vector does not directly reference the _data.

            if in_vec._alloc_complex:
                in_petsc.array = in_vec._data

            if out_vec._alloc_complex:
                out_petsc.array = out_vec._data

            self._transfer.scatter(out_petsc, in_petsc, addv=flag, mode=flag)

            if in_vec._alloc_complex:
                in_vec._data[:] = in_petsc.array

    def multi_transfer(self, in_vec, out_vec, mode='fwd'):
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
            in_petsc = in_vec._petsc
            out_petsc = out_vec._petsc

            # For Complex Step, need to disassemble real and imag parts, transfer them separately,
            # then reassemble them.
            if in_vec._under_complex_step and out_vec._alloc_complex:
                in_petsc_imag = in_vec._imag_petsc
                out_petsc_imag = out_vec._imag_petsc
                for i in range(in_vec._ncol):

                    # Real
                    in_petsc.array = in_vec._data[:, i].real
                    out_petsc.array = out_vec._data[:, i].real
                    self._transfer.scatter(out_petsc, in_petsc, addv=False, mode=False)

                    # Imaginary
                    in_petsc_imag.array = in_vec._data[:, i].imag
                    out_petsc_imag.array = out_vec._data[:, i].imag
                    self._transfer.scatter(out_petsc_imag, in_petsc_imag, addv=False, mode=False)

                    in_vec._data[:, i] = in_petsc.array + in_petsc_imag.array * 1j

            else:
                for i in range(in_vec._ncol):
                    in_petsc.array = in_vec._data[:, i]
                    out_petsc.array = out_vec._data[:, i]
                    self._transfer.scatter(out_petsc, in_petsc, addv=False, mode=False)
                    in_vec._data[:, i] = in_petsc.array

        elif mode == 'rev':
            in_petsc = in_vec._petsc
            out_petsc = out_vec._petsc
            for i in range(in_vec._ncol):
                in_petsc.array = in_vec._data[:, i]
                out_petsc.array = out_vec._data[:, i]
                self._transfer.scatter(in_petsc, out_petsc, addv=True, mode=True)
                out_vec._data[:, i] = out_petsc.array
