"""Define the PETSc Transfer class."""
import numpy as np
from petsc4py import PETSc
from itertools import product, chain
from collections import defaultdict

from openmdao.vectors.transfer import Transfer
from openmdao.vectors.default_transfer import DefaultTransfer, _merge
from openmdao.vectors.vector import INT_DTYPE
from openmdao.utils.mpi import MPI
from openmdao.utils.array_utils import convert_neg, _flatten_src_indices
from openmdao.utils.general_utils import _is_slice, _slice_indices

_empty_idx_array = np.array([], dtype=INT_DTYPE)


class PETScTransfer(DefaultTransfer):
    """
    PETSc Transfer implementation for running in parallel.

    Attributes
    ----------
    _scatter : method
        Method that performs a PETSc scatter.
    _transfer : method
        Method that performs either a normal transfer or a multi-transfer.
    """

    def __init__(self, in_vec, out_vec, in_inds, out_inds, comm):
        """
        Initialize all attributes.

        Parameters
        ----------
        in_vec : <Vector>
            pointer to the input vector.
        out_vec : <Vector>
            pointer to the output vector.
        in_inds : int ndarray
            input indices for the transfer.
        out_inds : int ndarray
            output indices for the transfer.
        comm : MPI.Comm or <FakeComm>
            communicator of the system that owns this transfer.
        """
        super(PETScTransfer, self).__init__(in_vec, out_vec, in_inds, out_inds, comm)
        in_indexset = PETSc.IS().createGeneral(self._in_inds, comm=self._comm)
        out_indexset = PETSc.IS().createGeneral(self._out_inds, comm=self._comm)

        self._scatter = PETSc.Scatter().create(out_vec._petsc, out_indexset, in_vec._petsc,
                                               in_indexset).scatter

        if in_vec._ncol > 1:
            self._transfer = self._multi_transfer

    @staticmethod
    def _setup_transfers(group):
        """
        Compute all transfers that are owned by our parent group.

        Parameters
        ----------
        group : <Group>
            Parent group.
        """
        rev = group._mode != 'fwd'

        for subsys in group._subgroups_myproc:
            subsys._setup_transfers()

        abs2meta = group._var_abs2meta
        allprocs_abs2meta = group._var_allprocs_abs2meta
        myproc = group.comm.rank

        transfers = group._transfers = {}
        vectors = group._vectors
        offsets = group._get_var_offsets()

        vec_names = group._lin_rel_vec_name_list if group._use_derivatives else group._vec_names

        mypathlen = len(group.pathname + '.' if group.pathname else '')
        sub_inds = group._subsystems_inds

        for vec_name in vec_names:
            relvars, _ = group._relevant[vec_name]['@all']

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

            # Loop through all connections owned by this system
            for abs_in, abs_out in group._conn_abs_in2out.items():
                if abs_out not in relvars['output'] or abs_in not in relvars['input']:
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
                        owner = group._owning_rank[abs_out]
                        # if the input is larger than the output on a single proc, we have
                        # to just loop over the procs in the same way we do when src_indices
                        # is defined.
                        if meta_in['size'] > sizes_out[owner, idx_out]:
                            src_indices = np.arange(meta_in['size'], dtype=INT_DTYPE)
                    elif src_indices.ndim == 1:
                        if _is_slice(src_indices):
                            indices = _slice_indices(src_indices, meta_out['global_size'],
                                                     meta_out['global_shape'])
                            src_indices = convert_neg(indices, meta_out['global_size'])
                    else:
                        src_indices = _flatten_src_indices(src_indices, meta_in['shape'],
                                                           meta_out['global_shape'],
                                                           meta_out['global_size'])

                    # 1. Compute the output indices
                    # NOTE: src_indices are relative to a single, possibly distributed variable,
                    # while the output_inds that we compute are relative to the full distributed
                    # array that contains all local variables from each rank stacked in rank order.
                    if src_indices is None:
                        if meta_out['distributed']:
                            # input in this case is non-distributed (else src_indices would be
                            # defined by now).  The input size must match the full
                            # distributed size of the output.
                            for rank, sz in enumerate(sizes_out[:, idx_out]):
                                if sz > 0:
                                    out_offset = offsets_out[rank, idx_out]
                                    break
                            output_inds = np.arange(out_offset,
                                                    out_offset + meta_out['global_size'],
                                                    dtype=INT_DTYPE)
                        else:
                            rank = myproc if abs_out in abs2meta else owner
                            offset = offsets_out[rank, idx_out]
                            output_inds = np.arange(offset, offset + meta_in['size'],
                                                    dtype=INT_DTYPE)
                    else:
                        output_inds = np.zeros(src_indices.size, INT_DTYPE)
                        start = end = 0
                        for iproc in range(group.comm.size):
                            end += sizes_out[iproc, idx_out]
                            if start == end:
                                continue

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
                    sub_in = abs_in[mypathlen:].split('.', 1)[0]
                    isub = sub_inds[sub_in]
                    fwd_xfer_in[isub].append(input_inds)
                    fwd_xfer_out[isub].append(output_inds)
                    if rev:
                        sub_out = abs_out[mypathlen:].split('.', 1)[0]
                        isub = sub_inds[sub_out]
                        rev_xfer_in[isub].append(input_inds)
                        rev_xfer_out[isub].append(output_inds)

            transfers[vec_name] = {}

            for isub in range(nsub_allprocs):
                fwd_xfer_in[isub] = _merge(fwd_xfer_in[isub])
                fwd_xfer_out[isub] = _merge(fwd_xfer_out[isub])
                if rev:
                    rev_xfer_in[isub] = _merge(rev_xfer_in[isub])
                    rev_xfer_out[isub] = _merge(rev_xfer_out[isub])

            xfer_in = np.concatenate(fwd_xfer_in)
            xfer_out = np.concatenate(fwd_xfer_out)

            out_vec = vectors['output'][vec_name]

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
            self._scatter(out_petsc, in_petsc, addv=flag, mode=flag)

            # Imaginary
            in_petsc_imag = in_vec._imag_petsc
            out_petsc_imag = out_vec._imag_petsc
            in_petsc_imag.array = in_vec._data.imag
            out_petsc_imag.array = out_vec._data.imag
            self._scatter(out_petsc_imag, in_petsc_imag, addv=flag, mode=flag)

            in_vec._data[:] = in_petsc.array + in_petsc_imag.array * 1j

        else:

            # Anything that has been allocated complex requires an additional step because
            # the petsc vector does not directly reference the _data.

            if in_vec._alloc_complex:
                in_petsc.array = in_vec._data

            if out_vec._alloc_complex:
                out_petsc.array = out_vec._data

            self._scatter(out_petsc, in_petsc, addv=flag, mode=flag)

            if in_vec._alloc_complex:
                in_vec._data[:] = in_petsc.array

    def _multi_transfer(self, in_vec, out_vec, mode='fwd'):
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
                    self._scatter(out_petsc, in_petsc, addv=False, mode=False)

                    # Imaginary
                    in_petsc_imag.array = in_vec._data[:, i].imag
                    out_petsc_imag.array = out_vec._data[:, i].imag
                    self._scatter(out_petsc_imag, in_petsc_imag, addv=False, mode=False)

                    in_vec._data[:, i] = in_petsc.array + in_petsc_imag.array * 1j

            else:
                for i in range(in_vec._ncol):
                    in_petsc.array = in_vec._data[:, i]
                    out_petsc.array = out_vec._data[:, i]
                    self._scatter(out_petsc, in_petsc, addv=False, mode=False)
                    in_vec._data[:, i] = in_petsc.array

        elif mode == 'rev':
            in_petsc = in_vec._petsc
            out_petsc = out_vec._petsc
            for i in range(in_vec._ncol):
                in_petsc.array = in_vec._data[:, i]
                out_petsc.array = out_vec._data[:, i]
                self._scatter(in_petsc, out_petsc, addv=True, mode=True)
                out_vec._data[:, i] = out_petsc.array
