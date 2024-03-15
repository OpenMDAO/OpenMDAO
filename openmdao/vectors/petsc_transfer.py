"""Define the PETSc Transfer class."""
import numpy as np
from openmdao.utils.mpi import check_mpi_env
from openmdao.core.constants import INT_DTYPE

use_mpi = check_mpi_env()
_empty_idx_array = np.array([], dtype=INT_DTYPE)


if use_mpi is False:
    PETScTransfer = None
else:
    try:
        import petsc4py
        from petsc4py import PETSc
    except ImportError:
        PETSc = None
        if use_mpi is True:
            raise ImportError("Importing petsc4py failed and OPENMDAO_USE_MPI is true.")

    from petsc4py import PETSc
    from collections import defaultdict

    from openmdao.vectors.default_transfer import DefaultTransfer, _setup_index_views

    class PETScTransfer(DefaultTransfer):
        """
        PETSc Transfer implementation for running in parallel.

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
        comm : MPI.Comm or <FakeComm>
            Communicator of the system that owns this transfer.

        Attributes
        ----------
        _scatter : method
            Method that performs a PETSc scatter.
        """

        def __init__(self, in_vec, out_vec, in_inds, out_inds, comm):
            """
            Initialize all attributes.
            """
            super().__init__(in_vec, out_vec, in_inds, out_inds)
            in_indexset = PETSc.IS().createGeneral(self._in_inds, comm=comm)
            out_indexset = PETSc.IS().createGeneral(self._out_inds, comm=comm)

            self._scatter = PETSc.Scatter().create(out_vec._petsc, out_indexset, in_vec._petsc,
                                                   in_indexset).scatter

        @staticmethod
        def _setup_transfers(group):
            """
            Compute all transfers that are owned by our parent group.

            Parameters
            ----------
            group : <Group>
                Parent group.
            """
            rev = group._orig_mode != 'fwd'

            group._transfers = {
                'fwd': PETScTransfer._setup_transfers_fwd(group)
            }

            if rev:
                group._transfers['rev'] = PETScTransfer._setup_transfers_rev(group)

        @staticmethod
        def _setup_transfers_fwd(group):
            transfers = {}

            if not group._conn_abs_in2out:
                return transfers

            abs2meta_in = group._var_abs2meta['input']
            myrank = group.comm.rank

            offsets_in = group._get_var_offsets()['input'][myrank, :]
            mypathlen = len(group.pathname) + 1 if group.pathname else 0

            xfer_in = defaultdict(list)
            xfer_out = defaultdict(list)

            allprocs_abs2idx = group._var_allprocs_abs2idx
            sizes_in = group._var_sizes['input'][myrank, :]

            total_len = 0

            # Loop through all connections owned by this system
            for abs_in, abs_out in group._conn_abs_in2out.items():
                sub_in = abs_in[mypathlen:].partition('.')[0]

                # Only continue if the input exists on this processor
                if abs_in in abs2meta_in:

                    output_inds, _ = _get_output_inds(group, abs_out, abs_in)

                    idx_in = allprocs_abs2idx[abs_in]
                    input_inds = range(offsets_in[idx_in], offsets_in[idx_in] + sizes_in[idx_in])

                    total_len += len(input_inds)

                    xfer_in[sub_in].append(input_inds)
                    xfer_out[sub_in].append(output_inds)
                else:
                    # not a local input but still need entries in the transfer dicts to
                    # avoid hangs
                    xfer_in[sub_in]  # defaultdict will create an empty list there
                    xfer_out[sub_in]

            if xfer_in:
                full_xfer_in, full_xfer_out = _setup_index_views(total_len, xfer_in, xfer_out)
                # full transfer (transfer to all subsystems at once)
                transfers[None] = PETScTransfer(group._vectors['input']['nonlinear'],
                                                group._vectors['output']['nonlinear'],
                                                full_xfer_in, full_xfer_out, group.comm)

                # transfers to individual subsystems
                for sname, inds in xfer_in.items():
                    transfers[sname] = PETScTransfer(group._vectors['input']['nonlinear'],
                                                     group._vectors['output']['nonlinear'],
                                                     inds, xfer_out[sname], group.comm)

            return transfers

        @staticmethod
        def _setup_transfers_rev(group):
            abs2meta_in = group._var_abs2meta['input']
            abs2meta_out = group._var_abs2meta['output']
            allprocs_abs2prom = group._var_allprocs_abs2prom

            # for an FD group, we use the relevance graph to determine which inputs on the
            # boundary of the group are upstream of responses within the group so
            # that we can perform any necessary corrections to the derivative inputs.
            if group._owns_approx_jac:
                if group.comm.size > 1 and group.pathname != '' and group._has_distrib_vars:
                    all_abs2meta_out = group._var_allprocs_abs2meta['output']
                    all_abs2meta_in = group._var_allprocs_abs2meta['input']

                    # connections internal to this group and all direct/indirect subsystems
                    conns = group._conn_global_abs_in2out

                    inp_boundary_set = set(all_abs2meta_in).difference(conns)

                    if inp_boundary_set:
                        for dv, resp, rel in group._relevance.iter_seed_pair_relevance(inputs=True):
                            if resp in all_abs2meta_out and dv not in allprocs_abs2prom:
                                # response is continuous and inside this group and
                                # dv is outside this group
                                if all_abs2meta_out[resp]['distributed']:  # a distributed response
                                    for inp in inp_boundary_set.intersection(rel):
                                        if inp in abs2meta_in:
                                            if resp not in group._fd_rev_xfer_correction_dist:
                                                group._fd_rev_xfer_correction_dist[resp] = set()
                                            group._fd_rev_xfer_correction_dist[resp].add(inp)

                # FD groups don't need reverse transfers
                return {}

            myrank = group.comm.rank
            allprocs_abs2idx = group._var_allprocs_abs2idx
            transfers = group._transfers
            vectors = group._vectors
            offsets = group._get_var_offsets()
            mypathlen = len(group.pathname) + 1 if group.pathname else 0

            has_par_coloring = group._problem_meta['has_par_deriv_color']

            xfer_in = defaultdict(list)
            xfer_out = defaultdict(list)

            # xfers that are only active when parallel coloring is not
            xfer_in_nocolor = defaultdict(list)
            xfer_out_nocolor = defaultdict(list)

            sizes_in = group._var_sizes['input']
            offsets_in = offsets['input']
            offsets_out = offsets['output']

            total_size = total_size_nocolor = 0

            # Loop through all connections owned by this system
            for abs_in, abs_out in group._conn_abs_in2out.items():
                sub_out = abs_out[mypathlen:].partition('.')[0]

                # Only continue if the input exists on this processor
                if abs_in in abs2meta_in:
                    meta_in = abs2meta_in[abs_in]
                    idx_in = allprocs_abs2idx[abs_in]
                    idx_out = allprocs_abs2idx[abs_out]

                    output_inds, src_indices = _get_output_inds(group, abs_out, abs_in)

                    # 2. Compute the input indices
                    input_inds = range(offsets_in[myrank, idx_in],
                                       offsets_in[myrank, idx_in] + sizes_in[myrank, idx_in])

                    # Now the indices are ready - input_inds, output_inds
                    inp_is_dup, inp_missing, distrib_in = group.get_var_dup_info(abs_in, 'input')
                    out_is_dup, _, distrib_out = group.get_var_dup_info(abs_out, 'output')

                    iowninput = myrank == group._owning_rank[abs_in]

                    if inp_is_dup and (abs_out not in abs2meta_out or
                                       (distrib_out and not iowninput)):
                        xfer_in[sub_out]
                        xfer_out[sub_out]
                    elif out_is_dup and inp_missing > 0 and (iowninput or distrib_in):
                        # if this rank owns the input or the input is distributed,
                        # and the output is duplicated, then we send the owning/distrib input
                        # to each duplicated output that doesn't have a corresponding connected
                        # input on the same rank.
                        oidxlist = []
                        iidxlist = []
                        oidxlist_nc = []
                        iidxlist_nc = []
                        size = size_nc = 0
                        for rnk, osize, isize in zip(range(group.comm.size),
                                                     group.get_var_sizes(abs_out, 'output'),
                                                     group.get_var_sizes(abs_in, 'input')):
                            if rnk == myrank:  # transfer to output on same rank
                                oidxlist.append(output_inds)
                                iidxlist.append(input_inds)
                                size += len(input_inds)
                            elif osize > 0 and isize == 0:
                                # dup output exists on this rank but there is no corresponding
                                # input, so we send the owning/distrib input to the dup output
                                offset = offsets_out[rnk, idx_out]
                                if src_indices is None:
                                    oarr = range(offset, offset + meta_in['size'])
                                elif src_indices.size > 0:
                                    oarr = np.asarray(src_indices + offset, dtype=INT_DTYPE)
                                else:
                                    continue

                                if has_par_coloring:
                                    # these transfers will only happen if parallel coloring is
                                    # not active for the current seed response
                                    oidxlist_nc.append(oarr)
                                    iidxlist_nc.append(input_inds)
                                    size_nc += len(input_inds)
                                else:
                                    oidxlist.append(oarr)
                                    iidxlist.append(input_inds)
                                    size += len(input_inds)

                        if len(iidxlist) > 1:
                            input_inds = _merge(iidxlist, size)
                            output_inds = _merge(oidxlist, size)
                        else:
                            input_inds = iidxlist[0]
                            output_inds = oidxlist[0]

                        total_size += len(input_inds)

                        xfer_in[sub_out].append(input_inds)
                        xfer_out[sub_out].append(output_inds)

                        if has_par_coloring and iidxlist_nc:
                            # keep transfers separate that shouldn't happen when parallel
                            # deriv coloring is active
                            if len(iidxlist_nc) > 1:
                                input_inds = _merge(iidxlist_nc, size_nc)
                                output_inds = _merge(oidxlist_nc, size_nc)
                            else:
                                input_inds = iidxlist_nc[0]
                                output_inds = oidxlist_nc[0]

                            total_size_nocolor += len(input_inds)

                            xfer_in_nocolor[sub_out].append(input_inds)
                            xfer_out_nocolor[sub_out].append(output_inds)
                    else:
                        if (inp_is_dup and out_is_dup and src_indices is not None and
                                src_indices.size > 0):
                            offset = offsets_out[myrank, idx_out]
                            output_inds = np.asarray(src_indices + offset, dtype=INT_DTYPE)

                        total_size += len(input_inds)

                        xfer_in[sub_out].append(input_inds)
                        xfer_out[sub_out].append(output_inds)
                else:
                    # remote input but still need entries in the transfer dicts to avoid hangs
                    xfer_in[sub_out]
                    xfer_out[sub_out]
                    if has_par_coloring:
                        xfer_in_nocolor[sub_out]
                        xfer_out_nocolor[sub_out]

            full_xfer_in, full_xfer_out = _setup_index_views(total_size, xfer_in, xfer_out)

            transfers = {
                None: PETScTransfer(vectors['input']['nonlinear'],
                                    vectors['output']['nonlinear'],
                                    full_xfer_in, full_xfer_out, group.comm)
            }

            for sname, inds in xfer_out.items():
                transfers[sname] = PETScTransfer(vectors['input']['nonlinear'],
                                                 vectors['output']['nonlinear'],
                                                 xfer_in[sname], inds, group.comm)

            if xfer_in_nocolor:
                full_xfer_in, full_xfer_out = _setup_index_views(total_size_nocolor,
                                                                 xfer_in_nocolor,
                                                                 xfer_out_nocolor)

                transfers[(None, '@nocolor')] = PETScTransfer(vectors['input']['nonlinear'],
                                                              vectors['output']['nonlinear'],
                                                              full_xfer_in, full_xfer_out,
                                                              group.comm)

                for sname, inds in xfer_out_nocolor.items():
                    transfers[(sname, '@nocolor')] = PETScTransfer(vectors['input']['nonlinear'],
                                                                   vectors['output']['nonlinear'],
                                                                   xfer_in_nocolor[sname], inds,
                                                                   group.comm)

            return transfers

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
                    in_petsc.array = in_vec._get_data()

                if out_vec._alloc_complex:
                    out_petsc.array = out_vec._get_data()

                self._scatter(out_petsc, in_petsc, addv=flag, mode=flag)

                if in_vec._alloc_complex:
                    data = in_vec._get_data()
                    data[:] = in_petsc.array


def _merge(inds_list, tot_size):
    """
    Convert a list of indices and/or ranges into an array.

    Parameters
    ----------
    inds_list : list of ranges or ndarrays
        List of indices.
    tot_size : int
        Total size of the indices in the list.

    Returns
    -------
    ndarray
        Array of indices.
    """
    if inds_list:
        arr = np.empty(tot_size, dtype=INT_DTYPE)
        start = end = 0
        for inds in inds_list:
            end += len(inds)
            arr[start:end] = inds
            start = end

        return arr

    return _empty_idx_array


def _get_output_inds(group, abs_out, abs_in):
    owner = group._owning_rank[abs_out]
    meta_in = group._var_abs2meta['input'][abs_in]
    out_dist = group._var_allprocs_abs2meta['output'][abs_out]['distributed']
    in_dist = meta_in['distributed']
    src_indices = meta_in['src_indices']

    rank = group.comm.rank if abs_out in group._var_abs2meta['output'] else owner
    out_idx = group._var_allprocs_abs2idx[abs_out]
    offsets = group._get_var_offsets()['output'][:, out_idx]
    sizes = group._var_sizes['output'][:, out_idx]

    if src_indices is None:
        orig_src_inds = src_indices
    else:
        src_indices = src_indices.shaped_array()
        orig_src_inds = src_indices
        if not out_dist and not in_dist:  # convert from local to distributed src_indices
            off = np.sum(sizes[:rank])
            if off > 0.:  # adjust for local offsets
                # don't do += to avoid modifying stored value
                src_indices = src_indices + off

    # NOTE: src_indices are relative to a single, possibly distributed variable,
    # while the output_inds that we compute are relative to the full distributed
    # array that contains all local variables from each rank stacked in rank order.
    if src_indices is None:
        if out_dist:
            # input in this case is non-distributed (else src_indices would be
            # defined by now).  dist output to non-distributed input conns w/o
            # src_indices are not allowed.
            raise RuntimeError(f"{group.msginfo}: Can't connect distributed output "
                               f"'{abs_out}' to non-distributed input '{abs_in}' "
                               "without declaring src_indices.", ident=(abs_out, abs_in))
        else:
            offset = offsets[rank]
            output_inds = range(offset, offset + sizes[rank])
    else:
        output_inds = np.empty(src_indices.size, INT_DTYPE)
        start = end = 0
        for iproc in range(group.comm.size):
            end += sizes[iproc]
            if start == end:
                continue

            # The part of src on iproc
            on_iproc = np.logical_and(start <= src_indices, src_indices < end)

            if np.any(on_iproc):
                # This converts from global to variable specific ordering
                output_inds[on_iproc] = src_indices[on_iproc] + (offsets[iproc] - start)

            start = end

    return output_inds, orig_src_inds
