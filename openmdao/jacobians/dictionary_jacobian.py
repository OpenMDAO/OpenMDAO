"""Define the DictionaryJacobian class."""
import numpy as np
import scipy.sparse as sp

from openmdao.jacobians.jacobian import Jacobian
from openmdao.core.constants import INT_DTYPE


class DictionaryJacobian(Jacobian):
    """
    No global <Jacobian>; use dictionary of user-supplied sub-Jacobians.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    _iter_keys : list of (vname, vname) tuples
        List of tuples of variable names that match subjacs in the this Jacobian.
    _key_owner : dict
        Dict mapping subjac keys to the rank where that subjac is local.
    """

    def __init__(self, system, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(system, **kwargs)
        self._iter_keys = None
        self._key_owner = None

    def _iter_abs_keys(self, system):
        """
        Iterate over subjacs keyed by absolute names.

        This includes only subjacs that have been set and are part of the current system.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.

        Returns
        -------
        list
            List of keys matching this jacobian for the current system.
        """
        if self._iter_keys is None:
            # determine the set of remote keys (keys where either of or wrt is remote somewhere)
            # only if we're under MPI with comm size > 1 and the given system is a Group that
            # computes its derivatives using finite difference or complex step.
            include_remotes = system.pathname and \
                system.comm.size > 1 and system._owns_approx_jac and system._subsystems_allprocs
            subjacs = self._subjacs_info
            keys = []
            if include_remotes:
                ofnames = system._var_allprocs_abs2meta['output']
                wrtnames = system._var_allprocs_abs2meta
            else:
                ofnames = system._var_abs2meta['output']
                wrtnames = system._var_abs2meta

            for res_name in ofnames:
                for type_ in ('output', 'input'):
                    for name in wrtnames[type_]:
                        key = (res_name, name)
                        if key in subjacs:
                            keys.append(key)

            self._iter_keys = keys

            if include_remotes:
                local_out = system._var_abs2meta['output']
                local_in = system._var_abs2meta['input']
                remote_keys = []
                for key in keys:
                    of, wrt = key
                    if of not in local_out or (wrt not in local_in and wrt not in local_out):
                        remote_keys.append(key)

                abs2idx = system._var_allprocs_abs2idx
                sizes_out = system._var_sizes['output']
                sizes_in = system._var_sizes['input']
                owner_dict = {}
                for keys in system.comm.allgather(remote_keys):
                    for key in keys:
                        if key not in owner_dict:
                            of, wrt = key
                            ofsizes = sizes_out[:, abs2idx[of]]
                            if wrt in ofnames:
                                wrtsizes = sizes_out[:, abs2idx[wrt]]
                            else:
                                wrtsizes = sizes_in[:, abs2idx[wrt]]
                            for rank, (ofsz, wrtsz) in enumerate(zip(ofsizes, wrtsizes)):
                                # find first rank where both of and wrt are local
                                if ofsz and wrtsz:
                                    owner_dict[key] = rank
                                    break
                            else:  # no rank was found where both were local. Use 'of' local rank
                                owner_dict[key] = np.min(np.nonzero(ofsizes)[0])

                self._key_owner = owner_dict
            else:
                self._key_owner = {}

        return self._iter_keys

    def _apply(self, system, d_inputs, d_outputs, d_residuals, mode):
        """
        Compute matrix-vector product.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        d_inputs : Vector
            inputs linear vector.
        d_outputs : Vector
            outputs linear vector.
        d_residuals : Vector
            residuals linear vector.
        mode : str
            'fwd' or 'rev'.
        """
        fwd = mode == 'fwd'
        d_out_names = d_outputs._names
        d_res_names = d_residuals._names
        d_inp_names = d_inputs._names

        if not d_out_names and not d_inp_names:
            return

        rflat = d_residuals._abs_get_val
        oflat = d_outputs._abs_get_val
        iflat = d_inputs._abs_get_val
        subjacs_info = self._subjacs_info
        is_explicit = system.is_explicit()
        randgen = self._randgen

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            for abs_key in self._iter_abs_keys(system):
                res_name, other_name = abs_key
                ofvec = rflat(res_name) if res_name in d_res_names else None

                if other_name in d_out_names:
                    wrtvec = oflat(other_name)
                elif other_name in d_inp_names:
                    wrtvec = iflat(other_name)
                else:
                    wrtvec = None

                if fwd:
                    if is_explicit and res_name is other_name and wrtvec is not None:
                        # skip the matvec mult completely for identity subjacs
                        ofvec -= wrtvec
                        continue
                    left_vec = ofvec
                    right_vec = wrtvec
                else:  # rev
                    left_vec = wrtvec
                    right_vec = ofvec

                if abs_key in self._key_owner and abs_key in system._cross_keys:
                    wrtowner = system._owning_rank[other_name]
                    if system.comm.rank == wrtowner:
                        system.comm.bcast(right_vec, root=wrtowner)
                    else:
                        right_vec = system.comm.bcast(None, root=wrtowner)

                if left_vec is not None and right_vec is not None:
                    subjac_info = subjacs_info[abs_key]
                    if randgen:
                        subjac = self._randomize_subjac(subjac_info['val'], abs_key)
                    else:
                        subjac = subjac_info['val']
                    rows = subjac_info['rows']
                    if rows is not None:  # our homegrown COO format
                        linds, rinds = rows, subjac_info['cols']
                        if not fwd:
                            linds, rinds = rinds, linds
                        if self._under_complex_step:
                            # bincount only works with float, so split into parts
                            prod = right_vec[rinds] * subjac
                            left_vec[:].real += np.bincount(linds, prod.real,
                                                            minlength=left_vec.size)
                            left_vec[:].imag += np.bincount(linds, prod.imag,
                                                            minlength=left_vec.size)
                        else:
                            left_vec[:] += np.bincount(linds, right_vec[rinds] * subjac,
                                                       minlength=left_vec.size)

                    else:
                        if fwd:
                            left_vec += subjac.dot(right_vec)
                        else:  # rev
                            subjac = subjac.transpose()
                            left_vec += subjac.dot(right_vec)

                if abs_key in self._key_owner:
                    owner = self._key_owner[abs_key]
                    if owner == system.comm.rank:
                        system.comm.bcast(left_vec, root=owner)
                    elif owner is not None:
                        left_vec = system.comm.bcast(None, root=owner)
                        if fwd:
                            if res_name in d_res_names:
                                d_residuals._abs_set_val(res_name, left_vec)
                        else:  # rev
                            if other_name in d_out_names:
                                d_outputs._abs_set_val(other_name, left_vec)
                            elif other_name in d_inp_names:
                                d_inputs._abs_set_val(other_name, left_vec)


class _CheckingJacobian(DictionaryJacobian):
    """
    A special type of Jacobian that we use only inside of check_partials.

    It checks during set_col to make sure that any user specified rows/cols don't mask any
    nonzero values found in the column being set.
    """

    def __init__(self, system):
        super().__init__(system)
        self._subjacs_info = self._subjacs_info.copy()

        # Convert any scipy.sparse subjacs to OpenMDAO's interal COO specification.
        for key, subjac in self._subjacs_info.items():
            if sp.issparse(subjac['val']):
                coo_val = subjac['val'].tocoo()
                self._subjacs_info[key]['rows'] = coo_val.row
                self._subjacs_info[key]['cols'] = coo_val.col
                self._subjacs_info[key]['val'] = coo_val.data

        self._errors = []

    def __iter__(self):
        for key, _ in self.items():
            yield key

    def items(self):
        from openmdao.core.explicitcomponent import ExplicitComponent
        explicit = isinstance(self._system(), ExplicitComponent)

        for key, meta in self._subjacs_info.items():
            if explicit and key[0] == key[1]:
                continue
            rows = meta['rows']
            if rows is None:
                yield key, meta['val']
            elif 'directional' in meta:
                yield key, np.atleast_2d(meta['val']).T
            else:
                dense = np.zeros(meta['shape'])
                dense[rows, meta['cols']] = meta['val']
                yield key, dense

    def _setup_index_maps(self, system):
        super()._setup_index_maps(system)
        from openmdao.core.component import Component

        if isinstance(system, Component):
            local_opts = system._get_check_partial_options()
        else:
            local_opts = None

        for of, start, end, _, _ in system._jac_of_iter():
            nrows = end - start
            for wrt, wstart, wend, _, _, _ in system._jac_wrt_iter():
                if local_opts:
                    loc_wrt = wrt.rsplit('.', 1)[-1]
                    directional = (loc_wrt in local_opts and
                                   local_opts[loc_wrt]['directional'])
                else:
                    directional = False
                key = (of, wrt)
                if key not in self._subjacs_info:
                    ncols = wend - wstart
                    # create subjacs_info objects for matrix_free systems that don't have them
                    self._subjacs_info[key] = {
                        'rows': None,
                        'cols': None,
                        'val': np.zeros((nrows, 1 if directional else ncols)),
                    }
                elif directional:
                    shape = self._subjacs_info[key]['val'].shape
                    if shape[-1] != 1:
                        self._subjacs_info[key] = meta = self._subjacs_info[key].copy()
                        if len(shape) > 1:
                            meta['val'] = np.atleast_2d(meta['val'][:, 0]).T
                        else:
                            meta['val'] = np.atleast_1d(meta['val'])

    def set_col(self, system, icol, column):
        """
        Set a column of the jacobian.

        The column is assumed to be the same size as a column of the jacobian.

        If the column has any nonzero values that are outside of specified sparsity patterns for
        any of the subjacs, an exception will be raised.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        icol : int
            Column index.
        column : ndarray
            Column value.
        """
        if self._col_varnames is None:
            self._setup_index_maps(system)

        wrt = self._col_varnames[self._col2name_ind[icol]]
        loc_idx = icol - self._col_var_offset[wrt]  # local col index into subjacs

        scratch = np.zeros(column.shape)

        # If we are doing a directional derivative, then the sparsity will be violated.
        # Skip sparsity check if that is the case.
        options = system._get_check_partial_options()
        loc_wrt = wrt.rpartition('.')[2]
        directional = (options is not None and loc_wrt in options and
                       options[loc_wrt]['directional'])

        for of, start, end, _, _ in system._jac_of_iter():
            key = (of, wrt)
            if key in self._subjacs_info:
                subjac = self._subjacs_info[key]
                if subjac['cols'] is None:
                    if subjac['val'] is None:  # can happen for matrix free comp
                        subjac['val'] = np.zeros(subjac['shape'])
                    subjac['val'][:, loc_idx] = column[start:end]
                else:
                    match_inds = np.nonzero(subjac['cols'] == loc_idx)[0]
                    if match_inds.size > 0:
                        row_inds = subjac['rows'][match_inds]
                        if subjac['val'] is None:
                            subjac['val'] = np.zeros(len(subjac['rows']))
                        subjac['val'][match_inds] = column[start:end][row_inds]
                    else:
                        row_inds = np.zeros(0, dtype=INT_DTYPE)

                    if directional:
                        subjac['directional'] = True
                        continue

                    arr = scratch[start:end]
                    arr[:] = column[start:end]
                    arr[row_inds] = 0.
                    nzs = np.nonzero(arr)
                    if nzs[0].size > 0:
                        self._errors.append(f"{system.msginfo}: User specified sparsity (rows/cols)"
                                            f" for subjac '{of}' wrt '{wrt}' is incorrect. There "
                                            f"are non-covered nonzeros in column {loc_idx} at "
                                            f"row(s) {nzs[0]}.")
