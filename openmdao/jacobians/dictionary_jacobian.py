"""Define the DictionaryJacobian class."""
import numpy as np

from openmdao.jacobians.jacobian import Jacobian
from openmdao.jacobians.subjac import SUBJAC_META_DEFAULTS


class DictionaryJacobian(Jacobian):
    """
    No global <Jacobian>; use dictionary of user-supplied sub-Jacobians.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.

    Attributes
    ----------
    _iter_keys : list of (vname, vname) tuples
        List of tuples of variable names that match subjacs in the this Jacobian.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        super().__init__(system)
        self._iter_keys = None
        self._setup(system)

    def _setup(self, system):
        self._subjacs = self._get_subjacs()

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
        if self._update_needed:
            self._update(system)

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
        subjacs = self._get_subjacs()
        randgen = self._randgen

        do_reset = False
        key_owners = system._get_subjac_owners()

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            for abs_key in self._get_ordered_subjac_keys(system):
                if abs_key not in subjacs_info:
                    # for components that compute sparsity at first linearization, some subjacs
                    # will be determined to be zero and removed from subjacs_info., so we need to
                    # update our iteration keys.
                    do_reset = True
                    continue

                res_name, other_name = abs_key

                ofvec = rflat(res_name) if res_name in d_res_names else None

                if other_name in d_out_names:
                    wrtvec = oflat(other_name)
                    # if fwd and is_explicit and res_name is other_name:
                    #     # skip the matvec mult completely for identity subjacs
                    #     ofvec -= wrtvec
                    #     continue
                elif other_name in d_inp_names:
                    wrtvec = iflat(other_name)
                else:
                    wrtvec = None

                if abs_key in system._cross_keys and abs_key in key_owners:
                    wrtowner = key_owners[abs_key]
                    if system.comm.rank == wrtowner:
                        system.comm.bcast(wrtvec, root=wrtowner)
                    else:
                        wrtvec = system.comm.bcast(None, root=wrtowner)

                if fwd:
                    left_vec = ofvec
                    right_vec = wrtvec
                else:  # rev
                    left_vec = wrtvec
                    right_vec = ofvec

                if left_vec is not None and right_vec is not None:
                    if fwd:
                        subjacs[abs_key].apply_fwd(d_inputs, d_outputs, d_residuals, randgen)
                    else:
                        subjacs[abs_key].apply_rev(d_inputs, d_outputs, d_residuals, randgen)

                if abs_key in key_owners:
                    owner = key_owners[abs_key]
                    if owner == system.comm.rank:
                        system.comm.bcast(left_vec, root=owner)
                    elif owner is not None:
                        left_vec = system.comm.bcast(None, root=owner)
                        if left_vec is not None:
                            if fwd:
                                if res_name in d_res_names:
                                    d_residuals._abs_set_val(res_name, left_vec)
                            else:  # rev
                                if other_name in d_out_names:
                                    d_outputs._abs_set_val(other_name, left_vec)
                                elif other_name in d_inp_names:
                                    d_inputs._abs_set_val(other_name, left_vec)

            if do_reset:
                self._iter_keys = None  # subjacs_info has been reduced, so update iter keys


class _CheckingJacobian(DictionaryJacobian):
    """
    A special type of Jacobian that we use only inside of check_partials.

    It checks during set_col to make sure that any user specified rows/cols don't mask any
    nonzero values found in the column being set.
    """

    def __init__(self, system, uncovered_threshold=1.0E-16):
        self._uncovered_threshold = uncovered_threshold
        super().__init__(system)

    def _setup(self, system):
        self._subjacs_info = self._subjacs_info.copy()

        # # Convert any scipy.sparse subjacs to OpenMDAO's interal COO specification.
        # for key, subjac in self._subjacs_info.items():
        #     if sp.issparse(subjac['val']):
        #         coo_val = subjac['val'].tocoo()
        #         self._subjacs_info[key]['rows'] = coo_val.row
        #         self._subjacs_info[key]['cols'] = coo_val.col
        #         self._subjacs_info[key]['val'] = coo_val.data

        self._setup_index_maps(system)
        self._subjacs = self._get_subjacs()

    def __iter__(self):
        for key, _ in self.items():
            yield key

    def items(self):
        from openmdao.core.explicitcomponent import ExplicitComponent
        explicit = isinstance(self._system(), ExplicitComponent)

        self._get_subjacs()

        for key, subjac in self._subjacs.items():
            meta = subjac.info
            if explicit and key[0] == key[1]:
                continue
            if 'directional' in meta:
                yield key, np.atleast_2d(meta['val']).T
            else:
                yield key, subjac.todense()

    def _setup_index_maps(self, system):
        super()._setup_index_maps(system)
        from openmdao.core.component import Component

        if isinstance(system, Component):
            local_opts = system._get_check_partial_options()
        else:
            local_opts = None

        for of, start, end, _, _ in system._jac_of_iter():
            nrows = end - start
            for wrt, wstart, wend, vec, _, _ in system._jac_wrt_iter():
                if local_opts:
                    loc_wrt = wrt.rsplit('.', 1)[-1]
                    directional = (loc_wrt in local_opts and
                                   local_opts[loc_wrt]['directional'])
                else:
                    directional = False
                key = (of, wrt)
                if vec is not None and key not in self._subjacs_info:
                    ncols = wend - wstart
                    # create subjacs_info objects for matrix_free systems that don't have them.
                    # Note that this is our own copy of _subjacs_info so when this instance is
                    # destroyed, any extra allocated subjacs will be garbage collected.
                    self._subjacs_info[key] = subjac = SUBJAC_META_DEFAULTS.copy()
                    subjac['val'] = np.zeros((nrows, 1 if directional else ncols))

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
        any of the subjacs, the information will be saved in subjacs_info so we can report it
        during the derivative test.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        icol : int
            Column index.
        column : ndarray
            Column value.
        """
        self._update_needed = True
        wrt, loc_idx = self._col_mapper.index2key_rel(icol)  # local col index into subjacs

        # If we are doing a directional derivative, then the sparsity will be violated.
        # Skip sparsity check if that is the case.
        options = system._get_check_partial_options()
        loc_wrt = wrt.rpartition('.')[2]
        directional = (options is not None and loc_wrt in options and
                       options[loc_wrt]['directional'])

        system = self._system()
        subjacs = self._get_subjacs()

        for of, start, end, _, _ in system._jac_of_iter():
            key = (of, wrt)
            if key in subjacs:
                subjac = subjacs[key]
                info = subjac.info
                if info['diagonal']:
                    subjac.set_col(loc_idx, column[start:end], self._uncovered_threshold)
                    if directional:
                        info['directional'] = True
                elif info['cols'] is not None:
                    subjac.set_col(loc_idx, column[start:end], self._uncovered_threshold)
                    if directional:
                        info['directional'] = True
                        continue
                else:
                    subjac.set_col(loc_idx, column[start:end], self._uncovered_threshold)
