"""Define the DictionaryJacobian class."""
import numpy as np

from openmdao.jacobians.jacobian import Jacobian
from openmdao.jacobians.subjac import SUBJAC_META_DEFAULTS


class DictionaryJacobian(Jacobian):
    """
    A Jacobian that stores nonzero subjacobians in a dictionary.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.

    Attributes
    ----------
    _has_children : bool
        True if the system has children, False otherwise.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        super().__init__(system)
        self._has_children = bool(system._subsystems_allprocs)
        self._setup(system)

    def _setup(self, system):
        self._subjacs = self._get_subjacs(system)

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

        abs_resids = d_residuals._abs_get_val
        abs_outs = d_outputs._abs_get_val
        abs_ins = d_inputs._abs_get_val
        subjacs = self._get_subjacs(system)
        randgen = self._randgen

        key_owners = system._get_subjac_owners()

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            for abs_key in self._get_ordered_subjac_keys(system):
                res_name, other_name = abs_key

                ofvec = abs_resids(res_name) if res_name in d_res_names else None

                if other_name in d_out_names:
                    wrtvec = abs_outs(other_name)
                elif other_name in d_inp_names:
                    wrtvec = abs_ins(other_name)
                else:
                    wrtvec = None

                if self._has_children and abs_key in system._cross_keys and abs_key in key_owners:
                    wrtowner = key_owners[abs_key]
                    if system.comm.rank == wrtowner:
                        system.comm.bcast(wrtvec, root=wrtowner)
                    else:
                        wrtvec = system.comm.bcast(None, root=wrtowner)

                if fwd:
                    left_vec = ofvec
                    right_vec = wrtvec
                    if left_vec is not None and right_vec is not None:
                        subjacs[abs_key].apply_fwd(d_inputs, d_outputs, d_residuals, randgen)
                else:  # rev
                    left_vec = wrtvec
                    right_vec = ofvec
                    if left_vec is not None and right_vec is not None:
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

        self._setup_index_maps(system)
        self._subjacs = self._get_subjacs(system)

    def __iter__(self):
        for key, _ in self.items():
            yield key

    def items(self):
        for key, subjac in self._subjacs.items():
            meta = subjac.info
            if self._is_explicitcomp and key[0] == key[1]:
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

        for of, start, end, _, _ in system._get_jac_ofs():
            nrows = end - start
            for wrt, wstart, wend, vec, _, _ in system._get_jac_wrts():
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
                    subjac['shape'] = subjac['val'].shape

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
        if self._col_mapper is None:
            self._setup_index_maps(system)

        wrt, loc_idx = self._col_mapper.index2key_rel(icol)  # local col index into subjacs

        # If we are doing a directional derivative, then the sparsity will be violated.
        # Skip sparsity check if that is the case.
        options = system._get_check_partial_options()
        loc_wrt = wrt.rpartition('.')[2]
        directional = (options is not None and loc_wrt in options and
                       options[loc_wrt]['directional'])

        subjacs = self._get_subjacs(system)

        for of, start, end, _, _ in system._get_jac_ofs():
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


class ExplicitDictionaryJacobian(Jacobian):
    """
    A DictionaryJacobian that is a collection of sub-Jacobians.

    It is intended to be used with ExplicitComponents only because dr/do is assumed to be -I.

    Parameters
    ----------
    system : System
        System that is updating this jacobian.  Must be an ExplicitComponent.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        super().__init__(system)
        self._subjacs = self._get_subjacs(system)

    def _get_subjacs(self, system=None):
        """
        Get the subjacs for the current system, creating them if necessary based on _subjacs_info.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.

        Returns
        -------
        dict
            Dictionary of subjacs keyed by absolute names.
        """
        if not self._initialized:
            if not self._is_explicitcomp:
                msginfo = system.msginfo if system else ''
                raise RuntimeError(f"{msginfo}: ExplicitDictionaryJacobian is only intended to be "
                                   "used with ExplicitComponents.")

            rel_subjacs, irrelevant_subjacs = self._get_relevant_subjacs_info(system)
            self._subjacs = {}
            self._irrelevant_subjacs = {}
            for key, meta, dtype in rel_subjacs:
                # only keep dr/di subjacs.  dr/do matrix is always -I
                if key[1] in self._input_slices:
                    self._subjacs[key] = self.create_subjac(key, meta, dtype)
            for key, meta, dtype in irrelevant_subjacs:
                self._irrelevant_subjacs[key] = self.create_subjac(key, meta, dtype)

            self._initialized = True

        return self._subjacs

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
        randgen = self._randgen

        d_inp_names = d_inputs._names

        with system._unscaled_context(outputs=(d_outputs,), residuals=(d_residuals,)):
            dresids = d_residuals.asarray()

            if mode == 'fwd':
                if d_outputs._names:
                    # apply dr/do of -I
                    dresids -= d_outputs.asarray()

                if d_inp_names:
                    for key, subjac in self._get_subjacs(system).items():
                        if key[1] in d_inp_names:
                            subjac.apply_fwd(d_inputs, d_outputs, d_residuals, randgen)
            else:  # rev
                if d_outputs._names:
                    # apply dr/do of -I
                    doutarr = d_outputs.asarray()
                    doutarr -= dresids

                if d_inp_names:
                    for key, subjac in self._get_subjacs(system).items():
                        if key[1] in d_inp_names:
                            subjac.apply_rev(d_inputs, d_outputs, d_residuals, randgen)

    def todense(self):
        """
        Return a dense version of the jacobian.

        Returns
        -------
        ndarray
            Dense version of the jacobian.
        """
        if self._subjacs:
            lst = [-np.eye(self.shape[0])]
            drdi_shape = (self.shape[0], self.shape[1] - self.shape[0])
            J_dr_di = np.zeros(drdi_shape)
            lst.append(J_dr_di)

            for subjac in self._subjacs.values():
                J_dr_di[subjac.row_slice, subjac.col_slice] = subjac.todense()

            return np.hstack(lst)

        return -np.eye(self.shape[0])
