"""Define the DictionaryJacobian class."""
import numpy as np

from openmdao.jacobians.jacobian import Jacobian
from openmdao.core.constants import INT_DTYPE


class DictionaryJacobian(Jacobian):
    """
    No global <Jacobian>; use dictionary of user-supplied sub-Jacobians.

    Attributes
    ----------
    _iter_keys : list of (vname, vname) tuples
        List of tuples of variable names that match subjacs in the this Jacobian.
    """

    def __init__(self, system, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        system : System
            Parent system to this jacobian.
        **kwargs : dict
            options dictionary.
        """
        super().__init__(system, **kwargs)
        self._iter_keys = {}

    def _iter_abs_keys(self, system, vec_name):
        """
        Iterate over subjacs keyed by absolute names.

        This includes only subjacs that have been set and are part of the current system.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        vec_name : str
            The name of the current RHS vector.

        Returns
        -------
        list
            List of keys matching this jacobian for the current system.
        """
        entry = (system.pathname, vec_name)

        if entry not in self._iter_keys:
            subjacs = self._subjacs_info
            keys = []
            for res_name in system._var_relevant_names[vec_name]['output']:
                for type_ in ('output', 'input'):
                    for name in system._var_relevant_names[vec_name][type_]:
                        key = (res_name, name)
                        if key in subjacs:
                            keys.append(key)

            self._iter_keys[entry] = keys

        return self._iter_keys[entry]

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
        # avoid circular import
        from openmdao.core.explicitcomponent import ExplicitComponent

        fwd = mode == 'fwd'
        d_res_names = d_residuals._names
        d_out_names = d_outputs._names
        d_inp_names = d_inputs._names

        if not d_out_names and not d_inp_names:
            return

        rflat = d_residuals._abs_get_val
        oflat = d_outputs._abs_get_val
        iflat = d_inputs._abs_get_val
        ncol = d_residuals._ncol
        subjacs_info = self._subjacs_info
        is_explicit = isinstance(system, ExplicitComponent)

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            for abs_key in self._iter_abs_keys(system, d_residuals._name):
                res_name, other_name = abs_key
                if res_name in d_res_names:
                    if other_name in d_out_names:
                        # skip the matvec mult completely for identity subjacs
                        if is_explicit and res_name is other_name:
                            if fwd:
                                val = rflat(res_name)
                                val -= oflat(other_name)
                            else:
                                val = oflat(other_name)
                                val -= rflat(res_name)
                            continue
                        if fwd:
                            left_vec = rflat(res_name)
                            right_vec = oflat(other_name)
                        else:
                            left_vec = oflat(other_name)
                            right_vec = rflat(res_name)
                    elif other_name in d_inp_names:
                        if fwd:
                            left_vec = rflat(res_name)
                            right_vec = iflat(other_name)
                        else:
                            left_vec = iflat(other_name)
                            right_vec = rflat(res_name)
                    else:
                        continue

                    subjac_info = subjacs_info[abs_key]
                    if self._randomize:
                        subjac = self._randomize_subjac(subjac_info['value'], abs_key)
                    else:
                        subjac = subjac_info['value']
                    rows = subjac_info['rows']
                    if rows is not None:  # our homegrown COO format
                        linds, rinds = rows, subjac_info['cols']
                        if not fwd:
                            linds, rinds = rinds, linds
                        if self._under_complex_step:
                            # bincount only works with float, so split into parts
                            if ncol > 1:
                                for i in range(ncol):
                                    prod = right_vec[:, i][rinds] * subjac
                                    left_vec[:, i].real += np.bincount(linds, prod.real,
                                                                       minlength=left_vec.shape[0])
                                    left_vec[:, i].imag += np.bincount(linds, prod.imag,
                                                                       minlength=left_vec.shape[0])
                            else:
                                prod = right_vec[rinds] * subjac
                                left_vec[:].real += np.bincount(linds, prod.real,
                                                                minlength=left_vec.size)
                                left_vec[:].imag += np.bincount(linds, prod.imag,
                                                                minlength=left_vec.size)
                        else:
                            if ncol > 1:
                                for i in range(ncol):
                                    left_vec[:, i] += np.bincount(linds,
                                                                  right_vec[:, i][rinds] * subjac,
                                                                  minlength=left_vec.shape[0])
                            else:
                                left_vec[:] += np.bincount(linds, right_vec[rinds] * subjac,
                                                           minlength=left_vec.size)

                    else:
                        if not fwd:
                            subjac = subjac.transpose()

                        left_vec += subjac.dot(right_vec)


class _CheckingJacobian(DictionaryJacobian):
    """
    A special type of Jacobian that we use only inside of check_partials.

    It checks during set_col to make sure that any user specified rows/cols don't mask any
    nonzero values found in the column being set.
    """

    def __init__(self, system):
        super().__init__(system)
        self._subjacs_info = self._subjacs_info.copy()

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
                yield key, meta['value']
            else:
                dense = np.zeros(meta['shape'])
                dense[rows, meta['cols']] = meta['value']
                yield key, dense

    def _setup_index_maps(self, system):
        super()._setup_index_maps(system)
        from openmdao.core.component import Component

        if isinstance(system, Component):
            local_opts = system._get_check_partial_options()
        else:
            local_opts = None

        for of, start, end, _ in system._jac_of_iter():
            nrows = end - start
            for wrt, wstart, wend, _, _ in system._jac_wrt_iter():
                ncols = wend - wstart
                loc_wrt = wrt.rsplit('.', 1)[-1]
                directional = (local_opts is not None and loc_wrt in local_opts and
                               local_opts[loc_wrt]['directional'])
                key = (of, wrt)
                if key not in self._subjacs_info:
                    # create subjacs_info objects for matrix_free systems that don't have them
                    self._subjacs_info[key] = {
                        'rows': None,
                        'cols': None,
                        'value': np.zeros((nrows, 1 if directional else ncols)),
                    }
                elif directional and self._subjacs_info[key]['value'].shape[1] != 1:
                    self._subjacs_info[key] = meta = self._subjacs_info[key].copy()
                    meta['value'] = np.atleast_2d(meta['value'][:, 0]).T

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
        if self._colnames is None:
            self._setup_index_maps(system)

        wrt = self._colnames[self._col2name_ind[icol]]
        _, offset, _, _, _ = self._col_var_info[wrt]
        loc_idx = icol - offset  # local col index into subjacs

        scratch = np.zeros(column.shape)

        for of, start, end, _ in system._jac_of_iter():
            key = (of, wrt)
            if key in self._subjacs_info:
                subjac = self._subjacs_info[key]
                if subjac['cols'] is None:
                    subjac['value'][:, loc_idx] = column[start:end]
                else:
                    match_inds = np.nonzero(subjac['cols'] == loc_idx)[0]
                    if match_inds.size > 0:
                        row_inds = subjac['rows'][match_inds]
                        subjac['value'][match_inds] = column[start:end][row_inds]
                    else:
                        row_inds = np.zeros(0, dtype=INT_DTYPE)
                    arr = scratch[start:end]
                    arr[:] = column[start:end]
                    arr[row_inds] = 0.
                    nzs = np.nonzero(arr)
                    if nzs[0].size > 0:
                        raise ValueError(f"{system.msginfo}: User specified sparsity (rows/cols) "
                                         f"for subjac '{of}' wrt '{wrt}' is incorrect. There are "
                                         f"non-covered nonzeros in column {loc_idx} at "
                                         f"row(s) {nzs[0]}.")
