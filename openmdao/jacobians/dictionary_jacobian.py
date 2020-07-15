"""Define the DictionaryJacobian class."""
import numpy as np
from scipy.sparse import csc_matrix

from openmdao.jacobians.jacobian import Jacobian, _full_slice


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
        super(DictionaryJacobian, self).__init__(system, **kwargs)
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
            ncol = system._vectors['residual'][vec_name]._ncol
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

        rflat = d_residuals._views_flat
        oflat = d_outputs._views_flat
        iflat = d_inputs._views_flat
        ncol = d_residuals._ncol
        subjacs_info = self._subjacs_info
        is_explicit = isinstance(system, ExplicitComponent)

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            for abs_key in self._iter_abs_keys(system, d_residuals._name):
                subjac_info = subjacs_info[abs_key]
                if self._randomize:
                    subjac = self._randomize_subjac(subjac_info['value'], abs_key)
                else:
                    subjac = subjac_info['value']
                res_name, other_name = abs_key
                if res_name in d_res_names:

                    if other_name in d_out_names:
                        # skip the matvec mult completely for identity subjacs
                        if is_explicit and res_name is other_name:
                            if fwd:
                                rflat[res_name] -= oflat[other_name]
                            else:
                                oflat[other_name] -= rflat[res_name]
                            continue

                        if fwd:
                            left_vec = rflat[res_name]
                            right_vec = oflat[other_name]
                        else:
                            left_vec = oflat[other_name]
                            right_vec = rflat[res_name]
                    elif other_name in d_inp_names:
                        if fwd:
                            left_vec = rflat[res_name]
                            right_vec = iflat[other_name]
                        else:
                            left_vec = iflat[other_name]
                            right_vec = rflat[res_name]
                    else:
                        continue

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
