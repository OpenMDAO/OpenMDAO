"""Define the base Jacobian class."""
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from six.moves import range

from openmdao.utils.generalized_dict import OptionsDictionary
from openmdao.utils.name_maps import key2abs_key


class Jacobian(object):
    """
    Base Jacobian class.

    This class provides a dictionary interface for sub-Jacobians and
    performs matrix-vector products when apply_linear is called.

    Attributes
    ----------
    _system : <System>
        Pointer to the system that is currently operating on this Jacobian.
    _subjacs : dict
        Dictionary of the user-supplied sub-Jacobians keyed by absolute names.
    _subjacs_info : dict
        Dictionary of the sub-Jacobian metadata keyed by absolute names.
    _int_mtx : <Matrix>
        Global internal Jacobian.
    _ext_mtx : <Matrix>
        Global external Jacobian.
    _keymap : dict
        Mapping of original (output, input) key to (output, source) in cases
        where the input has src_indices.
    _iter_list : [(out_name, in_name), ...]
        List of output-input pairs to iterate over where the keys are absolute names.
    options : <OptionsDictionary>
        Options dictionary.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        self._system = None

        self._subjacs = {}
        self._subjacs_info = {}
        self._int_mtx = None
        self._ext_mtx = None
        self._keymap = {}
        self._iter_list = None

        self.options = OptionsDictionary()
        self.options.update(kwargs)

    def _abs_key2shape(self, abs_key):
        """
        Return shape of sub-jacobian for variables making up the key tuple.

        Parameters
        ----------
        abs_key : (str, str)
            Absolute name pair of sub-Jacobian.

        Returns
        -------
        out_size : int
            local size of the output variable.
        in_size : int
            local size of the input variable.
        """
        return (np.prod(self._system._var_abs2data_io[abs_key[0]]['metadata']['shape']),
                np.prod(self._system._var_abs2data_io[abs_key[1]]['metadata']['shape']))

    def _multiply_subjac(self, abs_key, val):
        """
        Multiply this sub-Jacobian by val.

        Parameters
        ----------
        abs_key : (str, str)
            Absolute name pair of sub-Jacobian.
        val : float
            value to multiply by.
        """
        jac = self._subjacs[abs_key]

        if isinstance(jac, np.ndarray):
            self._subjacs[abs_key] = val * jac
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            self._subjacs[abs_key].data *= val  # DOK not supported
        elif len(jac) == 3:
            self._subjacs[abs_key][0] *= val

    def _pre_and_post_multiply_subjac(self, abs_key, left_vec, right_vec):
        """
        Compute left_vec transposed times this sub-Jacobian times right_vec.

        Parameters
        ----------
        abs_key : (str, str)
            Absolute name pair of sub-Jacobian.
        left_vec : ndarray
            Array to pre-multiply by.
        right_vec : ndarray
            Array to post-multiply by.
        """
        jac = self._subjacs[abs_key]
        left_vec = np.atleast_2d(left_vec).T

        if isinstance(jac, np.ndarray):
            self._subjacs[abs_key] = left_vec * jac / right_vec
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            # DOK not supported
            self._subjacs[abs_key].data = left_vec * self._subjacs[abs_key].data / right_vec
        else:
            # TODO: This is currently untested because support for scaler specification of a
            # subjac larger than 1x1 is not implemented.
            self._subjacs[abs_key] = left_vec * self._subjacs[abs_key][0] / right_vec

    def _precompute_iter(self):
        """
        Cache list of absolute name pairs found in the jacobian for the current System.
        """
        system = self._system

        iter_list = []
        for res_name in system._var_abs_names['output']:
            for out_name in system._var_abs_names['output']:
                if (res_name, out_name) in self._subjacs:
                    iter_list.append((res_name, out_name))
            for in_name in system._var_abs_names['input']:
                if (res_name, in_name) in self._subjacs:
                    iter_list.append((res_name, in_name))

        self._iter_list = iter_list

    def __contains__(self, key):
        """
        Return whether there is a subjac for the given promoted or relative name pair.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.

        Returns
        -------
        boolean
            return whether sub-Jacobian has been defined.
        """
        abs_key = key2abs_key(self._system, key)
        return abs_key in self._subjacs

    def __getitem__(self, key):
        """
        Get sub-Jacobian.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.

        Returns
        -------
        ndarray or spmatrix or list[3]
            sub-Jacobian as an array, sparse mtx, or AIJ/IJ list or tuple.
        """
        abs_key = key2abs_key(self._system, key)
        if abs_key in self._subjacs:
            return self._subjacs[abs_key]
        else:
            msg = 'Variable name pair ("{}", "{}") not found.'
            raise KeyError(msg.format(key[0], key[1]))

    def __setitem__(self, key, subjac):
        """
        Set sub-Jacobian.

        Parameters
        ----------
        key : (str, str)
            Promoted or relative name pair of sub-Jacobian.
        subjac : int or float or ndarray or sparse matrix
            sub-Jacobian as a scalar, vector, array, or AIJ list or tuple.
        """
        abs_key = key2abs_key(self._system, key)
        if abs_key is not None:
            self._set_abs(abs_key, subjac)
        else:
            msg = 'Variable name pair ("{}", "{}") not found.'
            raise KeyError(msg.format(key[0], key[1]))

    def _set_abs(self, abs_key, subjac):
        """
        Set sub-Jacobian.

        Parameters
        ----------
        abs_key : (str, str)
            Absolute name pair of sub-Jacobian.
        subjac : int or float or ndarray or sparse matrix
            sub-Jacobian as a scalar, vector, array, or AIJ list or tuple.
        """
        if np.isscalar(subjac) or isinstance(subjac, np.ndarray):
            shape = self._abs_key2shape(abs_key)
            subjac = np.atleast_2d(subjac).reshape(shape)
            # np.promote_types will choose the smallest dtype that can contain both arguments
            safe_dtype = np.promote_types(subjac.dtype, float)
            subjac = subjac.astype(safe_dtype, copy=False)
        elif isinstance(subjac, (coo_matrix, csr_matrix)):
            pass
        elif isinstance(subjac, (tuple, list)):
            if len(subjac) != 3:
                raise ValueError("Sub-jacobian of type '%s' for key %s has "
                                 "the wrong size (%d)." %
                                 (type(subjac).__name__, prom_key, len(subjac)))
            if isinstance(subjac, tuple):
                subjac = list(subjac)
        else:
            raise TypeError("Sub-jacobian of type '%s' for key %s is "
                            "not supported." % (type(subjac).__name__, prom_key))

        self._subjacs[abs_key] = subjac

    def _iter_abs_keys(self):
        """
        Iterate over subjacs keyed by absolute names.

        This includes only subjacs that have been set and are part of the current system.

        Returns
        -------
        iterator
            Iterator over subjacs keyed by absolute names that have been set on this Jacobian.
        """
        return iter(self._iter_list)

    def _scale_subjac(self, abs_key, coeffs):
        """
        Change the scaling state of a single subjac.

        Parameters
        ----------
        abs_key : (str, str)
            Absolute name pair of sub-Jacobian.
        coeffs : dict of ndarray[nvar_myproc, 2]
            0th and 1st order coefficients for scaling/unscaling.
            The keys are 'input', 'output', and 'residual'
        """
        data0 = self._system._var_abs2data_io[abs_key[0]]
        data1 = self._system._var_abs2data_io[abs_key[1]]

        ind_of0, ind_of1 = data0['resid_scale_idx']

        # Implicit states are the only wrt that will have this
        try:
            ind_wrt0, ind_wrt1 = data1['output_scale_idx']
        except KeyError:
            ind_wrt0 = data1['my_idx']
            ind_wrt1 = ind_wrt0 + 1

        type_ = data1['type']

        # A vector scale factor on the residual needs to be transposed and pre-multiplied.
        if (ind_of1 - ind_of0) > 1:
            self._pre_and_post_multiply_subjac(abs_key, coeffs['residual'][ind_of0:ind_of1, 1],
                                               coeffs[type_][ind_wrt0:ind_wrt1, 1])
        else:
            val = coeffs['residual'][ind_of0:ind_of1, 1] / coeffs[type_][ind_wrt0:ind_wrt1, 1]
            self._multiply_subjac(abs_key, val)

    def _scale(self, coeffs):
        """
        Change the scaling state for all subjacs under the current system.

        Parameters
        ----------
        coeffs : dict of ndarray[nvar_myproc, 2]
            0th and 1st order coefficients for scaling/unscaling.
            The keys are 'input', 'output', and 'residual'.
        """
        for abs_key in self._iter_abs_keys():
            self._scale_subjac(abs_key, coeffs)

    def _initialize(self):
        """
        Allocate the global matrices.
        """
        pass

    def _update(self):
        """
        Read the user's sub-Jacobians and set into the global matrix.
        """
        pass

    def _apply(self, d_inputs, d_outputs, d_residuals, mode):
        """
        Compute matrix-vector product.

        Parameters
        ----------
        d_inputs : Vector
            inputs linear vector.
        d_outputs : Vector
            outputs linear vector.
        d_residuals : Vector
            residuals linear vector.
        mode : str
            'fwd' or 'rev'.
        """
        pass

    def _set_partials_meta(self, abs_key, meta, negate=False):
        """
        Store subjacobian metadata.

        Note: this method MUST be called by a Component because prom_key is otherwise non-unique.

        Parameters
        ----------
        abs_key : (str, str)
            Absolute name pair of sub-Jacobian.
        meta : dict
            Metadata dictionary for the subjacobian.
        negate : bool
            If True negate the given value, if any.
        """
        shape = self._abs_key2shape(abs_key)
        self._subjacs_info[abs_key] = (meta, shape)

        val = meta['value']
        if val is not None:
            if negate:
                val *= -1.
            if meta['rows'] is not None:
                val = [val, meta['rows'], meta['cols']]
            self._set_abs(abs_key, val)
