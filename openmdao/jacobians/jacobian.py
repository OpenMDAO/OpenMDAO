"""Define the base Jacobian class."""
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from six.moves import range

from openmdao.utils.generalized_dict import OptionsDictionary


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
        return (np.prod(self._system._varx_abs2data_io[abs_key[0]]['metadata']['shape']),
                np.prod(self._system._varx_abs2data_io[abs_key[1]]['metadata']['shape']))

    def _prom_key2abs_key(self, prom_key):
        """
        Map output-input name pair from promoted names to absolute names.

        This is only valid when the key is unique; otherwise, a KeyError is thrown.

        Parameters
        ----------
        prom_key : (str, str)
            Promoted name pair of sub-Jacobian.

        Returns
        -------
        (str, str)
            Absolute output, input name tuple of sub-Jacobian.
        """
        prom2abs_list = self._system._varx_allprocs_prom2abs_list

        # First promoted name is invalid
        if prom_key[0] not in prom2abs_list['output']:
            msg = 'The first entry in key ("{}", "{}") is invalid'
            raise KeyError(msg.format(prom_key[0], prom_key[1]))
        # First promoted name is valid
        else:
            abs_key0 = prom2abs_list['output'][prom_key[0]][0]

        # Second promoted name is invalid
        if not (prom_key[1] in prom2abs_list['output'] or prom_key[1] in prom2abs_list['input']):
            msg = 'The second entry in key ("{}", "{}") is invalid'
            raise KeyError(msg.format(prom_key[0], prom_key[1]))
        # Second promoted name is an output
        elif prom_key[1] not in prom2abs_list['input']:
            abs_key1 = prom2abs_list['output'][prom_key[1]][0]
        # Second promoted name is an input
        elif prom_key[1] not in prom2abs_list['output']:
            if len(prom2abs_list['input'][prom_key[1]]) > 1:
                msg = 'The second entry in key ("{}", "{}") is non-unique' \
                    + 'so it must be accessed from a lower-level system.'
                raise KeyError(msg.format(prom_key[0], prom_key[1]))
            else:
                abs_key1 = prom2abs_list['input'][prom_key[1]][0]
        # Second promoted name is both an output and an input - non-unique
        else:
            msg = 'The second entry in key ("{}", "{}") is non-unique' \
                + 'so it must be accessed from a lower-level system.'
            raise KeyError(msg.format(prom_key[0], prom_key[1]))

        return (abs_key0, abs_key1)

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

    def _precompute_iter(self):
        """
        Cache list of absolute name pairs found in the jacobian for the current System.
        """
        system = self._system

        iter_list = []
        for res_name in system._varx_abs_names['output']:
            for out_name in system._varx_abs_names['output']:
                if (res_name, out_name) in self._subjacs:
                    iter_list.append((res_name, out_name))
            for in_name in system._varx_abs_names['input']:
                if (res_name, in_name) in self._subjacs:
                    iter_list.append((res_name, in_name))

        self._iter_list = iter_list

    def __contains__(self, prom_key):
        """
        Return if self contains a sub-jac for the given promoted output-input name pair.

        Parameters
        ----------
        prom_key : (str, str)
            Promoted name pair of sub-Jacobian.

        Returns
        -------
        boolean
            return whether sub-Jacobian has been defined.
        """
        abs_key = self._prom_key2abs_key(prom_key)
        return abs_key in self._subjacs

    def __setitem__(self, prom_key, subjac):
        """
        Set sub-Jacobian.

        Parameters
        ----------
        prom_key : (str, str)
            Promoted name pair of sub-Jacobian.
        subjac : int or float or ndarray or sparse matrix
            sub-Jacobian as a scalar, vector, array, or AIJ list or tuple.
        """
        abs_key = self._prom_key2abs_key(prom_key)

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

    def __getitem__(self, prom_key):
        """
        Get sub-Jacobian.

        Parameters
        ----------
        prom_key : (str, str)
            Promoted name pair of sub-Jacobian.

        Returns
        -------
        ndarray or spmatrix or list[3]
            sub-Jacobian as an array, sparse mtx, or AIJ/IJ list or tuple.
        """
        abs_key = self._prom_key2abs_key(prom_key)
        return self._subjacs[abs_key]

    def _iter_abs_names(self):
        """
        Iterate over subjacs keyed by absolute names.

        This includes only subjacs that have been set and are part of the current system.
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
        data0 = self._system._varx_abs2data_io[abs_key[0]]
        data1 = self._system._varx_abs2data_io[abs_key[1]]

        ind0 = data0['my_idx']
        ind1 = data1['my_idx']
        type_ = data1['type_']

        val = coeffs['residual'][ind0, 1] / coeffs[type_][ind1, 1]
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
        for abs_key in self._iter_pathnames():
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

    def _set_partials_meta(self, prom_key, meta, negate=False):
        """
        Store subjacobian metadata.

        Note: this method MUST be called by a Component because prom_key is otherwise non-unique.

        Parameters
        ----------
        prom_key : (str, str)
            Promoted name pair of sub-Jacobian.
        meta : dict
            Metadata dictionary for the subjacobian.
        negate : bool
            If True negate the given value, if any.
        """
        abs_key = self._prom_key2abs_key(prom_key)
        shape = self._abs_key2shape(abs_key)
        self._subjacs_info[abs_key] = (meta, shape)

        val = meta['value']
        if val is not None:
            if negate:
                val *= -1.
            if meta['rows'] is not None:
                val = [val, meta['rows'], meta['cols']]
            self[prom_key] = val
