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
        self._iter_list_rel_unprom = None
        self._iter_list_pathnames = None

        self.options = OptionsDictionary()
        self.options.update(kwargs)

    def _key2shape(self, key):
        """
        Return shape of sub-jacobian for variables making up the key tuple.

        Parameters
        ----------
        key : (str, str)
            Absolute output name, input name of sub-Jacobian.

        Returns
        -------
        out_size : int
            local size of the output variable.
        in_size : int
            local size of the input variable.
        """
        out_name, in_name = key
        return (np.prod(self._system._varx_abs2data_io[out_name]['metadata']['shape']),
                np.prod(self._system._varx_abs2data_io[in_name]['metadata']['shape']))

    def _key2unique(self, key):
        """
        Map output-input name pair from promoted names to absolute names.

        This should only be called when self._system is a Component or
        key parts are all outputs.  If the key contains an input name, that
        may not be unique in a Group context.

        Parameters
        ----------
        key : (str, str)
            Promoted output, input name tuple of sub-Jacobian.

        Returns
        -------
        (str, str)
            Absolute output, input name tuple of sub-Jacobian.
        """
        prom2abs_set = self._system._varx_allprocs_prom2abs_set
        prom_out, prom_in = key

        if prom_out not in prom2abs_set['output']:
            msg = 'The first entry in key ("{}", "{}") is invalid'
            raise ValueError(msg.format(prom_out, prom_in))
        else:
            abs_out = prom2abs_set['output'][prom_out]

        if prom_in not in prom2abs_set['output'] and prom_in not in prom2abs_set['input']:
            msg = 'The second entry in key ("{}", "{}") is invalid'
            raise ValueError(msg.format(prom_out, prom_in))
        elif prom_in not in prom2abs_set['input']:
            abs_out = prom2abs_set['output'][prom_in]

        if key[1] in self._system._var_name2path['input']:
            return (self._system._var_name2path['output'][key[0]],
                    self._system._var_name2path['input'][key[1]][0])
        else:
            return (self._system._var_name2path['output'][key[0]],
                    self._system._var_name2path['output'][key[1]])

    def _multiply_subjac(self, ukey, val):
        """
        Multiply this sub-Jacobian by val.

        Parameters
        ----------
        ukey : (str, str)
            Output name, input name of sub-Jacobian. The names are the full global pathnames.
        val : float
            value to multiply by.
        """
        jac = self._subjacs[ukey]

        if isinstance(jac, np.ndarray):
            self._subjacs[ukey] = val * jac
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            self._subjacs[ukey].data *= val  # DOK not supported
        elif len(jac) == 3:
            self._subjacs[ukey][0] *= val

    def _precompute_iter(self):
        """
        Cache lists of (output, input) pairs found in the jacobian for the current System.

        The two lists are:
        1. _iter_list_rel_unprom : list of unpromoted names viewed from this system
        2. _iter_list_pathnames : list of global, unique pathnames
        """
        system = self._system
        start = len(system.pathname) + 1 if system.pathname else 0
        out_paths = system._var_allprocs_pathnames['output']
        in_paths = system._var_allprocs_pathnames['input']
        out_offset = system._var_allprocs_range['output'][0]
        in_offset = system._var_allprocs_range['input'][0]
        in_indices = system._var_myproc_indices['input']
        out_indices = system._var_myproc_indices['output']

        iter_list_rel_unprom = []
        iter_list_pathnames = []
        for re_ind in out_indices:
            re_path = out_paths[re_ind - out_offset]
            re_unprom = re_path[start:]

            for out_ind in out_indices:
                out_path = out_paths[out_ind - out_offset]
                out_unprom = out_path[start:]

                if (re_path, out_path) in self._subjacs:
                    iter_list_rel_unprom.append((re_unprom, out_unprom))
                    iter_list_pathnames.append((re_path, out_path))

            for in_ind in in_indices:
                in_path = in_paths[in_ind - in_offset]
                in_unprom = in_path[start:]

                if (re_path, in_path) in self._subjacs:
                    iter_list_rel_unprom.append((re_unprom, in_unprom))
                    iter_list_pathnames.append((re_path, in_path))

        self._iter_list_rel_unprom = iter_list_rel_unprom
        self._iter_list_pathnames = iter_list_pathnames

    def __contains__(self, key):
        """
        Map output-input pairs names to indices.

        Parameters
        ----------
        key : (str, str)
            output name, input name of sub-Jacobian. Names are promoted names.

        Returns
        -------
        boolean
            return whether sub-Jacobian has been defined.
        """
        outname2path = self._system._var_name2path['output']
        if key[0] in outname2path:
            if key[1] in self._system._var_name2path['input']:
                out_path = outname2path[key[0]]
                for ipath in self._system._var_name2path['input'][key[1]]:
                    return (out_path, ipath) in self._subjacs
            elif key[1] in outname2path:
                return (outname2path[key[0]], outname2path[key[1]]) in self._subjacs
        return False

    def __setitem__(self, key, jac):
        """
        Set sub-Jacobian.

        Parameters
        ----------
        key : (str, str)
            output name, input name of sub-Jacobian. Names are promoted names.
        jac : int or float or ndarray or sparse matrix
            sub-Jacobian as a scalar, vector, array, or AIJ list or tuple.
        """
        ukey = self._key2unique(key)

        if np.isscalar(jac) or isinstance(jac, np.ndarray):
            shape = self._key2shape(key)
            jac = np.atleast_2d(jac).reshape(shape)
            # np.promote_types will choose the smallest dtype that can contain both arguments
            safe_dtype = np.promote_types(jac.dtype, float)
            jac = jac.astype(safe_dtype, copy=False)
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            pass
        elif isinstance(jac, (tuple, list)):
            if len(jac) != 3:
                raise ValueError("Sub-jacobian of type '%s' for key %s has "
                                 "the wrong size (%d)." %
                                 (type(jac).__name__, key, len(jac)))
            if isinstance(jac, tuple):
                jac = list(jac)
        else:
            raise TypeError("Sub-jacobian of type '%s' for key %s is "
                            "not supported." % (type(jac).__name__, key))

        self._subjacs[ukey] = jac

    def __getitem__(self, key):
        """
        Get sub-Jacobian.

        Parameters
        ----------
        key : (str, str)
            output name, input name of sub-Jacobian. Names are promoted names.

        Returns
        -------
        jac : ndarray or spmatrix or list[3]
            sub-Jacobian as an array, sparse mtx, or AIJ/IJ list or tuple.
        """
        ukey = self._key2unique(key)
        return self._subjacs[ukey]

    def _iter_pathnames(self):
        return iter(self._iter_list_pathnames)

    def _iter_rel_unprom(self):
        return iter(self._iter_list_rel_unprom)

    def _scale_subjac(self, ukey, coeffs):
        """
        Change the scaling state of a single subjac.

        Parameters
        ----------
        ukey : (str, str)
            Output name, input name of sub-Jacobian. The names are the full global pathnames.
        coeffs : dict of ndarray[nvar_myproc, 2]
            0th and 1st order coefficients for scaling/unscaling.
            The keys are 'input', 'output', and 'residual'
        """
        ind0 = self._system._var_pathdict[ukey[0]].myproc_idx
        ind1 = self._system._var_pathdict[ukey[1]].myproc_idx
        typ = self._system._var_pathdict[ukey[1]].typ

        val = coeffs['residual'][ind0, 1] / coeffs[typ][ind1, 1]
        self._multiply_subjac(ukey, val)

    def _scale(self, coeffs):
        """
        Change the scaling state for all subjacs under the current system.

        Parameters
        ----------
        coeffs : dict of ndarray[nvar_myproc, 2]
            0th and 1st order coefficients for scaling/unscaling.
            The keys are 'input', 'output', and 'residual'.
        """
        for ukey in self._iter_pathnames():
            self._scale_subjac(ukey, coeffs)

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

    def _set_partials_meta(self, key, meta, negate=False):
        """
        Store subjacobian metadata.

        Parameters
        ----------
        key : (str, str)
            output name, input name of sub-Jacobian. The names are promoted names.
        meta : dict
            Metadata dictionary for the subjacobian.
        negate : bool
            If True negate the given value, if any.
        """
        ukey = self._key2unique(key)
        if ukey is None:
            raise KeyError("Could not find unique key for %s." % (key,))
        self._subjacs_info[ukey] = (meta, self._key2shape(key))

        val = meta['value']
        if val is not None:
            if negate:
                val *= -1.
            if meta['rows'] is not None:
                val = [val, meta['rows'], meta['cols']]
            self.__setitem__(key, val)
