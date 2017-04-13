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
        abs2meta = self._system._var_abs2meta

        meta0 = abs2meta['output'][abs_key[0]]
        if abs_key[1] in abs2meta['output']:
            meta1 = abs2meta['output'][abs_key[1]]
        elif abs_key[1] in abs2meta['input']:
            meta1 = abs2meta['input'][abs_key[1]]

        return (np.prod(meta0['shape']), np.prod(meta1['shape']))

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
        return key2abs_key(self._system, key) in self._subjacs

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
        """
        system = self._system
        subjacs = self._subjacs

        # FIXME: these keys should really be cached in system, not as they were previously
        # in precompute_iter_keys, since they had to be constantly recomputed whenever the
        # jacobian's system changed.  There is an ordering issue with caching them in system
        # because this method gets called once (for scaling) prior to the first call to
        # linearize for each system which can add keys to the jacobian, so we'll need to
        # make sure we recompute the keys for each system after the first call to lineraize
        # after a new jacobian has been set.
        for res_name in system._var_abs_names['output']:
            for type_ in ('output', 'input'):
                for name in system._var_abs_names[type_]:
                    key = (res_name, name)
                    if key in subjacs:
                        yield key

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
