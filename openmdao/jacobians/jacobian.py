"""Define the base Jacobian class."""
from __future__ import division
import numpy as np
from numpy.random import rand

from collections import OrderedDict, defaultdict
from scipy.sparse import issparse
from six import itervalues

from openmdao.utils.name_maps import key2abs_key
from openmdao.matrices.matrix import sparse_types


class Jacobian(object):
    """
    Base Jacobian class.

    This class provides a dictionary interface for sub-Jacobians and
    performs matrix-vector products when apply_linear is called.

    Attributes
    ----------
    _system : <System>
        Pointer to the system that is currently operating on this Jacobian.
    _subjacs_info : dict
        Dictionary of the sub-Jacobian metadata keyed by absolute names.
    _override_checks : bool
        If we are approximating a jacobian at the top level and we have specified indices on the
        functions or designvars, then we need to disable the size checking temporarily so that we
        can assign a jacobian with less rows or columns than the variable sizes.
    _abs_keys : defaultdict
        A cache dict for key to absolute key.
    _randomize : bool
        If True, sparsity is being computed for simultaneous derivative coloring.
    """

    def __init__(self, system):
        """
        Initialize all attributes.

        Parameters
        ----------
        system : System
            Parent system to this jacobian.
        """
        self._system = system
        self._subjacs_info = system._subjacs_info
        self._override_checks = False
        self._abs_keys = defaultdict(bool)
        self._randomize = False

    def _get_abs_key(self, key):
        abskey = self._abs_keys[key]
        if not abskey:
            self._abs_keys[key] = abskey = key2abs_key(self._system, key)
        return abskey

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
        abs2meta = self._system._var_allprocs_abs2meta
        return (abs2meta[abs_key[0]]['size'], abs2meta[abs_key[1]]['size'])

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
        return self._get_abs_key(key) in self._subjacs_info

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
        abs_key = self._get_abs_key(key)
        if abs_key in self._subjacs_info:
            return self._subjacs_info[abs_key]['value']
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
        abs_key = self._get_abs_key(key)
        if abs_key is not None:

            # You can only set declared subjacobians.
            if abs_key not in self._subjacs_info:
                msg = 'Variable name pair ("{}", "{}") must first be declared.'
                raise KeyError(msg.format(key[0], key[1]))

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
        subjacs_info = self._subjacs_info[abs_key]

        if not issparse(subjac):
            # np.promote_types will choose the smallest dtype that can contain both arguments
            subjac = np.atleast_1d(subjac)
            safe_dtype = np.promote_types(subjac.dtype, float)
            subjac = subjac.astype(safe_dtype, copy=False)

            # Bail here so that we allow top level jacobians to be of reduced size when indices are
            # specified on driver vars.
            if self._override_checks:
                subjacs_info['value'] = subjac
                return

            rows = subjacs_info['rows']

            if rows is None:
                # Dense subjac
                shape = self._abs_key2shape(abs_key)
                subjac = np.atleast_2d(subjac)
                if subjac.shape == (1, 1):
                    subjac = subjac[0, 0] * np.ones(shape, dtype=safe_dtype)
                else:
                    subjac = subjac.reshape(shape)
            else:
                # Sparse subjac
                if subjac.shape == (1,):
                    subjac = subjac[0] * np.ones(rows.shape, dtype=safe_dtype)

                if subjac.shape != rows.shape:
                    raise ValueError("Sub-jacobian for key %s has "
                                     "the wrong shape (%s), expected (%s)." %
                                     (abs_key, subjac.shape, rows.shape))

            np.copyto(subjacs_info['value'], subjac)
        else:
            subjacs_info['value'] = subjac

    def _initialize(self):
        """
        Allocate the global matrices.
        """
        pass

    def _update(self, system):
        """
        Read the user's sub-Jacobians and set into the global matrix.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
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

    def _randomize_subjac(self, subjac):
        """
        Return a subjac that is the given subjac filled with random values.

        Parameters
        ----------
        subjac : ndarray or csc_matrix
            Sub-jacobian to be randomized.

        Returns
        -------
        ndarray or csc_matrix
            Randomized version of the subjac.
        """
        if isinstance(subjac, sparse_types):  # sparse
            sparse = subjac.copy()
            sparse.data = rand(sparse.data.size) + 1.0
            return sparse

        return rand(*subjac.shape) + 1.0

    def _reset_mats(self):
        """
        Zero out internal matrices if needed.
        """
        pass
