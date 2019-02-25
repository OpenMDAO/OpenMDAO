"""Define the base Jacobian class."""
from __future__ import division
import numpy as np
from numpy.random import rand

from collections import OrderedDict, defaultdict
from scipy.sparse import issparse
from six import itervalues, iteritems

from openmdao.utils.name_maps import key2abs_key
from openmdao.matrices.matrix import sparse_types

SUBJAC_META_DEFAULTS = {
    'rows': None,
    'cols': None,
    'value': None,
    'dependent': False,
}


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
    _under_complex_step : bool
        When True, this Jacobian is under complex step, using a complex jacobian.
    _abs_keys : defaultdict
        A cache dict for key to absolute key.
    _randomize : bool
        If True, sparsity is being computed for simultaneous derivative coloring.
    _jac_summ : dict or None
        A dict containing a summation of some number of instantaneous absolute values of this
        jacobian, for use later to determine jacobian sparsity and simultaneous coloring.
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
        self._under_complex_step = False
        self._abs_keys = defaultdict(bool)
        self._randomize = False
        self._jac_summ = None

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

    def _update(self, system):
        """
        Read the user's sub-Jacobians and set into the global matrix.

        Parameters
        ----------
        system : System
            System that is updating this jacobian.
        """
        pass

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

    def _get_ranges(self, system, vtype):
        iproc = system.comm.rank
        abs2idx = system._var_allprocs_abs2idx['linear']
        sizes = system._var_sizes['linear'][vtype]
        start = end = 0
        ranges = OrderedDict()
        for name in system._var_allprocs_abs_names[vtype]:
            end += sizes[iproc, abs2idx[name]]
            ranges[name] = (start, end)
            start = end
        return ranges

    def _save_sparsity(self):
        subjacs = self._subjacs_info
        if self._jac_summ is None:
            # create _jac_summ structure
            self._jac_summ = summ = {}
            for key in subjacs:
                summ[key] = np.abs(self[key])
        else:
            summ = self._jac_summ
            for key in subjacs:
                summ[key] += np.abs(self[key])

    def _compute_sparsity(self, system, wrt_matches, tol=1e-15, orders=5):
        """
        Compute a dense sparsity matrix for this jacobian using saved absolute summations.

        The sparsity matrix will contain only those columns that match the wrt variables in
        wrt_matches.

        Parameters
        ----------
        system : System
            The System containing the jacobian whose sparsity will be computed.
        wrt_matches : set of str
            Set of wrt variables to compute sparity for.
        tol : float
            Tolerance used to determine if an array entry is zero or nonzero.
        orders : int
            Number of orders +/- for the tolerance sweep.

        Returns
        -------
        ndarray
            Boolean sparsity matrix.
        """
        from openmdao.utils.coloring import _tol_sweep

        summ = self._jac_summ
        subjacs = self._subjacs_info
        ofs = list(system._var_allprocs_abs_names['output'])
        wrts = list(system._var_allprocs_abs_names['input'])

        iproc = system.comm.rank
        abs2idx = system._var_allprocs_abs2idx['linear']
        osizes = system._var_sizes['linear']['output']
        isizes = system._var_sizes['linear']['input']

        wrt_info = ((ofs, osizes), (wrts, isizes))

        ncols = nrows = 0
        locs = {}
        roffset = rend = 0
        coffset = cend = 0
        for of in ofs:
            rend += osizes[iproc, abs2idx[of]]
            for wrts, sizes in wrt_info:
                for wrt in wrts:
                    if wrt in wrt_matches:
                        cend += sizes[iproc, abs2idx[wrt]]
                        key = (of, wrt)
                        if key in subjacs:
                            locs[key] = (slice(roffset, rend), slice(coffset, cend))
                        coffset = cend
            roffset = rend

        J = np.zeros((rend, cend))

        for key in locs:
            J[locs[key]] = summ[key]

        # normalize by largest value
        J /= np.max(J)

        good_tol, nz_matches, n_tested, zero_entries = _tol_sweep(J, tol, orders)

        boolJ = np.zeros(J.shape, dtype=bool)
        boolJ[J > good_tol] = True

        return boolJ

    def set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        When turned on, the value in each subjac is cast as complex, and when turned
        off, they are returned to real values.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        for key, meta in iteritems(self._subjacs_info):
            if active:
                meta['value'] = meta['value'].astype(np.complex)
            else:
                meta['value'] = meta['value'].real

        self._under_complex_step = active
