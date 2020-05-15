"""Define the base Jacobian class."""
import weakref

import numpy as np
from numpy.random import rand

from collections import OrderedDict, defaultdict
from scipy.sparse import issparse

from openmdao.utils.name_maps import key2abs_key, rel_name2abs_name
from openmdao.matrices.matrix import sparse_types
from openmdao.vectors.vector import _full_slice

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
        self._system = weakref.ref(system)
        self._subjacs_info = system._subjacs_info
        self._override_checks = False
        self._under_complex_step = False
        self._abs_keys = defaultdict(bool)
        self._randomize = False
        self._jac_summ = None

    def _get_abs_key(self, key):
        abskey = self._abs_keys[key]
        if not abskey:
            self._abs_keys[key] = abskey = key2abs_key(self._system(), key)
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
        system = self._system()
        abs2meta = system._var_allprocs_abs2meta
        of, wrt = abs_key
        return (abs2meta[of]['size'], abs2meta[wrt]['size'])

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
            msg = '{}: Variable name pair ("{}", "{}") not found.'
            raise KeyError(msg.format(self.msginfo, key[0], key[1]))

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
                msg = '{}: Variable name pair ("{}", "{}") must first be declared.'
                raise KeyError(msg.format(self.msginfo, key[0], key[1]))

            subjacs_info = self._subjacs_info[abs_key]

            if issparse(subjac):
                subjacs_info['value'] = subjac
            else:
                # np.promote_types will choose the smallest dtype that can contain both arguments
                subjac = np.atleast_1d(subjac)
                safe_dtype = np.promote_types(subjac.dtype, float)
                subjac = subjac.astype(safe_dtype, copy=False)

                # Bail here so that we allow top level jacobians to be of reduced size when indices
                # are specified on driver vars.
                if self._override_checks:
                    subjacs_info['value'] = subjac
                    return

                rows = subjacs_info['rows']

                if rows is None:
                    # Dense subjac
                    subjac = np.atleast_2d(subjac)
                    if subjac.shape != (1, 1):
                        shape = self._abs_key2shape(abs_key)
                        subjac = subjac.reshape(shape)
                else:
                    # Sparse subjac
                    if subjac.shape != (1,) and subjac.shape != rows.shape:
                        msg = '{}: Sub-jacobian for key {} has the wrong shape ({}), expected ({}).'
                        raise ValueError(msg.format(self.msginfo, abs_key,
                                                    subjac.shape, rows.shape))

                subjacs_info['value'][:] = subjac

        else:
            msg = '{}: Variable name pair ("{}", "{}") not found.'
            raise KeyError(msg.format(self.msginfo, key[0], key[1]))

    def __iter__(self):
        """
        Yield next name pair of sub-Jacobian.
        """
        for key in self._subjacs_info.keys():
            yield key

    def items(self):
        """
        Yield name pair and value of sub-Jacobian.
        """
        for key, val in self._subjacs_info.items():
            yield key, val['value']

    @property
    def msginfo(self):
        """
        Return info to prepend to messages.

        Returns
        -------
        str
            Info to prepend to messages.
        """
        if self._system() is None:
            return type(self).__name__
        return '{} in {}'.format(type(self).__name__, self._system().msginfo)

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

    def _randomize_subjac(self, subjac, key):
        """
        Return a subjac that is the given subjac filled with random values.

        Parameters
        ----------
        subjac : ndarray or csc_matrix
            Sub-jacobian to be randomized.
        key : tuple (of, wrt)
            Key for subjac within the jacobian.

        Returns
        -------
        ndarray or csc_matrix
            Randomized version of the subjac.
        """
        if isinstance(subjac, sparse_types):  # sparse
            sparse = subjac.copy()
            sparse.data = rand(sparse.data.size)
            sparse.data += 1.0
            return sparse

        # if a subsystem has computed a dynamic partial or semi-total coloring,
        # we use that sparsity information to set the sparsity of the randomized
        # subjac.  Otherwise all subjacs that didn't have sparsity declared by the
        # user will appear completely dense, which will lead to a total jacobian that
        # is more dense than it should be, causing any total coloring that we compute
        # to be overly conservative.
        subjac_info = self._subjacs_info[key]
        if 'sparsity' in subjac_info:
            assert subjac_info['rows'] is None
            rows, cols, shape = subjac_info['sparsity']
            r = np.zeros(shape)
            val = rand(len(rows))
            val += 1.0
            r[rows, cols] = val
        else:
            r = rand(*subjac.shape)
            r += 1.0
        return r

    def _save_sparsity(self, system):
        """
        Add the current jacobian to a running absolute summation.

        Parameters
        ----------
        system : System
            System owning this jacobian.
        """
        subjacs = self._subjacs_info
        if self._jac_summ is None:
            # create _jac_summ structure
            self._jac_summ = summ = {}
            for key in subjacs:
                summ[key] = np.abs(subjacs[key]['value'])
        else:
            summ = self._jac_summ
            for key in subjacs:
                summ[key] += np.abs(subjacs[key]['value'])

    def _compute_sparsity(self, ordered_of_info, ordered_wrt_info, num_full_jacs, tol, orders):
        """
        Compute a dense sparsity matrix for this jacobian using saved absolute summations.

        The sparsity matrix will contain only those columns that match the wrt variables in
        wrt_matches, but will contain rows for all outputs in the given system.

        Parameters
        ----------
        ordered_of_info : list of (name, offset, end, idxs)
            Name, offset, etc. of row variables in the order that they appear in the jacobian.
        ordered_wrt_info : list of (name, offset, end, idxs)
            Name, offset, etc. of column variables in the order that they appear in the jacobian.
        num_full_jacs : int
            Number of times to compute partial jacobian when computing sparsity.
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

        subjacs = self._subjacs_info
        summ = self._jac_summ

        rend = ordered_of_info[-1][2]
        cend = ordered_wrt_info[-1][2]
        J = np.zeros((rend, cend))

        for of, roffset, rend, _ in ordered_of_info:
            for wrt, coffset, cend, _ in ordered_wrt_info:
                key = (of, wrt)
                if key in subjacs:
                    meta = subjacs[key]
                    if meta['rows'] is not None:
                        rows = meta['rows'] + roffset
                        cols = meta['cols'] + coffset
                        J[rows, cols] = summ[key]
                    elif issparse(summ[key]):
                        raise NotImplementedError("{}: scipy sparse arrays are not "
                                                  "supported yet.".format(self.msginfo))
                    else:
                        J[roffset:rend, coffset:cend] = summ[key]

        J *= (1.0 / np.max(J))

        tol_info = _tol_sweep(J, tol, orders)

        boolJ = np.zeros(J.shape, dtype=bool)
        boolJ[J > tol_info['good_tol']] = True

        return boolJ, tol_info

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
        for meta in self._subjacs_info.values():
            if active:
                meta['value'] = meta['value'].astype(np.complex)
            else:
                meta['value'] = meta['value'].real

        self._under_complex_step = active
