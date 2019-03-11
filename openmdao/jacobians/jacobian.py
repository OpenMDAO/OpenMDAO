"""Define the base Jacobian class."""
from __future__ import division
import numpy as np
from numpy.random import rand

from collections import OrderedDict, defaultdict
from scipy.sparse import issparse
from six import itervalues, iteritems

from openmdao.utils.name_maps import key2abs_key, rel_name2abs_name
from openmdao.matrices.matrix import sparse_types

SUBJAC_META_DEFAULTS = {
    'rows': None,
    'cols': None,
    'value': None,
    'dependent': False,
}

_full_slice = slice(None)


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
            msg = 'Variable name pair ("{}", "{}") not found.'
            raise KeyError(msg.format(key[0], key[1]))

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

    def _save_sparsity(self, system):
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

    def _compute_sparsity(self, system, wrt_matches, tol=1e-15, orders=5):
        """
        Compute a dense sparsity matrix for this jacobian using saved absolute summations.

        The sparsity matrix will contain only those columns that match the wrt variables in
        wrt_matches, but will contain rows for all outputs in the given system.

        Parameters
        ----------
        system : System
            The System containing the jacobian whose sparsity will be computed.
        wrt_matches : set of str
            Set of wrt variables to compute sparsity for.
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

        iproc = system.comm.rank
        abs2idx = system._var_allprocs_abs2idx['linear']
        approx_of_idx = system._owns_approx_of_idx
        approx_wrt_idx = system._owns_approx_wrt_idx
        ofsizes = system._var_sizes['linear']['output'][iproc]

        if system._owns_approx_of or system._owns_approx_wrt:
            # we're computing totals
            ofs = [n for n in system._var_allprocs_abs_names['output']
                   if n in system._owns_approx_of]
            wrts = [n for n in system._var_allprocs_abs_names['output']
                    if n in system._owns_approx_wrt]
            wrt_info = ((wrts, ofsizes, approx_wrt_idx),)
        else:
            from openmdao.core.implicitcomponent import ImplicitComponent
            ofs = system._var_allprocs_abs_names['output']
            wrts = system._var_allprocs_abs_names['input']
            isizes = system._var_sizes['linear']['input'][iproc]
            wrt_info = [(ofs, ofsizes, ())]
            if isinstance(system, ImplicitComponent):
                wrt_info.append(((ofs, ofsizes, ())))
            wrt_info.append((wrts, isizes, ()))

        ncols = nrows = 0
        locs = {}
        roffset = rend = 0
        for of in ofs:
            if of in approx_of_idx:
                sub_of_idx = approx_of_idx[of]
                rend += len(sub_of_idx)
            else:
                rend += ofsizes[abs2idx[of]]
                sub_of_idx = _full_slice
            coffset = cend = 0
            for wrts, sizes, approx_idx in wrt_info:
                for wrt in wrts:
                    if wrt in wrt_matches:
                        if wrt in approx_idx:
                            sub_wrt_idx = approx_idx[wrt]
                            cend += len(sub_wrt_idx)
                        else:
                            cend += sizes[abs2idx[wrt]]
                            sub_wrt_idx = _full_slice
                        key = (of, wrt)
                        if key in subjacs:
                            locs[key] = ((slice(roffset, rend), slice(coffset, cend)),
                                         sub_of_idx, sub_wrt_idx)
                        coffset = cend
            roffset = rend

        J = np.zeros((rend, cend))

        for key in locs:
            jslice, sub_of_idx, sub_wrt_idx = locs[key]
            meta = subjacs[key]
            if meta['rows'] is not None:
                rows = meta['rows'] + jslice[0].start
                # if sub_of_idx is not _full_slice:
                #     rows = rows[sub_of_idx]
                cols = meta['cols'] + jslice[1].start
                # if sub_wrt_idx is not _full_slice:
                #     cols = cols[sub_wrt_idx]
                J[rows, cols] = summ[key]
            elif issparse(summ[key]):
                raise NotImplementedError("don't support scipy sparse arrays yet")
            else:
                J[jslice] = summ[key]

        # normalize by number of saved jacs, giving a sort of 'average' jac
        J /= system.options['dynamic_derivs_repeats']

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
