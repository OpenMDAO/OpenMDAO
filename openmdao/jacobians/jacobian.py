"""Define the base Jacobian class."""
import weakref

import numpy as np
from numpy.random import rand

from collections import OrderedDict, defaultdict
from scipy.sparse import issparse, coo_matrix

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
    _col_var_info : dict
        Maps column name to start, end, and slice/indices into the result array.
    _colnames : list
        List of column var names.
    _col2name_ind : ndarray
        Array that maps jac col index to index of column name.
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
        self._col_var_info = None
        self._colnames = None
        self._col2name_ind = None

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
        abs2meta = self._system()._var_allprocs_abs2meta
        of, wrt = abs_key
        if wrt in abs2meta['input']:
            sz = abs2meta['input'][wrt]['size']
        else:
            sz = abs2meta['output'][wrt]['size']
        return (abs2meta['output'][of]['size'], sz)

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

    def keys(self):
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
        if self._jac_summ is None:
            fdtypes = ('cs', 'fd')
            # create _jac_summ structure
            self._jac_summ = summ = {}
            for key, meta in self._subjacs_info.items():
                if key[0] in system._owns_approx_of_idx or ('method' in meta and
                                                            meta['method'] in fdtypes):
                    summ[key] = np.abs(meta['value'])
        else:
            subjacs = self._subjacs_info
            for key, summ in self._jac_summ.items():
                summ += np.abs(subjacs[key]['value'])

    def _compute_sparsity(self, ordered_of_info, ordered_wrt_info, tol, orders):
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
        sys_subjacs = self._system()._subjacs_info
        summ = self._jac_summ

        Jrows = []
        Jcols = []
        Jdata = []

        # TODO: this currently doesn't use indices info for total approx derivs that
        #       could greatly reduce the size of data we need to save
        for of, roffset, rend, _ in ordered_of_info:
            for wrt, coffset, cend, _, _ in ordered_wrt_info:
                key = (of, wrt)
                if key in summ:
                    subsum = summ[key]
                    meta = subjacs[key]
                    if meta['rows'] is not None:
                        sysmeta = sys_subjacs[key]
                        Jrows.append(meta['rows'] + roffset)
                        Jcols.append(meta['cols'] + coffset)
                        sprows = sysmeta['rows']
                        spcols = sysmeta['cols']
                        if sysmeta['rows'].size != meta['rows'].size:
                            mask = np.zeros(sprows.size, dtype=bool)
                            for r in meta['rows']:
                                mask |= sprows == r
                            sprows = sprows[mask]
                            spcols = spcols[mask]
                            subsum = subsum[mask]
                        if sysmeta['cols'].size != meta['cols'].size:
                            mask = np.zeros(spcols.size, dtype=bool)
                            for c in meta['cols']:
                                mask |= spcols == c
                            sprows = sprows[mask]
                        Jdata.append(subsum)
                    elif issparse(subsum):
                        raise NotImplementedError("{}: scipy sparse arrays are not "
                                                  "supported yet.".format(self.msginfo))
                    else:  # dense
                        for i, r in enumerate(range(roffset, rend)):
                            Jrows.append([r] * (cend - coffset))
                            Jcols.append(range(coffset, cend))
                            Jdata.append(subsum[i, :])

        summ = self._jac_summ = None  # free up some memory

        Jrows = np.hstack(Jrows)
        Jcols = np.hstack(Jcols)
        Jdata = np.hstack([d.flat for d in Jdata])
        shape = (rend, cend)

        # TODO: for now, convert to dense, but later keep as COO
        J = coo_matrix((Jdata, (Jrows, Jcols)), shape=shape).toarray()

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

    def _setup_index_maps(self, system):
        self._col_var_info = col_var_info = {t[0]: t for t in system._jac_wrt_iter()}
        self._colnames = list(col_var_info)   # map var id to varname

        ncols = np.sum(end - start for _, start, end, _, _ in col_var_info.values())
        self._col2name_ind = np.empty(ncols, dtype=int)  # jac col to var id
        start = end = 0
        for i, (of, _start, _end, _, _) in enumerate(col_var_info.values()):
            end += _end - _start
            self._col2name_ind[start:end] = i
            start = end

        # for total derivs, we can have sub-indices making some subjacs smaller
        if system.pathname == '':
            for key, meta in system._subjacs_info.items():
                nrows, ncols = meta['shape']
                if key[0] in system._owns_approx_of_idx:
                    ridxs = system._owns_approx_of_idx[key[0]]
                    if len(ridxs) == nrows:
                        ridxs = _full_slice  # value was already changed
                else:
                    ridxs = _full_slice
                if key[1] in system._owns_approx_wrt_idx:
                    cidxs = system._owns_approx_wrt_idx[key[1]]
                    if len(cidxs) == ncols:
                        cidxs = _full_slice  # value was already changed
                else:
                    cidxs = _full_slice

                if ridxs is not _full_slice or cidxs is not _full_slice:
                    # replace our local subjac with a smaller one but don't
                    # change the subjac belonging to the system (which has values
                    # shared with subsystems)
                    if self._subjacs_info is system._subjacs_info:
                        self._subjacs_info = system._subjacs_info.copy()
                    self._subjacs_info[key] = meta.copy()
                    meta = self._subjacs_info[key]

                    if ridxs is not _full_slice:
                        nrows = len(ridxs)
                    if cidxs is not _full_slice:
                        ncols = len(cidxs)
                    if meta['rows'] is None:  # dense
                        val = meta['value']
                        val = val[ridxs, :]
                        val = val[:, cidxs]
                        meta['value'] = val
                    else:  # sparse
                        sprows = meta['rows']
                        spcols = meta['cols']
                        if ridxs is not _full_slice:
                            mask = np.zeros(sprows.size, dtype=bool)
                            for r in ridxs:
                                mask |= sprows == r
                            sprows = sprows[mask]
                            spcols = spcols[mask]
                        if cidxs is not _full_slice:
                            mask = np.zeros(sprows.size, dtype=bool)
                            for c in cidxs:
                                mask |= spcols == c
                            sprows = sprows[mask]
                            spcols = spcols[mask]
                        meta['rows'] = sprows
                        meta['cols'] = spcols
                    meta['shape'] = (nrows, ncols)

    def set_col(self, system, icol, column):
        """
        Set a column of the jacobian.

        The column is assumed to be the same size as a column of the jacobian.

        This also assumes that the column does not attempt to set any nonzero values that are
        outside of specified sparsity patterns for any of the subjacs.

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

        for of, start, end, _ in system._jac_of_iter():
            key = (of, wrt)
            if key in self._subjacs_info:
                subjac = self._subjacs_info[key]
                # TODO: support other sparse subjac types
                if subjac['rows'] is None:
                    subjac['value'][:, loc_idx] = column[start:end]
                else:
                    match_inds = np.nonzero(subjac['cols'] == loc_idx)[0]
                    subjac['value'][match_inds] = column[start:end][subjac['rows'][match_inds]]
