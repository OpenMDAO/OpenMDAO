"""Define the base Jacobian class."""
import weakref

import numpy as np

from collections import defaultdict
from scipy.sparse import issparse

from openmdao.core.constants import INT_DTYPE
from openmdao.utils.name_maps import key2abs_key, rel_name2abs_name
from openmdao.utils.array_utils import sparse_subinds
from openmdao.matrices.matrix import sparse_types
from openmdao.vectors.vector import _full_slice

SUBJAC_META_DEFAULTS = {
    'rows': None,
    'cols': None,
    'val': None,
    'dependent': False,
}


class Jacobian(object):
    """
    Base Jacobian class.

    This class provides a dictionary interface for sub-Jacobians and
    performs matrix-vector products when apply_linear is called.

    Parameters
    ----------
    system : System
        Parent system to this jacobian.

    Attributes
    ----------
    _system : <System>
        Pointer to the system that is currently operating on this Jacobian.
    _subjacs_info : dict
        Dictionary of the sub-Jacobian metadata keyed by absolute names.
    _under_complex_step : bool
        When True, this Jacobian is under complex step, using a complex jacobian.
    _abs_keys : dict
        A cache dict for key to absolute key.
    _randgen : Generator or None
        If not None, use the generator to generate random numbers during computation of
        sparsity for for simultaneous derivative coloring.
    _col_var_offset : dict
        Maps column name to offset into the result array.
    _col_varnames : list
        List of column var names.
    _col2name_ind : ndarray
        Array that maps jac col index to index of column name.
    """

    def __init__(self, system):
        """
        Initialize all attributes.
        """
        self._system = weakref.ref(system)
        self._subjacs_info = system._subjacs_info
        self._under_complex_step = False
        self._abs_keys = {}
        self._randgen = None
        self._col_var_offset = None
        self._col_varnames = None
        self._col2name_ind = None

    def _get_abs_key(self, key):
        if key in self._abs_keys:
            return self._abs_keys[key]
        abskey = key2abs_key(self._system(), key)
        if abskey is not None:
            self._abs_keys[key] = abskey
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
        bool
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
            return self._subjacs_info[abs_key]['val']
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
        if abs_key is None:
            msg = '{}: Variable name pair ("{}", "{}") not found.'
            raise KeyError(msg.format(self.msginfo, key[0], key[1]))

        # You can only set declared subjacobians.
        if abs_key not in self._subjacs_info:
            msg = '{}: Variable name pair ("{}", "{}") must first be declared.'
            raise KeyError(msg.format(self.msginfo, key[0], key[1]))

        subjacs_info = self._subjacs_info[abs_key]

        if issparse(subjac):
            subjacs_info['val'] = subjac
        else:
            rows = subjacs_info['rows']

            if rows is None:
                # Dense subjac
                subjac = np.atleast_2d(subjac)
                if subjac.shape != (1, 1):
                    shape = self._abs_key2shape(abs_key)
                    subjac = subjac.reshape(shape)

                subjacs_info['val'][:] = subjac

            else:
                try:
                    subjacs_info['val'][:] = subjac
                except ValueError:
                    subjac = np.atleast_1d(subjac)
                    msg = '{}: Sub-jacobian for key {} has the wrong shape ({}), expected ({}).'
                    raise ValueError(msg.format(self.msginfo, abs_key,
                                                subjac.shape, rows.shape))

    def __iter__(self):
        """
        Yield next name pair of sub-Jacobian.

        Yields
        ------
        str
        """
        yield from self._subjacs_info.keys()

    def keys(self):
        """
        Yield next name pair of sub-Jacobian.

        Yields
        ------
        str
        """
        yield from self._subjacs_info.keys()

    def items(self):
        """
        Yield name pair and value of sub-Jacobian.

        Yields
        ------
        str
        """
        for key, meta in self._subjacs_info.items():
            yield key, meta['val']

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
            sparse.data = self._randgen.random(sparse.data.size)
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
            val = self._randgen.random(len(rows))
            val += 1.0
            r[rows, cols] = val
        else:
            r = self._randgen.random(subjac.shape)
            r += 1.0
        return r

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
                meta['val'] = meta['val'].astype(complex)
            else:
                meta['val'] = meta['val'].real

        self._under_complex_step = active

    def _setup_index_maps(self, system):
        self._col_var_offset = {}
        col_var_info = []
        for wrt, start, end, _, _, _ in system._jac_wrt_iter():
            self._col_var_offset[wrt] = start
            col_var_info.append(end)

        self._col_varnames = list(self._col_var_offset)
        self._col2name_ind = np.empty(end, dtype=INT_DTYPE)  # jac col to var id
        start = 0
        for i, end in enumerate(col_var_info):
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
                        ridxs = ridxs.shaped_array()
                else:
                    ridxs = _full_slice
                if key[1] in system._owns_approx_wrt_idx:
                    cidxs = system._owns_approx_wrt_idx[key[1]]
                    if len(cidxs) == ncols:
                        cidxs = _full_slice  # value was already changed
                    else:
                        cidxs = cidxs.shaped_array()
                else:
                    cidxs = _full_slice

                if ridxs is not _full_slice or cidxs is not _full_slice:
                    # replace our local subjac with a smaller one but don't
                    # change the subjac belonging to the system (which has values
                    # shared with subsystems)
                    if self._subjacs_info is system._subjacs_info:
                        self._subjacs_info = system._subjacs_info.copy()
                    meta = self._subjacs_info[key] = meta.copy()
                    val = meta['val']

                    if ridxs is not _full_slice:
                        nrows = len(ridxs)
                    if cidxs is not _full_slice:
                        ncols = len(cidxs)

                    if meta['rows'] is None:  # dense
                        val = val[ridxs, :]
                        val = val[:, cidxs]
                        meta['val'] = val
                    else:  # sparse
                        sprows = meta['rows']
                        spcols = meta['cols']
                        if ridxs is not _full_slice:
                            sprows, mask = sparse_subinds(sprows, ridxs)
                            spcols = spcols[mask]
                            val = val[mask]
                        if cidxs is not _full_slice:
                            spcols, mask = sparse_subinds(spcols, cidxs)
                            sprows = sprows[mask]
                            val = val[mask]
                        meta['rows'] = sprows
                        meta['cols'] = spcols
                        meta['val'] = val

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
        if self._col_varnames is None:
            self._setup_index_maps(system)

        wrt = self._col_varnames[self._col2name_ind[icol]]
        loc_idx = icol - self._col_var_offset[wrt]  # local col index into subjacs

        for of, start, end, _, _ in system._jac_of_iter():
            key = (of, wrt)
            if key in self._subjacs_info:
                subjac = self._subjacs_info[key]
                if subjac['cols'] is None:  # dense
                    subjac['val'][:, loc_idx] = column[start:end]
                else:  # our COO format
                    match_inds = np.nonzero(subjac['cols'] == loc_idx)[0]
                    if match_inds.size > 0:
                        subjac['val'][match_inds] = column[start:end][subjac['rows'][match_inds]]

    def set_dense_jac(self, system, jac):
        """
        Assign a dense jacobian to this jacobian.

        This assumes that any column does not attempt to set any nonzero values that are
        outside of specified sparsity patterns for any of the subjacs.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        jac : ndarray
            Dense jacobian.
        """
        if self._col_varnames is None:
            self._setup_index_maps(system)

        wrtiter = list(system._jac_wrt_iter())
        for of, start, end, _, _ in system._jac_of_iter():
            for wrt, wstart, wend, _, _, _ in wrtiter:
                key = (of, wrt)
                if key in self._subjacs_info:
                    subjac = self._subjacs_info[key]
                    if subjac['cols'] is None:  # dense
                        subjac['val'][:, :] = jac[start:end, wstart:wend]
                    else:  # our COO format
                        subj = jac[start:end, wstart:wend]
                        subjac['val'][:] = subj[subjac['rows'], subjac['cols']]

    def _restore_approx_sparsity(self):
        """
        Revert all subjacs back to the way they were as declared by the user.
        """
        self._subjacs_info = self._system()._subjacs_info
        self._col_varnames = None  # force recompute of internal index maps on next set_col
