"""
Subjacobian classes.

Subjacobian classes are used to store the subjacobian information for a given variable pair.
They are used to store the subjacobian in a variety of formats, including dense, sparse, and
OpenMDAO's internal COO format.

"""

import numpy as np
from scipy.sparse import coo_matrix, issparse


class Subjac(object):
    """
    Base class for subjacobians.

    Parameters
    ----------
    key : tuple
        The (of, wrt) key.
    info : dict
        Metadata for the subjacobian.
    row_slice : slice
        Slice of the row indices.
    col_slice : slice
        Slice of the column indices.  Note that this is the slice into either the input or output
        vector, not the absolute indices with respect to the full jacobian columns.
    wrt_is_input : bool
        Whether the wrt variable is an input.
    src_indices : array or None
        Source indices for the subjacobian.  None unless the jacobian is split into a square
        and non-square part where the square part has outputs as rows and columns, requiring
        a mapping of inputs to their source outputs via the src_indices array.
    factor : float or None
        Unit conversion factor for the subjacobian if, as with src_indices, we have a square
        part of the jacobian requiring a mapping of inputs to their source outputs for the
        jacobian columns where the input and output have different units.

    Attributes
    ----------
    key : tuple
        The (of, wrt) key.
    info : dict
        Metadata for the subjacobian.
    row_slice : slice
        Slice of the row indices.
    col_slice : slice
        Slice of the column indices.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    randval : ndarray or None
        Random value for the subjacobian.
    shape : tuple
        Shape of the subjacobian.
    """

    def __init__(self, key, info, row_slice, col_slice, wrt_is_input, src_indices=None,
                 factor=None):
        """
        Initialize the subjacobian.

        Parameters
        ----------
        key : tuple
            The (of, wrt) key.
        info : dict
            Metadata for the subjacobian.
        row_slice : slice
            Slice of the row indices.
        col_slice : slice
            Slice of the column indices.
        wrt_is_input : bool
            Whether the wrt variable is an input.
        src_indices : array or None
            Source indices for the subjacobian.
        factor : float or None
            Unit conversion factor for the subjacobian.
        """
        self.key = key
        self.info = info
        self.row_slice = row_slice
        self.col_slice = col_slice
        self.src_indices = src_indices
        self.factor = factor
        self.randval = None
        self.shape = (row_slice.stop - row_slice.start, col_slice.stop - col_slice.start)

        self._map_functions(wrt_is_input)
        self._init_val()

    @staticmethod
    def get_subjac_class(pattern_meta):
        """
        Get the subjacobian class for the given pattern metadata.

        Parameters
        ----------
        pattern_meta : dict
            Pattern metadata.

        Returns
        -------
        Subjac
            The subjacobian class.
        """
        if not pattern_meta['dependent']:
            return ZeroSubjac
        elif pattern_meta.get('diagonal'):
            return DiagonalSubjac
        elif pattern_meta.get('rows') is None:
            if issparse(pattern_meta.get('val')):
                try:
                    return _sparse_subjac_types[pattern_meta['val'].format]
                except KeyError:
                    raise NotImplementedError(f"Subjac format {pattern_meta['val'].format} not "
                                              "supported.")
            else:
                return DenseSubjac
        else:
            return OMCOOSubjac

    @staticmethod
    def get_instance_metadata(pattern_meta, prev_inst_meta, shape, system, key):
        """
        Get the instance metadata for the given pattern metadata.

        Parameters
        ----------
        pattern_meta : dict
            Pattern metadata.
        prev_inst_meta : dict
            Previous instance metadata, if any.
        shape : tuple
            Shape of the subjacobian.
        system : System
            The system containing the subjac.
        key : str
            The (of, wrt) key indicating which subjac is being updated.

        Returns
        -------
        dict
            Instance metadata.
        """
        subjac_class = Subjac.get_subjac_class(pattern_meta)
        meta = _init_meta(pattern_meta, prev_inst_meta)
        pat_val = pattern_meta.get('val')
        if pat_val is None:
            meta['val'] = None
        meta['shape'] = shape
        return subjac_class._update_instance_meta(meta, system, key)

    def _init_val(self):
        pass

    def set_dtype(self, dtype):
        """
        Set the dtype of the subjacobian.

        Parameters
        ----------
        dtype : dtype
            The type to set the subjacobian to.
        """
        if dtype is self.info['val'].dtype:
            return

        if dtype is float:
            self.info['val'] = self.info['val'].real
        elif dtype is complex:
            self.info['val'] = np.asarray(self.info['val'], dtype=dtype)
        else:
            raise ValueError(f"Subjacobian {self.key}: Unsupported dtype: {dtype}")

    def _map_functions(self, wrt_is_input):
        if wrt_is_input:
            self.apply_fwd = self._apply_fwd_input
            self.apply_rev = self._apply_rev_input
            self.apply_rand_fwd = self._apply_rand_fwd_input
            self.apply_rand_rev = self._apply_rand_rev_input
        else:
            self.apply_fwd = self._apply_fwd_output
            self.apply_rev = self._apply_rev_output
            self.apply_rand_fwd = self._apply_rand_fwd_output
            self.apply_rand_rev = self._apply_rand_rev_output

    def get_val(self):
        """
        Get the value of the subjacobian.

        This is the value set by a System and it may not be a dense array. For example, if the
        subjac is a diagonal subjac, the value will be a 1D array of the diagonal values.

        Returns
        -------
        ndarray
            Value of the subjacobian.
        """
        return self.info['val']

    def todense(self):
        """
        Return the subjacobian as a dense array.

        Returns
        -------
        ndarray
            Subjacobian as a dense array.
        """
        return self.info['val']

    def get_random_subjac(self, randgen):
        """
        Get a random subjacobian.

        Parameters
        ----------
        randgen : RandomNumberGenerator
            Random number generator.

        Returns
        -------
        ndarray
            Random subjacobian.
        """
        if self.info['sparsity'] is not None:
            rows, cols, shape = self.info['sparsity']
            r = np.zeros(shape)
            val = randgen.random(len(rows))
            val += 1.0
            r[rows, cols] = val
        else:
            r = randgen.random(self.shape)
            r += 1.0

        return r

    def set_val(self, val):
        """
        Set the value of the subjacobian.

        Parameters
        ----------
        val : ndarray
            Value to set the subjacobian to.
        """
        myval = self.info['val']
        if np.isscalar(val):
            myval[:] = val
        else:
            try:
                myval[:] = np.atleast_2d(val).reshape(myval.shape)
            except ValueError as err:
                if val.size == 1:  # allow for backwards compatability
                    myval[:] = val[0]
                else:
                    raise err

    def set_col(self, icol, column, uncovered_threshold=None):
        """
        Set a column of the subjacobian.

        Parameters
        ----------
        icol : int
            Column index to set.
        column : ndarray
            Column to set.
        uncovered_threshold : float or None
            Threshold for uncovered elements. Only used in _CheckingJacobian for sparse subjacs.
        """
        self.get_val()[:, icol] = column

    def get_full_size(self):
        """
        Get the full dense size of the subjacobian.

        Returns
        -------
        int
            Full dense size of the subjacobian.
        """
        return self.shape[0] * self.shape[1]

    def get_sparse_data_size(self):
        """
        Get the size of the subjacobian in COO format.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        shape = self.info['shape']  # this is shape of dr/di subjac, not dr/do subjac
        return shape[0] * shape[1]

    def _apply_rand_fwd_input(self, d_inputs, d_outputs, d_residuals, randgen):
        d_residuals.add_to_slice(
            self.row_slice, self.get_random_subjac(randgen) @ d_inputs.get_slice(self.col_slice))

    def _apply_rand_fwd_output(self, d_inputs, d_outputs, d_residuals, randgen):
        d_residuals.add_to_slice(
            self.row_slice, self.get_random_subjac(randgen) @ d_outputs.get_slice(self.col_slice))

    def _apply_rand_rev_input(self, d_inputs, d_outputs, d_residuals, randgen):
        d_inputs.add_to_slice(
            self.col_slice,
            self.get_random_subjac(randgen).T @ d_residuals.get_slice(self.row_slice))

    def _apply_rand_rev_output(self, d_inputs, d_outputs, d_residuals, randgen):
        d_outputs.add_to_slice(
            self.col_slice,
            self.get_random_subjac(randgen).T @ d_residuals.get_slice(self.row_slice))

    def _apply_fwd_input(self, d_inputs, d_outputs, d_residuals):
        d_residuals.add_to_slice(self.row_slice,
                                 self.get_val() @ d_inputs.get_slice(self.col_slice))

    def _apply_fwd_output(self, d_inputs, d_outputs, d_residuals):
        d_residuals.add_to_slice(self.row_slice,
                                 self.get_val() @ d_outputs.get_slice(self.col_slice))

    def _apply_rev_input(self, d_inputs, d_outputs, d_residuals):
        d_inputs.add_to_slice(self.col_slice,
                              self.get_val().T @ d_residuals.get_slice(self.row_slice))

    def _apply_rev_output(self, d_inputs, d_outputs, d_residuals):
        d_outputs.add_to_slice(self.col_slice,
                               self.get_val().T @ d_residuals.get_slice(self.row_slice))


class DenseSubjac(Subjac):
    """
    Dense subjacobian.

    Parameters
    ----------
    key : tuple
        The (of, wrt) key.
    info : dict
        Metadata for the subjacobian.
    row_slice : slice
        Slice of the row indices.
    col_slice : slice
        Slice of the column indices.
    wrt_is_input : bool
        Whether the wrt variable is an input.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    """

    def _init_val(self):
        if self.info['val'] is None:
            self.info['val'] = np.zeros(self.shape)

    @classmethod
    def _update_instance_meta(cls, meta, system, key):
        """
        Update the given instance metadata.

        Parameters
        ----------
        meta : dict
            Instance metadata.
        system : System
            The system containing the subjac.
        key : str
            The (of, wrt) key indicating which subjac is being updated.

        Returns
        -------
        dict
            Updated instance metadata.
        """
        val = meta['val']
        if val is None:
            meta['val'] = np.zeros(meta['shape'])
        elif np.isscalar(val):
            meta['val'] = np.full(meta['shape'], val, dtype=float)
        else:
            if val.shape == meta['shape']:
                meta['val'] = np.asarray(val, dtype=float).copy()
            else:
                if meta['shape'] != val.shape:
                    if len(meta['shape']) != len(val.shape):
                        if meta['shape'] == (1, ) + val.shape:
                            val = val.reshape((1, ) + val.shape)
                        elif meta['shape'] == val.shape + (1, ):
                            val = val.reshape(val.shape + (1, ))

                    if meta['shape'] == val.shape:
                        meta['val'] = np.asarray(val, dtype=float).copy().reshape(meta['shape'])
                    else:
                        pathlen = len(system.pathname) + 1 if system.pathname else 0
                        relkey = (key[0][pathlen:], key[1][pathlen:])
                        of, wrt = relkey
                        raise ValueError(f"{system.msginfo}: d({of})/d({wrt}), Expected shape "
                                         f"{meta['shape']} but got {val.shape}.")

        return meta


class SparseSubjac(Subjac):
    """
    Sparse subjacobian.

    Parameters
    ----------
    key : tuple
        The (of, wrt) key.
    info : dict
        Metadata for the subjacobian.
    row_slice : slice
        Slice of the row indices.
    col_slice : slice
        Slice of the column indices.  Note that this is the slice into either the input or output
        vector, not the absolute indices with respect to the full jacobian columns.
    wrt_is_input : bool
        Whether the wrt variable is an input.
    src_indices : array or None
        Source indices for the subjacobian.  None unless the jacobian is split into a square
        and non-square part where the square part has outputs as rows and columns, requiring
        a mapping of inputs to their source outputs via the src_indices array.
    factor : float or None
        Unit conversion factor for the subjacobian if, as with src_indices, we have a square
        part of the jacobian requiring a mapping of inputs to their source outputs for the
        jacobian columns where the input and output have different units.

    Attributes
    ----------
    mask : array or None
        Mask to apply to the subjacobian rows/cols/data for an 'internal' matrix.
    nnz_src_inds : array or None
        Number of nonzero elements when only including
    """

    def __init__(self, key, info, row_slice, col_slice, wrt_is_input, src_indices=None,
                 factor=None):
        """
        Initialize the sparse subjac.
        """
        super().__init__(key, info, row_slice, col_slice, wrt_is_input, src_indices, factor)
        self.mask = None
        self.nnz_src_inds = None

    @classmethod
    def _update_instance_meta(cls, meta, system, key):
        """
        Update the given instance metadata.

        Parameters
        ----------
        meta : dict
            Instance metadata.
        system : System
            The system containing the subjac.
        key : str
            The (of, wrt) key indicating which subjac is being updated.

        Returns
        -------
        dict
            Updated instance metadata.
        """
        meta['val'] = meta['val'].copy()
        meta['val'].data = np.asarray(meta['val'].data, dtype=float)
        return meta

    def set_dtype(self, dtype):
        """
        Set the dtype of the subjacobian.

        Parameters
        ----------
        dtype : dtype
            The type to set the subjacobian to.
        """
        if dtype is self.info['val'].dtype:
            return

        if dtype is float:
            self.info['val'].data = self.info['val'].data.real
        elif dtype is complex:
            self.info['val'].data = np.asarray(self.info['val'].data, dtype=dtype)
        else:
            raise ValueError(f"Subjacobian {self.key}: Unsupported dtype: {dtype}")

    def todense(self):
        """
        Return the subjacobian as a dense array.

        Returns
        -------
        ndarray
            Subjacobian as a dense array.
        """
        return self.info['val'].toarray()

    def get_sparse_data_size(self):
        """
        Get the size the subjacobian would be if flattened.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        return self.info['val'].data.size

    def get_random_subjac(self, randgen):
        """
        Get a random subjacobian with the same sparsity pattern as this one.

        Parameters
        ----------
        randgen : RandomNumberGenerator
            Random number generator.

        Returns
        -------
        ndarray
            Random subjacobian.
        """
        sparse = self.info['val'].copy()
        sparse.data = randgen.random(sparse.data.size) + 1.0
        return sparse

    def set_val(self, val):
        """
        Set the value of the subjacobian.

        Parameters
        ----------
        val : ndarray
            Value to set the subjacobian to.
        """
        if issparse(val):
            if val.format == self.info['val'].format:
                self.info['val'].data[:] = val.data
            else:
                raise ValueError(f"Sparse subjacobian format {self.info['val'].format} does not "
                                 f"match the format of the provided array ({val.format}).")
        else:
            raise ValueError(f"Can't set sparse subjac with value of type {type(val).__name__}.")


class COOSubjac(SparseSubjac):
    """
    Sparse subjacobian in COO format.

    Parameters
    ----------
    key : tuple
        The (of, wrt) key.
    info : dict
        Metadata for the subjacobian.
    row_slice : slice
        Slice of the row indices.
    col_slice : slice
        Slice of the column indices.
    wrt_is_input : bool
        Whether the wrt variable is an input.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    """

    def tocoo(self, mask=None):
        """
        Convert the subjac to a COO matrix.

        Parameters
        ----------
        mask : array or None
            Mask to apply to the nonzero elements so only those corresponding to src_indices columns
            are included.

        Returns
        -------
        coo_matrix
            Subjacobian in COO format.
        """
        if mask is None:
            return self.get_val()
        else:
            return coo_matrix((self.info['val'].data[mask], (self.info['val'].row[mask],
                                                             self.info['val'].col[mask])),
                              shape=self.shape)

    def set_col(self, icol, column, uncovered_threshold=None):
        """
        Set a column of the subjacobian.

        Parameters
        ----------
        icol : int
            Column index to set.
        column : ndarray
            Column to set.
        uncovered_threshold : float or None
            Threshold for uncovered elements. Only used in _CheckingJacobian.
        """
        coo = self.info['val']
        self._set_coo_col(icol, column, coo.data, coo.row, coo.col, uncovered_threshold)

    def _set_coo_col(self, icol, column, data, row, col, uncovered_threshold=None):
        mask = col == icol
        row_inds = row[mask]
        data[mask] = column[row_inds]

        if uncovered_threshold is not None:
            arr = column.copy()
            arr[row_inds] = 0.  # zero out the rows that are covered by sparsity
            nzs = np.where(np.abs(arr) > uncovered_threshold)[0]
            if nzs.size > 0:
                if 'uncovered_nz' not in self.info:
                    self.info['uncovered_nz'] = []
                    self.info['uncovered_threshold'] = uncovered_threshold
                    self.info['uncovered_nz'].extend(list(zip(nzs, icol * np.ones_like(nzs))))

    def set_dtype(self, dtype):
        """
        Set the dtype of the subjacobian.

        Parameters
        ----------
        dtype : dtype
            The type to set the subjacobian to.
        """
        if dtype is self.info['val'].dtype:
            return

        if dtype is float:
            self.info['val'] = self.info['val'].real
        elif dtype is complex:
            self.info['val'] = np.asarray(self.info['val'], dtype=dtype)
        else:
            raise ValueError(f"Subjacobian {self.key}: Unsupported dtype: {dtype}")


class CSRSubjac(SparseSubjac):
    """
    Sparse subjacobian in CSR format.

    Parameters
    ----------
    key : tuple
        The (of, wrt) key.
    info : dict
        Metadata for the subjacobian.
    row_slice : slice
        Slice of the row indices.
    col_slice : slice
        Slice of the column indices.
    wrt_is_input : bool
        Whether the wrt variable is an input.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    """

    def set_col(self, icol, column, uncovered_threshold=None):
        """
        Set a column of the subjacobian.

        Parameters
        ----------
        icol : int
            Column index to set.
        column : ndarray
            Column to set.
        uncovered_threshold : float or None
            Threshold for uncovered elements. Only used in _CheckingJacobian.
        """
        # This isn't very efficient...
        csc = self.info['val'].tocsc()
        rowinds = csc.indices[csc.indptr[icol]:csc.indptr[icol + 1]]
        csc.data[csc.indptr[icol]:csc.indptr[icol + 1]] = column[rowinds]

        if uncovered_threshold is not None:
            arr = column.copy()
            arr[rowinds] = 0.  # zero out the rows that are covered by sparsity
            nzs = np.where(np.abs(arr) > uncovered_threshold)[0]
            if nzs.size > 0:
                if 'uncovered_nz' not in self.info:
                    self.info['uncovered_nz'] = []
                    self.info['uncovered_threshold'] = uncovered_threshold

        self.info['val'].data = csc.tocsr().data


class CSCSubjac(SparseSubjac):
    """
    Sparse subjacobian in CSC format.

    Parameters
    ----------
    key : tuple
        The (of, wrt) key.
    info : dict
        Metadata for the subjacobian.
    row_slice : slice
        Slice of the row indices.
    col_slice : slice
        Slice of the column indices.
    wrt_is_input : bool
        Whether the wrt variable is an input.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    """

    def set_col(self, icol, column, uncovered_threshold=None):
        """
        Set a column of the subjacobian.

        Parameters
        ----------
        icol : int
            Column index to set.
        column : ndarray
            Column to set.
        uncovered_threshold : float or None
            Threshold for uncovered elements. Only used in _CheckingJacobian.
        """
        csc = self.info['val']
        rowinds = csc.indices[csc.indptr[icol]:csc.indptr[icol + 1]]
        csc.data[csc.indptr[icol]:csc.indptr[icol + 1]] = column[rowinds]

        if uncovered_threshold is not None:
            arr = column.copy()
            arr[rowinds] = 0.  # zero out the rows that are covered by sparsity
            nzs = np.where(np.abs(arr) > uncovered_threshold)[0]
            if nzs.size > 0:
                if 'uncovered_nz' not in self.info:
                    self.info['uncovered_nz'] = []
                    self.info['uncovered_threshold'] = uncovered_threshold
                    self.info['uncovered_nz'].extend(list(zip(nzs, icol * np.ones_like(nzs))))


class OMCOOSubjac(COOSubjac):
    """
    Sparse subjacobian in OpenMDAO's internal COO format.

    Parameters
    ----------
    key : tuple
        The (of, wrt) key.
    info : dict
        Metadata for the subjacobian.
    row_slice : slice
        Slice of the row indices.
    col_slice : slice
        Slice of the column indices.
    wrt_is_input : bool
        Whether the wrt variable is an input.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.

    Attributes
    ----------
    mask : slice or array
        Mask to apply to the rows and columns when src_indices is not None.
        If None, no mask is applied.
    rows : array or None
        Rows of the subjacobian after applying the mask.
    cols : array or None
        Columns of the subjacobian after applying the mask.
    """

    def __init__(self, key, info, row_slice, col_slice, wrt_is_input, src_indices=None,
                 factor=None):
        """
        Initialize the subjacobian.

        Parameters
        ----------
        key : tuple
            The (of, wrt) key.
        info : dict
            Metadata for the subjacobian.
        row_slice : slice
            Slice of the row indices.
        col_slice : slice
            Slice of the column indices.
        wrt_is_input : bool
            Whether the wrt variable is an input.
        src_indices : array or None
            Source indices for the subjacobian.
        factor : float or None
            Unit conversion factor for the subjacobian.
        """
        super().__init__(key, info, row_slice, col_slice, wrt_is_input, src_indices, factor)
        self.rows = info['rows']
        self.cols = info['cols']
        self.mask = slice(None)

        if info['rows'] is not None and src_indices is not None:
            self.mask = np.isin(self.cols, src_indices.shaped_array(flat=True))
            self.rows = self.rows[self.mask]
            self.cols = self.cols[self.mask]

        self.set_val(info['val'])

    def get_col_inds(self):
        """
        Get the column indices of the subjacobian.

        Repeated entries are allowed.

        Returns
        -------
        array
            Column indices of the subjacobian.
        """
        return self.info['cols']

    @classmethod
    def _update_instance_meta(cls, meta, system, key):
        """
        Update the given instance metadata.

        Parameters
        ----------
        meta : dict
            Instance metadata.
        system : System
            The system containing the subjac.
        key : str
            The (of, wrt) key indicating which subjac is being updated.

        Returns
        -------
        dict
            Updated instance metadata.
        """
        val = meta['val']
        rows = meta['rows']
        if val is None:
            val = np.zeros(rows.size)
        elif np.isscalar(val):
            val = np.full(rows.size, val, dtype=float)
        else:
            val = np.asarray(val, dtype=float).copy().reshape(rows.size)

        meta['val'] = val

        return meta

    def todense(self):
        """
        Return the subjacobian as a dense array.

        Returns
        -------
        ndarray
            Subjacobian as a dense array.
        """
        return self.coo.toarray()

    def get_random_subjac(self, randgen):
        """
        Get a random subjacobian with the same sparsity pattern as this one.

        Parameters
        ----------
        randgen : RandomNumberGenerator
            Random number generator.

        Returns
        -------
        ndarray
            Random subjacobian.
        """
        val = randgen.random(len(self.rows)) + 1.0
        return coo_matrix((val, (self.rows, self.cols)), shape=self.shape)

    def set_val(self, val):
        """
        Set the value of the subjacobian.

        Parameters
        ----------
        val : ndarray
            Value to set the subjacobian to.
        """
        self.info['val'][:] = val
        self.coo = coo_matrix((self.info['val'][self.mask], (self.rows, self.cols)),
                              shape=self.shape)

    def set_col(self, icol, column, uncovered_threshold=None):
        """
        Set a column of the subjacobian.

        Parameters
        ----------
        icol : int
            Column index to set.
        column : ndarray
            Column to set.
        uncovered_threshold : float or None
            Threshold for uncovered elements. Only used in _CheckingJacobian.
        """
        self._set_coo_col(icol, column, self.info['val'], self.info['rows'], self.info['cols'],
                          uncovered_threshold)

    def get_sparse_data_size(self):
        """
        Get the size the subjacobian would be if flattened.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        return self.info['val'].size

    def _apply_fwd_input(self, d_inputs, d_outputs, d_residuals):
        self.coo.data[:] = self.get_val()
        d_residuals.add_to_slice(self.row_slice, self.coo @ d_inputs.get_slice(self.col_slice))

    def _apply_fwd_output(self, d_inputs, d_outputs, d_residuals):
        self.coo.data[:] = self.get_val()
        d_residuals.add_to_slice(self.row_slice, self.coo @ d_outputs.get_slice(self.col_slice))

    def _apply_rev_input(self, d_inputs, d_outputs, d_residuals):
        self.coo.data[:] = self.get_val()
        d_inputs.add_to_slice(self.col_slice, self.coo.T @ d_residuals.get_slice(self.row_slice))

    def _apply_rev_output(self, d_inputs, d_outputs, d_residuals):
        self.coo.data[:] = self.get_val()
        d_outputs.add_to_slice(self.col_slice, self.coo.T @ d_residuals.get_slice(self.row_slice))


class DiagonalSubjac(Subjac):
    """
    Diagonal subjacobian.

    Parameters
    ----------
    key : tuple
        The (of, wrt) key.
    info : dict
        Metadata for the subjacobian.
    row_slice : slice
        Slice of the row indices.
    col_slice : slice
        Slice of the column indices.
    wrt_is_input : bool
        Whether the wrt variable is an input.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    """

    def _init_val(self):
        """
        Initialize the val in the subjacobian's metadata, converting to correct shape if necessary.
        """
        if self.info['val'] is None:
            self.info['val'] = np.zeros(self.shape[0])
        elif np.isscalar(self.info['val']):
            self.info['val'] = np.full(self.shape[0], self.info['val'])

    @classmethod
    def _update_instance_meta(cls, meta, system, key):
        """
        Update the given instance metadata.

        Parameters
        ----------
        meta : dict
            Instance metadata.
        system : System
            The system containing the subjac.
        key : str
            The (of, wrt) key indicating which subjac is being updated.

        Returns
        -------
        dict
            Updated instance metadata.
        """
        val = meta['val']
        meta['rows'] = meta['cols'] = None
        if val is None:
            val = np.zeros(meta['shape'][0])
        elif np.isscalar(val):
            val = np.full(meta['shape'][0], val, dtype=float)
        else:
            val = np.asarray(val, dtype=float).copy().reshape(meta['shape'][0])

        meta['val'] = val

        return meta

    def todense(self):
        """
        Return the subjacobian as a dense array.

        Returns
        -------
        ndarray
            Subjacobian as a dense array.
        """
        return np.diag(self.info['val'])

    def get_sparse_data_size(self):
        """
        Get the size the subjacobian would be if flattened.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        return self.info['val'].size

    def set_val(self, val):
        """
        Set the value of the subjacobian.

        Parameters
        ----------
        val : ndarray
            Value to set the subjacobian to.
        """
        self.info['val'][:] = val

    def set_col(self, icol, column, uncovered_threshold=None):
        """
        Set a column of the subjacobian.

        Parameters
        ----------
        icol : int
            Column index to set.
        column : ndarray
            Column to set.
        uncovered_threshold : float or None
            Threshold for uncovered elements. Only used in _CheckingJacobian.
        """
        self.info['val'][icol] = column[icol]
        if uncovered_threshold is not None:
            save = column[icol]
            column[icol] = 0.  # zero out the row that is covered by sparsity
            nzs = np.where(np.abs(column) > uncovered_threshold)[0]
            if nzs.size > 0:
                self.info['uncovered_nz'].extend(list(zip(nzs, icol * np.ones_like(nzs))))
            column[icol] = save

    def get_random_subjac(self, randgen):
        """
        Get a random subjacobian with the same sparsity pattern as this one.

        Parameters
        ----------
        randgen : RandomNumberGenerator
            Random number generator.

        Returns
        -------
        ndarray
            Random subjacobian.
        """
        size = self.info['val'].size
        return coo_matrix((randgen.random(size) + 1.0, (range(size), range(size))),
                          shape=self.shape)

    def _apply_fwd_input(self, d_inputs, d_outputs, d_residuals):
        d_residuals.add_to_slice(self.row_slice,
                                 self.get_val() * d_inputs.get_slice(self.col_slice))

    def _apply_fwd_output(self, d_inputs, d_outputs, d_residuals):
        d_residuals.add_to_slice(self.row_slice,
                                 self.get_val() * d_outputs.get_slice(self.col_slice))

    def _apply_rev_input(self, d_inputs, d_outputs, d_residuals):
        d_inputs.add_to_slice(self.col_slice,
                              self.get_val() * d_residuals.get_slice(self.row_slice))

    def _apply_rev_output(self, d_inputs, d_outputs, d_residuals):
        d_outputs.add_to_slice(self.col_slice,
                               self.get_val() * d_residuals.get_slice(self.row_slice))


class ZeroSubjac(Subjac):
    """
    Zero subjacobian.

    Parameters
    ----------
    key : tuple
        The (of, wrt) key.
    info : dict
        Metadata for the subjacobian.
    row_slice : slice
        Slice of the row indices.
    col_slice : slice
        Slice of the column indices.
    wrt_is_input : bool
        Whether the wrt variable is an input.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    """

    @classmethod
    def _update_instance_meta(cls, meta, system, key):
        """
        Update the given instance metadata.

        Parameters
        ----------
        meta : dict
            Instance metadata.
        system : System
            The system containing the subjac.
        key : str
            The (of, wrt) key indicating which subjac is being updated.

        Returns
        -------
        dict
            Updated instance metadata.
        """
        meta['val'] = None
        meta['rows'] = meta['cols'] = None
        return meta

    def get_val(self):
        """
        Get the value of the subjacobian.

        This is the value set by a System and it may not be a dense array. For example, if the
        subjac is a diagonal subjac, the value will be a 1D array of the diagonal values.

        Returns
        -------
        coo_matrix
            Value of the subjacobian.
        """
        zro = np.zeros(0)
        return coo_matrix((zro, (zro, zro)), shape=self.shape)

    def todense(self):
        """
        Return the subjacobian as a dense array.

        Returns
        -------
        ndarray
            Subjacobian as a dense array.
        """
        return np.zeros(self.shape)

    def get_sparse_data_size(self):
        """
        Get the size the subjacobian would be if flattened.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        return 0

    def set_val(self, val):
        """
        Set the value of the subjacobian.

        Parameters
        ----------
        val : ndarray
            Value to set the subjacobian to.
        """
        pass

    def set_col(self, icol, column, uncovered_threshold=None):
        """
        Set a column of the subjacobian.

        Parameters
        ----------
        icol : int
            Column index to set.
        column : ndarray
            Column to set.
        uncovered_threshold : float or None
            Threshold for uncovered elements. Only used in _CheckingJacobian for sparse subjacs.
        """
        # treat this subjac as a completely sparse subjac, i.e. no rows/cols/data
        if uncovered_threshold is not None:
            nzs = np.where(np.abs(column) > uncovered_threshold)[0]
            if nzs.size > 0:
                self.info['uncovered_nz'].extend(list(zip(nzs, icol * np.ones_like(nzs))))

    def get_random_subjac(self, randgen):
        """
        Get a random subjacobian.

        Parameters
        ----------
        randgen : RandomNumberGenerator
            Random number generator.

        Returns
        -------
        ndarray
            Random subjacobian.
        """
        return np.zeros(self.shape)

    def _apply_fwd_input(self, d_inputs, d_outputs, d_residuals):
        pass

    def _apply_fwd_output(self, d_inputs, d_outputs, d_residuals):
        pass

    def _apply_rev_input(self, d_inputs, d_outputs, d_residuals):
        pass

    def _apply_rev_output(self, d_inputs, d_outputs, d_residuals):
        pass


def _init_meta(pattern_meta, prev_inst_meta):
    """
    Initialize the instance metadata for the subjacobian.

    Parameters
    ----------
    pattern_meta : dict
        Pattern metadata.
    prev_inst_meta : dict or None
        Previous instance metadata, if any.

    Returns
    -------
    dict
        Instance metadata.
    """
    if prev_inst_meta is not None:
        meta = prev_inst_meta.copy()
        for key, val in pattern_meta.items():
            if val is not None:
                meta[key] = val
    else:
        meta = SUBJAC_META_DEFAULTS.copy()
        meta['dependent'] = False
        meta.update(pattern_meta)

    return meta


SUBJAC_META_DEFAULTS = {
    'rows': None,
    'cols': None,
    'val': None,
    'dependent': True,
    'diagonal': False,
    'sparsity': None,
}

_sparse_subjac_types = {
    'coo': COOSubjac,
    'csr': CSRSubjac,
    'csc': CSCSubjac
}
