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
    src_shape : tuple
        Shape of the subjacobian of the row var with respect to wrt's source.

    Attributes
    ----------
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

    def __init__(self, info, row_slice, col_slice, wrt_is_input, src_indices=None, factor=None,
                 src_shape=None):
        """
        Initialize the subjacobian.

        Parameters
        ----------
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
        src_shape : tuple
            Shape of the subjacobian of the row var with respect to wrt's source.
        """
        self.info = info
        self.row_slice = row_slice
        self.col_slice = col_slice
        self.src_indices = src_indices
        self.factor = factor
        self.randval = None
        if src_shape is None:
            self.shape = (row_slice.stop - row_slice.start, col_slice.stop - col_slice.start)
        else:
            self.shape = src_shape

        self._map_functions(wrt_is_input)
        self._init_val()

    def _init_val(self):
        pass

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

        This is the value set by a System and it may not be a 2D array. For example, if the subjac
        is a diagonal subjac, the value will be a 1D array of the diagonal values.

        Returns
        -------
        ndarray
            Value of the subjacobian.
        """
        return self.info['val']

    def as_2d(self):
        """
        Return the subjacobian as a 2D array.

        Returns
        -------
        ndarray
            Subjacobian as a 2D array.
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
        if issparse(val):
            myval[:] = val.toarray()
        else:
            myval[:] = np.atleast_2d(val).reshape(myval.shape)

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
        return self._shape[0] * self._shape[1]

    def get_sparse_data_size(self):
        """
        Get the size of the subjacobian in COO format.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        # for dense, this is the same as get_full_size if src_indices is None
        if self.src_indices is None:
            return self.get_full_size()
        else:
            # each src_ind is a column in the subjac, so the full subjac size is
            # the number of rows times the number of matching columns
            return self.shape[0] * self.src_indices.indexed_src_size

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
    src_shape : tuple
        Shape of the subjacobian of the row var with respect to wrt's source.
    """

    def _init_val(self):
        if self.info['val'] is None:
            self.info['val'] = np.zeros(self.shape)


class SparseSubjac(Subjac):
    """
    Sparse subjacobian.

    Parameters
    ----------
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
    src_shape : tuple
        Shape of the subjacobian of the row var with respect to wrt's source.
    """

    def as_2d(self):
        """
        Return the subjacobian as a 2D array.

        Returns
        -------
        ndarray
            Subjacobian as a 2D array.
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
        if self.src_indices is None:
            return self.info['val'].data.size
        else:
            coo = self.info['val'].tocoo()
            return coo.col[np.isin(coo.col, self.src_indices.asarray())].size

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
    src_shape : tuple
        Shape of the subjacobian of the row var with respect to wrt's source.
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


class CSRSubjac(SparseSubjac):
    """
    Sparse subjacobian in CSR format.

    Parameters
    ----------
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
    src_shape : tuple
        Shape of the subjacobian of the row var with respect to wrt's source.
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
    src_shape : tuple
        Shape of the subjacobian of the row var with respect to wrt's source.
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
    src_shape : tuple
        Shape of the subjacobian of the row var with respect to wrt's source.

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

    def __init__(self, info, row_slice, col_slice, wrt_is_input, src_indices=None, factor=None,
                 src_shape=None):
        """
        Initialize the subjacobian.

        Parameters
        ----------
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
        src_shape : tuple
            Shape of the subjacobian of the row var with respect to wrt's source.
        """
        super().__init__(info, row_slice, col_slice, wrt_is_input, src_indices, factor, src_shape)
        self.rows = info['rows']
        self.cols = info['cols']
        self.mask = slice(None)

        if info['rows'] is not None and src_indices is not None:
            colset = set(src_indices.shaped_array())
            self.mask = np.isin(self.cols, colset)
            self.rows = self.rows[self.mask]
            self.cols = self.cols[self.mask]

        self.set_val(info['val'])

    def as_2d(self):
        """
        Return the subjacobian as a 2D array.

        Returns
        -------
        ndarray
            Subjacobian as a 2D array.
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
        if self.src_indices is None:
            return len(self.cols)
        else:
            cols = np.asarray(self.cols)
            return cols[np.isin(cols, self.src_indices.asarray())].size

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
    src_shape : tuple
        Shape of the subjacobian of the row var with respect to wrt's source.
    """

    def as_2d(self):
        """
        Return the subjacobian as a 2D array.

        Returns
        -------
        ndarray
            Subjacobian as a 2D array.
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
        if self.src_indices is None:
            return self.info['val'].size
        else:
            cols = np.arange(self.info['val'].size)
            return cols[np.isin(cols, self.src_indices.asarray())].size

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
                          shape=(size, size))

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
    src_shape : tuple
        Shape of the subjacobian of the row var with respect to wrt's source.
    """

    def as_2d(self):
        """
        Return the subjacobian as a 2D array.

        Returns
        -------
        ndarray
            Subjacobian as a 2D array.
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


SUBJAC_META_DEFAULTS = {
    'rows': None,
    'cols': None,
    'val': None,
    'dependent': True,
    'diagonal': False,
    'sparsity': None,
}
