"""
Subjacobian classes.

Subjacobian classes are used to store the subjacobian information for a given variable pair.
They are used to store the subjacobian in a variety of formats, including dense, sparse, and
OpenMDAO's internal COO format.

"""

from pprint import pformat

import numpy as np
from numpy import bincount, isscalar
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, issparse

# from openmdao.devtools.debug import DebugDict


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
    dtype : dtype
        The dtype of the subjacobian.
    src_indices : array or None
        Source indices for the subjacobian.  None unless the jacobian is split into a square
        and non-square part where the square part has outputs as rows and columns, requiring
        a mapping of inputs to their source outputs via the src_indices array.
    factor : float or None
        Unit conversion factor for the subjacobian if, as with src_indices, we have a square
        part of the jacobian requiring a mapping of inputs to their source outputs for the
        jacobian columns where the input and output have different units.
    src : str or None
        Source name for the subjacobian.

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
    dtype : dtype
        The dtype of the subjacobian.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    src : str or None
        Source name for the subjacobian.
    dense : bool
        Whether the subjacobian is dense.
    nrows : int
        Number of rows in the subjacobian.
    ncols : int
        Number of columns in the subjacobian.
    parent_ncols : int
        Number of columns this subjacobian occupies in the parent jacobian.  If this subjac has
        src_indices then parent_ncols may differ from ncols.
    _in_view : ndarray or None
        View of this subjac's slice of the input vector.
    _out_view : ndarray or None
        View of this subjac's slice of the output vector.
    _res_view : ndarray or None
        View of this subjac's slice of the residual vector.
    """

    def __init__(self, key, info, row_slice, col_slice, wrt_is_input, dtype, src_indices=None,
                 factor=None, src=None):
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
        dtype : dtype
            The dtype of the subjacobian.
        src_indices : array or None
            Source indices for the subjacobian.  If not None, this is a subjac in the dr/do matrix
            of a SplitJacobian.
        factor : float or None
            Unit conversion factor for the subjacobian.
        src : str or None
            Source name for the subjacobian.
        """
        self.key = key
        self.info = info
        self.row_slice = row_slice
        self.col_slice = col_slice
        self.dtype = dtype
        self.nrows = row_slice.stop - row_slice.start
        self.ncols = info['shape'][1]
        self.parent_ncols = col_slice.stop - col_slice.start
        if src_indices is not None:
            src_indices = src_indices.shaped_array(flat=True)
        self.src_indices = src_indices
        self.factor = factor
        self.src = src
        self.dense = False

        self._in_view = None
        self._out_view = None
        self._res_view = None

        self._map_functions(wrt_is_input)
        self._init_val()

    def __repr__(self):
        """
        Return a string representation of the subjacobian.

        Returns
        -------
        str
            String representation of the subjacobian.
        """
        return (f"{type(self).__name__}(key={self.key}, nrows={self.nrows}, ncols={self.ncols}, "
                f"parent_ncols={self.parent_ncols}, row_slice={self.row_slice}, "
                f"col_slice={self.col_slice}, dtype={self.dtype}, "
                f"src_indices={self.src_indices}, factor={self.factor}, src={self.src}, "
                f"info:\n{pformat(self.info)})\n")

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
        if pattern_meta.get('diagonal'):
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

    def get_val(self):
        """
        Get the value of the subjacobian.

        Returns
        -------
        ndarray
            Subjac data.
        """
        return self.info['val']

    def set_dtype(self, dtype):
        """
        Set the dtype of the subjacobian.

        Parameters
        ----------
        dtype : dtype
            The type to set the subjacobian to.
        """
        if dtype.kind == self.info['val'].dtype.kind:
            return

        self._in_view = None
        self._out_view = None
        self._res_view = None

        if dtype.kind == 'f':
            self.info['val'] = np.ascontiguousarray(self.info['val'].real)
        elif dtype.kind == 'c':
            self.info['val'] = np.asarray(self.info['val'], dtype=dtype)
        else:
            raise ValueError(f"Subjacobian {self.key}: Unsupported dtype: {dtype}")

    def _map_functions(self, wrt_is_input):
        if wrt_is_input:
            self.apply_fwd = self._apply_fwd_input
            self.apply_rev = self._apply_rev_input
        else:
            self.apply_fwd = self._apply_fwd_output
            self.apply_rev = self._apply_rev_output

    def _apply_fwd_input(self, d_inputs, d_outputs, d_residuals, randgen=None):
        if self._in_view is None:
            self._in_view = d_inputs.get_slice(self.col_slice)
            self._res_view = d_residuals.get_slice(self.row_slice)

        val = self.info['val'] if randgen is None else self.get_rand_val(randgen)
        self._res_view += val @ self._in_view

    def _apply_fwd_output(self, d_inputs, d_outputs, d_residuals, randgen=None):
        if self._out_view is None:
            self._out_view = d_outputs.get_slice(self.col_slice)
            self._res_view = d_residuals.get_slice(self.row_slice)

        val = self.info['val'] if randgen is None else self.get_rand_val(randgen)
        self._res_view += val @ self._out_view

    def _apply_rev_input(self, d_inputs, d_outputs, d_residuals, randgen=None):
        if self._in_view is None:
            self._in_view = d_inputs.get_slice(self.col_slice)
            self._res_view = d_residuals.get_slice(self.row_slice)

        val = self.info['val'].T if randgen is None else self.get_rand_val(randgen).T
        self._in_view += val @ self._res_view

    def _apply_rev_output(self, d_inputs, d_outputs, d_residuals, randgen=None):
        if self._out_view is None:
            self._out_view = d_outputs.get_slice(self.col_slice)
            self._res_view = d_residuals.get_slice(self.row_slice)

        val = self.info['val'].T if randgen is None else self.get_rand_val(randgen).T
        self._out_view += val @ self._res_view


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
    dtype : dtype
        The dtype of the subjacobian.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    src : str or None
        Source name for the subjacobian.
    """

    def _init_val(self):
        self.dense = True
        if self.info['val'] is None:
            self.info['val'] = np.zeros(self.info['shape'])

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
        elif isscalar(val):
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

    def get_rand_val(self, randgen):
        """
        Get the value of the subjacobian.

        If randgen is not None then we're computing sparsity and we return a matrix of random
        values based on our 'sparsity' metadata.

        Parameters
        ----------
        randgen : RandomNumberGenerator
            Random number generator.

        Returns
        -------
        ndarray
            Subjacobian value.
        """
        # we're generating a random subjac during total derivative sparsity computation, so we need
        # to check on the value of the 'sparsity' metadata so that our actual sparsity will be
        # reported up to the top level.  Otherwise we'll report a fully dense subjac and the
        # total derivative sparsity will be too conservative.
        if self.info['sparsity'] is not None:
            rows, cols, _ = self.info['sparsity']
            # since the subjac is dense, we need to create a dense random array
            r = np.zeros(self.info['shape'])
            randval = randgen.random(rows.size)
            randval += 1.0
            r[rows, cols] = randval
        else:
            r = randgen.random(self.info['shape'])
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
        if isscalar(val):
            self.info['val'][:] = val
        else:
            myval = self.info['val']
            try:
                myval[:] = np.atleast_2d(val).reshape(myval.shape)
            except ValueError as err:
                if val.size == 1:  # allow for backwards compatability
                    myval[:] = val.item()
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
        self.info['val'][:, icol] = column

    def todense(self):
        """
        Return the subjacobian as a dense array.

        Returns
        -------
        ndarray
            Subjacobian as a dense array.
        """
        return self.info['val']

    def get_as_coo_data(self, randgen=None):
        """
        Get the subjac as data from a COO matrix.

        Parameters
        ----------
        randgen : RandomNumberGenerator or None
            Random number generator.

        Returns
        -------
        ndarray
            Subjac data.
        """
        if randgen is None:
            return self.info['val'].ravel()

        return self.get_rand_val(randgen).ravel()

    def get_coo_data_size(self):
        """
        Get the size of the subjacobian in COO format.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        return self.nrows * self.ncols

    def as_coo_info(self, full=False, randgen=None):
        """
        Get the subjac as COO data.

        This is here for completeness, but it shouldn't normally be called.

        Parameters
        ----------
        full : bool
            Whether to offset the row and column indices by the row and column slice start so the
            rows and columns will then represent the rows and columns of the full jacobian.
        randgen : RandomNumberGenerator or None
            Random number generator.

        Returns
        -------
        tuple
            (data, rows, cols).
        """
        nrows, ncols = self.info['shape']
        roffset = self.row_slice.start if full else 0
        size = self.row_slice.stop - self.row_slice.start
        a = np.arange(roffset, roffset + size).reshape((nrows, 1))
        rows = np.repeat(a, ncols, axis=1).ravel()

        coffset = self.col_slice.start if full else 0
        if self.src_indices is None:
            colrange = range(coffset, coffset + ncols)
        else:
            colrange = self.src_indices
            if full:
                colrange = colrange + coffset

        cols = np.tile(colrange, nrows)

        if randgen is None:
            return self.info['val'].ravel(), rows, cols

        return self.get_rand_val(randgen).ravel(), rows, cols


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
    dtype : dtype
        The dtype of the subjacobian.
    src_indices : array or None
        Source indices for the subjacobian.  None unless the jacobian is split into a square
        and non-square part where the square part has outputs as rows and columns, requiring
        a mapping of inputs to their source outputs via the src_indices array.
    factor : float or None
        Unit conversion factor for the subjacobian if, as with src_indices, we have a square
        part of the jacobian requiring a mapping of inputs to their source outputs for the
        jacobian columns where the input and output have different units.
    src : str or None
        Source name for the subjacobian.
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
        if dtype.kind == self.info['val'].dtype.kind:
            return

        self._in_view = None
        self._out_view = None
        self._res_view = None

        if dtype.kind == 'f':
            self.info['val'].data = np.ascontiguousarray(self.info['val'].data.real, dtype=dtype)
        elif dtype.kind == 'c':
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

    def as_coo_info(self, full=False, randgen=None):
        """
        Get the subjac as COO data.

        Parameters
        ----------
        full : bool
            Whether to offset the row and column indices by the row and column slice start so the
            rows and columns will then represent the rows and columns of the full jacobian.
        randgen : RandomNumberGenerator or None
            Random number generator.

        Returns
        -------
        tuple
            (data, rows, cols).
        """
        coo = self.info['val'].tocoo(copy=True)
        row = coo.row
        col = coo.col

        if randgen is None:
            data = coo.data
        else:
            data = randgen.random(coo.data.size)
            data += 1.0

        if self.src_indices is not None:
            col = self.src_indices[col]

        if full:
            row = row + self.row_slice.start
            col = col + self.col_slice.start

        return data, row, col

    def get_coo_data_size(self):
        """
        Get the size the subjacobian would be if flattened.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        return self.info['val'].data.size

    def get_as_coo_data(self, randgen=None):
        """
        Get the subjac as data from a COO matrix.

        Parameters
        ----------
        randgen : RandomNumberGenerator or None
            Random number generator.

        Returns
        -------
        ndarray
            Subjac data.
        """
        if randgen is None:
            return self.as_coo_info()[0]

        # our data size will be the same as the COO data size, so we don't have to
        # convert to COO first
        randval = randgen.random(self.get_coo_data_size())
        randval += 1.0

        return randval

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
                self.info['val'] = val
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
    dtype : dtype
        The dtype of the subjacobian.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    src : str or None
        Source name for the subjacobian.
    """

    def get_rand_val(self, randgen):
        """
        Get the value of the subjacobian.

        Parameters
        ----------
        randgen : RandomNumberGenerator
            Random number generator.

        Returns
        -------
        coo_matrix
            Subjacobian value.
        """
        submat = self.info['val']
        randval = randgen.random(submat.data.size)
        randval += 1.0

        return coo_matrix((randval, (submat.row, submat.col)), shape=self.info['shape'])

    def as_coo_info(self, full=False, randgen=None):
        """
        Get the subjac as COO data.

        Parameters
        ----------
        full : bool
            Whether to offset the row and column indices by the row and column slice start so the
            rows and columns will then represent the rows and columns of the full jacobian.
        randgen : RandomNumberGenerator or None
            Random number generator.

        Returns
        -------
        tuple
            (data, rows, cols).
        """
        coo = self.info['val']
        data, rows, cols = coo.data, coo.row, coo.col
        if randgen is not None:
            data = randgen.random(data.size)
            data += 1.0

        if self.src_indices is not None:
            # if src_indices is not None, we are part of the dr/do matrix, so columns correspond
            # to source variables and we have to convert columns using src_indices.
            cols = self.src_indices[cols]

        if full:
            rows = rows + self.row_slice.start
            cols = cols + self.col_slice.start

        return data, rows, cols

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
        col_match = col == icol
        row_inds = row[col_match]
        data[col_match] = column[row_inds]

        if uncovered_threshold is not None:  # do a sparsity check
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
        if dtype.kind == self.info['val'].dtype.kind:
            return

        self._in_view = None
        self._out_view = None
        self._res_view = None

        if dtype.kind == 'f':
            self.info['val'] = np.ascontiguousarray(self.info['val'].real, dtype=dtype)
        elif dtype.kind == 'c':
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
    dtype : dtype
        The dtype of the subjacobian.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    src : str or None
        Source name for the subjacobian.
    """

    def get_rand_val(self, randgen):
        """
        Get the value of the subjacobian.

        Parameters
        ----------
        randgen : RandomNumberGenerator
            Random number generator.

        Returns
        -------
        csr_matrix
            Subjacobian value.
        """
        submat = self.info['val']
        randval = randgen.random(submat.data.size)
        randval += 1.0

        return csr_matrix((randval, submat.indices, submat.indptr), shape=self.info['shape'])

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
    dtype : dtype
        The dtype of the subjacobian.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    src : str or None
        Source name for the subjacobian.
    """

    def get_rand_val(self, randgen):
        """
        Get the value of the subjacobian.

        Parameters
        ----------
        randgen : RandomNumberGenerator
            Random number generator.

        Returns
        -------
        csc_matrix
            Subjacobian value.
        """
        submat = self.info['val']
        randval = randgen.random(submat.data.size)
        randval += 1.0

        return csc_matrix((randval, submat.indices, submat.indptr), shape=self.info['shape'])

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
    dtype : dtype
        The dtype of the subjacobian.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    src : str or None
        Source name for the subjacobian.

    Attributes
    ----------
    rows : ndarray
        Row indices of the subjacobian.
    cols : ndarray
        Column indices of the subjacobian.
    """

    def __init__(self, key, info, row_slice, col_slice, wrt_is_input, dtype, src_indices=None,
                 factor=None, src=None):
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
        dtype : dtype
            The dtype of the subjacobian.
        src_indices : array or None
            Source indices for the subjacobian.
        factor : float or None
            Unit conversion factor for the subjacobian.
        src : str or None
            Source name for the subjacobian.
        """
        super().__init__(key, info, row_slice, col_slice, wrt_is_input, dtype, src_indices,
                         factor, src)
        self.rows = info['rows']
        self.cols = info['cols']
        self.set_val(info['val'])

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
        elif isscalar(val):
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
        arr = np.zeros(self.info['shape'])
        arr[self.rows, self.cols] = self.info['val']
        return arr

    def get_coo_data_size(self):
        """
        Get the size the subjacobian would be if flattened.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        return self.rows.size

    def as_coo_info(self, full=False, randgen=None):
        """
        Get the subjac as COO data.

        Parameters
        ----------
        full : bool
            Whether to offset the row and column indices by the row and column slice start so the
            rows and columns will then represent the rows and columns of the full jacobian.
        randgen : RandomNumberGenerator or None
            Random number generator.

        Returns
        -------
        tuple
            (data, rows, cols).
        """
        if randgen is None:
            data = self.info['val']
        else:
            data = self.get_rand_val(randgen)
        rows, cols = self.rows, self.cols

        if self.src_indices is not None:
            # if src_indices is not None, we are part of the dr/do matrix, so columns correspond
            # to source variables and we have to convert columns using src_indices.
            cols = self.src_indices[cols]

        if full:
            rows = rows + self.row_slice.start
            cols = cols + self.col_slice.start

        return data, rows, cols

    def get_rand_val(self, randgen):
        """
        Get the value of the subjacobian.

        Parameters
        ----------
        randgen : RandomNumberGenerator
            Random number generator.

        Returns
        -------
        coo_matrix
            Subjacobian value.
        """
        randval = randgen.random(self.info['val'].size)
        randval += 1.0

        return randval

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
        self._set_coo_col(icol, column, self.info['val'], self.rows, self.cols,
                          uncovered_threshold)

    def _apply_fwd_input(self, d_inputs, d_outputs, d_residuals, randgen=None):
        if self._in_view is None:
            self._in_view = d_inputs.get_slice(self.col_slice)
            self._res_view = d_residuals.get_slice(self.row_slice)

        val = self.info['val'] if randgen is None else self.get_rand_val(randgen)
        # bincount allows rows and cols to contain repeated (row, col) pairs.
        self._res_view += bincount(self.rows, self._in_view[self.cols] * val, minlength=self.nrows)

    def _apply_fwd_output(self, d_inputs, d_outputs, d_residuals, randgen=None):
        if self._out_view is None:
            self._out_view = d_outputs.get_slice(self.col_slice)
            self._res_view = d_residuals.get_slice(self.row_slice)

        val = self.info['val'] if randgen is None else self.get_rand_val(randgen)
        # bincount allows rows and cols to contain repeated (row, col) pairs.
        self._res_view += bincount(self.rows, self._out_view[self.cols] * val, minlength=self.nrows)

    def _apply_rev_input(self, d_inputs, d_outputs, d_residuals, randgen=None):
        if self._in_view is None:
            self._in_view = d_inputs.get_slice(self.col_slice)
            self._res_view = d_residuals.get_slice(self.row_slice)

        val = self.info['val'] if randgen is None else self.get_rand_val(randgen)
        self._in_view += bincount(self.cols, self._res_view[self.rows] * val,
                                  minlength=self.parent_ncols)

    def _apply_rev_output(self, d_inputs, d_outputs, d_residuals, randgen=None):
        if self._out_view is None:
            self._out_view = d_outputs.get_slice(self.col_slice)
            self._res_view = d_residuals.get_slice(self.row_slice)

        val = self.info['val'] if randgen is None else self.get_rand_val(randgen)
        self._out_view += bincount(self.cols, self._res_view[self.rows] * val,
                                   minlength=self.parent_ncols)


class DiagonalSubjac(SparseSubjac):
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
    dtype : dtype
        The dtype of the subjacobian.
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    src : str or None
        Source name for the subjacobian.
    """

    def _init_val(self):
        """
        Initialize the val in the subjacobian's metadata, converting to correct shape if necessary.
        """
        if self.info['val'] is None:
            self.info['val'] = np.zeros(self.nrows)
        elif isscalar(self.info['val']):
            self.info['val'] = np.full(self.nrows, self.info['val'])

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
        elif isscalar(val):
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

    def get_coo_data_size(self):
        """
        Get the size the subjacobian would be if flattened.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        return self.info['val'].size

    def as_coo_info(self, full=False, randgen=None):
        """
        Get the subjac as COO data, rows, and cols.

        Parameters
        ----------
        full : bool
            Whether to offset the row and column indices by the row and column slice start so the
            rows and columns will then represent the rows and columns of the full jacobian.
        randgen : RandomNumberGenerator or None
            Random number generator.

        Returns
        -------
        tuple
            (data, rows, cols).
        """
        if full:
            rows = np.arange(self.row_slice.start, self.row_slice.stop)
            if self.src_indices is None:
                cols = np.arange(self.col_slice.start, self.col_slice.stop)
            else:
                cols = self.src_indices + self.col_slice.start
        else:
            rows = cols = np.arange(self.nrows)
            if self.src_indices is not None:
                cols = self.src_indices

        if randgen is None:
            return self.info['val'], rows, cols

        return self.get_rand_val(randgen), rows, cols

    def get_rand_val(self, randgen):
        """
        Get the value of the subjacobian.

        Parameters
        ----------
        randgen : RandomNumberGenerator
            Random number generator.

        Returns
        -------
        ndarray
            Subjac data.
        """
        val = randgen.random(self.info['val'].size)
        val += 1.0
        return val

    def get_as_coo_data(self, randgen=None):
        """
        Get the subjac as COO data.

        Parameters
        ----------
        randgen : RandomNumberGenerator or None
            Random number generator.

        Returns
        -------
        ndarray
            Subjacobian data.
        """
        if randgen is None:
            return self.info['val']

        return self.get_rand_val(randgen)

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
                if 'uncovered_nz' not in self.info:
                    self.info['uncovered_nz'] = []
                self.info['uncovered_nz'].extend(list(zip(nzs, icol * np.ones_like(nzs))))
            column[icol] = save

    def _apply_fwd_input(self, d_inputs, d_outputs, d_residuals, randgen=None):
        if self._in_view is None:
            self._in_view = d_inputs.get_slice(self.col_slice)
            self._res_view = d_residuals.get_slice(self.row_slice)

        val = self.info['val'] if randgen is None else self.get_rand_val(randgen)
        self._res_view += self._in_view * val

    def _apply_fwd_output(self, d_inputs, d_outputs, d_residuals, randgen=None):
        if self._out_view is None:
            self._out_view = d_outputs.get_slice(self.col_slice)
            self._res_view = d_residuals.get_slice(self.row_slice)

        val = self.info['val'] if randgen is None else self.get_rand_val(randgen)
        self._res_view += self._out_view * val

    def _apply_rev_input(self, d_inputs, d_outputs, d_residuals, randgen=None):
        if self._in_view is None:
            self._in_view = d_inputs.get_slice(self.col_slice)
            self._res_view = d_residuals.get_slice(self.row_slice)

        val = self.info['val'] if randgen is None else self.get_rand_val(randgen)
        self._in_view += self._res_view * val

    def _apply_rev_output(self, d_inputs, d_outputs, d_residuals, randgen=None):
        if self._out_view is None:
            self._out_view = d_outputs.get_slice(self.col_slice)
            self._res_view = d_residuals.get_slice(self.row_slice)

        val = self.info['val'] if randgen is None else self.get_rand_val(randgen)
        self._out_view += self._res_view * val

    def set_dtype(self, dtype):
        """
        Set the dtype of the subjacobian.

        Parameters
        ----------
        dtype : dtype
            The type to set the subjacobian to.
        """
        if dtype.kind == self.info['val'].dtype.kind:
            return

        self._in_view = None
        self._out_view = None
        self._res_view = None

        if dtype.kind == 'f':
            self.info['val'] = np.ascontiguousarray(self.info['val'].real, dtype=dtype)
        elif dtype.kind == 'c':
            self.info['val'] = np.asarray(self.info['val'], dtype=dtype)
        else:
            raise ValueError(f"Subjacobian {self.key}: Unsupported dtype: {dtype}")


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
