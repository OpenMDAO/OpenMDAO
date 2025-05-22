"""
Subjacobian classes.

Subjacobian classes are used to store the subjacobian information for a given variable pair.
They are used to store the subjacobian in a variety of formats, including dense, sparse, and
OpenMDAO's internal COO format.

"""

from pprint import pformat

import numpy as np
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
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    src : str or None
        Source name for the subjacobian.
    shape : tuple
        Shape of the subjacobian.
    _randval : ndarray or None
        Random value of the subjacobian that will be used for later apply calls.
    """

    def __init__(self, key, info, row_slice, col_slice, wrt_is_input, src_indices=None,
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
        src_indices : array or None
            Source indices for the subjacobian.
        factor : float or None
            Unit conversion factor for the subjacobian.
        src : str or None
            Source name for the subjacobian.
        """
        self.key = key
        self.info = info
        self.row_slice = row_slice
        self.col_slice = col_slice
        self.src_indices = src_indices
        self.factor = factor
        self.src = src
        self.shape = (row_slice.stop - row_slice.start, col_slice.stop - col_slice.start)
        self._randval = None

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
        return (f"{type(self).__name__}(key={self.key}, shape={self.shape}, "
                f"row_slice={self.row_slice}, col_slice={self.col_slice}, "
                f"src_indices={self.src_indices}, factor={self.factor}, src={self.src}, "
                f"info:\n{pformat(self.info)})")

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

    def reset_random(self):
        """
        Reset cached random subjac.
        """
        self._randval = None

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
        else:
            self.apply_fwd = self._apply_fwd_output
            self.apply_rev = self._apply_rev_output

    def _matvec_fwd(self, vec, randgen=None):
        return self.get_val(randgen) @ vec

    def _matvec_rev(self, vec, randgen=None):
        return self.get_val(randgen).T @ vec

    def _apply_fwd_input(self, d_inputs, d_outputs, d_residuals, randgen=None):
        d_residuals.add_to_slice(self.row_slice,
                                 self._matvec_fwd(d_inputs.get_slice(self.col_slice), randgen))

    def _apply_fwd_output(self, d_inputs, d_outputs, d_residuals, randgen=None):
        d_residuals.add_to_slice(self.row_slice,
                                 self._matvec_fwd(d_outputs.get_slice(self.col_slice), randgen))

    def _apply_rev_input(self, d_inputs, d_outputs, d_residuals, randgen=None):
        d_inputs.add_to_slice(self.col_slice,
                              self._matvec_rev(d_residuals.get_slice(self.row_slice), randgen))

    def _apply_rev_output(self, d_inputs, d_outputs, d_residuals, randgen=None):
        d_outputs.add_to_slice(self.col_slice,
                               self._matvec_rev(d_residuals.get_slice(self.row_slice), randgen))


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
    src : str or None
        Source name for the subjacobian.
    """

    def _init_val(self):
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

    def get_val(self, randgen=None):
        """
        Get the value of the subjacobian.

        If randgen is not None then we're computing sparsity and we return a matrix of random
        values based on our 'sparsity' metadata.

        Parameters
        ----------
        randgen : RandomNumberGenerator or None
            Random number generator.

        Returns
        -------
        ndarray
            Subjacobian value.
        """
        if randgen is None:
            return self.info['val']

        # we're generating a random subjac
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
        if np.isscalar(val):
            self.info['val'][:] = val
        else:
            try:
                myval = self.info['val']
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
        self.get_val()[:, icol] = column

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
        return self.get_val(randgen).ravel()

    def get_coo_data_size(self):
        """
        Get the size of the subjacobian in COO format.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        if self.src_indices is None:
            ncols = self.shape[1]
        else:
            ncols = self.src_indices.indexed_src_size
        return self.shape[0] * ncols

    def cols(self):
        """
        Get the COO column indices of the subjacobian.

        Returns
        -------
        ndarray
            COO column indices of the subjacobian.
        """
        if self.src_indices is None:
            colrange = range(self.col_slice.stop - self.col_slice.start)
        else:
            colrange = self.src_indices.shaped_array(flat=True)

        return np.tile(colrange, self.shape[0])

    def rows(self):
        """
        Get the COO row indices of the subjacobian.

        Returns
        -------
        ndarray
            COO row indices of the subjacobian.
        """
        nrows = self.shape[0]
        a = np.arange(self.row_slice.stop - self.row_slice.start).reshape((nrows, 1))
        repeats = self.shape[1] if self.src_indices is None else self.src_indices.indexed_src_size
        return np.repeat(a, repeats, axis=1).ravel()


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

    def rows(self):
        """
        Get the COO row indices of the subjacobian.

        Returns
        -------
        ndarray
            COO row indices of the subjacobian.
        """
        _, rows, _ = self.as_coo_info()
        return rows

    def cols(self):
        """
        Get the COO column indices of the subjacobian.

        Returns
        -------
        ndarray
            COO column indices of the subjacobian.
        """
        _, _, cols = self.as_coo_info()
        return cols

    def as_coo_info(self):
        """
        Get the subjac as COO data.

        Returns
        -------
        tuple
            (data, rows, cols).
        """
        coo = self.info['val'].tocoo(copy=True)
        if self.src_indices is None:
            return coo.data, coo.row, coo.col
        else:
            # if src_indices is not None, we are part of the 'int_mtx' of an assembled jac, so
            # columns correspond to source variables and we have to convert columns using
            # src_indices.
            return coo.data, coo.row, self.src_indices.shaped_array(flat=True)[coo.col]

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

        if self._randval is None:
            # our data size will be the same as the COO data size, so we don't have to
            # convert to COO first
            self._randval = randgen.random(self.get_coo_data_size())
            self._randval += 1.0

        return self._randval

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
    src_indices : array or None
        Source indices for the subjacobian.
    factor : float or None
        Unit conversion factor for the subjacobian.
    src : str or None
        Source name for the subjacobian.
    """

    def get_val(self, randgen=None):
        """
        Get the value of the subjacobian.

        Parameters
        ----------
        randgen : RandomNumberGenerator or None
            Random number generator.

        Returns
        -------
        coo_matrix
            Subjacobian value.
        """
        if randgen is None:
            return self.info['val']

        submat = self.info['val']
        if self._randval is None:
            self._randval = randgen.random(submat.data.size) + 1.

        return coo_matrix((self._randval, (submat.row, submat.col)),
                          shape=self.info['shape'])

    def as_coo_info(self):
        """
        Get the subjac as COO data.

        Returns
        -------
        tuple
            (data, rows, cols).
        """
        coo = self.info['val']
        if self.src_indices is None:
            return coo.data, coo.row, coo.col
        else:
            # if src_indices is not None, we are part of the 'int_mtx' of an assembled jac, so
            # columns correspond to source variables and we have to convert columns using
            # src_indices.
            return coo.data, coo.row, self.src_indices.shaped_array(flat=True)[coo.col]

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
    src : str or None
        Source name for the subjacobian.
    """

    def get_val(self, randgen=None):
        """
        Get the value of the subjacobian.

        Parameters
        ----------
        randgen : RandomNumberGenerator or None
            Random number generator.

        Returns
        -------
        csr_matrix
            Subjacobian value.
        """
        if randgen is None:
            return self.info['val']

        submat = self.info['val']
        if self._randval is None:
            self._randval = randgen.random(submat.data.size) + 1.

        return csr_matrix((self._randval, submat.indices, submat.indptr),
                          shape=self.info['shape'])

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
    src : str or None
        Source name for the subjacobian.
    """

    def get_val(self, randgen=None):
        """
        Get the value of the subjacobian.

        Parameters
        ----------
        randgen : RandomNumberGenerator or None
            Random number generator.

        Returns
        -------
        csc_matrix
            Subjacobian value.
        """
        if randgen is None:
            return self.info['val']

        submat = self.info['val']
        if self._randval is None:
            self._randval = randgen.random(submat.data.size) + 1.

        return csc_matrix((self._randval, submat.indices, submat.indptr),
                          shape=self.info['shape'])

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
    src : str or None
        Source name for the subjacobian.
    """

    def __init__(self, key, info, row_slice, col_slice, wrt_is_input, src_indices=None,
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
        src_indices : array or None
            Source indices for the subjacobian.
        factor : float or None
            Unit conversion factor for the subjacobian.
        src : str or None
            Source name for the subjacobian.
        """
        super().__init__(key, info, row_slice, col_slice, wrt_is_input, src_indices, factor, src)
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
        arr = np.zeros(self.info['shape'])
        arr[self.info['rows'], self.info['cols']] = self.info['val']
        return arr

    def get_coo_data_size(self):
        """
        Get the size the subjacobian would be if flattened.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        return self.info['val'].size

    def as_coo_info(self):
        """
        Get the subjac as COO data.

        Returns
        -------
        tuple
            (data, rows, cols).
        """
        if self.src_indices is None:
            return self.info['val'], self.info['rows'], self.info['cols']
        else:
            # if src_indices is not None, we are part of the 'int_mtx' of an assembled jac, so
            # columns correspond to source variables and we have to convert columns using
            # src_indices.
            return self.info['val'], self.info['rows'], \
                self.src_indices.shaped_array(flat=True)[self.info['cols']]

    def get_val(self, randgen=None):
        """
        Get the value of the subjacobian.

        Parameters
        ----------
        randgen : RandomNumberGenerator or None
            Random number generator.

        Returns
        -------
        coo_matrix
            Subjacobian value.
        """
        if randgen is None:
            return self.info['val']

        if self._randval is None:
            self._randval = randgen.random(self.info['val'].size)
            self._randval += 1.0

        return self._randval

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
        self._set_coo_col(icol, column, self.info['val'], self.info['rows'], self.info['cols'],
                          uncovered_threshold)

    def _matvec_fwd(self, vec, randgen=None):
        return np.bincount(self.info['rows'], vec[self.info['cols']] * self.get_val(randgen),
                           minlength=self.shape[0])

    def _matvec_rev(self, vec, randgen=None):
        return np.bincount(self.info['cols'], vec[self.info['rows']] * self.get_val(randgen),
                           minlength=self.shape[1])


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

    def get_coo_data_size(self):
        """
        Get the size the subjacobian would be if flattened.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        return self.info['val'].size

    def rows(self):
        """
        Get the COO row indices of the subjacobian.

        Returns
        -------
        ndarray
            COO row indices of the subjacobian.
        """
        return np.arange(self.shape[0])

    def cols(self):
        """
        Get the COO column indices of the subjacobian.

        Returns
        -------
        ndarray
            COO column indices of the subjacobian.
        """
        if self.src_indices is None:
            return np.arange(self.shape[0])
        else:
            return self.src_indices.shaped_array(flat=True)[np.arange(self.shape[0])]

    def get_val(self, randgen=None):
        """
        Get the value of the subjacobian.

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
            return self.info['val']

        val = randgen.random(self.info['val'].size)
        val += 1.0
        return val

    get_as_coo_data = get_val

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

    def _matvec_fwd(self, vec, randgen=None):
        return self.get_val(randgen) * vec

    _matvec_rev = _matvec_fwd

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
        meta['val'] = coo_matrix((np.zeros(0), (np.zeros(0, dtype=int), np.zeros(0, dtype=int))),
                                 shape=meta['shape'])
        return meta

    def get_coo_data_size(self):
        """
        Get the size the subjacobian would be if flattened.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        return 0

    def get_val(self, randgen=None):
        """
        Get the subjac as data from a COO matrix.

        Parameters
        ----------
        randgen : RandomNumberGenerator or None
            Random number generator.

        Returns
        -------
        ndarray
            Value of the subjacobian.
        """
        return np.zeros(0)

    get_as_coo_data = get_val

    def todense(self):
        """
        Return the subjacobian as a dense array.

        Returns
        -------
        ndarray
            Subjacobian as a dense array.
        """
        return np.zeros(self.shape)

    def rows(self):
        """
        Get the COO row indices of the subjacobian.

        Returns
        -------
        ndarray
            COO row indices of the subjacobian.
        """
        return np.zeros(0)

    def cols(self):
        """
        Get the COO column indices of the subjacobian.

        Returns
        -------
        ndarray
            COO column indices of the subjacobian.
        """
        return np.zeros(0)

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

    def _matvec_fwd(self, vec, randgen=None):
        return np.zeros_like(vec)

    def _matvec_rev(self, vec, randgen=None):
        return np.zeros_like(vec)

    # override these to avoid doing *any* operations on the subjac
    def _apply_fwd_input(self, d_inputs, d_outputs, d_residuals, randgen=None):
        pass

    def _apply_fwd_output(self, d_inputs, d_outputs, d_residuals, randgen=None):
        pass

    def _apply_rev_input(self, d_inputs, d_outputs, d_residuals, randgen=None):
        pass

    def _apply_rev_output(self, d_inputs, d_outputs, d_residuals, randgen=None):
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
