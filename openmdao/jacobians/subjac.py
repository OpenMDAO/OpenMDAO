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

        Returns
        -------
        ndarray
            Value of the subjacobian.
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

    def get_full_size(self):
        """
        Get the full dense size of the subjacobian.

        Returns
        -------
        int
            Full dense size of the subjacobian.
        """
        return self._shape[0] * self._shape[1]

    def get_coo_size(self):
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

    def get_coo_size(self):
        """
        Get the size the subjacobian would be if stored in a COO formatted jacobian.

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
        self.info['val'].data[:] = val.data

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


class OMCOOSubjac(Subjac):
    """
    Sparse subjacobian in OpenMDAO's internal COO format.

    Parameters
    ----------
    meta : dict
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

    def __init__(self, meta, row_slice, col_slice, wrt_is_input, src_indices=None, factor=None,
                 src_shape=None):
        """
        Initialize the subjacobian.

        Parameters
        ----------
        meta : dict
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
        super().__init__(meta, row_slice, col_slice, wrt_is_input, src_indices, factor, src_shape)
        self.set_val(meta['val'])

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
        val = randgen.random(len(self.info['rows'])) + 1.0
        return coo_matrix((val, (self.info['rows'], self.info['cols'])),
                          shape=self.shape)

    def set_val(self, val):
        """
        Set the value of the subjacobian.

        Parameters
        ----------
        val : ndarray
            Value to set the subjacobian to.
        """
        self.info['val'][:] = val
        self.coo = coo_matrix((self.info['val'], (self.info['rows'], self.info['cols'])),
                              shape=(self.shape))

    def get_coo_size(self):
        """
        Get the size the subjacobian would be if stored in a COO formatted jacobian.

        Returns
        -------
        int
            Size of the subjacobian in COO format.
        """
        if self.src_indices is None:
            return len(self.info['cols'])
        else:
            cols = np.asarray(self.info['cols'])
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

    def get_coo_size(self):
        """
        Get the size the subjacobian would be if stored in a COO formatted jacobian.

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


SUBJAC_META_DEFAULTS = {
    'rows': None,
    'cols': None,
    'val': None,
    'dependent': True,
    'diagonal': False,
    'sparsity': None,
}
