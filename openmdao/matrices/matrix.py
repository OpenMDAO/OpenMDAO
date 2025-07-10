"""Define the base Matrix class."""

import numpy as np


class Matrix(object):
    """
    Base matrix class.

    This class is used for global Jacobians.

    Parameters
    ----------
    submats : dict
        Dictionary of sub-matrix data keyed by (row_name, col_name).

    Attributes
    ----------
    _matrix : object
        implementation-specific representation of the actual matrix.
    _submats : dict
        dictionary of sub-matrix data keyed by (out_name, in_name).
    dtype : dtype
        The dtype of the matrix.
    """

    def __init__(self, submats):
        """
        Initialize all attributes.
        """
        self._matrix = None
        self._submats = submats
        self.dtype = np.dtype(object)  # placeholder for dtype that's not float or complex

    def _pre_update(self, dtype):
        """
        Do anything that needs to be done at the beginning of jacobian _update.
        """
        self._update_dtype(dtype)

    def _post_update(self):
        """
        Do anything that needs to be done at the end of jacobian _update.
        """
        pass

    def _get_masked_arr(self, in_arr, mask):
        """
        Get a masked array for the given array and mode.

        The original array is not modified.

        Parameters
        ----------
        in_arr : ndarray
            incoming array to mask.
        mask : ndarray or None
            Index array or slice used to mask out part of the array.  If mask is not None then
            the masked values will be set to 0.0.

        Returns
        -------
        ndarray
            masked array.
        """
        if mask is None:
            return in_arr
        mask_arr = in_arr.copy()
        mask_arr[mask] = 0.0
        return mask_arr
