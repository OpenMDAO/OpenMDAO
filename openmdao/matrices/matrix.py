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
    _masked_arr_caches : dict
        Dictionary of cached masked arrays for each mode.
    dtype : dtype
        The dtype of the matrix.
    """

    def __init__(self, submats):
        """
        Initialize all attributes.
        """
        self._matrix = None
        self._submats = submats
        self._masked_arr_caches = {'fwd': None, 'rev': None}
        self.dtype = np.dtype(object)  # placeholder for dtype that's not float or complex

    def _pre_update(self, dtype):
        """
        Do anything that needs to be done at the beginning of AssembledJacobian._update.
        """
        self._update_dtype(dtype)

    def _post_update(self):
        """
        Do anything that needs to be done at the end of AssembledJacobian._update.
        """
        pass

    def _get_masked_arr(self, in_arr, mode, mask):
        """
        Get a masked array for the given array and mode.

        The original array is not modified.

        Parameters
        ----------
        in_arr : ndarray
            incoming array to mask.
        mode : str
            'fwd' or 'rev'.
        mask : ndarray or None
            Array used to mask out part of the array.  If mask is not None then
            the masked values will be set to 0.0.

        Returns
        -------
        ndarray
            masked array.
        """
        if mask is not None:
            mask_arr = self._masked_arr_caches[mode]
            if mask_arr is None or mask_arr.dtype != in_arr.dtype:
                # copy the array because we don't want to modify the original
                mask_arr = in_arr.copy()
                self._masked_arr_caches[mode] = mask_arr
            else:
                mask_arr[:] = in_arr

            mask_arr[mask] = 0.0

            return mask_arr

        return in_arr
