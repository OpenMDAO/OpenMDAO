"""Define the default Vector and Transfer classes."""
from __future__ import division
from copy import deepcopy
import numbers

from six import iteritems, itervalues
from six.moves import range, zip

import numpy as np

from openmdao.vectors.default_vector import DefaultVector, DefaultTransfer

real_types = tuple([numbers.Real, np.float32, np.float64])


class DefaultMultiTransfer(DefaultTransfer):
    """
    Default NumPy transfer.
    """

    def __call__(self, in_vec, out_vec, mode='fwd'):
        """
        Perform transfer.

        Parameters
        ----------
        in_vec : <Vector>
            pointer to the input vector.
        out_vec : <Vector>
            pointer to the output vector.
        mode : str
            'fwd' or 'rev'.

        """
        in_inds = self._in_inds
        out_inds = self._out_inds
        using_complex_step = in_vec._vector_info._under_complex_step and out_vec._alloc_complex

        if mode == 'fwd':
            for key in in_inds:
                in_set_name, out_set_name = key
                for i in range(in_vec._ncol):
                    in_vec._data[in_set_name][in_inds[key], i] = \
                        out_vec._data[out_set_name][:, i][out_inds[key]]

                # Imaginary transfer
                # (for CS, so only need in fwd)
                if using_complex_step:
                    in_vec._imag_data[in_set_name][in_inds[key], i] = \
                        out_vec._imag_data[out_set_name][out_inds[key]]

        elif mode == 'rev':
            for key in in_inds:
                in_set_name, out_set_name = key
                for i in range(out_vec._ncol):
                    np.add.at(
                        out_vec._data[out_set_name][:, i], out_inds[key],
                        in_vec._data[in_set_name][:, i][in_inds[key]])


class DefaultMultiVector(DefaultVector):
    """
    Default NumPy vector.
    """

    TRANSFER = DefaultMultiTransfer

    def get_data(self, new_array=None):
        """
        Get the array combining the data of all the varsets.

        Parameters
        ----------
        new_array : ndarray or None
            Array to fill in with the values; otherwise new array created.

        Returns
        -------
        ndarray
            Array combining the data of all the varsets.
        """
        if new_array is None:
            new_array = np.zeros((self._length, self._ncol))

        for set_name, data in iteritems(self._data):
            for i in range(self._ncol):
                new_array[:, i][self._indices[set_name]] = data[:, i]

        return new_array

    def elem_mult(self, vec):
        """
        Perform element-wise multiplication and store the result in this vector.

        Parameters
        ----------
        vec : <Vector>
            The vector to perform element-wise multiplication with.
        """
        for set_name, data in iteritems(self._data):
            for i in range(data.shape[1]):
                data[:, i] *= vec._data[set_name]
