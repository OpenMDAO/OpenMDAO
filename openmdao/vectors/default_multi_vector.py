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
    Default NumPy vector with multiple columns.

    Attributes
    ----------
    _icol : int or None
        Specifies the 'active' column for multi-vectors.
    __save_getitem : method
        Where the base class version of __getitem__ is saved
    __save_setitem : method
        Where the base class version of __setitem__ is saved
    """

    TRANSFER = DefaultMultiTransfer

    def __init__(self, name, typ, system, root_vector=None, resize=False, alloc_complex=False,
                 ncol=1):
        """
        Initialize all attributes.

        Parameters
        ----------
        name : str
            The name of the vector: 'nonlinear', 'linear', or right-hand side name.
        typ : str
            Type: 'input' for input vectors; 'output' for output/residual vectors.
        system : <System>
            Pointer to the owning system.
        root_vector : <Vector>
            Pointer to the vector owned by the root system.
        resize : bool
            If true, resize the root vector.
        alloc_complex : bool
            Whether to allocate any imaginary storage to perform complex step. Default is False.
        ncol : int
            Number of columns for multi-vectors.
        """
        super(DefaultMultiVector, self).__init__(name, typ, system, root_vector,
                                                 resize, alloc_complex, ncol)
        self._icol = None
        self.__save_getitem = self.__getitem__
        self.__save_setitem = self.__setitem__

    def _use_column(self, icol):
        if icol is None:
            # restore getitem/setitem to their original versions
            self.__getitem__ = self.__save_getitem
            self.__setitem__ = self.__save_setitem
        else:
            # change to column-wise versions of getitem/setitem
            self.__getitem__ = self.__col_getitem
            self.__setitem__ = self.__col_setitem

        self._icol = icol

    def __col_getitem(self, name):
        """
        Same as __getitem__ except it only returns the value for the 'current' column.

        Parameters
        ----------
        name : str
            The name of the variable being accessed.
        """
        return self.__save_getitem(name)[:, self._icol]

    def __col_setitem(self, name, value):
        """
        Set the unscaled variable value in true units.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.
        value : float or list or tuple or ndarray
            variable value to set (not scaled, not dimensionless)
        """
        abs_name = name2abs_name(self._system, name, self._names, self._typ)
        if abs_name is not None:
            value, shape = ensure_compatible(name, value, self._views[abs_name].shape)
            if self._vector_info._under_complex_step:

                # setitem overwrites anything you may have done with numpy indexing
                try:
                    del self._complex_view_cache[abs_name]
                except KeyError:
                    pass

                self._views[abs_name][:, self._icol] = value.real
                self._imag_views[abs_name][:, self._icol] = value.imag
            else:
                self._views[abs_name][:, self._icol] = value
        else:
            msg = 'Variable name "{}" not found.'
            raise KeyError(msg.format(name))

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
