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

    def _initialize_transfer(self):
        """
        Set up the transfer; do any necessary pre-computation.

        Optionally implemented by the subclass.
        """
        in_inds = self._in_inds
        out_inds = self._out_inds

        # filter out any empty transfers
        outs = {}
        ins = {}
        for key in in_inds:
            if len(in_inds[key]) > 0:
                ins[key] = in_inds[key]
                outs[key] = out_inds[key]

        self._in_inds = ins
        self._out_inds = outs

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

        if mode == 'fwd':
            for key in in_inds:
                in_set_name, out_set_name = key
                for i in range(in_vec._ncol):
                    in_vec._data[in_set_name][in_inds[key], i] = \
                        out_vec._data[out_set_name][:, i][out_inds[key]]

                # Imaginary transfer
                # (for CS, so only need in fwd)
                if in_vec._vector_info._under_complex_step and out_vec._alloc_complex:
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

    def _create_data(self):
        """
        Allocate list of arrays, one for each var_set.

        Returns
        -------
        [ndarray[:], ...]
            list of zeros arrays of correct size, one for each var_set.
        """
        system = self._system
        type_ = self._typ
        iproc = self._iproc
        ncol = self._ncol

        sizes_byset_t = system._var_sizes_byset[type_]

        data = {}
        indices = {}
        for set_name in system._var_set2iset[type_]:
            size = np.sum(sizes_byset_t[set_name][iproc, :])
            data[set_name] = np.zeros((size, ncol)) if ncol > 1 else np.zeros(size)
            indices[set_name] = np.zeros(size, int)

        sizes_t = system._var_sizes[type_]
        abs2meta_t = system._var_abs2meta[type_]
        allprocs_abs2idx_byset_t = system._var_allprocs_abs2idx_byset[type_]
        allprocs_abs2idx_t = system._var_allprocs_abs2idx[type_]
        for abs_name in system._var_abs_names[type_]:
            set_name = abs2meta_t[abs_name]['var_set']

            idx_byset = allprocs_abs2idx_byset_t[abs_name]
            ind_byset1 = np.sum(sizes_byset_t[set_name][iproc, :idx_byset])
            ind_byset2 = np.sum(sizes_byset_t[set_name][iproc, :idx_byset + 1])

            idx = allprocs_abs2idx_t[abs_name]
            ind1 = np.sum(sizes_t[iproc, :idx])
            ind2 = np.sum(sizes_t[iproc, :idx + 1])
            indices[set_name][ind_byset1:ind_byset2] = np.arange(ind1, ind2)

        return data, indices

    def _update_root_data(self):
        """
        Resize the root data if necesary (i.e., due to reconfiguration).
        """
        system = self._system
        type_ = self._typ
        iproc = self._iproc
        root_vec = self._root_vector

        _, tmp_indices = self._create_data()

        ext_sizes_t = system._ext_sizes[type_]
        int_sizes_t = np.sum(system._var_sizes[type_][iproc, :])
        old_sizes_total = np.sum([len(data) for data in itervalues(root_vec._data)])

        old_sizes = (
            ext_sizes_t[0],
            old_sizes_total - ext_sizes_t[0] - ext_sizes_t[1],
            ext_sizes_t[1],
        )
        new_sizes = (
            ext_sizes_t[0],
            int_sizes_t,
            ext_sizes_t[1],
        )

        for set_name in system._var_set2iset[type_]:
            ext_sizes_byset_t = system._ext_sizes_byset[type_][set_name]
            int_sizes_byset_t = np.sum(system._var_sizes_byset[type_][set_name][iproc, :])
            old_sizes_total_byset = len(root_vec._data[set_name])

            old_sizes_byset = (
                ext_sizes_byset_t[0],
                old_sizes_total_byset - ext_sizes_byset_t[0] - ext_sizes_byset_t[1],
                ext_sizes_byset_t[1],
            )
            new_sizes_byset = (
                ext_sizes_byset_t[0],
                int_sizes_byset_t,
                ext_sizes_byset_t[1],
            )

            root_vec._data[set_name] = np.concatenate([
                root_vec._data[set_name][:old_sizes_byset[0]],
                np.zeros(new_sizes_byset[1]),
                root_vec._data[set_name][old_sizes_byset[0] + old_sizes_byset[1]:],
            ])

            root_vec._indices[set_name] = np.concatenate([
                root_vec._indices[set_name][:old_sizes_byset[0]],
                tmp_indices[set_name] + new_sizes[0],
                root_vec._indices[set_name][old_sizes_byset[0] + old_sizes_byset[1]:]
                    + new_sizes[1] - old_sizes[1],
            ])

        root_vec._initialize_views()

    def _extract_data(self):
        """
        Extract views of arrays from root_vector.

        Returns
        -------
        [ndarray[:], ...]
            list of zeros arrays of correct size, one for each var_set.
        """
        system = self._system
        type_ = self._typ
        iproc = self._iproc
        root_vec = self._root_vector
        ncol = self._ncol

        offset = system._ext_sizes[type_][0]

        data = {}
        imag_data = {}
        indices = {}
        for set_name in system._var_set2iset[type_]:
            offset_byset = system._ext_sizes_byset[type_][set_name][0]
            ind_byset1 = offset_byset
            ind_byset2 = offset_byset + np.sum(system._var_sizes_byset[type_][set_name][iproc, :])

            data[set_name] = root_vec._data[set_name][ind_byset1:ind_byset2]
            indices[set_name] = root_vec._indices[set_name][ind_byset1:ind_byset2] - offset

            # Extract view for imaginary part too
            if self._alloc_complex:
                if root_vec._alloc_complex:
                    imag_data[set_name] = root_vec._imag_data[set_name][ind_byset1:ind_byset2]
                else:
                    shape = root_vec._data[set_name][ind_byset1:ind_byset2].shape
                    imag_data[set_name] = np.zeros(shape)

        return data, imag_data, indices

    def _initialize_data(self, root_vector):
        """
        Internally allocate vectors.

        Sets the following attributes:
        _data
        _indices

        Parameters
        ----------
        root_vector : Vector or None
            the root's vector instance or None, if we are at the root.
        """
        if root_vector is None:
            self._data, self._indices = self._create_data()

            # Allocate imaginary for complex step
            if self._alloc_complex:
                self._imag_data = deepcopy(self._data)

        else:
            self._data, self._imag_data, self._indices = self._extract_data()

    def _initialize_views(self):
        """
        Internally assemble views onto the vectors.

        Sets the following attributes:
        _views
        _views_flat
        _idxs
        """
        system = self._system
        type_ = self._typ
        iproc = self._iproc
        ncol = self._ncol

        views = {}
        views_flat = {}

        alloc_complex = self._alloc_complex
        if alloc_complex:
            imag_views = {}
            imag_views_flat = {}

        allprocs_abs2idx_byset_t = system._var_allprocs_abs2idx_byset[type_]
        sizes_byset_t = system._var_sizes_byset[type_]
        abs2meta_t = system._var_abs2meta[type_]
        for abs_name in system._var_abs_names[type_]:
            idx_byset = allprocs_abs2idx_byset_t[abs_name]
            set_name = abs2meta_t[abs_name]['var_set']

            ind_byset1 = np.sum(sizes_byset_t[set_name][iproc, :idx_byset])
            ind_byset2 = np.sum(sizes_byset_t[set_name][iproc, :idx_byset + 1])
            shape = abs2meta_t[abs_name]['shape']
            if not isinstance(shape, tuple):
                shape = (shape,)
            shape = tuple(list(shape) + [ncol])

            views_flat[abs_name] = v = self._data[set_name][ind_byset1:ind_byset2]
            if shape != v.shape:
                v = v.view()
                v.shape = shape
            views[abs_name] = v

            if alloc_complex:
                imag_views_flat[abs_name] = v = self._imag_data[set_name][ind_byset1:ind_byset2]
                if shape != v.shape:
                    v = v.view()
                    v.shape = shape
                imag_views[abs_name] = v

        self._views = self._names = views
        self._views_flat = views_flat

        if self._alloc_complex:
            self._imag_views = imag_views
            self._imag_views_flat = imag_views_flat

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
