"""Define the default Vector and Transfer classes."""
from __future__ import division
import numpy

import numbers
from six.moves import range, zip

from openmdao.vectors.vector import Vector, Transfer

real_types = tuple([numbers.Real, numpy.float32, numpy.float64])


class DefaultTransfer(Transfer):
    """Default NumPy transfer."""

    def __call__(self, ip_vec, op_vec, mode='fwd'):
        """Perform transfer.

        Args
        ----
        ip_vec : <Vector>
            pointer to the input vector.
        op_vec : <Vector>
            pointer to the output vector.
        mode : str
            'fwd' or 'rev'.

        """
        ip_inds = self._ip_inds
        op_inds = self._op_inds
        if mode == 'fwd':
            for ip_iset, op_iset in self._ip_inds:
                key = (ip_iset, op_iset)
                if len(self._ip_inds[key]) > 0:
                    ip_inds = self._ip_inds[key]
                    op_inds = self._op_inds[key]
                    tmp = op_vec._root_vector._data[op_iset][op_inds]
                    ip_vec._root_vector._data[ip_iset][ip_inds] = tmp
        elif mode == 'rev':
            for ip_iset, op_iset in self._ip_inds:
                key = (ip_iset, op_iset)
                if len(self._ip_inds[key]) > 0:
                    ip_inds = self._ip_inds[key]
                    op_inds = self._op_inds[key]
                    tmp = ip_vec._root_vector._data[ip_iset][ip_inds]
                    numpy.add.at(op_vec._root_vector._data[op_iset],
                                 op_inds, tmp)


class DefaultVector(Vector):
    """Default NumPy vector."""

    TRANSFER = DefaultTransfer

    def _create_data(self):
        """Allocate list of arrays, one for each var_set.

        Returns
        -------
        [ndarray[:], ...]
            list of zeros arrays of correct size, one for each var_set.
        """
        data = [numpy.zeros(numpy.sum(sizes[self._iproc, :]))
                for sizes in self._assembler._variable_sizes[self._typ]]
        indices = [numpy.zeros(numpy.sum(sizes[self._iproc, :]), int)
                   for sizes in self._assembler._variable_sizes[self._typ]]

        variable_indices = self._system._variable_myproc_indices[self._typ]
        variable_names = self._system._variable_myproc_names[self._typ]
        set_indices = self._assembler._variable_set_indices[self._typ]
        sizes_all = self._assembler._variable_sizes_all[self._typ]
        sizes = self._assembler._variable_sizes[self._typ]

        for ind, ivar_all in enumerate(variable_indices):
            var_name = variable_names[ind]
            ivar_set, ivar = set_indices[ivar_all, :]

            ind1 = numpy.sum(sizes[ivar_set][self._iproc, :ivar])
            ind2 = numpy.sum(sizes[ivar_set][self._iproc, :ivar + 1])
            ind1_all = numpy.sum(sizes_all[self._iproc, :ivar_all])
            ind2_all = numpy.sum(sizes_all[self._iproc, :ivar_all + 1])
            indices[ivar_set][ind1:ind2] = numpy.arange(ind1_all, ind2_all)

        return data, indices

    def _extract_data(self):
        """Extract views of arrays from root_vector.

        Returns
        -------
        [ndarray[:], ...]
            list of zeros arrays of correct size, one for each var_set.
        """
        variable_sizes = self._assembler._variable_sizes[self._typ]
        variable_set_indices = self._assembler._variable_set_indices[self._typ]

        ind1, ind2 = self._system._variable_allprocs_range[self._typ]
        sub_variable_set_indices = variable_set_indices[ind1:ind2, :]

        data = []
        indices = []
        for iset in range(len(variable_sizes)):
            bool_vector = sub_variable_set_indices[:, 0] == iset
            data_inds = sub_variable_set_indices[bool_vector, 1]
            if len(data_inds) > 0:
                sizes_array = variable_sizes[iset]
                ind1 = numpy.sum(sizes_array[self._iproc, :data_inds[0]])
                ind2 = numpy.sum(sizes_array[self._iproc, :data_inds[-1] + 1])
                data.append(self._root_vector._data[iset][ind1:ind2])
                indices.append(self._root_vector._indices[iset][ind1:ind2] -
                               ind1)
            else:
                data.append(numpy.zeros(0))
                indices.append(numpy.zeros(0, int))

        return data, indices

    def _initialize_data(self, root_vector):
        """Internally allocate vectors.

        Sets the following attributes:

        - _data

        Args
        ----
        root_vector : Vector or None
            the root's vector instance or None, if we are at the root.
        """
        if root_vector is None:
            self._data, self._indices = self._create_data()
        else:
            self._data, self._indices = self._extract_data()

    def _initialize_views(self):
        """Internally assemble views onto the vectors.

        Sets the following attributes:

        - _views
        - _views_flat
        - _idxs

        """
        variable_sizes = self._assembler._variable_sizes[self._typ]
        variable_set_indices = self._assembler._variable_set_indices[self._typ]

        system = self._system
        variable_myproc_names = system._variable_myproc_names[self._typ]
        variable_myproc_indices = system._variable_myproc_indices[self._typ]
        meta = system._variable_myproc_metadata[self._typ]

        views = {}
        views_flat = {}

        # contains a 0 index for floats or a slice(None) for arrays so getitem
        # will return either a float or a properly shaped array respectively.
        idxs = {}

        for ind, name in enumerate(variable_myproc_names):
            ivar_all = variable_myproc_indices[ind]
            iset, ivar = variable_set_indices[ivar_all, :]
            ind1 = numpy.sum(variable_sizes[iset][self._iproc, :ivar])
            ind2 = numpy.sum(variable_sizes[iset][self._iproc, :ivar + 1])
            views[name] = self._root_vector._data[iset][ind1:ind2]
            views_flat[name] = self._root_vector._data[iset][ind1:ind2]
            views[name].shape = meta[ind]['shape']
            val = meta[ind]['value']

            # The shape entry overrides value's shape, which is why we don't
            # use the shape of val as the reference
            if numpy.prod(meta[ind]['shape']) == 1:
                idxs[name] = 0
            else:
                idxs[name] = slice(None)

        self._views = self._names = views
        self._views_flat = views_flat
        self._idxs = idxs

    def _clone_data(self):
        """For each item in _data, replace it with a copy of the data."""
        for iset in range(len(self._data)):
            data = self._data[iset]
            self._data[iset] = numpy.array(data)

    def __iadd__(self, vec):
        """Perform in-place vector addition.

        Args
        ----
        vec : <Vector>
            vector to add to self.

        Returns
        -------
        <Vector>
            self + vec
        """
        for data, vec_data in zip(self._data, vec._data):
            data += vec_data
        return self

    def __isub__(self, vec):
        """Perform in-place vector substraction.

        Args
        ----
        vec : <Vector>
            vector to subtract from self.

        Returns
        -------
        <Vector>
            self - vec
        """
        for data, vec_data in zip(self._data, vec._data):
            data -= vec_data
        return self

    def __imul__(self, val):
        """Perform in-place scalar multiplication.

        Args
        ----
        val : int or float
            scalar to multiply self.

        Returns
        -------
        <Vector>
            self * val
        """
        for data in self._data:
            data *= val
        return self

    def add_scal_vec(self, val, vec):
        """Perform in-place addition of a vector times a scalar.

        Args
        ----
        val : int or float
            scalar.
        vec : <Vector>
            this vector times val is added to self.
        """
        for data, vec_data in zip(self._data, vec._data):
            data += val * vec_data

    def set_vec(self, vec):
        """Set the value of this vector to that of the incoming vector.

        Args
        ----
        vec : <Vector>
            the vector whose values self is set to.
        """
        for data, vec_data in zip(self._data, vec._data):
            data[:] = vec_data

    def set_const(self, val):
        """Set the value of this vector to a constant scalar value.

        Args
        ----
        val : int or float
            scalar to set self to.
        """
        for data in self._data:
            data[:] = val

    def get_norm(self):
        """Return the norm of this vector.

        Returns
        -------
        float
            norm of this vector.
        """
        global_sum = 0
        for data in self._data:
            global_sum += numpy.sum(data**2)
        return global_sum ** 0.5

    def scale(self, coeffs):
        """Change the scaling state.

        Args
        ----
        coeffs : int ndarray[nvar_myproc, 2]
            0th and 1st order coefficients for scaling/unscaling.
        """
        for iset, data in enumerate(self._data):
            data[:] = coeffs[self._ivar_map[iset], 0] + \
                coeffs[self._ivar_map[iset], 1] * data
