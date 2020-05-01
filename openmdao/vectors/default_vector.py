"""Define the default Vector class."""
from copy import deepcopy
import numbers

import numpy as np

from openmdao.vectors.vector import Vector, INT_DTYPE, _full_slice
from openmdao.vectors.default_transfer import DefaultTransfer
from openmdao.utils.mpi import MPI, multi_proc_exception_check


class DefaultVector(Vector):
    """
    Default NumPy vector.
    """

    TRANSFER = DefaultTransfer

    def _create_data(self):
        """
        Allocate data array.

        This happens only in the top level system.  Child systems use views of the array
        we allocate here.

        Returns
        -------
        ndarray
            zeros array of correct size to hold all of this vector's variables.
        """
        ncol = self._ncol
        size = np.sum(self._system()._var_sizes[self._name][self._typ][self._iproc, :])
        return np.zeros(size) if ncol == 1 else np.zeros((size, ncol))

    def _update_root_data(self):
        """
        Resize the root data if necesary (i.e., due to reconfiguration).
        """
        system = self._system()
        type_ = self._typ
        vec_name = self._name
        root_vec = self._root_vector

        sys_offset, size_after_sys = system._ext_sizes[vec_name][type_]
        sys_size = np.sum(system._var_sizes[vec_name][type_][self._iproc, :])
        old_sizes_total = root_vec._data.size

        root_vec._data = np.concatenate([
            root_vec._data[:sys_offset],
            np.zeros(sys_size),
            root_vec._data[old_sizes_total - size_after_sys:],
        ])

        if self._alloc_complex and root_vec._cplx_data.size != root_vec._data.size:
            root_vec._cplx_data = np.zeros(root_vec._data.size, dtype=complex)

        root_vec._initialize_views()

    def _extract_root_data(self):
        """
        Extract views of arrays from root_vector.

        Returns
        -------
        ndarray
            zeros array of correct size.
        """
        system = self._system()
        root_vec = self._root_vector

        cplx_data = None
        scaling = {}
        if self._do_scaling:
            scaling['phys'] = {}
            scaling['norm'] = {}

        sizes = system._var_sizes[self._name][self._typ]
        ind1 = system._ext_sizes[self._name][self._typ][0]
        ind2 = ind1 + np.sum(sizes[self._iproc, :])

        data = root_vec._data[ind1:ind2]

        # Extract view for complex storage too.
        if self._alloc_complex:
            cplx_data = root_vec._cplx_data[ind1:ind2]

        if self._do_scaling:
            for typ in ('phys', 'norm'):
                root_scale = root_vec._scaling[typ]
                rs0 = root_scale[0]
                if rs0 is None:
                    scaling[typ] = (rs0, root_scale[1][ind1:ind2])
                else:
                    scaling[typ] = (rs0[ind1:ind2], root_scale[1][ind1:ind2])

        return data, cplx_data, scaling

    def _initialize_data(self, root_vector):
        """
        Internally allocate data array.

        Parameters
        ----------
        root_vector : Vector or None
            the root's vector instance or None, if we are at the root.
        """
        if root_vector is None:  # we're the root
            self._data = self._create_data()

            if self._do_scaling:
                self._scaling = {}
                data = self._data
                if self._name == 'nonlinear':
                    self._scaling['phys'] = (np.zeros(data.size), np.ones(data.size))
                    self._scaling['norm'] = (np.zeros(data.size), np.ones(data.size))
                elif self._name == 'linear':
                    # reuse the nonlinear scaling vecs since they're the same as ours
                    nlvec = self._system()._root_vecs[self._kind]['nonlinear']
                    self._scaling['phys'] = (None, nlvec._scaling['phys'][1])
                    self._scaling['norm'] = (None, nlvec._scaling['norm'][1])
                else:
                    self._scaling['phys'] = (None, np.ones(data.size))
                    self._scaling['norm'] = (None, np.ones(data.size))

            # Allocate imaginary for complex step
            if self._alloc_complex:
                self._cplx_data = np.zeros(self._data.shape, dtype=np.complex)

        else:
            self._data, self._cplx_data, self._scaling = self._extract_root_data()

    def _initialize_views(self):
        """
        Internally assemble views onto the vectors.

        Sets the following attributes:
        _views
        _views_flat
        """
        system = self._system()
        type_ = self._typ
        kind = self._kind
        ncol = self._ncol

        do_scaling = self._do_scaling
        if do_scaling:
            factors = system._scale_factors
            scaling = self._scaling

        self._views = views = {}
        self._views_flat = views_flat = {}

        alloc_complex = self._alloc_complex
        self._cplx_views = cplx_views = {}
        self._cplx_views_flat = cplx_views_flat = {}

        abs2meta = system._var_abs2meta
        start = end = 0
        for abs_name in system._var_relevant_names[self._name][type_]:
            meta = abs2meta[abs_name]
            end = start + meta['size']
            shape = meta['shape']
            if ncol > 1:
                if not isinstance(shape, tuple):
                    shape = (shape,)
                shape = tuple(list(shape) + [ncol])

            views_flat[abs_name] = v = self._data[start:end]
            if shape != v.shape:
                v = v.view()
                v.shape = shape
            views[abs_name] = v

            if alloc_complex:
                cplx_views_flat[abs_name] = v = self._cplx_data[start:end]
                if shape != v.shape:
                    v = v.view()
                    v.shape = shape
                cplx_views[abs_name] = v

            if do_scaling:
                for scaleto in ('phys', 'norm'):
                    scale0, scale1 = factors[abs_name][kind, scaleto]
                    vec = scaling[scaleto]
                    if vec[0] is not None:
                        vec[0][start:end] = scale0
                    vec[1][start:end] = scale1

            start = end

        self._names = frozenset(views)
        self._len = end

    def __iadd__(self, vec):
        """
        Perform in-place vector addition.

        Parameters
        ----------
        vec : <Vector>
            vector to add to self.

        Returns
        -------
        <Vector>
            self + vec
        """
        if isinstance(vec, Vector):
            self.iadd(vec.asarray())
        else:
            self._data += vec
        return self

    def __isub__(self, vec):
        """
        Perform in-place vector substraction.

        Parameters
        ----------
        vec : <Vector>
            vector to subtract from self.

        Returns
        -------
        <Vector>
            self - vec
        """
        if isinstance(vec, Vector):
            self.isub(vec.asarray())
        else:
            self._data -= vec
        return self

    def __imul__(self, vec):
        """
        Perform in-place multiplication.

        Parameters
        ----------
        vec : Vector, int, float or ndarray
            Value to multiply self.

        Returns
        -------
        <Vector>
            self * vec
        """
        if isinstance(vec, Vector):
            self.imul(vec.asarray())
        else:
            self._data *= vec
        return self

    def add_scal_vec(self, val, vec):
        """
        Perform in-place addition of a vector times a scalar.

        Parameters
        ----------
        val : int or float
            scalar.
        vec : <Vector>
            this vector times val is added to self.
        """
        self._data += (val * vec.asarray())

    def set_vec(self, vec):
        """
        Set the value of this vector to that of the incoming vector.

        Parameters
        ----------
        vec : <Vector>
            the vector whose values self is set to.
        """
        self._data[:] = vec.asarray()

    def set_val(self, val, idxs=_full_slice):
        """
        Set the data array of this vector to a value, with optional indexing.

        Parameters
        ----------
        val : float or ndarray
            scalar or array to set data array to.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        self._data[idxs] = val

    def asarray(self, idxs=_full_slice):
        """
        Return parts of the data array at the specified indices or slice(s).

        Parameters
        ----------
        idxs : int or slice or tuple of ints and/or slices.
            The locations to pull from the data array.

        Returns
        -------
        ndarray
            Array of values.
        """
        return self._data[idxs]

    def iadd(self, val, idxs=_full_slice):
        """
        Add the value to the data array at the specified indices or slice(s).

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        self._data[idxs] += val

    def isub(self, val, idxs=_full_slice):
        """
        Subtract the value from the data array at the specified indices or slice(s).

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        self._data[idxs] -= val

    def imul(self, val, idxs=_full_slice):
        """
        Multiply the value to the data array at the specified indices or slice(s).

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices.
            The locations where the data array should be updated.
        """
        self._data[idxs] *= val

    def dot(self, vec):
        """
        Compute the dot product of the real parts of the current vec and the incoming vec.

        Parameters
        ----------
        vec : <Vector>
            The incoming vector being dotted with self.

        Returns
        -------
        float
            The computed dot product value.
        """
        return np.dot(self._data, vec.asarray())

    def get_norm(self):
        """
        Return the norm of this vector.

        Returns
        -------
        float
            norm of this vector.
        """
        return np.linalg.norm(self._data)

    def get_slice_dict(self):
        """
        Return a dict of var names mapped to their slice in the local data array.

        Returns
        -------
        dict
            Mapping of var name to slice.
        """
        if self._slices is None:
            slices = {}
            start = end = 0
            for name in self._system()._var_relevant_names[self._name][self._typ]:
                end += self._views_flat[name].size
                slices[name] = slice(start, end)
                start = end
            self._slices = slices

        return self._slices

    def __getstate__(self):
        """
        Return state as a dict.

        For pickling vectors in case recording, we want to get rid of
        the system contained within Vectors, because MPI Comm objects cannot
        be pickled using Python3's pickle module.

        Returns
        -------
        dict
            state minus system member.
        """
        state = self.__dict__.copy()
        del state['_system']
        return state
