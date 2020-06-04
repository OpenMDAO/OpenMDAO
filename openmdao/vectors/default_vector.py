"""Define the default Vector class."""
from copy import deepcopy
import numbers

import numpy as np

from openmdao.vectors.vector import Vector, INT_DTYPE
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
        type_ = self._typ
        iproc = self._iproc
        root_vec = self._root_vector

        cplx_data = None
        scaling = {}
        if self._do_scaling:
            scaling['phys'] = {}
            scaling['norm'] = {}

        sizes = system._var_sizes[self._name][type_]
        ind1 = system._ext_sizes[self._name][type_][0]
        ind2 = ind1 + np.sum(sizes[iproc, :])

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
        iproc = self._iproc
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

        allprocs_abs2idx_t = system._var_allprocs_abs2idx[self._name]
        sizes_t = system._var_sizes[self._name][type_]
        offs = system._get_var_offsets()[self._name][type_]
        if offs.size > 0:
            offs = offs[iproc].copy()
            # turn global offset into local offset
            start = offs[0]
            offs -= start
        else:
            offs = offs[0].copy()
        offsets_t = offs

        abs2meta = system._var_abs2meta
        for abs_name in system._var_relevant_names[self._name][type_]:
            idx = allprocs_abs2idx_t[abs_name]

            ind1 = offsets_t[idx]
            ind2 = ind1 + sizes_t[iproc, idx]
            shape = abs2meta[abs_name]['shape']
            if ncol > 1:
                if not isinstance(shape, tuple):
                    shape = (shape,)
                shape = tuple(list(shape) + [ncol])

            views_flat[abs_name] = v = self._data[ind1:ind2]
            if shape != v.shape:
                v = v.view()
                v.shape = shape
            views[abs_name] = v

            if alloc_complex:
                cplx_views_flat[abs_name] = v = self._cplx_data[ind1:ind2]
                if shape != v.shape:
                    v = v.view()
                    v.shape = shape
                cplx_views[abs_name] = v

            if do_scaling:
                for scaleto in ('phys', 'norm'):
                    scale0, scale1 = factors[abs_name][kind, scaleto]
                    vec = scaling[scaleto]
                    if vec[0] is not None:
                        vec[0][ind1:ind2] = scale0
                    vec[1][ind1:ind2] = scale1

        self._names = frozenset(views)

    def _clone_data(self):
        """
        For each item in _data, replace it with a copy of the data.
        """
        self._data = self._data.copy()

        if self._under_complex_step and self._cplx_data is not None:
            self._cplx_data = self._cplx_data.copy()

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
        self._data += vec._data
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
        self._data -= vec._data
        return self

    def __imul__(self, val):
        """
        Perform in-place scalar multiplication.

        Parameters
        ----------
        val : int or float
            scalar to multiply self.

        Returns
        -------
        <Vector>
            self * val
        """
        self._data *= val
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
        self._data += val * vec._data

    def set_vec(self, vec):
        """
        Set the value of this vector to that of the incoming vector.

        Parameters
        ----------
        vec : <Vector>
            the vector whose values self is set to.
        """
        self._data[:] = vec._data

    def set_const(self, val):
        """
        Set the value of this vector to a constant scalar value.

        Parameters
        ----------
        val : int or float
            scalar to set self to.
        """
        self._data[:] = val

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
        return np.dot(self._data, vec._data)

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
