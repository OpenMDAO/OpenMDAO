"""Define the default Vector class."""
import numpy as np

from openmdao.vectors.vector import Vector, _full_slice
from openmdao.vectors.default_transfer import DefaultTransfer


class DefaultVector(Vector):
    """
    Default NumPy vector.

    Parameters
    ----------
    name : str
        The name of the vector: 'nonlinear' or 'linear'.
    kind : str
        The kind of vector, 'input', 'output', or 'residual'.
    system : <System>
        Pointer to the owning system.
    root_vector : <Vector>
        Pointer to the vector owned by the root system.
    alloc_complex : bool
        Whether to allocate any imaginary storage to perform complex step. Default is False.
    """

    TRANSFER = DefaultTransfer

    def _get_data(self):
        """
        Return either the data array or its real part.

        Note that this is intended to return only the _data array and not, for example,
        to return a combined array in the case of an input vector that shares entries with
        a connected output vector (for no-copy transfers).

        Returns
        -------
        ndarray
            The data array or its real part.
        """
        if self._under_complex_step:
            return self._data
        return self._data.real

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
        system = self._system()
        size = np.sum(system._var_sizes[self._typ][system.comm.rank, :])
        dtype = complex if self._alloc_complex else float
        return np.zeros(size, dtype=dtype)

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
        root_vec = self._root_vector

        slices = root_vec.get_slice_dict()

        mynames = list(system._var_abs2meta[type_])
        if mynames:
            myslice = slice(slices[mynames[0]].start, slices[mynames[-1]].stop)
        else:
            myslice = slice(0, 0)

        data = root_vec._data[myslice]

        scaling = None
        if self._do_scaling:
            root_scale = root_vec._scaling
            rs0 = root_scale[0]
            if rs0 is None:
                scaling = (rs0, root_scale[1][myslice])
            else:
                scaling = (rs0[myslice], root_scale[1][myslice])

        return data, scaling

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
                data = self._data
                if self._name == 'nonlinear':
                    if self._do_adder:
                        self._scaling = (np.zeros(data.size), np.ones(data.size))
                    else:
                        self._scaling = (None, np.ones(data.size))
                elif self._name == 'linear':
                    if self._has_solver_ref:
                        # We only allocate an extra scaling vector when we have output scaling
                        # somewhere in the model.
                        self._scaling = (None, np.ones(data.size))
                    else:
                        # Reuse the nonlinear scaling vecs since they're the same as ours.
                        nlvec = self._system()._root_vecs[self._kind]['nonlinear']
                        self._scaling = (None, nlvec._scaling[1])
                else:
                    self._scaling = (None, np.ones(data.size))

        else:
            self._data, self._scaling = self._extract_root_data()

    def _initialize_views(self):
        """
        Internally assemble views onto the vectors.

        Sets the following attributes:
        _views
        _views_flat
        """
        system = self._system()
        io = self._typ
        kind = self._kind

        do_scaling = self._do_scaling
        if do_scaling:
            factors = system._scale_factors
            scaling = self._scaling

        self._views = views = {}
        self._views_flat = views_flat = {}

        start = end = 0
        for abs_name, meta in system._var_abs2meta[io].items():
            end = start + meta['size']
            shape = meta['shape']
            views_flat[abs_name] = v = self._data[start:end]
            if shape != v.shape:
                v = v.view()
                v.shape = shape
            views[abs_name] = v

            if do_scaling:
                factor_tuple = factors[abs_name][kind]

                if len(factor_tuple) == 4:
                    # Input vector unit conversion.
                    a0, a1, factor, offset = factor_tuple

                    if self._name == 'linear':
                        scale0 = None
                        scale1 = a1 / factor

                    else:
                        scale0 = (a0 + offset) * factor
                        scale1 = a1 * factor
                else:
                    scale0, scale1 = factor_tuple

                if scaling[0] is not None:
                    scaling[0][start:end] = scale0
                scaling[1][start:end] = scale1

            start = end

        self._names = frozenset(views)
        self._len = end

    def _in_matvec_context(self):
        """
        Return True if this vector is inside of a matvec_context.

        Returns
        -------
        bool
            Whether or not this vector is in a matvec_context.
        """
        return len(self._names) != len(self._views)

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
            data = self.asarray()
            data += vec
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
            data = self.asarray()
            data -= vec
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
            data = self.asarray()
            data *= vec
        return self

    def add_scal_vec(self, val, vec):
        """
        Perform in-place addition of a vector times a scalar.

        Parameters
        ----------
        val : int or float
            Scalar.
        vec : <Vector>
            This vector times val is added to self.
        """
        data = self.asarray()
        data += (val * vec.asarray())

    def set_vec(self, vec):
        """
        Set the value of this vector to that of the incoming vector.

        Parameters
        ----------
        vec : <Vector>
            The vector whose values self is set to.
        """
        self.set_val(vec.asarray())

    def set_val(self, val, idxs=_full_slice):
        """
        Set the data array of this vector to a value, with optional indexing.

        Parameters
        ----------
        val : float or ndarray
            Scalar or array to set data array to.
        idxs : int or slice or tuple of ints and/or slices
            The locations where the data array should be updated.
        """
        # we use _data here specifically so that imaginary part
        # will get properly reset, e.g. when the array is zeroed out.
        self._data[idxs] = val

    def scale_to_norm(self, mode='fwd'):
        """
        Scale this vector to normalized form.

        Parameters
        ----------
        mode : str
            Derivative direction.
        """
        if self._has_solver_ref and mode == 'fwd':
            scaler = self._scaling_nl_vec[1]
            adder = None
        else:
            adder, scaler = self._scaling

        if mode == 'rev' and not self._has_solver_ref:
            self._scale_reverse(scaler, adder)
        else:
            self._scale_forward(scaler, adder)

    def scale_to_phys(self, mode='fwd'):
        """
        Scale this vector to physical form.

        Parameters
        ----------
        mode : str
            Derivative direction.
        """
        if self._has_solver_ref and mode == 'fwd':
            scaler = self._scaling_nl_vec[1]
            adder = None
        else:
            adder, scaler = self._scaling

        if mode == 'rev' and not self._has_solver_ref:
            self._scale_forward(scaler, adder)
        else:
            self._scale_reverse(scaler, adder)

    def _scale_forward(self, scaler, adder):
        """
        Scale this vector by subtracting the adder and dividing by the scaler.

        Parameters
        ----------
        scaler : darray
            Vector of multiplicative scaling factors.
        adder : darray
            Vector of additive scaling factors.
        """
        data = self.asarray()
        if adder is not None:  # nonlinear only
            data -= adder
        data /= scaler

    def _scale_reverse(self, scaler, adder):
        """
        Scale this vector by multiplying by the scaler ahd adding the adder.

        Parameters
        ----------
        scaler : darray
            Vector of multiplicative scaling factors.
        adder : darray
            Vector of additive scaling factors.
        """
        data = self.asarray()
        data *= scaler
        if adder is not None:  # nonlinear only
            data += adder

    def asarray(self, copy=False):
        """
        Return an array representation of this vector.

        If copy is True, return a copy.  Otherwise, try to avoid it.

        Parameters
        ----------
        copy : bool
            If True, return a copy of the array.

        Returns
        -------
        ndarray
            Array representation of this vector.
        """
        if self._under_complex_step:
            arr = self._data
        else:
            arr = self._data.real

        if copy:
            return arr.copy()

        return arr

    def iscomplex(self):
        """
        Return True if this vector contains complex values.

        This checks the type of the values, not whether they have a nonzero imaginary part.

        Returns
        -------
        bool
            True if this vector contains complex values.
        """
        return np.iscomplexobj(self._get_data())

    def iadd(self, val, idxs=_full_slice):
        """
        Add the value to the data array at the specified indices or slice(s).

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices
            The locations where the data array should be updated.
        """
        data = self.asarray()
        data[idxs] += val

    def isub(self, val, idxs=_full_slice):
        """
        Subtract the value from the data array at the specified indices or slice(s).

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices
            The locations where the data array should be updated.
        """
        data = self.asarray()
        data[idxs] -= val

    def imul(self, val, idxs=_full_slice):
        """
        Multiply the value to the data array at the specified indices or slice(s).

        Parameters
        ----------
        val : ndarray
            Value to set into the data array.
        idxs : int or slice or tuple of ints and/or slices
            The locations where the data array should be updated.
        """
        data = self.asarray()
        data[idxs] *= val

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
        return np.dot(self.asarray(), vec.asarray())

    def get_norm(self):
        """
        Return the norm of this vector.

        Returns
        -------
        float
            Norm of this vector.
        """
        return np.linalg.norm(self.asarray())

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
            for name, arr in self._views_flat.items():
                end += arr.size
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
        if '_system' in state:
            del state['_system']
        return state
