"""Define the default Vector class."""
import hashlib
import numpy as np

from openmdao.vectors.vector import Vector, _VecData, _full_slice
from openmdao.vectors.default_transfer import DefaultTransfer
from openmdao.utils.array_utils import array_hash, shape_to_len


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
    parent_vector : <Vector>
        Parent vector.
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
        return self._data if self._under_complex_step else self._data.real

    def _initialize_data(self, parent_vector, system):
        """
        Set up internal data structures.

        Parameters
        ----------
        parent_vector : <Vector> or None
            Parent vector.
        system : <System>
            The owning system.
        """
        self._views = views = {}
        start = end = 0
        for name, shape in system._name_shape_iter(self._iotype):
            end += shape_to_len(shape)
            views[name] = _VecData(shape, (start, end))
            start = end

        if parent_vector is None:  # this is a root vector
            self._parent_slice = slice(0, end)
            self._data = np.zeros(end, dtype=complex if self._alloc_complex else float)
        else:
            for name in views:
                # just get our first name to get the starting index in the parent vector
                start = parent_vector._views[name].range[0]
                self._parent_slice = slice(start, start + end)
                self._data = parent_vector._data[self._parent_slice]
                break
            else:
                self._parent_slice = slice(0, 0)
                self._data = np.zeros(0, dtype=complex if self._alloc_complex else float)

            if parent_vector._scaling:
                parent_scaler, parent_adder = parent_vector._scaling
                if parent_adder is not None:
                    parent_adder = parent_adder[self._parent_slice]
                self._scaling = (parent_scaler[self._parent_slice], parent_adder)

        data = self._data
        for vinfo in views.values():
            vinfo.set_view(data)

        if self._name == 'linear' and self._kind in ('input', 'output'):
            self._names = frozenset(views)
        else:
            self._names = views

    def _set_scaling(self, system, do_adder, nlvec=None):
        """
        Set the scaling vectors.

        Parameters
        ----------
        system : <System>
            The system to set the scaling for.
        do_adder : bool
            Whether to initialize with an additive term.
        nlvec : <Vector> or None
            Nonlinear vector if this is a linear vector.
        """
        kind = self._kind
        islinear = self._name == 'linear'
        isinput = kind == 'input'
        factors = system._scale_factors

        # If we define 'ref' on an output, then we will need to allocate a separate scaling ndarray
        # for the linear and nonlinear input vectors.
        self._has_solver_ref = system._has_output_scaling and isinput and islinear
        self._nlvec = nlvec

        # if root, allocate space for scaling vectors
        if self._isroot:
            self._allocate_scaling_data(do_adder, nlvec)

        scaler_array, adder_array = self._scaling

        for abs_name, vinfo in self._views.items():
            if abs_name in factors:
                factor = factors[abs_name]
                if kind in factor:
                    a0, a1, factor, offset = factor[kind]

                    if factor is not None:
                        # Linear input vectors need to be able to handle the unit and solver scaling
                        # in opposite directions in reverse mode.

                        if islinear:
                            scale0 = None
                            scale1 = factor / a1
                        else:
                            scale0 = (a0 + offset) * factor
                            scale1 = a1 * factor
                    else:
                        if islinear and isinput:
                            scale0 = None
                            scale1 = 1.0 / a1
                        else:
                            scale0 = a0
                            scale1 = a1

                    start, end = vinfo.range
                    if adder_array is not None:
                        adder_array[start:end] = scale0
                    scaler_array[start:end] = scale1

    def _allocate_scaling_data(self, do_adder, nlvec):
        """
        Allocate root scaling arrays.

        Parameters
        ----------
        do_adder : bool
            Whether to initialize with an additive term.
        nlvec : <Vector> or None
            Nonlinear vector if this is a linear vector.
        """
        data = self._data
        if self._name == 'nonlinear':
            if do_adder:
                self._scaling = (np.ones(data.size), np.zeros(data.size))
            else:
                self._scaling = (np.ones(data.size), None)
        elif self._name == 'linear':
            if self._has_solver_ref:
                # We only allocate an extra scaling vector when we have output scaling
                # somewhere in the model.
                self._scaling = (np.ones(data.size), None)
            else:
                # Reuse the nonlinear scaling vecs since they're the same as ours.
                # The nonlinear vectors are created before the linear vectors
                self._scaling = (nlvec._scaling[0], None)
        else:
            raise NameError(f"Invalid vector name: {self._name}.")

    def __len__(self):
        """
        Return the flattened length of this Vector.

        Returns
        -------
        int
            Total flattened length of this vector.
        """
        return self._data.size

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
        if mode == 'rev':
            self._scale_reverse(*self._scaling)
        else:
            if self._has_solver_ref:
                self._scale_forward(self._nlvec._scaling[0], None)
            else:
                self._scale_forward(*self._scaling)

    def scale_to_phys(self, mode='fwd'):
        """
        Scale this vector to physical form.

        Parameters
        ----------
        mode : str
            Derivative direction.
        """
        if mode == 'rev':
            self._scale_forward(*self._scaling)
        else:
            if self._has_solver_ref:
                self._scale_reverse(self._nlvec._scaling[0], None)
            else:
                self._scale_reverse(*self._scaling)

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

        If copy is True, return a copy.

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

    def get_range(self, name):
        """
        Return the range of the variable in the local data array.

        Parameters
        ----------
        name : str
            Name of the variable.

        Returns
        -------
        tuple
            Start and stop indices of the variable in the local data array.
        """
        return self._views[name].range

    def get_hash(self, alg=hashlib.sha1):
        """
        Return a hash string for the array contained in this Vector.

        Parameters
        ----------
        alg : function
            Algorithm used to generate the hash.  Default is hashlib.sha1.

        Returns
        -------
        str
            The hash string.
        """
        if self._data.size == 0:
            return ''
        # we must use self._data here because the hashing alg requires array to be C-contiguous
        return array_hash(self._data, alg)
