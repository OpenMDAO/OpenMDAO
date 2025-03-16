"""Define the default Vector class."""
from collections import defaultdict
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
    root_vector : <Vector>
        Pointer to the vector owned by the root system.
    alloc_complex : bool
        Whether to allocate any imaginary storage to perform complex step. Default is False.

    Attributes
    ----------
    _views_rel : dict or None
        If owning system is a component, this will contain a mapping of relative names to views.
    """

    TRANSFER = DefaultTransfer

    def __init__(self, name, kind, system, name_shape_iter, parent_vector=None, msginfo='', path='',
                 alloc_complex=False, do_scaling=False):
        """
        Initialize all attributes.
        """
        self._views_rel = None
        super().__init__(name, kind, system, name_shape_iter, parent_vector, msginfo, path,
                         alloc_complex, do_scaling)

    def __getitem__(self, name):
        """
        Get the variable value.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.

        Returns
        -------
        float or ndarray
            variable value.
        """
        if self._views_rel is not None:
            try:
                vinfo = self._views_rel[name]
                val = vinfo.view[0] if vinfo.is_scalar else vinfo.view
                if self._under_complex_step:
                    return val
                return val.real
            except KeyError:
                pass  # try normal lookup after rel lookup failed

        return super().__getitem__(name)

    def __setitem__(self, name, value):
        """
        Set the variable value.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.
        value : float or list or tuple or ndarray
            variable value to set
        """
        if self._views_rel is not None and not self.read_only:
            try:
                self._views_rel[name].view[:] = value
                return
            except Exception:
                pass  # fall through to normal set if fast one failed in any way

        self.set_var(name, value)

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

    def _initialize_data(self, parent_vector, name_shape_iter):
        """
        Set up internal data structures.

        Parameters
        ----------
        parent_vector : <Vector> or None
            Parent vector.
        name_shape_iter : iterator of (str, tuple)
            Iterator of (name, shape) tuples.
        """
        self._views = {}
        start = end = 0
        for name, shape in name_shape_iter:
            end += shape_to_len(shape)
            self._views[name] = _VecData(shape, (start, end))
            start = end

        if parent_vector is None:  # this is a root vector
            self._parent_slice = slice(0, end)
            self._data = np.zeros(end, dtype=complex if self._alloc_complex else float)
        else:
            for name in self._views:
                # just get our first name to get the starting index in the parent vector
                start = parent_vector._views[name].range[0]
                self._parent_slice = slice(start, start + end)
                self._data = parent_vector._data[self._parent_slice]
                break
            else:
                self._parent_slice = slice(0, 0)
                self._data = np.zeros(0, dtype=complex if self._alloc_complex else float)

            if self._do_scaling:
                ps0, ps1 = parent_vector._scaling
                if ps0 is not None:
                    ps0 = ps0[self._parent_slice]
                self._scaling = (ps0, ps1[self._parent_slice])

        # print(self._kind, self._name, 'scaling', self._scaling)

    def _initialize_scaling(self, nlvec, do_adder):
        """
        Initialize root scaling vectors.

        Parameters
        ----------
        nlvec : <Vector>
            nonlinear vector.
        do_adder : bool
            Whether to initialize with an additive term.
        """
        data = self._data
        if self._name == 'nonlinear':
            if do_adder:
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
                # The nonlinear vectors are created before the linear vectors
                self._scaling = (None, nlvec._scaling[1])
        else:
            self._scaling = (None, np.ones(data.size))

    def _initialize_views(self, system):
        """
        Internally assemble views onto the vectors.
        """
        kind = self._kind
        islinear = self._name == 'linear'
        rel_lookup = system._has_fast_rel_lookup()

        scaling = self._scaling
        if scaling is not None:
            factors = system._scale_factors

        if rel_lookup:
            self._views_rel = views_rel = {}
            relstart = len(self._pathname) + 1 if self._pathname else 0
        else:
            self._views_rel = None

        views = self._views
        start = end = 0
        for abs_name, vinfo in self._views.items():
            end += vinfo.size
            vflat = v = self._data[start:end]
            if vinfo.shape != vflat.shape and vinfo.shape != ():
                v = vflat.view().reshape(vinfo.shape)

            # print(abs_name, v.shape, vflat.shape)

            vinfo.view = v
            vinfo.flat = vflat

            if rel_lookup:
                views_rel[abs_name[relstart:]] = vinfo

            if scaling is not None:
                factor_tuple = factors[abs_name][kind]

                if len(factor_tuple) == 4:
                    # Only input vectors can have 4 factors. Linear input vectors need to be able
                    # to handle the unit and solver scaling in opposite directions in reverse mode.
                    a0, a1, factor, offset = factor_tuple

                    if islinear:
                        scale0 = None
                        scale1 = factor / a1
                    else:
                        scale0 = (a0 + offset) * factor
                        scale1 = a1 * factor
                else:
                    if self._name == 'linear' and self._typ == 'input':
                        scale0 = None
                        scale1 = 1.0 / factor_tuple[1]
                    else:
                        scale0, scale1 = factor_tuple

                if scaling[0] is not None:
                    scaling[0][start:end] = scale0
                scaling[1][start:end] = scale1

                # print('scaling for', abs_name, self._kind, self._name, self._scaling)

            start = end

        self._names = frozenset(views) if islinear else views

    def _name2abs_name(self, name):
        """
        Map the given absolute or relative name to the absolute name.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.

        Returns
        -------
        str or None
            Absolute variable name if unique abs_name found or None otherwise.
        """
        # try relative name first
        if self._views_rel is not None and name in self._views_rel:
            return self._pathname + '.' + name if self._pathname else name

        return super()._name2abs_name(name)

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
            self._scale_reverse(self._scaling[1], self._scaling[0])
        else:
            # if self._has_solver_ref:
            #     self._scale_forward(self._scaling_nl_vec[1], None)
            # else:
            self._scale_forward(self._scaling[1], self._scaling[0])

    def scale_to_phys(self, mode='fwd'):
        """
        Scale this vector to physical form.

        Parameters
        ----------
        mode : str
            Derivative direction.
        """
        if mode == 'rev':
            self._scale_forward(self._scaling[1], self._scaling[0])
        else:
            # if self._has_solver_ref:
            #     self._scale_reverse(self._scaling_nl_vec[1], None)
            # else:
            self._scale_reverse(self._scaling[1], self._scaling[0])

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
            for name, vinfo in self._views.items():
                slices[name] = slice(*vinfo.range)
            self._slices = slices

        return self._slices

    def idxs2nameloc(self, idxs):
        """
        Given some indices, return a dict mapping variable name to corresponding local indices.

        This is slow and is meant to be used only for debugging or maybe error messages.

        Parameters
        ----------
        idxs : list of int
            Vector indices to be converted to local indices for each corresponding variable.

        Returns
        -------
        dict
            Mapping of variable name to a list of local indices into that variable.
        """
        name2inds = defaultdict(list)
        start = end = 0
        for name, vinfo in self._views.items():
            start, end = vinfo.range
            for idx in idxs:
                if start <= idx < end:
                    name2inds[name].append(idx - start)
            start = end

        return name2inds

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
