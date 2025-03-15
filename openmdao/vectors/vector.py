"""Define the base Vector and Transfer classes."""
import weakref
import hashlib

import numpy as np

from openmdao.utils.indexer import Indexer, indexer
from openmdao.utils.array_utils import shape_to_len

_full_slice = slice(None)
_flat_full_indexer = indexer(_full_slice, flat_src=True)
_full_indexer = indexer(_full_slice, flat_src=False)

_type_map = {  # map vector type to iotype
    'input': 'input',
    'output': 'output',
    'residual': 'output'
}


class _VecData(object):
    """
    Internal data structure for each variable in a Vector.
    """
    __slots__ = ['shape', 'size', 'is_scalar', 'range', 'view', 'flat']

    def __init__(self, shape, rng):
        self.shape = shape
        self.size = shape_to_len(shape)
        self.is_scalar = shape == ()
        self.range = rng
        self.view = None
        self.flat = None


class Vector(object):
    """
    Base Vector class.

    This class is instantiated for inputs, outputs, and residuals.
    It provides a dictionary interface and an arithmetic operations interface.
    Implementations:

    - <DefaultVector>
    - <PETScVector>

    Parameters
    ----------
    name : str
        The name of the vector: 'nonlinear' or 'linear'.
    kind : str
        The kind of vector, 'input', 'output', or 'residual'.
    system : <System>
        Pointer to the owning system.
    parent_vectors : dict of dict of Vector
        Parent vectors: first key is 'input', 'output', or 'residual'; second key is vec_name.
        If None, this is a root vector.
    alloc_complex : bool
        Whether to allocate any imaginary storage to perform complex step. Default is False.

    Attributes
    ----------
    _name : str
        The name of the vector: 'nonlinear' or 'linear'.
    _typ : str
        Type: 'input' for input vectors; 'output' for output/residual vectors.
    _kind : str
        Specific kind of vector, either 'input', 'output', or 'residual'.
    _views : dict
        Dictionary mapping absolute variable names to the ndarray views.
    _names : set([str, ...])
        Set of variables that are relevant in the current context.
    _parent_slice : slice
        Slice of the parent vector that this vector represents.
    _alloc_complex : bool
        If True, then space for the complex vector is also allocated.
    _data : ndarray
        Actual allocated data.
    _slices : dict
        Mapping of var name to slice.
    _under_complex_step : bool
        When True, this vector is under complex step, and data is swapped with the complex data.
    _scaling : dict
        Contains scale factors to convert data arrays.
    read_only : bool
        When True, values in the vector cannot be changed via the user __setitem__ API.
    _len : int
        Total length of data vector (including shared memory parts).
    _has_solver_ref : bool
        This is set to True only when a ref is defined on a solver.
    """

    # Listing of relevant citations
    cite = ""
    # Indicator whether a vector class is MPI-distributed
    distributed = False

    def __init__(self, name, kind, system, name_shape_iter, parent_vector=None,
                 msginfo='', path='', alloc_complex=False):
        """
        Initialize all attributes.
        """
        # TODO: remove system ref...
        self._system = weakref.ref(system)
        self._name = name
        self._typ = _type_map[kind]
        self._kind = kind
        self._pathname = path
        self.msginfo = msginfo
        self._prom2abs = system._var_allprocs_prom2abs_list[self._typ]

        self._views = {}

        # self._names will either contain the same names as self._views or to the
        # set of variables relevant to the current matvec product.
        self._names = self._views

        self._data = None
        self._slices = None
        self._parent_slice = None

        # Support for Complex Step
        self._alloc_complex = alloc_complex
        self._under_complex_step = False

        self._scaling = None

        # If we define 'ref' on an output, then we will need to allocate a separate scaling ndarray
        # for the linear and nonlinear input vectors.
        self._has_solver_ref = system._has_output_scaling and kind == 'input' and name == 'linear'

        self._initialize_data(parent_vector, name_shape_iter)
        self._initialize_views(system)

        self.read_only = False

    def __str__(self):
        """
        Return a string representation of the Vector object.

        Returns
        -------
        str
            String rep of this object.
        """
        return str(self.asarray())

    def __len__(self):
        """
        Return the flattened length of this Vector.

        Returns
        -------
        int
            Total flattened length of this vector.
        """
        raise NotImplementedError('__len__ not defined for vector type '
                                  f'{type(self).__name__}')

    def nvars(self):
        """
        Return the number of variables in this Vector.

        Returns
        -------
        int
            Number of variables in this Vector.
        """
        return len(self._views)

    def _copy_vars(self):
        """
        Return a dictionary containing the variable values.

        Returns
        -------
        dict
            Dictionary containing the variable values.
        """
        return {n: vinfo.view[0] if vinfo.is_scalar else vinfo.view.copy()
                for n, vinfo in self._views.items()}

    def keys(self):
        """
        Return variable names of variables contained in this vector (relative names).

        Returns
        -------
        listiterator (Python 3.x) or list (Python 2.x)
            The variable names.
        """
        return self.__iter__()

    def values(self):
        """
        Return values of variables contained in this vector.

        Yields
        ------
        ndarray or float
            Value of each variable.
        """
        if self._under_complex_step:
            for n, vinfo in self._views.items():
                if n in self._names:
                    yield vinfo.view[0] if vinfo.is_scalar else vinfo.view
                else:
                    yield 0.0j if vinfo.is_scalar else np.zeros_like(vinfo.view)
        else:
            for n, vinfo in self._views.items():
                if n in self._names:
                    yield vinfo.view[0].real if vinfo.is_scalar else vinfo.view.real
                else:
                    yield 0.0 if vinfo.is_scalar else np.zeros_like(vinfo.view.real)

    def items(self):
        """
        Return (name, value) for variables contained in this vector.

        Yields
        ------
        str
            Relative name of each variable.
        ndarray or float
            Value of each variable.
        """
        if self._pathname:
            plen = len(self._pathname) + 1
        else:
            plen = 0

        if self._under_complex_step:
            for n, vinfo in self._views.items():
                if n in self._names:
                    yield n[plen:], vinfo.view[0] if vinfo.is_scalar else vinfo.view
        else:
            for n, vinfo in self._views.items():
                if n in self._names:
                    yield n[plen:], vinfo.view[0].real if vinfo.is_scalar else vinfo.view.real

    def _name2abs_name(self, name):
        """
        Map the given promoted or relative name to the absolute name.

        This is only valid when the name is unique; otherwise, a KeyError is thrown.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.

        Returns
        -------
        str or None
            Absolute variable name if unique abs_name found or None otherwise.
        """
        if name in self._prom2abs:
            lst = self._prom2abs[name]
            if len(lst) > 1:
                model = self._system()._problem_meta['model_ref']()
                src_name = model._var_abs2prom['output'][model._conn_global_abs_in2out[lst[0]]]
                raise RuntimeError(f"The promoted name {name} is invalid because it refers to "
                                   f"multiple inputs: [{' ,'.join(lst)}]. Access the value from "
                                   f"the connected output variable {src_name} instead.")

            for abs_name in self._prom2abs[name]:
                if abs_name in self._views:
                    return abs_name

        ## test abs name
        #if name in self._views:
            #return name

        # test relative name
        abs_name = self._pathname + '.' + name if self._pathname else name
        if abs_name in self._views:
            return abs_name

    def __iter__(self):
        """
        Return an iterator over variables involved in the current mat-vec product (relative names).

        Returns
        -------
        listiterator
            iterator over the variable names.
        """
        path = self._pathname
        idx = len(path) + 1 if path else 0

        return (n[idx:] for n in self._views if n in self._names)

    def _abs_item_iter(self, flat=True):
        """
        Iterate over the items in the vector, using absolute names.

        Parameters
        ----------
        flat : bool
            If True, return the flattened values.

        Yields
        ------
        str
            Name of each variable.
        ndarray or float
            Value of each variable.
        """
        if flat:
            if self._under_complex_step:
                for name, vinfo in self._views.items():
                    yield name, vinfo.flat
            else:
                for name, vinfo in self._views.items():
                    yield name, vinfo.flat.real
        else:
            for name, vinfo in self._views.items():
                if vinfo.is_scalar:
                    if self._under_complex_step:
                        yield name, vinfo.view[0]
                    else:
                        yield name, vinfo.view[0].real
                else:
                    if self._under_complex_step:
                        yield name, vinfo.view
                    else:
                        yield name, vinfo.view.real

    def _abs_iter(self):
        """
        Iterate over the absolute names in the vector.

        Yields
        ------
        str
            Name of each variable.
        """
        yield from self._views

    def __contains__(self, name):
        """
        Check if the variable is found in this vector.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.

        Returns
        -------
        bool
            True or False.
        """
        return self._name2abs_name(name) in self._names

    def _contains_abs(self, name):
        """
        Check if the variable is found in this vector.

        Parameters
        ----------
        name : str
            Absolute variable name.

        Returns
        -------
        bool
            True or False.
        """
        return name in self._names

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
        abs_name = self._name2abs_name(name)
        if abs_name is not None:
            return self._abs_get_val(abs_name, flat=False)
        else:
            raise KeyError(f"{self._system().msginfo}: Variable name '{name}' not found.")

    def get_val(self, name, flat=True):
        """
        Get the variable value.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.
        flat : bool
            If True, return the flat value.

        Returns
        -------
        float or ndarray
            Variable value.
        """
        return self._abs_get_val(name, flat=flat)

    def _abs_get_val(self, name, flat=True):
        """
        Get the variable value using the absolute name.

        No error checking is performed on the name.

        Parameters
        ----------
        name : str
            Absolute name in the owning system's namespace.
        flat : bool
            If True, return the flat value.

        Returns
        -------
        float or ndarray
            variable value.
        """
        if flat:
            if self._under_complex_step:
                return self._views[name].flat
            else:
                return self._views[name].flat.real

        vinfo = self._views[name]
        if vinfo.is_scalar:
            return vinfo.view[0]

        return vinfo.view if self._under_complex_step else vinfo.view.real

    def _abs_set_val(self, name, val):
        """
        Set the variable value using the absolute name.

        No error checking is performed on the name.

        Parameters
        ----------
        name : str
            Absolute name in the owning system's namespace.
        val : float or ndarray
            Value to set.
        """
        if self._under_complex_step:
            self._views[name].view[:] = val
        else:
            self._views[name].view.real[:] = val

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
        self.set_var(name, value)

    def _initialize_data(self, parent_vectors):
        """
        Internally allocate vectors.

        Must be implemented by the subclass.

        Parameters
        ----------
        parent_vectors : dict of dict of Vector
            Parent vectors: first key is 'input', 'output', or 'residual'; second key is vec_name.
        """
        raise NotImplementedError('_initialize_data not defined for vector type '
                                  f'{type(self).__name__}')

    def _initialize_views(self):
        """
        Internally assemble views onto the vectors.

        Must be implemented by the subclass.
        """
        raise NotImplementedError('_initialize_views not defined for vector type '
                                  f'{type(self).__name__}')

    def __iadd__(self, vec):
        """
        Perform in-place vector addition.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            vector to add to self.
        """
        raise NotImplementedError(f'__iadd__ not defined for vector type {type(self).__name__}')

    def __isub__(self, vec):
        """
        Perform in-place vector substraction.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            vector to subtract from self.
        """
        raise NotImplementedError(f'__isub__ not defined for vector type {type(self).__name__}')

    def __imul__(self, val):
        """
        Perform in-place scalar multiplication.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : int or float
            scalar to multiply self.
        """
        raise NotImplementedError(f'__imul__ not defined for vector type {type(self).__name__}')

    def add_scal_vec(self, val, vec):
        """
        Perform in-place addition of a vector times a scalar.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : int or float
            Scalar.
        vec : <Vector>
            This vector times val is added to self.
        """
        raise NotImplementedError('add_scale_vec not defined for vector type '
                                  f'{type(self).__name__}')

    def asarray(self, copy=False):
        """
        Return a flat array representation of this vector.

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
        raise NotImplementedError(f'asarray not defined for vector type {type(self).__name__}')
        return None  # silence lint warning

    def iscomplex(self):
        """
        Return True if this vector contains complex values.

        This checks the type of the values, not whether they have a nonzero imaginary part.

        Returns
        -------
        bool
            True if this vector contains complex values.
        """
        raise NotImplementedError(f'iscomplex not defined for vector type {type(self).__name__}')
        return False  # silence lint warning

    def set_vec(self, vec):
        """
        Set the value of this vector to that of the incoming vector.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            The vector whose values self is set to.
        """
        raise NotImplementedError(f'set_vec not defined for vector type {type(self).__name__}')

    def set_val(self, val, idxs=_full_slice):
        """
        Set the data array of this vector to a scalar or array value, with optional indices.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : float or ndarray
            Scalar or array to set data array to.
        idxs : int or slice or tuple of ints and/or slices
            The locations where the data array should be updated.
        """
        raise NotImplementedError(f'set_arr not defined for vector type {type(self).__name__}')

    def set_vals(self, vals):
        """
        Set the data array of this vector using a value or iter of values, one for each variable.

        The values must be in the same order and size as the variables appear in this Vector.

        Parameters
        ----------
        vals : iter of ndarrays
            Values for each variable contained in this vector, in the proper order.
        """
        for vinfo, val in zip(self._views.values(), vals):
            vinfo.flat[:] = val.ravel()

    def set_var(self, name, val, idxs=_full_slice, flat=False, var_name=None):
        """
        Set the array view corresponding to the named variable, with optional indexing.

        Parameters
        ----------
        name : str
            The name of the variable.
        val : float or ndarray
            Scalar or array to set data array to.
        idxs : int or slice or tuple of ints and/or slices
            The locations where the data array should be updated.
        flat : bool
            If True, set into flattened variable.
        var_name : str or None
            If specified, the variable name to use when reporting errors. This is useful
            when setting an AutoIVC value that the user only knows by a connected input name.
        """
        abs_name = self._name2abs_name(name)
        if abs_name is None:
            raise KeyError(f"{self._system().msginfo}: Variable name "
                           f"'{var_name if var_name else name}' not found.")

        if self.read_only:
            raise ValueError(f"{self._system().msginfo}: Attempt to set value of "
                             f"'{var_name if var_name else name}' in "
                             f"{self._kind} vector when it is read only.")

        if idxs is _full_slice:
            if flat:
                idxs = _flat_full_indexer
            else:
                idxs = _full_indexer

        elif not isinstance(idxs, Indexer):
            idxs = indexer(idxs, flat_src=flat)

        vinfo = self._views[abs_name]
        value = np.asarray(val)

        if flat:
            vinfo.flat[idxs.flat()] = value.flat
        else:
            try:
                vinfo.view[idxs()] = value
            except Exception as err:
                try:
                    value = value.reshape(vinfo.view[idxs()].shape)
                except Exception:
                    raise ValueError(f"{self._system().msginfo}: Failed to set value of "
                                     f"'{var_name if var_name else name}': {str(err)}.")
                vinfo.view[idxs()] = value

    def dot(self, vec):
        """
        Compute the dot product of the current vec and the incoming vec.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            The incoming vector being dotted with self.
        """
        raise NotImplementedError(f'dot not defined for vector type {type(self).__name__}')

    def get_norm(self):
        """
        Return the norm of this vector.

        Must be implemented by the subclass.

        Returns
        -------
        float
            Norm of this vector.
        """
        raise NotImplementedError(f'get_norm not defined for vector type {type(self).__name__}')
        return None  # silence lint warning about missing return value.

    def _in_matvec_context(self):
        """
        Return True if this vector is inside of a matvec_context.
        """
        raise NotImplementedError('_in_matvec_context not defined for vector type '
                                  f'{type(self).__name__}')

    def set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        self._under_complex_step = active

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
        raise NotImplementedError(f'get_hash not defined for vector type {type(self).__name__}')
        return ''  # silence lint warning about missing return value.

    def _get_local_views(self, arr=None):
        """
        Return a dict of views into an array using local names.

        If arr is not supplied, use our existing internal data array.
        Note that if arr is not specified, the array used will depend upon the value of
        _under_complex_step.

        Parameters
        ----------
        arr : ndarray or None
            If not None, create views into this array.

        Returns
        -------
        dict
            A dict of (view, is_scalar) tuples into the data array keyed using local names.
        """
        if arr is None:
            arr = self.asarray(copy=False)
        elif len(self) != arr.size:
            raise RuntimeError(f"{self._system().msginfo}: can't create local view dict because "
                               f"given array is size {arr.size} but expected size is {len(self)}.")

        dct = {}
        pathlen = len(self._pathname) + 1 if self._pathname else 0

        start = end = 0
        for name, vinfo in self._views.items():
            end += vinfo.size
            dct[name[pathlen:]] = (arr[start:end].reshape(vinfo.view.shape), vinfo.is_scalar)
            start = end

        return dct
