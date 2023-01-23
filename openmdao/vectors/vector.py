"""Define the base Vector and Transfer classes."""
from copy import deepcopy
import weakref
import hashlib

import numpy as np
from numpy import ndim

from openmdao.utils.name_maps import prom_name2abs_name
from openmdao.utils.indexer import Indexer, indexer


_full_slice = slice(None)
_flat_full_indexer = indexer(_full_slice, flat_src=True)
_full_indexer = indexer(_full_slice, flat_src=False)

_type_map = {
    'input': 'input',
    'output': 'output',
    'residual': 'output'
}


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
    root_vector : <Vector>
        Pointer to the vector owned by the root system.
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
    _system : System
        Weak ref to the owning system.
    _views : dict
        Dictionary mapping absolute variable names to the ndarray views.
    _views_flat : dict
        Dictionary mapping absolute variable names to the flattened ndarray views.
    _names : set([str, ...])
        Set of variables that are relevant in the current context.
    _root_vector : Vector
        Pointer to the vector owned by the root system.
    _root_offset : int
        Offset of this vector into the root vector.
    _alloc_complex : bool
        If True, then space for the complex vector is also allocated.
    _data : ndarray
        Actual allocated data.
    _slices : dict
        Mapping of var name to slice.
    _under_complex_step : bool
        When True, this vector is under complex step, and data is swapped with the complex data.
    _do_scaling : bool
        True if this vector performs scaling.
    _do_adder : bool
        True if this vector's scaling includes an additive term.
    _scaling : dict
        Contains scale factors to convert data arrays.
    _scaling_nl_vec : dict
        Reference to the scaling factors in the nonlinear vector. Only used for linear input
        vectors.
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

    def __init__(self, name, kind, system, root_vector=None, alloc_complex=False):
        """
        Initialize all attributes.
        """
        self._name = name
        self._typ = _type_map[kind]
        self._kind = kind
        self._len = 0

        self._system = weakref.ref(system)

        self._views = {}
        self._views_flat = {}

        # self._names will either contain the same names as self._views or to the
        # set of variables relevant to the current matvec product.
        self._names = self._views

        self._root_vector = None
        self._data = None
        self._slices = None
        self._root_offset = 0

        # Support for Complex Step
        self._alloc_complex = alloc_complex
        self._under_complex_step = False

        self._do_scaling = ((kind == 'input' and system._has_input_scaling) or
                            (kind == 'output' and system._has_output_scaling) or
                            (kind == 'residual' and system._has_resid_scaling))
        self._do_adder = ((kind == 'input' and system._has_input_adder) or
                          (kind == 'output' and system._has_output_adder) or
                          (kind == 'residual' and system._has_resid_scaling))

        self._scaling = None
        self._scaling_nl_vec = None

        # If we define 'ref' on an output, then we will need to allocate a separate scaling ndarray
        # for the linear and nonlinear input vectors.
        self._has_solver_ref = system._has_output_scaling and kind == 'input' and name == 'linear'

        if root_vector is None:
            self._root_vector = self
        else:
            self._root_vector = root_vector

        self._initialize_data(root_vector)
        self._initialize_views()

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
        return self._len

    def nvars(self):
        """
        Return the number of variables in this Vector.

        Returns
        -------
        int
            Number of variables in this Vector.
        """
        return len(self._views)

    def _copy_views(self):
        """
        Return a dictionary containing just the views.

        Returns
        -------
        dict
            Dictionary containing the _views.
        """
        return deepcopy(self._views)

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
            for n, v in self._views.items():
                if n in self._names:
                    yield v
        else:
            for n, v in self._views.items():
                if n in self._names:
                    yield v.real

    def items(self):
        """
        Return (name, value) for variables contained in this vector.

        Yields
        ------
        str
            Name of each variable.
        ndarray or float
            Value of each variable.
        """
        if self._under_complex_step:
            for n, v in self._views.items():
                if n in self._names:
                    yield n, v
        else:
            for n, v in self._views.items():
                if n in self._names:
                    yield n, v.real

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
        system = self._system()

        # try relative name first
        abs_name = system.pathname + '.' + name if system.pathname else name
        if abs_name in self._views:
            return abs_name

        abs_name = prom_name2abs_name(system, name, self._typ)
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
        system = self._system()
        path = system.pathname
        idx = len(path) + 1 if path else 0

        return (n[idx:] for n in system._var_abs2meta[self._typ] if n in self._names)

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
        arrs = self._views_flat if flat else self._views

        if self._under_complex_step:
            yield from arrs.items()
        else:
            for name, val in arrs.items():
                yield name, val.real

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
                return self._views_flat[name]
            else:
                return self._views_flat[name].real

        if self._under_complex_step:
            return self._views[name]
        else:
            return self._views[name].real

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

    def _initialize_data(self, root_vector):
        """
        Internally allocate vectors.

        Must be implemented by the subclass.

        Parameters
        ----------
        root_vector : <Vector> or None
            the root's vector instance or None, if we are at the root.
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

        The values must be in the same order as the variables appear in this Vector.

        Parameters
        ----------
        vals : ndarray, float, or iter of ndarrays and/or floats
            Values for each variable contained in this vector, in the proper order.
        """
        arr = self.asarray()

        if self.nvars() == 1:
            if ndim(vals) == 0:
                arr[:] = vals
            else:
                arr[:] = vals.ravel()
        else:
            start = end = 0
            for v in vals:
                if ndim(v) == 0:
                    end += 1
                    arr[start] = v
                else:
                    end += v.size
                    arr[start:end] = v.ravel()
                start = end

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

        if flat:
            if isinstance(val, float):
                self._views_flat[abs_name][idxs.flat()] = val
            else:
                self._views_flat[abs_name][idxs.flat()] = np.asarray(val).flat
        else:
            value = np.asarray(val)
            view = self._views[abs_name]
            try:
                if view.shape:
                    view[idxs()] = value
                else:
                    # view is a scalar so we can't update it without breaking its connection
                    # to the underlying array, so set the value into the
                    # array using the flat view, which is an array of size 1.
                    self._views_flat[abs_name][0] = value
            except Exception as err:
                try:
                    value = value.reshape(view[idxs()].shape)
                except Exception:
                    raise ValueError(f"{self._system().msginfo}: Failed to set value of "
                                     f"'{var_name if var_name else name}': {str(err)}.")
                view[idxs()] = value

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
            A dict of views into the data array keyed using local names.
        """
        if arr is None:
            arr = self.asarray(copy=False)
        elif len(self) != arr.size:
            raise RuntimeError(f"{self._system().msginfo}: can't create local view dict because "
                               f"given array is size {arr.size} but expected size is {len(self)}.")

        dct = {}
        path = self._system().pathname
        pathlen = len(path) + 1 if path else 0

        start = end = 0
        for name, val in self._abs_item_iter(flat=False):
            end += val.size
            view = arr[start:end]
            view.shape = val.shape
            dct[name[pathlen:]] = view
            start = end

        return dct
