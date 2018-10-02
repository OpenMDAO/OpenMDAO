"""Define the base Vector and Transfer classes."""
from __future__ import division, print_function

import os
import numpy as np
from six import iteritems, PY3

from openmdao.utils.name_maps import name2abs_name


_full_slice = slice(None)
_type_map = {
    'input': 'input',
    'output': 'output',
    'residual': 'output'
}

# This is the dtype we use for index arrays.  Petsc by default uses 32 bit ints
if os.environ.get('OPENMDAO_USE_BIG_INTS'):
    INT_DTYPE = np.dtype(np.int64)
else:
    INT_DTYPE = np.dtype(np.int32)


class Vector(object):
    """
    Base Vector class.

    This class is instantiated for inputs, outputs, and residuals.
    It provides a dictionary interface and an arithmetic operations interface.
    Implementations:

    - <DefaultVector>
    - <PETScVector>

    Attributes
    ----------
    _name : str
        The name of the vector: 'nonlinear', 'linear', or right-hand side name.
    _typ : str
        Type: 'input' for input vectors; 'output' for output/residual vectors.
    _kind : str
        Specific kind of vector, either 'input', 'output', or 'residual'.
    _system : System
        Pointer to the owning system.
    _iproc : int
        Global processor index.
    _length : int
        Length of flattened vector.
    _views : dict
        Dictionary mapping absolute variable names to the ndarray views.
    _views_flat : dict
        Dictionary mapping absolute variable names to the flattened ndarray views.
    _names : set([str, ...])
        Set of variables that are relevant in the current context.
    _root_vector : Vector
        Pointer to the vector owned by the root system.
    _alloc_complex : Bool
        If True, then space for the complex vector is also allocated.
    _data : ndarray
        Actual allocated data.
    _cplx_data : ndarray
        Actual allocated data under complex step.
    _cplx_views : dict
        Dictionary mapping absolute variable names to the ndarray views under complex step.
    _cplx_views_flat : dict
        Dictionary mapping absolute variable names to the flattened ndarray views under complex
        step.
    _under_complex_step : bool
        When True, this vector is under complex step, and data is swapped with the complex data.
    _ncol : int
        Number of columns for multi-vectors.
    _icol : int or None
        If not None, specifies the 'active' column of a multivector when interfaceing with
        a component that does not support multivectors.
    _relevant : dict
        Mapping of a VOI to a tuple containing dependent inputs, dependent outputs,
        and dependent systems.
    _do_scaling : bool
        True if this vector performs scaling.
    _scaling : dict
        Contains scale factors to convert data arrays.
    cite : str
        Listing of relevant citataions that should be referenced when
        publishing work that uses this class.
    read_only : bool
        When True, values in the vector cannot be changed via the user __setitem__ API.
    _under_complex_step : bool
        When True, self._data is replaced with self._cplx_data.
    """

    cite = ""

    def __init__(self, name, kind, system, root_vector=None, resize=False, alloc_complex=False,
                 ncol=1, relevant=None):
        """
        Initialize all attributes.

        Parameters
        ----------
        name : str
            The name of the vector: 'nonlinear', 'linear', or right-hand side name.
        kind : str
            The kind of vector, 'input', 'output', or 'residual'.
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
        relevant : dict
            Mapping of a VOI to a tuple containing dependent inputs, dependent outputs,
            and dependent systems.
        """
        self._name = name
        self._typ = _type_map[kind]
        self._kind = kind
        self._ncol = ncol
        self._icol = None
        self._relevant = relevant

        self._system = system

        self._iproc = system.comm.rank
        self._views = {}
        self._views_flat = {}

        # self._names will either be equivalent to self._views or to the
        # set of variables relevant to the current matvec product.
        self._names = self._views

        self._root_vector = None
        self._data = None

        # Support for Complex Step
        self._alloc_complex = alloc_complex
        self._cplx_data = None
        self._cplx_views = {}
        self._cplx_views_flat = {}
        self._under_complex_step = False

        self._do_scaling = ((kind == 'input' and system._has_input_scaling) or
                            (kind == 'output' and system._has_output_scaling) or
                            (kind == 'residual' and system._has_resid_scaling))

        self._scaling = {}

        if root_vector is None:
            self._root_vector = self
        else:
            self._root_vector = root_vector

        if resize:
            if root_vector is None:
                raise RuntimeError(
                    'Cannot resize the vector because the root vector has not yet '
                    'been created in system %s' % system.pathname)
            self._update_root_data()

        self._initialize_data(root_vector)
        self._initialize_views()

        self._length = np.sum(self._system._var_sizes[name][self._typ][self._iproc, :])

        self.read_only = False

    def __str__(self):
        """
        Return a string representation of the Vector object.

        Returns
        -------
        str
            String rep of this object.
        """
        try:
            return str(self._data)
        except Exception as err:
            return "<error during call to Vector.__str__>: %s" % err

    def __len__(self):
        """
        Return the flattened length of this Vector.

        Returns
        -------
        int
            Total flattened length of this vector.
        """
        return self._length

    def _clone(self, initialize_views=False):
        """
        Return a copy that optionally provides view access to its data.

        Parameters
        ----------
        initialize_views : bool
            Whether to initialize the views into the clone.

        Returns
        -------
        <Vector>
            instance of the clone; the data is copied.
        """
        vec = self.__class__(self._name, self._kind, self._system, self._root_vector,
                             alloc_complex=self._alloc_complex, ncol=self._ncol)
        vec._under_complex_step = self._under_complex_step
        vec._clone_data()
        if initialize_views:
            vec._initialize_views()
        return vec

    def keys(self):
        """
        Return variable names of variables contained in this vector (relative names).

        Returns
        -------
        listiterator (Python 3.x) or list (Python 2.x)
            the variable names.
        """
        return self.__iter__() if PY3 else list(self.__iter__())

    def __iter__(self):
        """
        Yield an iterator over variables involved in the current mat-vec product (relative names).

        Returns
        -------
        listiterator
            iterator over the variable names.
        """
        path = self._system.pathname
        idx = len(path) + 1 if path else 0

        return (n[idx:] for n in self._system._var_abs_names[self._typ] if n in self._names)

    def __contains__(self, name):
        """
        Check if the variable is involved in the current mat-vec product.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.

        Returns
        -------
        boolean
            True or False.
        """
        return name2abs_name(self._system, name, self._names, self._typ) is not None

    def __getitem__(self, name):
        """
        Get the unscaled variable value in true units.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the owning system's namespace.

        Returns
        -------
        float or ndarray
            variable value (not scaled, not dimensionless).
        """
        abs_name = name2abs_name(self._system, name, self._names, self._typ)
        if abs_name is not None:
            if self._icol is None:
                return self._views[abs_name]
            else:
                return self._views[abs_name][:, self._icol]
        else:
            msg = 'Variable name "{}" not found.'
            raise KeyError(msg.format(name))

    def __setitem__(self, name, value):
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
            if self.read_only:
                msg = "Attempt to set value of '{}' in {} vector when it is read only."
                raise ValueError(msg.format(name, self._kind))

            if self._icol is None:
                slc = _full_slice
                oldval = self._views[abs_name]
            else:
                slc = (_full_slice, self._icol)
                oldval = self._views[abs_name][slc]

            value = np.asarray(value)
            if value.shape != () and value.shape != (1,) and oldval.shape != value.shape:
                raise ValueError("Incompatible shape for '%s': "
                                 "Expected %s but got %s." %
                                 (name, oldval.shape, value.shape))

            self._views[abs_name][slc] = value

        else:
            msg = 'Variable name "{}" not found.'
            raise KeyError(msg.format(name))

    def _initialize_data(self, root_vector):
        """
        Internally allocate vectors.

        Must be implemented by the subclass.

        Sets the following attributes:

        - _data

        Parameters
        ----------
        root_vector : <Vector> or None
            the root's vector instance or None, if we are at the root.
        """
        pass

    def _initialize_views(self):
        """
        Internally assemble views onto the vectors.

        Must be implemented by the subclass.

        Sets the following attributes:

        - _views
        - _views_flat
        """
        pass

    def _clone_data(self):
        """
        For each item in _data, replace it with a copy of the data.

        Must be implemented by the subclass.
        """
        pass

    def __iadd__(self, vec):
        """
        Perform in-place vector addition.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            vector to add to self.
        """
        pass

    def __isub__(self, vec):
        """
        Perform in-place vector substraction.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            vector to subtract from self.
        """
        pass

    def __imul__(self, val):
        """
        Perform in-place scalar multiplication.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : int or float
            scalar to multiply self.
        """
        pass

    def add_scal_vec(self, val, vec):
        """
        Perform in-place addition of a vector times a scalar.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : int or float
            scalar.
        vec : <Vector>
            this vector times val is added to self.
        """
        pass

    def scale(self, scale_to):
        """
        Scale this vector to normalized or physical form.

        Parameters
        ----------
        scale_to : str
            Values are "phys" or "norm" to scale to physical or normalized.
        """
        scaling = self._scaling[scale_to]
        if self._ncol == 1:
            self._data *= scaling[1]
            if scaling[0] is not None:  # nonlinear only
                self._data += scaling[0]
        else:
            self._data *= scaling[1][:, np.newaxis]
            if scaling[0] is not None:  # nonlinear only
                self._data += scaling[0]

    def set_vec(self, vec):
        """
        Set the value of this vector to that of the incoming vector.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            the vector whose values self is set to.
        """
        pass

    def set_const(self, val):
        """
        Set the value of this vector to a constant scalar value.

        Must be implemented by the subclass.

        Parameters
        ----------
        val : int or float
            scalar to set self to.
        """
        pass

    def dot(self, vec):
        """
        Compute the dot product of the current vec and the incoming vec.

        Must be implemented by the subclass.

        Parameters
        ----------
        vec : <Vector>
            The incoming vector being dotted with self.
        """
        pass

    def get_norm(self):
        """
        Return the norm of this vector.

        Must be implemented by the subclass.

        Returns
        -------
        float
            norm of this vector.
        """
        pass

    def _enforce_bounds_vector(self, du, alpha, lower_bounds, upper_bounds):
        """
        Enforce lower/upper bounds, backtracking the entire vector together.

        This method modifies both self (u) and step (du) in-place.

        Parameters
        ----------
        du : <Vector>
            Newton step; the backtracking is applied to this vector in-place.
        alpha : float
            step size.
        lower_bounds : <Vector>
            Lower bounds vector.
        upper_bounds : <Vector>
            Upper bounds vector.
        """
        pass

    def _enforce_bounds_scalar(self, du, alpha, lower_bounds, upper_bounds):
        """
        Enforce lower/upper bounds on each scalar separately, then backtrack as a vector.

        This method modifies both self (u) and step (du) in-place.

        Parameters
        ----------
        du : <Vector>
            Newton step; the backtracking is applied to this vector in-place.
        alpha : float
            step size.
        lower_bounds : <Vector>
            Lower bounds vector.
        upper_bounds : <Vector>
            Upper bounds vector.
        """
        pass

    def _enforce_bounds_wall(self, du, alpha, lower_bounds, upper_bounds):
        """
        Enforce lower/upper bounds on each scalar separately, then backtrack along the wall.

        This method modifies both self (u) and step (du) in-place.

        Parameters
        ----------
        du : <Vector>
            Newton step; the backtracking is applied to this vector in-place.
        alpha : float
            step size.
        lower_bounds : <Vector>
            Lower bounds vector.
        upper_bounds : <Vector>
            Upper bounds vector.
        """
        pass

    def print_variables(self):
        """
        Print the names and values of all variables in this vector, one per line.
        """
        abs2prom = self._system._var_abs2prom[self._typ]
        print('-' * 35)
        print('   Vector %s, type %s' % (self._name, self._typ))
        for abs_name, view in iteritems(self._views):
            prom_name = abs2prom[abs_name]
            print(' ' * 3, prom_name, view)
        print('-' * 35)
        print()

    def set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        When turned on, the default real ndarray is replaced with a complex ndarray and all
        pointers are updated to point to it.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        if active:
            self._cplx_data[:] = self._data

        self._data, self._cplx_data = self._cplx_data, self._data
        self._views, self._cplx_views = self._cplx_views, self._views
        self._views_flat, self._cplx_views_flat = self._cplx_views_flat, self._views_flat
        self._under_complex_step = active
