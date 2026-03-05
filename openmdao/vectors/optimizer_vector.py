"""Lightweight vector wrapper for driver-level design variables and responses."""

from typing import Any, Literal, TYPE_CHECKING

import numpy as np

from openmdao.utils.mpi import MPI

if TYPE_CHECKING:
    from openmdao.core.driver import Driver


class OptimizerVector(object):
    """
    Provides name-based indexing over optimizer-level design variable or response vectors.

    This is a lightweight dict-like wrapper, not a System Vector. It provides convenient
    access for optimization algorithms.

    Parameters
    ----------
    voi_type : str ('design_var', 'constraint', 'objective', or 'lagrange_multiplier')
        A string specifying the type of optimization variable in the vector.
    data : ndarray
        Flat numpy array containing variable values.
    metadata : dict
        Metadata dict mapping variable names to index information. Each entry should contain
        'start_idx', 'end_idx', and 'size' keys.
    driver_scaling : bool
        True if the data provided is in driver/optimizer-scaled space.
        
    Attributes
    ----------
    voi_type : str ('design_var', 'constraint', or 'objective')
        A string specifying the type of optimization variable in the vector.
    _data : ndarray
        Flat numpy array containing variable values.
    _meta : dict
        Metadata dict mapping variable names to index information. Each entry should contain
        'start_idx', 'end_idx', and 'size' keys.
    _filters : dict[tuple, ndarray]
        Cache for computed filter indices. Keys are tuples of sorted (key, value)
        filter criteria, values are integer arrays of indices into _data.
        Used by asarray() to avoid recomputing indices for repeated filter calls.
    _dist_driver_vars : dict[str, tuple]
        Mapping of distributed driver variables to (local_indices, sizes, _)

    Examples
    --------
    >>> vec = OpimizerVector('design_var', numpy_array, metadata)
    >>> x_value = vec['x']  # Get design variable by name
    >>> vec['x'] = 2.5  # Set design variable by name
    >>> for name, value in vec.items():  # Iterate over variables
    ...     print(f"{name}: {value}")
    """

    def __init__(self, voi_type, data, metadata, driver_scaling=False):
        """Initialize OpimizerVector with data array and metadata."""
        self.voi_type: Literal['design_var', 'constraint', 'objective'] = voi_type
        self._data: np.ndarray = data
        self._meta: dict[str, Any] = metadata
        self._filters = {}
        self._driver_scaling: bool = driver_scaling
        self._dist_driver_vars = {}  # Mapping of distributed var names to (local_indices, sizes, _)

    def __getitem__(self, name):
        """
        Get variable value by name.

        Parameters
        ----------
        name : str
            Variable name (promoted or alias).

        Returns
        -------
        ndarray
            1D array of variable values.

        Raises
        ------
        KeyError
            If variable name not found.
        """
        if name not in self._meta:
            raise KeyError(f"Variable '{name}' not found in OpimizerVector")
        info = self._meta[name]
        return self._data[info['start_idx']:info['end_idx']].reshape(-1)

    def __setitem__(self, name, value):
        """
        Set variable value by name.

        Parameters
        ----------
        name : str
            Variable name (promoted or alias).
        value : float or ndarray
            New value for the variable.

        Raises
        ------
        KeyError
            If variable name not found.
        """
        if name not in self._meta:
            raise KeyError(f"Variable '{name}' not found in OptimizerVector")
        info = self._meta[name]
        self._data[info['start_idx']:info['end_idx']] = np.asarray(value).flat

    def __contains__(self, name):
        """
        Check if variable name exists.

        Parameters
        ----------
        name : str
            Variable name to check.

        Returns
        -------
        bool
            True if variable exists, False otherwise.
        """
        return name in self._meta

    def __len__(self):
        """
        Return number of variables.

        Returns
        -------
        int
            Number of variables in this vector.
        """
        return len(self._meta)

    def __iter__(self):
        """
        Iterate over variable names.

        Yields
        ------
        str
            Variable names in iteration order.
        """
        return iter(self._meta)

    def keys(self):
        """
        Return variable names.

        Returns
        -------
        dict_keys
            View of variable names.
        """
        return self._meta.keys()

    def values(self):
        """
        Iterate over variable values.

        Yields
        ------
        ndarray
            Variable values in iteration order.
        """
        for name in self._meta:
            yield self[name]

    def items(self):
        """
        Iterate over (name, value) pairs.

        Yields
        ------
        tuple
            (variable_name, variable_value) pairs in iteration order.
        """
        for name in self._meta:
            yield name, self[name]

    def _make_filter_key(self, **kwargs):
        """
        Create hashable cache key from filter kwargs.

        Parameters
        ----------
        **kwargs : dict
            Filter criteria.

        Returns
        -------
        tuple or None
            Sorted tuple of (key, value) pairs, or None if no kwargs.
        """
        if not kwargs:
            return None
        return tuple(sorted(kwargs.items()))

    def _compute_filter_indices(self, filters):
        """
        Compute array indices for variables matching filter criteria.

        Parameters
        ----------
        filters : dict
            Filter criteria to match.

        Returns
        -------
        ndarray
            1D integer array of indices into self._data for matching variables.
            Empty array if no variables match.
        """
        ranges = []
        for meta in self._meta.values():
            # Check if all filter criteria match this variable's metadata
            if all(meta.get(key) == value for key, value in filters.items()):
                ranges.append((meta['start_idx'], meta['end_idx']))

        if not ranges:
            return np.array([], dtype=np.intp)

        # Use np.concatenate with list of np.arange (follows OpenMDAO vector.py pattern)
        return np.concatenate([np.arange(start, end, dtype=np.intp)
                              for start, end in ranges])

    def set_data(self, val, driver_scaling=True, order='C'):
        """
        Set the values of the internal vector.

        The size of val must match the size of the internal data vector.
        val is flattened before being set into _data.

        Parameters
        ----------
        val : ArrayLike or dict[str, ArrayLike]
            Values to which the entire internal vector should be set. May be given
            as an array-like value or as a dict (as in pyoptsparse) that maps
            names to associated values.
        driver_scaling : bool
            If True, set the
        order : str
            The order in which val is flattened, as accepted by
        """
        if isinstance(val, dict):
            for n, v in val.items():
                self[n] = v
        else:
            self._data[:] = np.asarray(val).ravel(order=order)
        self._driver_scaling = driver_scaling
    
    @classmethod
    def create_from_model(cls, voi_type: Literal['design_var', 'constraint', 'objective'],
                          driver: 'Driver', driver_scaling: bool=True):
        """
        Populate the data in the vector based on values from the model.

        Note that this only returns the continuous values.

        Parameters
        ----------
        voi_type : str
            The kind of vector being created ('design_var', 'constraint', or 'objective').
        driver : Driver
            The driver that owns this OptimizerVector.
        driver_scaling : bool
            If True, return vector values in the optimizer-scaled space. Default is True.

        Returns
        -------
        OptimizerVector
            A new OptimizerVector.
        """
        varmeta_map = {'design_var': driver._designvars,
                       'constraint': driver._cons,
                       'objective': driver._objs}

        # Determine remote VOIs dict based on type
        if voi_type == 'design_var':
            remote_vois = driver._remote_dvs
        elif voi_type == 'constraint':
            remote_vois = driver._remote_cons
        else:  # objective
            remote_vois = driver._remote_objs

        # Build metadata for OptimizerVector with flat array indexing
        vecmeta = {}
        voi_array = []
        idx = 0

        for name, meta in varmeta_map[voi_type].items():
            if meta['discrete']:
                continue
            size = meta['global_size'] if meta['distributed'] else meta['size']

            vecmeta[name] = {
                'start_idx': idx,
                'end_idx': idx + size,
                'size': size,
            }

            # Meta that only applies to constraints and/or design vars
            if 'linear' in meta:
                vecmeta[name]['linear'] = meta.get('linear')
            if 'equals' in meta:
                vecmeta[name]['equals'] = meta.get('equals')
            if 'lower' in meta:
                vecmeta[name]['lower'] = meta.get('lower')
            if 'upper' in meta:
                vecmeta[name]['upper'] = meta.get('upper')

            val = driver._get_voi_val(name, meta, remote_vois,
                                      get_remote=True, driver_units=True)
            voi_array.append(np.atleast_1d(val).flat)

            idx += size

        flat_array = np.concatenate(voi_array) if voi_array else np.array([])
        out = cls(voi_type, flat_array, vecmeta)

        # Set distributed variable info for get_remote handling
        out._dist_driver_vars = driver._dist_driver_vars

        # Apply autoscaler to the vector
        if driver_scaling:
            if voi_type == 'design_var':
                driver._autoscaler.apply_design_var_scaling(out)
            elif voi_type == 'constraint':
                driver._autoscaler.apply_constraint_scaling(out)
            else:
                driver._autoscaler.apply_objective_scaling(out)

        out.driver_scaling = driver_scaling

        return out

    def update_from_model(self, driver, driver_scaling=True):
        """
        Populate the data in the vector based on values from the model.

        Note that this only returns the continuous values.

        Parameters
        ----------
        driver : Driver
            The driver that owns this OptimizerVector.
        driver_scaling : bool
            If True, return vector values in the optimizer-scaled space. Default is True.
        """
        varmeta_map = {'design_var': driver._designvars,
                       'constraint': driver._cons,
                       'objective': driver._objs}

        voi_type = self.voi_type

        # Determine remote VOIs dict based on type
        if voi_type == 'design_var':
            remote_vois = driver._remote_dvs
        elif voi_type == 'constraint':
            remote_vois = driver._remote_cons
        else:  # objective
            remote_vois = driver._remote_objs

        # Mark as unscaled since we're about to populate with new unscaled data
        self._driver_scaling = False

        # Populate the vector in-place
        idx = 0

        for name, meta in varmeta_map[voi_type].items():
            if meta['discrete']:
                continue
            size = meta['global_size'] if meta['distributed'] else meta['size']

            # The vector already exists, just populate it
            self.asarray()[idx:idx + size] = driver._get_voi_val(name, meta, remote_vois,
                                                                 get_remote=True,
                                                                 driver_units=True)

            idx += size

        # Apply autoscaler to the vector
        if driver_scaling:
            if voi_type == 'design_var':
                driver._autoscaler.apply_design_var_scaling(self)
            elif voi_type == 'constraint':
                driver._autoscaler.apply_constraint_scaling(self)
            else:
                driver._autoscaler.apply_objective_scaling(self)

    def asarray(self, **kwargs) -> np.ndarray:
        """
        Return underlying flat numpy array, optionally filtered by metadata.

        When filters are provided, returns only elements whose metadata matches
        ALL specified criteria (AND logic). Filtered results are copies due to
        NumPy fancy indexing. Unfiltered results (no kwargs) return a view.

        Parameters
        ----------
        **kwargs : dict
            Filter criteria based on metadata keys. For example:
            - linear=False : only nonlinear constraints
            - linear=True : only linear constraints
            - equals=None : only inequality constraints
            Multiple criteria are combined with AND logic.

        Returns
        -------
        ndarray
            The underlying data array. If no filters provided, returns a view
            of the full array. If filters provided, returns a filtered copy.

        Examples
        --------
        >>> vec.asarray()  # Full array (view)
        >>> vec.asarray(linear=False)  # Only nonlinear constraints (copy)
        >>> vec.asarray(linear=False, equals=None)  # Nonlinear inequalities (copy)

        Notes
        -----
        Filter results are cached in self._filters for performance. Repeated
        calls with the same filters reuse cached indices but still return a
        copy (unavoidable with NumPy fancy indexing).
        """
        # No filters: return full view
        if not kwargs:
            return self._data

        # Check cache
        cache_key = self._make_filter_key(**kwargs)
        if cache_key not in self._filters:
            self._filters[cache_key] = self._compute_filter_indices(kwargs)

        indices = self._filters[cache_key]
        return self._data[indices]

    def _from_dict(self, var_dict):
        """
        Populate this OptimizerVector from a dictionary of variables in-place.

        This method populates the underlying flat array by extracting values from a
        dictionary interface (as used by pyOptSparseDriver), ordering them according to
        the metadata index information.

        Parameters
        ----------
        var_dict : dict
            Dictionary mapping variable names (str) to numpy array values.
            Keys must exactly match the keys in this vector's metadata.

        Raises
        ------
        KeyError
            If a key in metadata is not present in var_dict.
        ValueError
            If the size of a variable in var_dict doesn't match the size in metadata.

        Examples
        --------
        >>> vec = OptimizerVector('design_var', np.zeros(3), metadata)
        >>> dv_dict = {'x': np.array([1.0, 2.0]), 'y': np.array([3.0])}
        >>> vec._from_dict(dv_dict)
        >>> vec.asarray()
        array([1., 2., 3.])
        """
        for name, meta in self._meta.items():
            if name not in var_dict:
                raise KeyError(f"Variable '{name}' in metadata not found in var_dict")

            value = np.asarray(var_dict[name]).ravel()
            expected_size = meta['end_idx'] - meta['start_idx']

            if value.size != expected_size:
                raise ValueError(
                    f"Size mismatch for variable '{name}': expected {expected_size}, "
                    f"got {value.size}"
                )

            self._data[meta['start_idx']:meta['end_idx']] = value

    def _to_dict(self, get_remote=True, **filters):
        """
        Convert this OptimizerVector to a dictionary of variables.

        Returns a dictionary mapping variable names to their numpy array values, extracting
        them from the underlying flat array according to the metadata index information.
        When filters are provided, only variables whose metadata matches ALL specified
        criteria are included in the returned dictionary.

        Parameters
        ----------
        get_remote : bool
            If False, only return those items or indices of each variable on the local proc.
        **filters : dict
            Optional filter criteria based on metadata keys. For example:
            - linear=False : only nonlinear constraints
            - linear=True : only linear constraints
            Multiple criteria are combined with AND logic.

        Returns
        -------
        dict
            Dictionary mapping variable names (str) to numpy array values. Each value is
            a copy, not a view, so modifications to the dictionary won't affect this vector.
            When filters are provided, only matching variables are included.

        Examples
        --------
        >>> vec = OptimizerVector('design_var', np.array([1., 2., 3.]), metadata)
        >>> dv_dict = vec._to_dict()  # All variables
        >>> dv_dict['x']
        array([1., 2.])

        >>> # Get only nonlinear constraints
        >>> con_dict = vec._to_dict(linear=False)
        """
        result = {}
        for name, meta in self._meta.items():
            # Check if all filter criteria match this variable's metadata
            if all(meta.get(key) == value for key, value in filters.items()):
                val = self[name].copy()  # Use copy to return independent array (gathered array)

                # For distributed variables with get_remote=False, extract only local portion
                if not get_remote and MPI and name in self._dist_driver_vars:
                    local_indices, sizes, distributed_indices = self._dist_driver_vars[name]
                    # distributed_indices is a slice/array indicating which elements in the full
                    # distributed array belong to this rank. Use it to extract only local elements.
                    val = val[distributed_indices]

                result[name] = val
        return result

    @property
    def metadata(self):
        """
        Access the internal metadata dictionary for each variable.

        Returns
        -------
        dict
            The dictionary of metadata keyed by optimization variable name/alias.
        """
        return self._meta

    @property
    def driver_scaling(self):
        """
        Get the current scaling status of the vector.

        Returns
        -------
        bool
            True if the vector is currently in driver/optimizer-scaled space.
        """
        return self._driver_scaling
    
    @driver_scaling.setter
    def driver_scaling(self, b):
        """
        Set the current scaling status of the vector.

        Parameters
        ----------
        b : bool
            True if the vector is currently in driver/optimizer-scaled space, otherwise False.
        """
        self._driver_scaling = b


# TODO: Remove these if unused
# class DesignVarVector(OptimizerVector):
#     """
#     An design variable-specific implementation of OptimizerVector

#     Parameters
#     ----------
#     data : ndarray
#         Flat numpy array containing variable values.
#     metadata : dict
#         Metadata dict mapping variable names to index information. Each entry should contain
#         'start_idx', 'end_idx', and 'size' keys.
#     driver_scaling : bool
#         True if the data provided is in driver/optimizer-scaled space.
#     """

#     def __init__(self, data, metadata, driver_scaling=False):
#         """Initialize DesignVarVector with data array and metadata."""
#         super().__init__('objective', data, metadata, driver_scaling)

#     def apply_scaling(self, driver):
#         driver._autoscaler.apply_design_var_scaling(self)

#     def apply_unscaling(self, driver):
#         driver._autoscaler.apply_design_var_unscaling(self)

# class Constraint(OptimizerVector):
#     """
#     An constraint-specific implementation of OptimizerVector

#     Parameters
#     ----------
#     data : ndarray
#         Flat numpy array containing variable values.
#     metadata : dict
#         Metadata dict mapping variable names to index information. Each entry should contain
#         'start_idx', 'end_idx', and 'size' keys.
#     driver_scaling : bool
#         True if the data provided is in driver/optimizer-scaled space.
#     """

#     def __init__(self, data, metadata, driver_scaling=False):
#         """Initialize ConstraintVector with data array and metadata."""
#         super().__init__('constraint', data, metadata, driver_scaling)

#     def apply_scaling(self, driver):
#         driver._autoscaler.apply_constraint_scaling(self)

# class ObjectiveVector(OptimizerVector):
#     """
#     An objective-specific implementation of OptimizerVector

#     Parameters
#     ----------
#     data : ndarray
#         Flat numpy array containing variable values.
#     metadata : dict
#         Metadata dict mapping variable names to index information. Each entry should contain
#         'start_idx', 'end_idx', and 'size' keys.
#     driver_scaling : bool
#         True if the data provided is in driver/optimizer-scaled space.
#     """

#     def __init__(self, data, metadata, driver_scaling=False):
#         """Initialize ObjectiveVector with data array and metadata."""
#         super().__init__('objective', data, metadata, driver_scaling)

#     def apply_scaling(self, driver):
#         driver._autoscaler.apply_objective_scaling(self)
