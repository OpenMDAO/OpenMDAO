"""
Base class for interpolation methods.  New methods should inherit from this class.
"""
import numpy as np

from openmdao.components.interp_util.outofbounds_error import OutOfBoundsError
from openmdao.utils.options_dictionary import OptionsDictionary


class InterpAlgorithm(object):
    """
    Base class for interpolation over data in an n-dimensional table.

    Parameters
    ----------
    grid : tuple(ndarray)
        Tuple containing ndarray of x grid locations for each table dimension.
    values : ndarray
        Array containing the values at all points in grid.
    interp : class
        Interpolation class to be used for subsequent table dimensions.
    **kwargs : dict
        Interpolator-specific options to pass onward.

    Attributes
    ----------
    grid : tuple(ndarray)
        Tuple containing x grid locations for this dimension.
    k : int
        Minimum number of points required for this algorithm.
    last_index : int
        Index of previous evaluation, used to start search for current index.
    options : <OptionsDictionary>
        Dictionary with general pyoptsparse options.
    subtable : <InterpAlgorithm>
        Table interpolation that handles child dimensions.
    values : ndarray
        Array containing the table values for all dimensions.
    _compute_d_dvalues : bool
        When set to True, compute gradients with respect to the grid values.
    _compute_d_dx : bool
        When set to True, compute gradients with respect to the interpolated point location.
    _full_slice : tuple of <Slice>
        Used to cache the full slice if training derivatives are computed.
    _name : str
        Algorithm name for error messages.
    _supports_d_dvalues : bool
        If True, this algorithm can compute the derivatives with respect to table values.
    _vectorized :bool
        If True, this method is vectorized and can simultaneously solve multiple interpolations.
    """

    def __init__(self, grid, values, interp, **kwargs):
        """
        Initialize table and subtables.
        """
        self.options = OptionsDictionary(parent_name=type(self).__name__)
        self.initialize()
        self.options.update(kwargs)

        self.subtable = None

        self.grid = grid[0]
        self.values = values

        if len(grid) > 1:
            self.subtable = interp(grid[1:], values, interp, **kwargs)

        self.last_index = 0
        self.k = None
        self._name = None
        self._vectorized = False
        self._compute_d_dvalues = False
        self._compute_d_dx = True
        self._full_slice = None
        self._supports_d_dvalues = True

    def initialize(self):
        """
        Declare options.

        Override to add options.
        """
        pass

    def check_config(self):
        """
        Verify that we have enough points for this interpolation algorithm.
        """
        if self.subtable:
            self.subtable.check_config()
        k = self.k
        n_p = len(self.grid)
        if n_p < k:
            raise ValueError("There are %d points in a data dimension,"
                             " but method %s requires at least %d "
                             "points per dimension."
                             "" % (n_p, self._name, k + 1))

    def vectorized(self, x):
        """
        Return whether this table will be run vectorized for the given requested input.

        Parameters
        ----------
        x : float
            Value of new independent to interpolate.

        Returns
        -------
        bool
            Returns True if this table can be run vectorized.
        """
        return self._vectorized

    def bracket(self, x):
        """
        Locate the interval of the new independent.

        Uses the following algorithm:
           1. Determine if the value is above or below the value at last_index
           2. Bracket the value between last_index and last_index +- inc, where
              inc has an increasing value of 1,2,4,8, etc.
           3. Once the value is bracketed, use bisection method within that bracket.

        The grid is assumed to increase in a monotonic fashion.

        Parameters
        ----------
        x : float
            Value of new independent to interpolate.

        Returns
        -------
        integer
            Grid interval index that contains x.
        integer
            Extrapolation flag, -1 if the bracket is below the first table element, 1 if the
            bracket is above the last table element, 0 for normal interpolation.
        """
        grid = self.grid
        last_index = self.last_index
        high = last_index + 1
        highbound = len(grid) - 1
        inc = 1

        while x <= grid[last_index]:
            high = last_index
            last_index -= inc
            if last_index < 0:
                last_index = 0

                # Check if we're off of the bottom end.
                if x < grid[0]:
                    return last_index, -1
                break

            inc += inc

        if high > highbound:
            high = highbound

        while x > grid[high]:
            last_index = high
            high += inc
            if high >= highbound:

                # Check if we're off of the top end
                if x > grid[highbound]:
                    last_index = highbound
                    return last_index, 1

                high = highbound
                break
            inc += inc

        # Bisection
        while high - last_index > 1:
            low = (high + last_index) // 2
            if x < grid[low]:
                high = low
            else:
                last_index = low

        return last_index, 0

    def evaluate(self, x, slice_idx=None):
        """
        Interpolate across this and subsequent table dimensions.

        Parameters
        ----------
        x : ndarray
            The coordinates to sample the gridded data at. First array element is the point to
            interpolate here. Remaining elements are interpolated on sub tables.
        slice_idx : list of <slice>
            Slice object containing indices of data points requested by parent interpolating
            tables.

        Returns
        -------
        ndarray
            Interpolated values.
        ndarray
            Derivative of interpolated values with respect to this independent and child
            independents.
        ndarray
            Derivative of interpolated values with respect to values for this and subsequent table
            dimensions.
        ndarray
            Derivative of interpolated values with respect to grid for this and subsequent table
            dimensions.
        """
        idx, _ = self.bracket(x[0])

        self.last_index = idx
        if slice_idx is None:
            slice_idx = []

        if self.subtable is not None:
            self.subtable._compute_d_dvalues = self._compute_d_dvalues

        result, d_dx, d_values, d_grid = self.interpolate(x, idx, slice_idx)

        return result, d_dx, d_values, d_grid

    def interpolate(self, x, idx, slice_idx):
        """
        Compute the interpolated value over this grid dimension.

        This method must be defined by child classes.

        Parameters
        ----------
        x : ndarray
            The coordinates to sample the gridded data at. First array element is the point to
            interpolate here. Remaining elements are interpolated on sub tables.
        idx : int
            Interval index for x.
        slice_idx : list of <slice>
            Slice object containing indices of data points requested by parent interpolating
            tables.

        Returns
        -------
        ndarray
            Interpolated values.
        ndarray
            Derivative of interpolated values with respect to this independent and child
            independents.
        ndarray
            Derivative of interpolated values with respect to values for this and subsequent table
            dimensions.
        ndarray
            Derivative of interpolated values with respect to grid for this and subsequent table
            dimensions.
        """
        raise NotImplementedError()


class InterpAlgorithmFixed(object):
    """
    Base class for interpolation over data on a table with a fixed dimension.

    Parameters
    ----------
    grid : tuple(ndarray)
        Tuple (x, y, z) of grid locations.
    values : ndarray
        Array containing the table values.
    interp : class
        Unused, but kept for API compatibility.
    **kwargs : dict
        Interpolator-specific options to pass onward.

    Attributes
    ----------
    grid : tuple(ndarray)
        Tuple (x, y, z) of grid locations.
    k : int
        Minimum number of points required for this algorithm.
    dim : int
        Required number of dimensions for this algorithm.
    last_index : list of int
        Indices of previous evaluation, used to start search for current index.
    options : <OptionsDictionary>
        Dictionary with general pyoptsparse options.
    values : ndarray
        Array containing the table values.
    _compute_d_dvalues : bool
        When set to True, compute gradients with respect to the grid values.
    _compute_d_dx : bool
        When set to True, compute gradients with respect to the interpolated point location.
    _name : str
        Algorithm name for error messages.
    _supports_d_dvalues : bool
        If True, this algorithm can compute the derivatives with respect to table values.
    _vectorized :bool
        If True, this method is vectorized and can simultaneously solve multiple interpolations.
    """

    def __init__(self, grid, values, interp, **kwargs):
        """
        Initialize interp algorithm.
        """
        self.options = OptionsDictionary(parent_name=type(self).__name__)
        self.initialize()
        self.options.update(kwargs)

        self.grid = grid
        self.values = values

        self.last_index = 0
        self.k = None
        self.dim = None
        self._name = None
        self._vectorized = True
        self._compute_d_dvalues = False
        self._supports_d_dvalues = False
        self._compute_d_dx = True

    def initialize(self):
        """
        Declare options.

        Override to add options.
        """
        pass

    def check_config(self):
        """
        Verify that we have enough points for this interpolation algorithm.
        """
        grid = self.grid
        if isinstance(grid, np.ndarray):
            grid = (grid, )
        k = self.k
        n_p = len(grid[0])
        if n_p < k:
            raise ValueError(f"There are {n_p} points in a data dimension, but method "
                             f"'{self._name}' requires at least {k} points per dimension.")
        dim = self.dim
        n_d = len(grid)
        if n_d != dim:
            raise ValueError(f"There are {n_d} dimensions, but method '{self._name}' only works"
                             f" with a fixed table dimension of {dim}.")

    def vectorized(self, x):
        """
        Return whether this table will be run vectorized for the given requested input.

        Parameters
        ----------
        x : float
            Value of new independent to interpolate.

        Returns
        -------
        bool
            Returns True if this table can be run vectorized.
        """
        return self._vectorized

    def bracket(self, x):
        """
        Locate the interval of the new independents.

        Uses the following algorithm:
           1. Determine if the value is above or below the value at last_index
           2. Bracket the value between last_index and last_index +- inc, where
              inc has an increasing value of 1,2,4,8, etc.
           3. Once the value is bracketed, use bisection method within that bracket.

        The grid is assumed to increase in a monotonic fashion.

        Parameters
        ----------
        x : float
            Value of new independent to interpolate.

        Returns
        -------
        integer
            Grid interval index that contains x.
        integer
            Extrapolation flag, -1 if the bracket is below the first table element, 1 if the
            bracket is above the last table element, 0 for normal interpolation.
        """
        for j in range(self.dim):
            if self.vectorized(x):
                self.last_index[j] = np.searchsorted(self.grid[j], x[..., j], side='left') - 1
            else:
                self.last_index[j], _ = self._bracket_dim(self.grid[j], x[j],
                                                          self.last_index[j])

        return self.last_index, None

    def _bracket_dim(self, grid, x, last_index):
        """
        Bracketing algorithm applied on a single dimension.

        Parameters
        ----------
        grid : ndarray
            Grid values at this dimension.
        x : float
            Value of independent at this dimension.
        last_index : float
            Cached index of last interpolated value at this dimension.

        """
        last_index = max(last_index, 0)
        high = last_index + 1
        highbound = len(grid) - 1
        inc = 1

        while x <= grid[last_index]:
            high = last_index
            last_index -= inc
            if last_index < 0:
                last_index = 0

                # Check if we're off of the bottom end.
                if x < grid[0]:
                    return -1, -1
                break

            inc += inc

        if high > highbound:
            high = highbound

        while x > grid[high]:
            last_index = high
            high += inc
            if high >= highbound:

                # Check if we're off of the top end
                if x > grid[highbound]:
                    return highbound, 1

                high = highbound
                break
            inc += inc

        # Bisection
        while high - last_index > 1:
            low = (high + last_index) // 2
            if x < grid[low]:
                high = low
            else:
                last_index = low

        return last_index, 0

    def evaluate(self, x, slice_idx=None):
        """
        Interpolate on this grid.

        Parameters
        ----------
        x : ndarray
            The coordinates to interpolate on this grid.
        slice_idx : None
            Only needed for API compatibility.

        Returns
        -------
        ndarray
            Interpolated values.
        ndarray
            Derivative of interpolated values with respect to independents.
        ndarray
            Derivative of interpolated values with respect to values.
        ndarray
            Derivative of interpolated values with respect to grid.
        """
        idx, _ = self.bracket(x)
        result, d_dx, d_values, d_grid = self.interpolate(x, idx)

        return result, d_dx, d_values, d_grid

    def evaluate_vectorized(self, x, slice_idx=None):
        """
        Interpolate on this grid.

        Parameters
        ----------
        x : ndarray
            The coordinates to interpolate on this grid.
        slice_idx : None
            Only needed for API compatibility.

        Returns
        -------
        ndarray
            Interpolated values.
        ndarray
            Derivative of interpolated values with respect to independents.
        ndarray
            Derivative of interpolated values with respect to values.
        ndarray
            Derivative of interpolated values with respect to grid.
        """
        idx, _ = self.bracket(x)
        result, d_dx, d_values, d_grid = self.interpolate_vectorized(x, idx)

        return result, d_dx, d_values, d_grid

    def interpolate(self, x, idx):
        """
        Compute the interpolated value.

        This method must be defined by child classes.

        Parameters
        ----------
        x : ndarray
            The coordinates to interpolate on this grid.
        idx : int
            List of interval indices for x.

        Returns
        -------
        ndarray
            Interpolated values.
        ndarray
            Derivative of interpolated values with respect to independents.
        ndarray
            Derivative of interpolated values with respect to values.
        ndarray
            Derivative of interpolated values with respect to grid.
        """
        raise NotImplementedError()

    def interpolate_vectorized(self, x_vec, idx):
        """
        Compute the interpolated value for multiple points.

        This method must be defined by child classes.

        Parameters
        ----------
        x_vec : ndarray
            The coordinates to interpolate on this grid.
        idx : int
            List of interval indices for x.

        Returns
        -------
        ndarray
            Interpolated values.
        ndarray
            Derivative of interpolated values with respect to independents.
        ndarray
            Derivative of interpolated values with respect to values.
        ndarray
            Derivative of interpolated values with respect to grid.
        """
        raise NotImplementedError()


class InterpAlgorithmSemi(object):
    """
    Base class for interpolation over semi structured data in an n-dimensional table.

    For semi structured data, this class represents a single dimension. However, every point in
    this dimension has its own InterpAlgorithmSemi object that interpolates the subsequent
    dimensions.

    Parameters
    ----------
    grid : tuple(ndarray)
        Tuple containing ndarray of x grid locations for each table dimension.
    values : ndarray
        Array containing the values at all points in grid.
    interp : class
        Interpolation class to be used for subsequent table dimensions.
    extrapolate : bool
        When False, raise an error if extrapolation occurs in this dimension.
    compute_d_dvalues : bool
        When True, compute gradients with respect to the table values.
    idx : list or None
        Maps values to their indices in the training data input. Only used during recursive
        calls.
    idim : int
        Integer corresponding to table depth. Used for error messages.
    **kwargs : dict
        Interpolator-specific options to pass onward.

    Attributes
    ----------
    idim : int
        Integer corresponding to table depth. Use for error messages.
    extrapolate : bool
        When False, raise an error if extrapolation occurs in this dimension.
    grid : tuple(ndarray)
        Tuple containing x grid locations for this dimension.
    k : int
        Minimum number of points required for this algorithm.
    last_index : int
        Index of previous evaluation, used to start search for current index.
    options : <OptionsDictionary>
        Dictionary with general pyoptsparse options.
    subtables : list of <InterpAlgorithmSemi>
        Table interpolation that handles child dimensions.
    values : ndarray
        Array containing the table values for all dimensions.
    _compute_d_dvalues : bool
        When set to True, compute gradients with respect to the table values.
    _idx : list
        Maps values to their indices in the training data input. Used when _compute_d_dvalues
        is True.
    _name : str
        Algorithm name for error messages.
    _supports_d_dvalues : bool
        If True, this algorithm can compute the derivatives with respect to table values.
    """

    def __init__(self, grid, values, interp, extrapolate=True, compute_d_dvalues=False, idx=None,
                 idim=0, **kwargs):
        """
        Initialize table and subtables.
        """
        self.options = OptionsDictionary(parent_name=type(self).__name__)
        self.initialize()
        self.options.update(kwargs)

        self.values = values
        self.extrapolate = extrapolate
        self.idim = idim
        self._supports_d_dvalues = True
        self._compute_d_dvalues = compute_d_dvalues

        if len(grid.shape) > 1 and grid.shape[1] > 1:

            if np.any(np.diff(grid[:, 0]) < 0.):
                msg = f"The points in dimension {idim} must be strictly ascending."
                raise ValueError(msg)

            # Build hieararchy of subtables.
            subtables = []
            i_pt = grid[0, 0]
            if idx is None:
                idx = [item for item in range(len(values))]

            i0, i1 = 0, 1
            sub_idx = [idx[0]]
            for point, jj in zip(grid[1:], idx[1:]):
                if point[0] != i_pt:
                    newtable = interp(grid[i0:i1, 1:], values[i0:i1], interp, idx=sub_idx,
                                      idim=idim + 1, extrapolate=extrapolate,
                                      compute_d_dvalues=compute_d_dvalues, **kwargs)
                    subtables.append(newtable)
                    i0 = i1
                    i_pt = point[0]
                    sub_idx = []

                i1 += 1
                sub_idx.append(jj)

            newtable = interp(grid[i0:i1, 1:], values[i0:i1], interp, idx=sub_idx,
                              idim=idim + 1, extrapolate=extrapolate,
                              compute_d_dvalues=compute_d_dvalues, **kwargs)
            subtables.append(newtable)

            self.subtables = subtables
            self.grid = np.unique(grid[:, 0])
        else:
            # A "leaf" of the hierarchy.
            self.grid = grid
            self.subtables = None
            self._idx = idx if idx is not None else np.arange(len(values))

            if not np.all(np.diff(grid) > 0.):
                msg = f"The points in dimension {idim} must be strictly ascending."
                raise ValueError(msg)

        self.last_index = 0
        self.k = None
        self._name = None

    def initialize(self):
        """
        Declare options.

        Override to add options.
        """
        pass

    def check_config(self):
        """
        Verify that we have enough points for this interpolation algorithm.
        """
        if self.subtables:
            for subtable in self.subtables:
                subtable.check_config()
        k = self.k
        n_p = len(self.grid)
        if n_p < k:
            raise ValueError("There are %d points in a data dimension,"
                             " but method %s requires at least %d "
                             "points per dimension."
                             "" % (n_p, self._name, k + 1))

    def vectorized(self, x):
        """
        Return whether this table will be run vectorized for the given requested input.

        Parameters
        ----------
        x : float
            Value of new independent to interpolate.

        Returns
        -------
        bool
            Returns True if this table can be run vectorized.
        """
        return self._vectorized

    def bracket(self, x):
        """
        Locate the interval of the new independent.

        Uses the following algorithm:
           1. Determine if the value is above or below the value at last_index
           2. Bracket the value between last_index and last_index +- inc, where
              inc has an increasing value of 1,2,4,8, etc.
           3. Once the value is bracketed, use bisection method within that bracket.

        The grid is assumed to increase in a monotonic fashion.

        Parameters
        ----------
        x : float
            Value of new independent to interpolate.

        Returns
        -------
        integer
            Grid interval index that contains x.
        integer
            Extrapolation flag, -1 if the bracket is below the first table element, 1 if the
            bracket is above the last table element, 0 for normal interpolation.
        """
        grid = self.grid
        last_index = self.last_index
        high = last_index + 1
        highbound = len(grid) - 1
        inc = 1

        while x < grid[last_index]:
            high = last_index
            last_index -= inc
            if last_index < 0:
                last_index = 0

                # Check if we're off of the bottom end.
                if x < grid[0]:
                    if not self.extrapolate:
                        msg = f"Extrapolation while evaluation dimension {self.idim}."
                        raise OutOfBoundsError(msg, self.idim, x, grid[0], grid[-1])

                    return last_index, -1
                break

            inc += inc

        if high > highbound:
            high = highbound

        while x > grid[high]:
            last_index = high
            high += inc
            if high >= highbound:

                # Check if we're off of the top end
                if x > grid[highbound]:
                    if not self.extrapolate:
                        msg = f"Extrapolation while evaluation dimension {self.idim}."
                        raise OutOfBoundsError(msg, self.idim, x, grid[0], grid[-1])

                    last_index = highbound
                    return last_index, 1

                high = highbound
                break
            inc += inc

        # Bisection
        while high - last_index > 1:
            low = (high + last_index) // 2
            if x < grid[low]:
                high = low
            else:
                last_index = low

        return last_index, 0

    def interpolate(self, x):
        """
        Compute the interpolated value over this grid dimension.

        This method must be defined by child classes.

        Parameters
        ----------
        x : ndarray
            Coordinate of the point being interpolated. First element is component in this
            dimension. Remaining elements are interpolated on sub tables.

        Returns
        -------
        ndarray
            Interpolated values.
        ndarray
            Derivative of interpolated values with respect to this independent and child
            independents.
        ndarray
            Derivative of interpolated values with respect to values for this and subsequent table
            dimensions.
        bool
            True if the coordinate is extrapolated in this dimension.
        """
        raise NotImplementedError()
