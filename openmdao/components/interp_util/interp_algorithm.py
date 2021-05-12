"""
Base class for interpolation methods.  New methods should inherit from this class.
"""
from openmdao.utils.options_dictionary import OptionsDictionary


class InterpAlgorithm(object):
    """
    Base class for interpolation over data in an n-dimensional table.

    Attributes
    ----------
    grid : tuple(ndarray)
        Tuple containing x grid locations for this dimension.
    k : int
        Minimum number of points required for this algorithm.
    last_index : integer
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
    _vectorized :bool
        If True, this method is vectorized and can simultaneously solve multiple interpolations.
    """

    def __init__(self, grid, values, interp, **kwargs):
        """
        Initialize table and subtables.

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
        idx : integer
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
        pass
