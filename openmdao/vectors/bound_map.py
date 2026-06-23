"""Name-keyed mapping of optimizer bounds, storing None for entirely-unbounded variables."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class _VarBounds:
    """
    Bounds for a single optimizer variable.

    Attributes
    ----------
    lower : ndarray or None
        Lower bound array, or None if entirely unbounded below.
    upper : ndarray or None
        Upper bound array, or None if entirely unbounded above.
    equals : ndarray or None
        Equality value array, or None if not an equality constraint.
    """

    lower: object
    upper: object
    equals: object


def _compact(arr, is_lower):
    """
    Return arr, or None if all elements are the unbounded infinity for this direction.

    Parameters
    ----------
    arr : ndarray or None
        Bound array.
    is_lower : bool
        True if this is a lower bound (checks for all -inf), False for upper (all +inf).

    Returns
    -------
    ndarray or None
    """
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float).ravel()
    if is_lower:
        return None if np.all(np.isneginf(arr)) else arr
    return None if np.all(np.isposinf(arr)) else arr


class BoundMap:
    """
    Name-keyed mapping of optimizer bounds.

    Each entry holds lower, upper, and equals bounds for one variable. Entirely-unbounded
    directions are stored as None rather than an array of ±inf, avoiding unnecessary
    memory allocation for large unbounded variables.

    Attributes
    ----------
    _data : dict[str, _VarBounds]
        Per-variable bounds objects.
    """

    def __init__(self):
        """Initialize BoundMap."""
        self._data = {}

    def set(self, name, lower, upper, equals, size):
        """
        Store bounds for a single variable.

        Parameters
        ----------
        name : str
            Variable name.
        lower : ndarray or None
            Lower bound in scaled units, or None if entirely unbounded below.
        upper : ndarray or None
            Upper bound in scaled units, or None if entirely unbounded above.
        equals : ndarray or None
            Equality value in scaled units, or None if not an equality constraint.
        size : int
            Number of elements in the variable (used for broadcasting scalars).
        """
        self._data[name] = _VarBounds(
            _compact(lower, is_lower=True),
            _compact(upper, is_lower=False),
            equals if equals is None else np.asarray(equals, dtype=float).ravel(),
        )

    def __getitem__(self, name):
        """
        Return the _VarBounds for a variable.

        Parameters
        ----------
        name : str
            Variable name.

        Returns
        -------
        _VarBounds
            Bounds object with .lower, .upper, .equals attributes.

        Raises
        ------
        KeyError
            If variable name not found.
        """
        if name not in self._data:
            raise KeyError(f"Variable '{name}' not found in BoundMap")
        return self._data[name]

    def __contains__(self, name):
        """Return True if name is in the dict."""
        return name in self._data

    def __iter__(self):
        """Iterate over variable names."""
        return iter(self._data)

    def keys(self):
        """Return variable names."""
        return self._data.keys()

    def items(self):
        """Iterate over (name, _VarBounds) pairs."""
        return self._data.items()
