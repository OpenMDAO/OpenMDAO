"""
Utility functions for sampling generators.
"""

import numpy as np


def _get_size(name, dct):
    """
    Get the size of a variable from its metadata dictionary.

    This relies on the presence of 'lower' and 'upper' keys in the metadata dictionary.
    Both 'lower' and 'upper' must be present anyway to compute the levels for a DOE.
    If either 'lower' or 'upper' is not found, a RuntimeError is raised.
    If they are both present but do not have the same size, a ValueError is raised.

    Parameters
    ----------
    name : str
        The name of the variable for which to determine the size.
    dct : dict
        Dictionary containing metadata for the variable, must include 'upper', and 'lower' keys.

    Returns
    -------
    int
        The size of the variable as determined from the lower and upper bounds of the range.
        Note that both 'lower' and 'upper' must be present in the dictionary and have the same size.

    Raises
    ------
    ValueError
        The size of the specified lower bound does not match the size of the upper bound.
    RuntimeError
        The required metadata was not found in the dictionary to determine the size of the
        variable. Both the lower and upper bounds must be specified in order to compute the
        levels for a DOE.
    """
    try:
        lower_size = np.size(dct['lower'])
        upper_size = np.size(dct['upper'])
        if lower_size != upper_size:
            raise ValueError(f"Size mismatch for factor '{name}': 'lower' bound size "
                             f"({lower_size}) does not match 'upper' bound size ({upper_size}).")
        return lower_size
    except KeyError:
        raise RuntimeError(f"Unable to determine levels for factor '{name}'. "
                           "Factors dictionary must contain both 'lower' and 'upper' keys.")


def _get_bounds(name, dct, size, inf_bound):
    """
    Return the bounds of a variable as arrays of length `size`, with infinities mapped out.

    DOE and sampling generators draw values from the range spanned by a variable's bounds,
    so they cannot work with an infinite range. An unset (None) or infinite bound is
    replaced with the generator's finite `inf_bound` sentinel, mirroring the way drivers
    map infinities onto whatever their optimizer understands.

    Parameters
    ----------
    name : str
        The name of the variable whose bounds are being retrieved.
    dct : dict
        Dictionary containing metadata for the variable, must include 'upper' and 'lower' keys.
    size : int
        The size of the variable, used to broadcast scalar bounds.
    inf_bound : float
        The finite magnitude substituted for an unset or infinite bound.

    Returns
    -------
    lower : ndarray
        The lower bound of the variable, as an array of length `size`.
    upper : ndarray
        The upper bound of the variable, as an array of length `size`.
    """
    bounds = []

    for bound_name, unbounded in (('lower', -inf_bound), ('upper', inf_bound)):
        bound = dct[bound_name]
        if bound is None:
            bounds.append(np.full(size, unbounded, dtype=float))
            continue
        if np.ndim(bound) == 0:
            bound = np.full(size, bound, dtype=float)
        else:
            bound = np.asarray(bound, dtype=float).ravel()
        # Only substitute for infinities of the direction that makes this bound unbounded,
        # so that a nonsensical bound such as lower=+inf is left alone rather than flipped.
        bound[np.isneginf(bound) if unbounded < 0 else np.isposinf(bound)] = unbounded
        bounds.append(bound)

    return bounds[0], bounds[1]
