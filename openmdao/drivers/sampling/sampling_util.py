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
