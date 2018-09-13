"""
Utility functions related to recording or execution metadata.
"""
from fnmatch import fnmatchcase
from six.moves import map, zip
from six import iteritems
import os
import json
import numpy as np


def create_local_meta(name):
    """
    Create the metadata dictionary for this level of execution.

    Parameters
    ----------
    name : str
        String to describe the current level of execution.

    Returns
    -------
    local_meta : dict
        Dictionary containing the metadata
    """
    local_meta = {
        'name': name,
        'timestamp': None,
        'success': 1,
        'msg': '',
    }

    return local_meta


def format_iteration_coordinate(coord, prefix=None):
    """
    Format the iteration coordinate to a human-readable string.

    Parameters
    ----------
    coord : list
        List containing the iteration coordinate.

    prefix : str or None
        Prefix to prepend to iteration coordinates.

    Returns
    -------
    str :
        iteration coordinate converted to a string
    """
    separator = '|'
    iteration_number_separator = '-'

    iteration_coordinate = []

    for name, local_coord in zip(coord[1::2], coord[2::2]):
        iteration_coordinate.append(name)
        iter_str = map(str, local_coord)
        coord_str = iteration_number_separator.join(iter_str)
        iteration_coordinate.append(coord_str)

    if prefix:
        prefix = "%s_rank%d" % (prefix, coord[0])
    else:
        prefix = "rank%d" % (coord[0])

    return ':'.join([prefix, separator.join(iteration_coordinate)])


def check_valid_sqlite3_db(filename):
    """
    Raises an IOError if the given filename does not reference a valid SQLite3 database file.

    Parameters
    ----------
    filename : str
        The path to the file to be tested

    Raises
    ------
    IOError
        If the given filename does not reference a valid SQLite3 database file.
    """
    # check that the file exists
    if not os.path.isfile(filename):
        raise IOError('File does not exist({0})'.format(filename))

    # check that the file is large enough (SQLite database file header is 100 bytes)
    if os.path.getsize(filename) < 100:
        raise IOError('File does not contain a valid sqlite database ({0})'.format(filename))

    # check that the first 100 bytes actually contains a valid SQLite database header
    with open(filename, 'rb') as fd:
        header = fd.read(100)
    if header[:16] != b'SQLite format 3\x00':
        raise IOError('File does not contain a valid sqlite database ({0})'.format(filename))


def check_path(path, includes, excludes, include_all_path=False):
    """
    Calculate whether `path` should be recorded.

    Parameters
    ----------
    path : str
        path proposed to be recorded
    includes : list
        list of things to be included in recording list.
    excludes : list
        list of things to be excluded from recording list.
    include_all_path : bool
        If set to True, will return True unless it is in excludes

    Returns
    -------
    boolean
        True if path should be recorded, False if it's been excluded.
    """
    # First see if it's included
    for pattern in includes:
        if fnmatchcase(path, pattern) or include_all_path:
            # We found a match. Check to see if it is excluded.
            for ex_pattern in excludes:
                if fnmatchcase(path, ex_pattern):
                    return False
            return True

    # the case where includes is empty but include_all_path is True
    if include_all_path:
        for ex_pattern in excludes:
            if fnmatchcase(path, ex_pattern):
                return False
        return True

    return False


def json_to_np_array(vals, abs2meta):
    """
    Convert from a JSON string to a numpy named array.

    Parameters
    ----------
    vals : string
        json string of data
    abs2meta : dict
        Dictionary mapping absolute variable names to variable metadata.

    Returns
    -------
    array: numpy named array
        named array containing the same names and values as the input values json string.
    """
    json_vals = json.loads(vals)
    if json_vals is None:
        return None

    for var in json_vals:
        json_vals[var] = convert_to_np_array(json_vals[var], var, abs2meta[var]['shape'])

    return values_to_array(json_vals)


def convert_to_np_array(val, varname, shape):
    """
    Convert list to numpy array.

    Parameters
    ----------
    val : list
        the list to be converted to an np.array
    varname : str
        name of variable to be converted
    shape : tuple
        the shape of the resulting np.array

    Returns
    -------
    numpy.array :
        The converted array.
    """
    if isinstance(val, list):
        array = np.array(val)
        array = np.resize(array, shape)
        return array

    return val


def values_to_array(values):
    """
    Convert a dict of variable names and values into a numpy named array.

    Parameters
    ----------
    values : dict
        dict of variable names and values

    Returns
    -------
    array: numpy named array
        named array containing the same names and values as the input values dict.
    """
    if values:
        dtype_tuples = []
        for name, value in iteritems(values):
            tple = (str(name), '{}f8'.format(value.shape))
            dtype_tuples.append(tple)

        array = np.zeros((1,), dtype=dtype_tuples)

        for name, value in iteritems(values):
            array[name] = value
    else:
        array = None

    return array
