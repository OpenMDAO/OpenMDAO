"""
Utility functions related to recording or execution metadata.
"""
from fnmatch import fnmatchcase
import os
import re
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


# regular expression used to determine if a node in an iteration coordinate represents a system
_coord_system_re = re.compile('(_solve_nonlinear|_apply_nonlinear)$')

# Regular expression used for splitting iteration coordinates, removes separator and iter counts
_coord_split_re = re.compile('\\|\\d+\\|*')


def get_source_system(iteration_coordinate):
    """
    Get pathname of system that is the source of the iteration.

    Parameters
    ----------
    iteration_coordinate : str
        The full unique identifier for this iteration.

    Returns
    -------
    str
        The pathname of the system that is the source of the iteration.
    """
    path = []
    parts = _coord_split_re.split(iteration_coordinate)
    for part in parts:
        if (_coord_system_re.search(part) is not None):
            if ':' in part:
                # get rid of 'rank#:'
                part = part.split(':')[1]
            path.append(part.split('.')[0])

    # return pathname of the system
    return '.'.join(path)


def check_valid_sqlite3_db(filename):
    """
    Raise an IOError if the given filename does not reference a valid SQLite3 database file.

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
    for ex_pattern in excludes:
        if fnmatchcase(path, ex_pattern):
            return False

    if not include_all_path:
        for pattern in includes:
            if fnmatchcase(path, pattern):
                return True

    return include_all_path


def deserialize(json_data, abs2meta, prom2abs, conns):
    """
    Deserialize recorded data from a JSON formatted string.

    If all data values are arrays then a numpy structured array will be returned,
    otherwise a dictionary mapping variable names to values will be returned.

    Parameters
    ----------
    json_data : string
        JSON encoded data
    abs2meta : dict
        Dictionary mapping absolute variable names to variable metadata
    prom2abs : dict
        Dictionary mapping promoted input names to absolute. Needed to resolve auto_ivc outputs
        that are recorded with their promoted input name.
    conns : dict
        Dictionary of all model connections.

    Returns
    -------
    array or dict
        Variable names and values parsed from the JSON string
    """
    values = json.loads(json_data)
    if values is None:
        return None

    all_array = True

    for name, value in values.items():
        try:
            has_shape = 'shape' in abs2meta[name]
        except KeyError:
            abs_name = prom2abs['input'][name]
            src_name = conns[abs_name[0]]
            has_shape = 'shape' in abs2meta[src_name]

        if isinstance(value, list) and has_shape:
            values[name] = np.asarray(value)  # array will be proper shape based on list structure
        else:
            all_array = False

    if all_array:
        return dict_to_structured_array(values)
    else:
        return values


def dict_to_structured_array(values):
    """
    Convert a dict of variable names and values into a numpy structured array.

    Parameters
    ----------
    values : dict
        dict of variable names and values

    Returns
    -------
    array
        numpy structured array containing the same names and values as the input values dict.
    """
    if values:
        dtype_tuples = []
        for name, value in values.items():
            tple = (str(name), '{}f8'.format(value.shape))
            dtype_tuples.append(tple)

        array = np.zeros((1,), dtype=dtype_tuples)

        for name, value in values.items():
            array[name] = value

        return array
    else:
        return None
