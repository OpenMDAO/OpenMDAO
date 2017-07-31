"""
Utility functions related to recording or execution metadata.
"""
from six.moves import map, zip
import os


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


def format_iteration_coordinate(coord):
    """
    Format the iteration coordinate to a human-readable string.

    Parameters
    ----------
    coord : list
        List containing the iteration coordinate.

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

    return ':'.join(["rank%d" % coord[0], separator.join(iteration_coordinate)])


def is_valid_sqlite3_db(filename):
    """
    Return true if the given filename contains a valid SQLite3 database file.

    Parameters
    ----------
    filename : str
        The path to the file to be tested

    Returns
    -------
    bool :
        True if the filename specifies a valid SQlite3 database.

    """
    if not os.path.isfile(filename):
        return False
    if os.path.getsize(filename) < 100:
        # SQLite database file header is 100 bytes
        return False

    with open(filename, 'rb') as fd:
        header = fd.read(100)

    return header[:16] == b'SQLite format 3\x00'
