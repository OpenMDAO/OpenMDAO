from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.recorders.hdf5_reader import HDF5CaseReader


def CaseReader(filename):
    """ A factory function that returns a CaseReader for the given file.

    Parameters
    ----------
    filename : str
        A path to the recorded file.  The file should have been recorded using
        either the SqliteRecorder or the HDF5Recorder.

    Returns
    -------
    An instance of SqliteCaseReader or HDF5CaseReader, depending on the
    contents of the given file.
    """

    try:
        reader = SqliteCaseReader(filename)
        return reader
    except IOError:
        # filename not a valid Sqlite database file
        pass

    try:
        reader = HDF5CaseReader(filename)
        return reader
    except IOError:
        raise IOError('Unable to load cases from file {0}'.format(filename))
