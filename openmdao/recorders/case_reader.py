"""
CaseReader class definition.
"""
from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.recorders.sqlite_reader_new import SqliteCaseReaderNew
# from openmdao.recorders.hdf5_reader import HDF5CaseReader


def CaseReaderNew(filename):
    """
    Return a CaseReader for the given file.

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
        reader = SqliteCaseReaderNew(filename)
        return reader
    except IOError:
        # filename not a valid Sqlite database file
        raise IOError('Unable to load cases from file {0}'.format(filename))
