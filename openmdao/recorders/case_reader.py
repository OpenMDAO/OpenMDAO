"""
CaseReader factory function.
"""
from openmdao.recorders.sqlite_reader_new import SqliteCaseReader


def CaseReader(filename):
    """
    Return a CaseReader for the given file.

    Parameters
    ----------
    filename : str
        A path to the recorded file.  The file should have been recorded using
        either the SqliteRecorder or the HDF5Recorder.

    Returns
    -------
    reader : BaseCaseReader
        An instance of a SqliteCaseReader that is reading filename.
    """
    reader = SqliteCaseReader(filename)
    return reader
