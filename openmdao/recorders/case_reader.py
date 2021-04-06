"""
CaseReader factory function.
"""
from openmdao.recorders.sqlite_reader import SqliteCaseReader


def CaseReader(filename, pre_load=True, metadata_filename=None):
    """
    Return a CaseReader for the given file.

    Parameters
    ----------
    filename : str
        A path to the recorded file.
        Currently only sqlite database files recorded via SqliteRecorder are supported.
    pre_load : bool
        If True, load all the data into memory during initialization.
    metadata_filename : str
        For separate metadata from parallel runs, the metadata database filename.

    Returns
    -------
    reader : BaseCaseReader
        An instance of a CaseReader.
    """
    return SqliteCaseReader(filename, pre_load, metadata_filename)
