"""
CaseReader factory function.
"""
from openmdao.recorders.sqlite_reader import SqliteCaseReader


def CaseReader(filename, metadata_filename=None, pre_load=True):
    """
    Return a CaseReader for the given file.

    Parameters
    ----------
    filename : str
        A path to the recorded file.
        Currently only sqlite database files recorded via SqliteRecorder are supported.
    metadata_filename : str
        For separate metadata from parallel runs, the metadata database filename.
    pre_load : bool
        If True, load all the data into memory during initialization.

    Returns
    -------
    reader : BaseCaseReader
        An instance of a CaseReader.
    """
    return SqliteCaseReader(filename, metadata_filename, pre_load)
