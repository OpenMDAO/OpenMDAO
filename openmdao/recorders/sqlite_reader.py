"""
Definition of the SqliteCaseReader.
"""
from __future__ import print_function, absolute_import

from openmdao.recorders.base_case_reader import BaseCaseReader
from openmdao.recorders.case import Case
from openmdao.utils.record_util import is_valid_sqlite3_db

import sqlite3


class SqliteCaseReader(BaseCaseReader):
    """
    A CaseReader specific to files created with SqliteRecorder.

    Parameters
    ----------
    filename : str
        The path to the filename containing the recorded data.
    """

    def __init__(self, filename):
        """
        Initialize.
        """
        super(SqliteCaseReader, self).__init__(filename)

        if filename is not None:
            if not is_valid_sqlite3_db(filename):
                raise IOError('File does not contain a valid '
                              'sqlite database ({0})'.format(filename))

        # with SqliteDict(self.filename, 'metadata', flag='r') as db:
        #     self.format_version = db.get('format_version', None)

        # TODO_RECORDERS - need to actually read this in
        self.format_version = 1

        self._load()

        self.num_cases = len(self._case_keys)

    def _load(self):
        """
        The initial load of data from the sqlite database file.

        Load the metadata from the sqlite file, populating the
        `format_version`, `parameters`, and `unknowns` attributes of this
        CaseReader.

        The `iterations` table is read to load the keys which identify
        the individual cases/iterations from the recorded file.

        Parameters
        ----------
        format_version : int
            The version of the format assumed when loading the file.
        """
        if self.format_version in (1,):
            # Read the metadata and save it in the reader
            # with SqliteDict(self.filename, 'metadata', flag='r') as db:
            #     self._parameters = db.get('Parameters', None)
            #     self._unknowns = db.get('Unknowns', None)

            # Store the identifier for each iteration in _case_keys
            # with SqliteDict(self.filename, 'iterations', flag='r') as db:
            #     self._case_keys = tuple(db.keys())

            con = sqlite3.connect(self.filename, detect_types=sqlite3.PARSE_DECLTYPES)
            cur = con.cursor()
            cur.execute("SELECT iteration_coordinate FROM driver_iterations")
            rows = cur.fetchall()

            self._case_keys = [coord[0] for coord in rows]
            print(self._case_keys)
            # returns this [(u'rank0:SLSQP|1',), (u'rank0:SLSQP|2',),
            # (u'rank0:SLSQP|3',), (u'rank0:SLSQP|4',)]

        else:
            raise ValueError('SQliteCaseReader encountered an unhandled '
                             'format version: {0}'.format(self.format_version))

    def get_case(self, case_id):
        """
        Get a case from the database.

        Parameters
        ----------
        case_id : int or str
            The integer index or string-identifier of the case to be retrieved.

        Returns
        -------
            An instance of Case populated with data from the
            specified case/iteration.
        """
        if isinstance(case_id, int):
            # If case_id is an integer, assume the user
            # wants a case as an index
            _case_id = self._case_keys[case_id]
        else:
            # Otherwise assume we were given the case string identifier
            _case_id = case_id

        # Initialize the Case object from the iterations data
        with SqliteDict(self.filename, 'iterations', flag='r') as iter_db:
            case = Case(self.filename, _case_id, iter_db[_case_id])

        con = sqlite3.connect(self.filename, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()
        cur.execute("SELECT * FROM driver_iterations WHERE "
                    "iteration_coordinate=:iteration_coordinate",
                    {"iteration_coordinate": _case_id})

        rows = cur.fetchall()

        # Set the derivs data for the case if available
        # with SqliteDict(self.filename, 'derivs', flag='r') as derivs_db:
        #     # If derivs weren't recorded then don't bother sending them
        #     # to the Case.
        #     if len(derivs_db) == 0:
        #         pass
        #     else:
        #         if _case_id in derivs_db:
        #             case._derivs = derivs_db[_case_id].get('Derivatives', None)

        return case
