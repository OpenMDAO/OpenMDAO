"""
Definition of the SqliteCaseReader.
"""
from __future__ import print_function, absolute_import

import sqlite3

from openmdao.recorders.base_case_reader import BaseCaseReader
from openmdao.recorders.case import Case
from openmdao.recorders.sqlite_recorder import blob_to_array
from openmdao.utils.record_util import is_valid_sqlite3_db


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

        with sqlite3.connect(self.filename) as con:
            cur = con.cursor()
            cur.execute("SELECT format_version FROM metadata")
            row = cur.fetchone()
            self.format_version = row[0]
        con.close()

        self._load()

        self.num_cases = len(self._case_keys)

    def _load(self):
        """
        Load data from the sqlite database file.

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
            with sqlite3.connect(self.filename) as con:
                cur = con.cursor()
                cur.execute("SELECT iteration_coordinate FROM driver_iterations")
                rows = cur.fetchall()
            con.close()

            self._case_keys = [coord[0] for coord in rows]
            # returns something like this [(u'rank0:SLSQP|1',), (u'rank0:SLSQP|2',),
            #          (u'rank0:SLSQP|3',), (u'rank0:SLSQP|4',)]

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
            iteration_coordinate = self._case_keys[case_id]  # handles negative indices for example
        else:
            # Otherwise assume we were given the case string identifier
            iteration_coordinate = case_id

        with sqlite3.connect(self.filename) as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM driver_iterations WHERE "
                        "iteration_coordinate=:iteration_coordinate",
                        {"iteration_coordinate": iteration_coordinate})
            # Initialize the Case object from the iterations data
            row = cur.fetchone()
        con.close()

        counter, iteration_coordinate, timestamp, success, msg, desvars_blob, responses_blob, \
            objectives_blob, constraints_blob = row

        desvars_array = blob_to_array(desvars_blob)
        responses_array = blob_to_array(responses_blob)
        objectives_array = blob_to_array(objectives_blob)
        constraints_array = blob_to_array(constraints_blob)

        case = Case(self.filename, counter, iteration_coordinate, timestamp, success, msg,
                    desvars_array, responses_array, objectives_array, constraints_array)

        # returns something like this
        # [(1, u'rank0:Driver|1', 1491860346.232551, 1, u'', array([([5.0, 2.0],)],
        #     dtype=[('design_var.pz.z', '<f8', (2,))]))]

        return case
