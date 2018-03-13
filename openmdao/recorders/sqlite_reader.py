"""
Definition of the SqliteCaseReader.
"""
from __future__ import print_function, absolute_import

import sqlite3

from openmdao.recorders.base_case_reader import BaseCaseReader
from openmdao.recorders.case import DriverCase, SystemCase, SolverCase
from openmdao.recorders.cases import BaseCases
from openmdao.recorders.sqlite_recorder import blob_to_array
from openmdao.utils.record_util import is_valid_sqlite3_db

from six import PY2, PY3

if PY2:
    import cPickle as pickle
if PY3:
    import pickle


class SqliteCaseReader(BaseCaseReader):
    """
    A CaseReader specific to files created with SqliteRecorder.

    Parameters
    ----------
    filename : str
        The path to the filename containing the recorded data.

    Attributes
    ----------
    format_version : int
        The version of the format assumed when loading the file.
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    """

    def __init__(self, filename):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The path to the filename containing the recorded data.
        """
        super(SqliteCaseReader, self).__init__(filename)

        if filename is not None:
            if not is_valid_sqlite3_db(filename):
                raise IOError('File does not contain a valid '
                              'sqlite database ({0})'.format(filename))

        with sqlite3.connect(self.filename) as con:
            cur = con.cursor()
            cur.execute("SELECT format_version, abs2prom, prom2abs FROM metadata")
            row = cur.fetchone()
            self.format_version = row[0]
            self._abs2prom = None
            self._prom2abs = None

            if PY2:
                self._abs2prom = pickle.loads(str(row[1])) if row[1] is not None else None
                self._prom2abs = pickle.loads(str(row[2])) if row[2] is not None else None
            if PY3:
                self._abs2prom = pickle.loads(row[1]) if row[1] is not None else None
                self._prom2abs = pickle.loads(row[2]) if row[2] is not None else None
        con.close()

        self._load()

    def _load(self):
        """
        Load data from the sqlite database file.

        Load the metadata from the sqlite file, populating the
        `format_version`, `parameters`, and `unknowns` attributes of this
        CaseReader.

        The `iterations` table is read to load the keys which identify
        the individual cases/iterations from the recorded file.
        """
        self.driver_cases = DriverCases(self.filename)
        self.system_cases = SystemCases(self.filename)
        self.solver_cases = SolverCases(self.filename)

        self.driver_cases._prom2abs = self._prom2abs
        self.system_cases._prom2abs = self._prom2abs
        self.solver_cases._prom2abs = self._prom2abs

        if self.format_version in (1,):
            with sqlite3.connect(self.filename) as con:

                # Read in iterations from Drivers, Systems, and Solvers
                cur = con.cursor()
                cur.execute("SELECT iteration_coordinate FROM driver_iterations")
                rows = cur.fetchall()
                self.driver_cases._case_keys = [coord[0] for coord in rows]
                self.driver_cases.num_cases = len(self.driver_cases._case_keys)

                cur.execute("SELECT iteration_coordinate FROM system_iterations")
                rows = cur.fetchall()
                self.system_cases._case_keys = [coord[0] for coord in rows]
                self.system_cases.num_cases = len(self.system_cases._case_keys)

                cur.execute("SELECT iteration_coordinate FROM solver_iterations")
                rows = cur.fetchall()
                self.solver_cases._case_keys = [coord[0] for coord in rows]
                self.solver_cases.num_cases = len(self.solver_cases._case_keys)

                # Read in metadata for Drivers, Systems, and Solvers
                cur.execute("SELECT model_viewer_data FROM driver_metadata")
                for row in cur:
                    if PY2:
                        self.driver_metadata = pickle.loads(str(row[0]))
                    if PY3:
                        self.driver_metadata = pickle.loads(row[0])

                cur.execute("SELECT id, scaling_factors FROM system_metadata")
                for row in cur:
                    id = row[0]
                    self.system_metadata[id] = {}

                    if PY2:
                        self.system_metadata[id]['scaling_factors'] = pickle.loads(str(row[1]))
                    if PY3:
                        self.system_metadata[id]['scaling_factors'] = pickle.loads(row[1])

                cur.execute("SELECT id, solver_options, solver_class FROM solver_metadata")
                for row in cur:
                    id = row[0]
                    if PY2:
                        solver_options = pickle.loads(str(row[1]))
                    if PY3:
                        solver_options = pickle.loads(row[1])
                    solver_class = row[2]
                    self.solver_metadata[id] = {
                        'solver_options': solver_options,
                        'solver_class': solver_class,
                    }
            con.close()
        else:
            raise ValueError('SQliteCaseReader encountered an unhandled '
                             'format version: {0}'.format(self.format_version))


class DriverCases(BaseCases):
    """
    Case specific to the entries that might be recorded in a Driver iteration.
    """

    def get_case(self, case_id):
        """
        Get a case from the database.

        Parameters
        ----------
        case_id : int or str
            The integer index or string-identifier of the case to be retrieved.

        Returns
        -------
            An instance of a Driver Case populated with data from the
            specified case/iteration.
        """
        iteration_coordinate = self.get_iteration_coordinate(case_id)

        with sqlite3.connect(self.filename) as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM driver_iterations WHERE "
                        "iteration_coordinate=:iteration_coordinate",
                        {"iteration_coordinate": iteration_coordinate})
            # Initialize the Case object from the iterations data
            row = cur.fetchone()
        con.close()

        idx, counter, iteration_coordinate, timestamp, success, msg, desvars_blob, responses_blob, \
            objectives_blob, constraints_blob, sysincludes_blob = row

        desvars_array = blob_to_array(desvars_blob)
        responses_array = blob_to_array(responses_blob)
        objectives_array = blob_to_array(objectives_blob)
        constraints_array = blob_to_array(constraints_blob)
        sysincludes_array = blob_to_array(sysincludes_blob)

        case = DriverCase(self.filename, counter, iteration_coordinate, timestamp, success, msg,
                          desvars_array, responses_array, objectives_array, constraints_array,
                          sysincludes_array, self._prom2abs)

        return case


class SystemCases(BaseCases):
    """
    Case specific to the entries that might be recorded in a System iteration.
    """

    def get_case(self, case_id):
        """
        Get a case from the database.

        Parameters
        ----------
        case_id : int or str
            The integer index or string-identifier of the case to be retrieved.

        Returns
        -------
            An instance of a System Case populated with data from the
            specified case/iteration.
        """
        iteration_coordinate = self.get_iteration_coordinate(case_id)

        with sqlite3.connect(self.filename) as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM system_iterations WHERE "
                        "iteration_coordinate=:iteration_coordinate",
                        {"iteration_coordinate": iteration_coordinate})
            # Initialize the Case object from the iterations data
            row = cur.fetchone()
        con.close()

        # inputs , outputs , residuals
        idx, counter, iteration_coordinate, timestamp, success, msg, inputs_blob,\
            outputs_blob, residuals_blob = row

        inputs_array = blob_to_array(inputs_blob)
        outputs_array = blob_to_array(outputs_blob)
        residuals_array = blob_to_array(residuals_blob)

        case = SystemCase(self.filename, counter, iteration_coordinate, timestamp, success, msg,
                          inputs_array, outputs_array, residuals_array, self._prom2abs)

        return case


class SolverCases(BaseCases):
    """
    Case specific to the entries that might be recorded in a Solver iteration.
    """

    def get_case(self, case_id):
        """
        Get a case from the database.

        Parameters
        ----------
        case_id : int or str
            The integer index or string-identifier of the case to be retrieved.

        Returns
        -------
            An instance of a solver Case populated with data from the
            specified case/iteration.
        """
        iteration_coordinate = self.get_iteration_coordinate(case_id)

        with sqlite3.connect(self.filename) as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM solver_iterations WHERE "
                        "iteration_coordinate=:iteration_coordinate",
                        {"iteration_coordinate": iteration_coordinate})
            # Initialize the Case object from the iterations data
            row = cur.fetchone()
        con.close()

        idx, counter, iteration_coordinate, timestamp, success, msg, abs_err, rel_err, \
            output_blob, residuals_blob = row

        output_array = blob_to_array(output_blob)
        residuals_array = blob_to_array(residuals_blob)

        case = SolverCase(self.filename, counter, iteration_coordinate, timestamp, success, msg,
                          abs_err, rel_err, output_array, residuals_array, self._prom2abs)

        return case
