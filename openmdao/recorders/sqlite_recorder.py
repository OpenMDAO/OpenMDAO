"""
Class definition for SqliteRecorder, which provides dictionary backed by SQLite.
"""

import io
import os
import sqlite3

import numpy as np
from six import iteritems
from six.moves import cPickle as pickle

from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.utils.mpi import MPI


def array_to_blob(array):
    """
    Make numpy array in to BLOB type.

    Convert a numpy array to something that can be written
    to a BLOB field in sqlite

    TODO: move this to a util file?
    """
    out = io.BytesIO()
    np.save(out, array)
    out.seek(0)
    return sqlite3.Binary(out.read())


def blob_to_array(blob):
    """
    Convert sqlite BLOB to numpy array.

    TODO: move this to a util file?
    """
    out = io.BytesIO(blob)
    out.seek(0)
    return np.load(out)


format_version = 1


class SqliteRecorder(BaseRecorder):
    """
    Recorder that saves cases in a sqlite db.

    Attributes
    ----------
    model_viewer_data : dict
        Dict that holds the data needed to generate N2 diagram.
    con
        Connection to the sqlite3 database.
    cursor
        Sqlite3 system cursor via the con.
    """

    def __init__(self, filepath, append=False):
        """
        Initialize the SqliteRecorder.

        Parameters
        ----------
        filepath: str
            Path to the recorder file.
        append : bool
            Optional. If True, append to an existing case recorder file.
        """
        super(SqliteRecorder, self).__init__()

        if MPI and MPI.COMM_WORLD.rank > 0:
            self._open_close_sqlite = False
        else:
            self._open_close_sqlite = True

        self.model_viewer_data = None

        if append:
            raise NotImplementedError("Append feature not implemented for SqliteRecorder")

        if self._open_close_sqlite and not append:
            try:
                os.remove(filepath)
            except OSError:
                pass
            # isolation_level=None causes autocommit
            self.con = sqlite3.connect(filepath, isolation_level=None)

            self.cursor = self.con.cursor()

            self.cursor.execute("CREATE TABLE metadata( format_version INT)")
            self.cursor.execute("INSERT INTO metadata(format_version) VALUES(?)", (format_version,))

            # used to keep track of the order of the case records across all three tables
            self.cursor.execute("CREATE TABLE global_iterations(id INTEGER PRIMARY KEY, "
                                "record_type TEXT, rowid INT)")
            self.cursor.execute("CREATE TABLE driver_iterations(id INTEGER PRIMARY KEY, "
                                "counter INT,iteration_coordinate TEXT, timestamp REAL, "
                                "success INT, msg TEXT, desvars BLOB, responses BLOB, "
                                "objectives BLOB, constraints BLOB, sysincludes BLOB)")
            self.cursor.execute("CREATE TABLE system_iterations(id INTEGER PRIMARY KEY, "
                                "counter INT, iteration_coordinate TEXT,  timestamp REAL, "
                                "success INT, msg TEXT, inputs BLOB, outputs BLOB, "
                                "residuals BLOB)")
            self.cursor.execute("CREATE TABLE solver_iterations(id INTEGER PRIMARY KEY, "
                                "counter INT, iteration_coordinate TEXT, timestamp REAL, "
                                "success INT, msg TEXT, abs_err REAL, rel_err REAL, "
                                "solver_output BLOB, solver_residuals BLOB)")

            self.cursor.execute("CREATE TABLE driver_metadata(id TEXT PRIMARY KEY, "
                                "model_viewer_data BLOB)")
            self.cursor.execute("CREATE TABLE system_metadata(id TEXT PRIMARY KEY,"
                                " scaling_factors BLOB)")
            self.cursor.execute("CREATE TABLE solver_metadata(id TEXT PRIMARY KEY, "
                                "solver_options BLOB, solver_class TEXT)")

    def record_iteration_driver_passing_vars(self, object_requesting_recording, desvars,
                                             responses, objectives, constraints, sysvars, metadata):
        """
        Record an iteration of a driver with the variables passed in.

        Parameters
        ----------
        object_requesting_recording: <Driver>
            The Driver object that wants to record an iteration.
        desvars : dict
            Dictionary containing design variables.
        responses : dict
            Dictionary containing response variables.
        objectives : dict
            Dictionary containing objective variables.
        constraints : dict
            Dictionary containing constraint variables.
        metadata : dict
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """
        # make a nested numpy named array using the example
        #   http://stackoverflow.com/questions/19201868/how-to-set-dtype-for-nested-numpy-ndarray
        # e.g.
        # table = np.array(data, dtype=[('instrument', 'S32'),
        #                        ('filter', 'S64'),
        #                        ('response', [('linenumber', 'i'),
        #                                      ('wavelength', 'f'),
        #                                      ('throughput', 'f')], (2,))
        #                       ])

        super(SqliteRecorder, self).record_iteration_driver_passing_vars(
            object_requesting_recording, desvars, responses,
            objectives, constraints, sysvars, metadata)

        # Just an example of the syntax for creating a numpy structured array
        # arr = np.zeros((1,), dtype=[('dv_x','(5,)f8'),('dv_y','(10,)f8')])

        # This returns a dict of names and values. Use this to build up the tuples of
        # used for the dtypes in the creation of the numpy structured array
        # we want to write to sqlite
        if self._desvars_values:
            dtype_tuples = []
            for name, value in iteritems(self._desvars_values):
                tple = (name, '{}f8'.format(value.shape))
                dtype_tuples.append(tple)

            desvars_array = np.zeros((1,), dtype=dtype_tuples)

            for name, value in iteritems(self._desvars_values):
                desvars_array[name] = value
        else:
            desvars_array = None

        # Cannot handle responses yet
        # if self._responses_values:
        #     dtype_tuples = []
        #     for name, value in iteritems(self._responses_values):
        #         tple = (name, '{}f8'.format(value.shape))
        #         dtype_tuples.append(tple)
        #
        #     responses_array = np.zeros((1,), dtype=dtype_tuples)
        #
        #     for name, value in iteritems(self._responses_values):
        #         responses_array[name] = value
        # else:
        #     responses_array = None
        responses_array = None

        if self._objectives_values:
            dtype_tuples = []
            for name, value in iteritems(self._objectives_values):
                tple = (name, '{}f8'.format(value.shape))
                dtype_tuples.append(tple)

            objectives_array = np.zeros((1,), dtype=dtype_tuples)

            for name, value in iteritems(self._objectives_values):
                objectives_array[name] = value
        else:
            objectives_array = None

        if self._constraints_values:
            dtype_tuples = []
            for name, value in iteritems(self._constraints_values):
                tple = (name, '{}f8'.format(value.shape))
                dtype_tuples.append(tple)

            constraints_array = np.zeros((1,), dtype=dtype_tuples)

            for name, value in iteritems(self._constraints_values):
                constraints_array[name] = value
        else:
            constraints_array = None

        if self._sysincludes_values:
            dtype_tuples = []
            for name, value in iteritems(self._sysincludes_values):
                tple = (name, '{}f8'.format(value.shape))
                dtype_tuples.append(tple)

            sysincludes_array = np.zeros((1,), dtype=dtype_tuples)

            for name, value in iteritems(self._sysincludes_values):
                sysincludes_array[name] = value
        else:
            sysincludes_array = None

        desvars_blob = array_to_blob(desvars_array)
        responses_blob = array_to_blob(responses_array)
        objectives_blob = array_to_blob(objectives_array)
        constraints_blob = array_to_blob(constraints_array)
        sysincludes_blob = array_to_blob(sysincludes_array)

        self.cursor.execute("INSERT INTO driver_iterations(counter, iteration_coordinate, "
                            "timestamp, success, msg, desvars , responses , objectives , "
                            "constraints, sysincludes ) VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (self._counter, self._iteration_coordinate,
                             metadata['timestamp'], metadata['success'],
                             metadata['msg'], desvars_blob,
                             responses_blob, objectives_blob,
                             constraints_blob, sysincludes_blob))
        self.con.execute("INSERT INTO global_iterations(record_type, rowid) VALUES(?,?)",
                         ('driver', self.cursor.lastrowid))

    def record_iteration_system(self, object_requesting_recording, metadata):
        """
        Record an iteration using system options.

        Parameters
        ----------
        object_requesting_recording: <System>
            The System object that wants to record an iteration.
        metadata : dict
            Dictionary containing execution metadata (e.g. iteration coordinate).
        method : str
            The method that called record_iteration. One of '_apply_linear', '_solve_linear',
            '_apply_nonlinear,' '_solve_nonlinear'. Behavior varies based on from which function
            record_iteration was called.
        """
        super(SqliteRecorder, self).record_iteration_system(object_requesting_recording, metadata)

        # Inputs
        if self._inputs:
            dtype_tuples = []
            for name, value in iteritems(self._inputs):
                tple = (name, '{}f8'.format(value.shape))
                dtype_tuples.append(tple)

            inputs_array = np.zeros((1,), dtype=dtype_tuples)
            for name, value in iteritems(self._inputs):
                inputs_array[name] = value
        else:
            inputs_array = None

        # Outputs
        if self._outputs:
            dtype_tuples = []
            for name, value in iteritems(self._outputs):
                tple = (name, '{}f8'.format(value.shape))
                dtype_tuples.append(tple)

            outputs_array = np.zeros((1,), dtype=dtype_tuples)
            for name, value in iteritems(self._outputs):
                outputs_array[name] = value
        else:
            outputs_array = None

        # Residuals
        if self._resids:
            dtype_tuples = []
            for name, value in iteritems(self._resids):
                tple = (name, '{}f8'.format(value.shape))
                dtype_tuples.append(tple)

            residuals_array = np.zeros((1,), dtype=dtype_tuples)
            for name, value in iteritems(self._resids):
                residuals_array[name] = value
        else:
            residuals_array = None

        inputs_blob = array_to_blob(inputs_array)
        outputs_blob = array_to_blob(outputs_array)
        residuals_blob = array_to_blob(residuals_array)

        self.cursor.execute("INSERT INTO system_iterations(counter, iteration_coordinate, "
                            "timestamp, success, msg, inputs , outputs , residuals ) "
                            "VALUES(?,?,?,?,?,?,?,?)",
                            (self._counter, self._iteration_coordinate,
                             metadata['timestamp'], metadata['success'],
                             metadata['msg'], inputs_blob,
                             outputs_blob, residuals_blob))
        self.cursor.execute("INSERT INTO global_iterations(record_type, rowid) VALUES(?,?)",
                            ('system', self.cursor.lastrowid))

    def record_iteration_solver(self, object_requesting_recording, metadata, **kwargs):
        """
        Record an iteration using solver options.

        Parameters
        ----------
        object_requesting_recording: <Solver>
            The Solver object that wants to record an iteration.
        metadata : dict
            Dictionary containing execution metadata (e.g. iteration coordinate).
        absolute : float
            The absolute error of the Solver requesting recording. It is not cached in
            the Solver object, so we pass it in here.
        relative : float
            The relative error of the Solver requesting recording. It is not cached in
            the Solver object, so we pass it in here.
        """
        super(SqliteRecorder, self).record_iteration_solver(object_requesting_recording,
                                                            metadata, **kwargs)

        if self._outputs:
            dtype_tuples = []
            for name, value in iteritems(self._outputs):
                tple = (name, '{}f8'.format(value.shape))
                dtype_tuples.append(tple)

            outputs_array = np.zeros((1,), dtype=dtype_tuples)

            for name, value in iteritems(self._outputs):
                outputs_array[name] = value
        else:
            outputs_array = None

        if self._resids:
            dtype_tuples = []
            for name, value in iteritems(self._resids):
                tple = (name, '{}f8'.format(value.shape))
                dtype_tuples.append(tple)

            residuals_array = np.zeros((1,), dtype=dtype_tuples)
            for name, value in iteritems(self._resids):
                residuals_array[name] = value
        else:
            residuals_array = None

        outputs_blob = array_to_blob(outputs_array)
        residuals_blob = array_to_blob(residuals_array)

        self.cursor.execute("INSERT INTO solver_iterations(counter, iteration_coordinate, "
                            "timestamp, success, msg, abs_err, rel_err, solver_output, "
                            "solver_residuals) VALUES(?,?,?,?,?,?,?,?,?)",
                            (self._counter, self._iteration_coordinate,
                             metadata['timestamp'],
                             metadata['success'], metadata['msg'],
                             self._abs_error, self._rel_error,
                             outputs_blob, residuals_blob))

        self.cursor.execute("INSERT INTO global_iterations(record_type, rowid) VALUES(?,?)",
                            ('solver', self.cursor.lastrowid))

    def record_metadata_driver(self, object_requesting_recording):
        """
        Record driver metadata.

        Parameters
        ----------
        object_requesting_recording: <Driver>
            The Driver that would like to record its metadata.
        """
        driver_class = type(object_requesting_recording).__name__
        model_viewer_data = pickle.dumps(object_requesting_recording._model_viewer_data,
                                         pickle.HIGHEST_PROTOCOL)

        self.con.execute("INSERT INTO driver_metadata(id, model_viewer_data) "
                         "VALUES(?,?)", (driver_class, sqlite3.Binary(model_viewer_data)))

    def record_metadata_system(self, object_requesting_recording):
        """
        Record system metadata.

        Parameters
        ----------
        object_requesting_recording: <System>
            The System that would like to record its metadata.
        """
        # Cannot handle PETScVector yet
        from openmdao.api import PETScVector
        if isinstance(object_requesting_recording._scaling_vecs[('input', 'norm0')]['linear'],
                      PETScVector):
            return  # Cannot handle PETScVector yet

        scaling_factors = pickle.dumps(object_requesting_recording._scaling_vecs,
                                       pickle.HIGHEST_PROTOCOL)

        path = object_requesting_recording.pathname
        if not path:
            path = 'root'
        self.con.execute("INSERT INTO system_metadata(id, scaling_factors) VALUES(?,?)",
                         (path, sqlite3.Binary(scaling_factors)))

    def record_metadata_solver(self, object_requesting_recording):
        """
        Record solver metadata.

        Parameters
        ----------
        object_requesting_recording: <Solver>
            The Solver that would like to record its metadata.
        """
        path = object_requesting_recording._system.pathname
        solver_class = type(object_requesting_recording).__name__
        if not path:
            path = 'root'
        id = "{}.{}".format(path, solver_class)

        solver_options = pickle.dumps(object_requesting_recording.options,
                                      pickle.HIGHEST_PROTOCOL)
        self.con.execute(
            "INSERT INTO solver_metadata(id, solver_options, solver_class) "
            "VALUES(?,?,?)", (id, sqlite3.Binary(solver_options), solver_class))

    def close(self):
        """
        Close `out`.
        """
        if self._open_close_sqlite:
            self.con.close()
