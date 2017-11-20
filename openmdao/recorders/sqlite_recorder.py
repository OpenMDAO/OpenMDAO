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

from openmdao.utils.record_util import values_to_array


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
            self.con = sqlite3.connect(filepath)
            with self.con:
                self.cursor = self.con.cursor()
                self.cursor.execute("CREATE TABLE metadata( format_version INT)")
                self.cursor.execute("INSERT INTO metadata(format_version) VALUES(?)",
                                    (format_version,))

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

    def record_iteration_driver(self, recording_requester, data, metadata):
        """
        Record data and metadata from a Driver.

        Parameters
        ----------
        recording_requester : object
            Driver in need of recording.
        data : dict
            Dictionary containing desvars, objectives, constraints, responses, and System vars.
        metadata : dict
            Dictionary containing execution metadata.
        """
        desvars = data['des']
        responses = data['res']
        objectives = data['obj']
        constraints = data['con']
        sysvars = data['sys']

        # Need to gather up the values from across the ranks, if MPI
        # if MPI:
        #     desvars = self._gather_vars(root, desvars)
        #     responses = self._gather_vars(root, responses)
        #     objectives = self._gather_vars(root, objectives)
        #     constraints = self._gather_vars(root, constraints)
        #     sysvars = self._gather_vars(root, sysvars)

        if MPI is None or MPI.COMM_WORLD.rank == 0:
            desvars_array = values_to_array(desvars)
            responses_array = values_to_array(responses)
            objectives_array = values_to_array(objectives)
            constraints_array = values_to_array(constraints)
            sysvars_array = values_to_array(sysvars)

            desvars_blob = array_to_blob(desvars_array)
            responses_blob = array_to_blob(responses_array)
            objectives_blob = array_to_blob(objectives_array)
            constraints_blob = array_to_blob(constraints_array)
            sysvars_blob = array_to_blob(sysvars_array)

            with self.con:
                self.cursor.execute("INSERT INTO driver_iterations(counter, iteration_coordinate, "
                                    "timestamp, success, msg, desvars , responses , objectives , "
                                    "constraints, sysincludes ) VALUES(?,?,?,?,?,?,?,?,?,?)",
                                    (self._counter, self._iteration_coordinate,
                                     metadata['timestamp'], metadata['success'],
                                     metadata['msg'], desvars_blob,
                                     responses_blob, objectives_blob,
                                     constraints_blob, sysvars_blob))
                self.con.execute("INSERT INTO global_iterations(record_type, rowid) VALUES(?,?)",
                                 ('driver', self.cursor.lastrowid))

    def record_iteration_system(self, recording_requester, data, metadata):
        """
        Record data and metadata from a System.

        Parameters
        ----------
        recording_requester : object
            Driver in need of recording.
        data : dict
            Dictionary containing inputs, outputs, and residuals.
        metadata : dict
            Dictionary containing execution metadata.
        """
        inputs = data['i']
        outputs = data['o']
        residuals = data['r']

        inputs_array = values_to_array(inputs)
        outputs_array = values_to_array(outputs)
        residuals_array = values_to_array(residuals)

        inputs_blob = array_to_blob(inputs_array)
        outputs_blob = array_to_blob(outputs_array)
        residuals_blob = array_to_blob(residuals_array)

        with self.con:
            self.cursor.execute("INSERT INTO system_iterations(counter, iteration_coordinate, "
                                "timestamp, success, msg, inputs , outputs , residuals ) "
                                "VALUES(?,?,?,?,?,?,?,?)",
                                (self._counter, self._iteration_coordinate,
                                 metadata['timestamp'], metadata['success'],
                                 metadata['msg'], inputs_blob,
                                 outputs_blob, residuals_blob))
            self.cursor.execute("INSERT INTO global_iterations(record_type, rowid) VALUES(?,?)",
                                ('system', self.cursor.lastrowid))

    def record_iteration_solver(self, recording_requester, data, metadata):
        """
        Record data and metadata from a Solver.

        Parameters
        ----------
        recording_requester : object
            Solver in need of recording.
        data : dict
            Dictionary containing outputs, residuals, and errors.
        metadata : dict
            Dictionary containing execution metadata.
        """
        abs = data['abs']
        rel = data['rel']
        outputs = data['o']
        residuals = data['r']

        outputs_array = values_to_array(outputs)
        residuals_array = values_to_array(residuals)

        outputs_blob = array_to_blob(outputs_array)
        residuals_blob = array_to_blob(residuals_array)

        with self.con:
            self.cursor.execute("INSERT INTO solver_iterations(counter, iteration_coordinate, "
                                "timestamp, success, msg, abs_err, rel_err, solver_output, "
                                "solver_residuals) VALUES(?,?,?,?,?,?,?,?,?)",
                                (self._counter, self._iteration_coordinate,
                                 metadata['timestamp'],
                                 metadata['success'], metadata['msg'],
                                 abs, rel,
                                 outputs_blob, residuals_blob))

            self.cursor.execute("INSERT INTO global_iterations(record_type, rowid) VALUES(?,?)",
                                ('solver', self.cursor.lastrowid))

    def record_metadata_driver(self, recording_requester):
        """
        Record driver metadata.

        Parameters
        ----------
        recording_requester: <Driver>
            The Driver that would like to record its metadata.
        """
        driver_class = type(recording_requester).__name__
        model_viewer_data = pickle.dumps(recording_requester._model_viewer_data,
                                         pickle.HIGHEST_PROTOCOL)

        with self.con:
            self.con.execute("INSERT INTO driver_metadata(id, model_viewer_data) "
                             "VALUES(?,?)", (driver_class, sqlite3.Binary(model_viewer_data)))

    def record_metadata_system(self, recording_requester):
        """
        Record system metadata.

        Parameters
        ----------
        recording_requester: <System>
            The System that would like to record its metadata.
        """
        # Cannot handle PETScVector yet
        from openmdao.api import PETScVector
        if PETScVector and isinstance(recording_requester._outputs, PETScVector):
            return  # Cannot handle PETScVector yet

        # collect scaling arrays
        scaling_vecs = {}
        for kind, odict in iteritems(recording_requester._vectors):
            scaling_vecs[kind] = scaling = {}
            for vecname, vec in iteritems(odict):
                scaling[vecname] = vec._scaling
        scaling_factors = pickle.dumps(scaling_vecs,
                                       pickle.HIGHEST_PROTOCOL)

        path = recording_requester.pathname
        if not path:
            path = 'root'
        with self.con:
            self.con.execute("INSERT INTO system_metadata(id, scaling_factors) VALUES(?,?)",
                             (path, sqlite3.Binary(scaling_factors)))

    def record_metadata_solver(self, recording_requester):
        """
        Record solver metadata.

        Parameters
        ----------
        recording_requester: <Solver>
            The Solver that would like to record its metadata.
        """
        path = recording_requester._system.pathname
        solver_class = type(recording_requester).__name__
        if not path:
            path = 'root'
        id = "{}.{}".format(path, solver_class)

        solver_options = pickle.dumps(recording_requester.options,
                                      pickle.HIGHEST_PROTOCOL)
        with self.con:
            self.con.execute(
                "INSERT INTO solver_metadata(id, solver_options, solver_class) "
                "VALUES(?,?,?)", (id, sqlite3.Binary(solver_options), solver_class))

    def close(self):
        """
        Close `out`.
        """
        if self._open_close_sqlite:
            self.con.close()
