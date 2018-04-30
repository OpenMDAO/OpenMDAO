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

from openmdao.utils.record_util import values_to_array, check_path
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.core.driver import Driver
from openmdao.core.system import System


def array_to_blob(array):
    """
    Make numpy array in to BLOB type.

    Convert a numpy array to something that can be written
    to a BLOB field in sqlite.

    TODO: move this to a util file?

    Parameters
    ----------
    array : array
        The array that will be converted to a blob.

    Returns
    -------
    blob :
        The blob created from the array.

    """
    out = io.BytesIO()
    np.save(out, array)
    out.seek(0)
    return sqlite3.Binary(out.read())


def blob_to_array(blob):
    """
    Convert sqlite BLOB to numpy array.

    TODO: move this to a util file?

    Parameters
    ----------
    blob : blob
        The blob that will be converted to an array.

    Returns
    -------
    array :
        The array created from the blob.
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
    con : sqlite connection object
        Connection to the sqlite3 database.
    cursor : sqlite cursor object
        Sqlite3 system cursor via the con.
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    _abs2meta : {'name': {}}
        Dictionary mapping absolute variable names to their metadata including units,
        bounds, and scaling.
    _open_close_sqlite : bool
        If True, open, write, and close the sqlite file. Needed for when running under MPI.
    _pickle_version : int
        The pickle protocol version to use when pickling metadata.
    _all_procs : bool
        If True, write to a separate sqlite file on each processor.
    """

    def __init__(self, filepath, append=False, pickle_version=2, all_procs=False):
        """
        Initialize the SqliteRecorder.

        Parameters
        ----------
        filepath : str
            Path to the recorder file.
        append : bool
            Optional. If True, append to an existing case recorder file.
        pickle_version : int
            Optional. The pickle protocol version to use when pickling metadata.
        all_procs : bool
            If True, run on all procs and write to a separate file on each processor.
        """
        super(SqliteRecorder, self).__init__()

        if MPI and MPI.COMM_WORLD.rank > 0 and not all_procs:
            self._open_close_sqlite = False
        else:
            self._open_close_sqlite = True

        self.model_viewer_data = None
        self._abs2prom = {'input': {}, 'output': {}}
        self._prom2abs = {'input': {}, 'output': {}}
        self._abs2meta = {}
        self._pickle_version = pickle_version
        self._all_procs = all_procs

        if all_procs:
            self._parallel = True
            filepath = '%s_%d' % (filepath, MPI.COMM_WORLD.rank)

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
                self.cursor.execute("CREATE TABLE metadata( format_version INT, "
                                    "abs2prom BLOB, prom2abs BLOB, abs2meta BLOB)")
                self.cursor.execute("INSERT INTO metadata(format_version, abs2prom, "
                                    "prom2abs) VALUES(?,?,?)",
                                    (format_version, None, None))

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
                self.cursor.execute("CREATE TABLE system_metadata(id TEXT PRIMARY KEY, "
                                    "scaling_factors BLOB, component_metadata BLOB)")
                self.cursor.execute("CREATE TABLE solver_metadata(id TEXT PRIMARY KEY, "
                                    "solver_options BLOB, solver_class TEXT)")

    def startup(self, recording_requester):
        """
        Prepare for a new run and create/update the abs2prom and prom2abs variables.

        Parameters
        ----------
        recording_requester : object
            Object to which this recorder is attached.
        """
        super(SqliteRecorder, self).startup(recording_requester)

        # grab the system
        if isinstance(recording_requester, Driver):
            system = recording_requester._problem.model
        elif isinstance(recording_requester, System):
            system = recording_requester
        else:
            system = recording_requester._system

        # merge current abs2prom and prom2abs with this system's version
        for io in ['input', 'output']:
            for v in system._var_abs2prom[io]:
                self._abs2prom[io][v] = system._var_abs2prom[io][v]
            for v in system._var_allprocs_prom2abs_list[io]:
                if v not in self._prom2abs[io]:
                    self._prom2abs[io][v] = system._var_allprocs_prom2abs_list[io][v]
                else:
                    self._prom2abs[io][v] = list(set(self._prom2abs[io][v]) |
                                                 set(system._var_allprocs_prom2abs_list[io][v]))

        # grab all of the units
        states = system._list_states_allprocs()
        for name in system._var_allprocs_abs2meta:
            self._abs2meta[name] = system._var_allprocs_abs2meta[name].copy()
            self._abs2meta[name]['type'] = 'Explicit'
            if name in states:
                self._abs2meta[name]['type'] = 'Implicit'

        # store the updated abs2prom and prom2abs
        abs2prom = pickle.dumps(self._abs2prom)
        prom2abs = pickle.dumps(self._prom2abs)
        abs2meta = pickle.dumps(self._abs2meta)
        if self._open_close_sqlite:
            with self.con:
                self.con.execute("UPDATE metadata SET abs2prom=?, prom2abs=?, abs2meta=?",
                                 (abs2prom, prom2abs, abs2meta))

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

        if MPI is None or MPI.COMM_WORLD.rank == 0 or self._all_procs:
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
        recording_requester : System
            System in need of recording.
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
        recording_requester : Solver
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
        recording_requester : Driver
            The Driver that would like to record its metadata.
        """
        driver_class = type(recording_requester).__name__
        model_viewer_data = pickle.dumps(recording_requester._model_viewer_data,
                                         self._pickle_version)
        with self.con:
            self.con.execute("INSERT INTO driver_metadata(id, model_viewer_data) VALUES(?,?)",
                             (driver_class, sqlite3.Binary(model_viewer_data)))

    def record_metadata_system(self, recording_requester):
        """
        Record system metadata.

        Parameters
        ----------
        recording_requester : System
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
        scaling_factors = pickle.dumps(scaling_vecs, self._pickle_version)

        # create a copy of the system's metadata excluding what is in 'metadata_excludes'
        user_metadata = OptionsDictionary()
        excludes = recording_requester.recording_options['metadata_excludes']
        for key in recording_requester.metadata._dict:
            if check_path(key, [], excludes, True):
                user_metadata._dict[key] = recording_requester.metadata._dict[key]
        user_metadata._read_only = recording_requester.metadata._read_only
        pickled_metadata = pickle.dumps(user_metadata, self._pickle_version)

        path = recording_requester.pathname
        if not path:
            path = 'root'
        with self.con:
            self.con.execute("INSERT INTO system_metadata(id, scaling_factors, component_metadata) \
                              VALUES(?,?, ?)",
                             (path, sqlite3.Binary(scaling_factors),
                              sqlite3.Binary(pickled_metadata)))

    def record_metadata_solver(self, recording_requester):
        """
        Record solver metadata.

        Parameters
        ----------
        recording_requester : Solver
            The Solver that would like to record its metadata.
        """
        path = recording_requester._system.pathname
        solver_class = type(recording_requester).__name__
        if not path:
            path = 'root'
        id = "{}.{}".format(path, solver_class)

        solver_options = pickle.dumps(recording_requester.options, self._pickle_version)

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
