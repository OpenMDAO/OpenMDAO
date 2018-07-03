"""
Class definition for SqliteRecorder, which provides dictionary backed by SQLite.
"""

import io
import os
import sqlite3

import warnings
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
    connection : sqlite connection object
        Connection to the sqlite3 database.
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    _abs2meta : {'name': {}}
        Dictionary mapping absolute variable names to their metadata including units,
        bounds, and scaling.
    _pickle_version : int
        The pickle protocol version to use when pickling metadata.
    _filepath : str
        Path to the recorder file.
    _database_initialized : bool
        Flag indicating whether or not the database has been initialized.
    _record_on_proc : bool
        Flag indicating whether to record on this processor when running in parallel.
    """

    def __init__(self, filepath, append=False, pickle_version=2):
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
        """
        if append:
            raise NotImplementedError("Append feature not implemented for SqliteRecorder")

        self.connection = None
        self.model_viewer_data = None

        self._abs2prom = {'input': {}, 'output': {}}
        self._prom2abs = {'input': {}, 'output': {}}
        self._abs2meta = {}
        self._pickle_version = pickle_version
        self._filepath = filepath
        self._database_initialized = False

        # default to record on all procs when running in parallel
        self._record_on_proc = True

        super(SqliteRecorder, self).__init__()

    def _initialize_database(self):
        """
        Initialize the database.
        """
        if MPI:
            rank = MPI.COMM_WORLD.rank
            if self._parallel and self._record_on_proc:
                filepath = '%s_%d' % (self._filepath, rank)
                print("Note: SqliteRecorder is running on multiple processors. "
                      "Cases from rank %d are being written to %s." %
                      (rank, filepath))
            elif rank == 0:
                filepath = self._filepath
            else:
                filepath = None
        else:
            filepath = self._filepath

        if filepath:
            try:
                os.remove(filepath)
            except OSError:
                pass

            self.connection = sqlite3.connect(filepath)
            with self.connection as c:
                c.execute("CREATE TABLE metadata( format_version INT, "
                          "abs2prom BLOB, prom2abs BLOB, abs2meta BLOB)")
                c.execute("INSERT INTO metadata(format_version, abs2prom, prom2abs) "
                          "VALUES(?,?,?)", (format_version, None, None))

                # used to keep track of the order of the case records across all three tables
                c.execute("CREATE TABLE global_iterations(id INTEGER PRIMARY KEY, "
                          "record_type TEXT, rowid INT)")
                c.execute("CREATE TABLE driver_iterations(id INTEGER PRIMARY KEY, "
                          "counter INT,iteration_coordinate TEXT, timestamp REAL, "
                          "success INT, msg TEXT, inputs BLOB, outputs BLOB)")
                c.execute("CREATE INDEX driv_iter_ind on driver_iterations(iteration_coordinate)")
                c.execute("CREATE TABLE system_iterations(id INTEGER PRIMARY KEY, "
                          "counter INT, iteration_coordinate TEXT, timestamp REAL, "
                          "success INT, msg TEXT, inputs BLOB, outputs BLOB, residuals BLOB)")
                c.execute("CREATE INDEX sys_iter_ind on system_iterations(iteration_coordinate)")
                c.execute("CREATE TABLE solver_iterations(id INTEGER PRIMARY KEY, "
                          "counter INT, iteration_coordinate TEXT, timestamp REAL, "
                          "success INT, msg TEXT, abs_err REAL, rel_err REAL, "
                          "solver_inputs BLOB, solver_output BLOB, solver_residuals BLOB)")
                c.execute("CREATE INDEX solv_iter_ind on solver_iterations(iteration_coordinate)")
                c.execute("CREATE TABLE driver_metadata(id TEXT PRIMARY KEY, "
                          "model_viewer_data BLOB)")
                c.execute("CREATE TABLE system_metadata(id TEXT PRIMARY KEY, "
                          "scaling_factors BLOB, component_metadata BLOB)")
                c.execute("CREATE TABLE solver_metadata(id TEXT PRIMARY KEY, "
                          "solver_options BLOB, solver_class TEXT)")

        self._database_initialized = True

    def startup(self, recording_requester):
        """
        Prepare for a new run and create/update the abs2prom and prom2abs variables.

        Parameters
        ----------
        recording_requester : object
            Object to which this recorder is attached.
        """
        super(SqliteRecorder, self).startup(recording_requester)

        if not self._database_initialized:
            self._initialize_database()

        # grab the system
        if isinstance(recording_requester, Driver):
            system = recording_requester._problem.model
        elif isinstance(recording_requester, System):
            system = recording_requester
        else:
            system = recording_requester._system

        # grab all of the units and type (collective calls)
        states = system._list_states_allprocs()
        desvars = system.get_design_vars(True)
        responses = system.get_responses(True)
        objectives = system.get_objectives(True)
        constraints = system.get_constraints(True)
        inputs = system._var_allprocs_abs_names['input']
        outputs = system._var_allprocs_abs_names['output']
        full_var_set = [(inputs, 'input'), (outputs, 'output'),
                        (desvars, 'desvar'), (responses, 'response'),
                        (objectives, 'objective'), (constraints, 'constraint')]

        if self.connection:
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

            for var_set, var_type in full_var_set:
                for name in var_set:
                    if name not in self._abs2meta:
                        self._abs2meta[name] = system._var_allprocs_abs2meta[name].copy()
                        self._abs2meta[name]['type'] = set()
                        if name in states:
                            self._abs2meta[name]['explicit'] = False

                    if var_type not in self._abs2meta[name]['type']:
                        self._abs2meta[name]['type'].add(var_type)
                    self._abs2meta[name]['explicit'] = True

            for name in inputs:
                self._abs2meta[name] = system._var_allprocs_abs2meta[name].copy()
                self._abs2meta[name]['type'] = set()
                self._abs2meta[name]['type'].add('input')
                self._abs2meta[name]['explicit'] = True
                if name in states:
                    self._abs2meta[name]['explicit'] = False

            # store the updated abs2prom and prom2abs
            abs2prom = pickle.dumps(self._abs2prom)
            prom2abs = pickle.dumps(self._prom2abs)
            abs2meta = pickle.dumps(self._abs2meta)

            with self.connection as c:
                c.execute("UPDATE metadata SET abs2prom=?, prom2abs=?, abs2meta=?",
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
        if self.connection:
            outputs = data['out']
            inputs = data['in']

            outputs_array = values_to_array(outputs)
            inputs_array = values_to_array(inputs)

            outputs_blob = array_to_blob(outputs_array)
            inputs_blob = array_to_blob(inputs_array)

            with self.connection as c:
                c = c.cursor()  # need a real cursor for lastrowid

                c.execute("INSERT INTO driver_iterations(counter, iteration_coordinate, "
                          "timestamp, success, msg, inputs, outputs) VALUES(?,?,?,?,?,?,?)",
                          (self._counter, self._iteration_coordinate,
                           metadata['timestamp'], metadata['success'], metadata['msg'],
                           inputs_blob, outputs_blob))

                c.execute("INSERT INTO global_iterations(record_type, rowid) VALUES(?,?)",
                          ('driver', c.lastrowid))

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
        if self.connection:
            inputs = data['i']
            outputs = data['o']
            residuals = data['r']

            inputs_array = values_to_array(inputs)
            outputs_array = values_to_array(outputs)
            residuals_array = values_to_array(residuals)

            inputs_blob = array_to_blob(inputs_array)
            outputs_blob = array_to_blob(outputs_array)
            residuals_blob = array_to_blob(residuals_array)

            with self.connection as c:
                c = c.cursor()  # need a real cursor for lastrowid

                c.execute("INSERT INTO system_iterations(counter, iteration_coordinate, "
                          "timestamp, success, msg, inputs , outputs , residuals ) "
                          "VALUES(?,?,?,?,?,?,?,?)",
                          (self._counter, self._iteration_coordinate,
                           metadata['timestamp'], metadata['success'], metadata['msg'],
                           inputs_blob, outputs_blob, residuals_blob))

                c.execute("INSERT INTO global_iterations(record_type, rowid) VALUES(?,?)",
                          ('system', c.lastrowid))

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
        if self.connection:
            abs = data['abs']
            rel = data['rel']
            inputs = data['i']
            outputs = data['o']
            residuals = data['r']

            inputs_array = values_to_array(inputs)
            outputs_array = values_to_array(outputs)
            residuals_array = values_to_array(residuals)

            inputs_blob = array_to_blob(inputs_array)
            outputs_blob = array_to_blob(outputs_array)
            residuals_blob = array_to_blob(residuals_array)

            with self.connection as c:
                c = c.cursor()  # need a real cursor for lastrowid

                c.execute("INSERT INTO solver_iterations(counter, iteration_coordinate, "
                          "timestamp, success, msg, abs_err, rel_err, "
                          "solver_inputs, solver_output, solver_residuals) "
                          "VALUES(?,?,?,?,?,?,?,?,?,?)",
                          (self._counter, self._iteration_coordinate,
                           metadata['timestamp'], metadata['success'], metadata['msg'],
                           abs, rel, inputs_blob, outputs_blob, residuals_blob))

                c.execute("INSERT INTO global_iterations(record_type, rowid) VALUES(?,?)",
                          ('solver', c.lastrowid))

    def record_metadata_driver(self, recording_requester):
        """
        Record driver metadata.

        Parameters
        ----------
        recording_requester : Driver
            The Driver that would like to record its metadata.
        """
        if self.connection:
            driver_class = type(recording_requester).__name__
            model_viewer_data = pickle.dumps(recording_requester._model_viewer_data,
                                             self._pickle_version)
            model_viewer_data = sqlite3.Binary(model_viewer_data)

            try:
                with self.connection as c:
                    c.execute("INSERT INTO driver_metadata(id, model_viewer_data) "
                              "VALUES(?,?)", (driver_class, model_viewer_data))
            except sqlite3.IntegrityError:
                print("Metadata has already been recorded for %s." % driver_class)

    def record_metadata_system(self, recording_requester):
        """
        Record system metadata.

        Parameters
        ----------
        recording_requester : System
            The System that would like to record its metadata.
        """
        if self.connection:
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

            # create a copy of the system's metadata excluding what is in 'options_excludes'
            user_options = OptionsDictionary()
            excludes = recording_requester.recording_options['options_excludes']
            for key in recording_requester.options._dict:
                if check_path(key, [], excludes, True):
                    user_options._dict[key] = recording_requester.options._dict[key]
            user_options._read_only = recording_requester.options._read_only

            # try to pickle the metadata, report if it failed
            try:
                pickled_metadata = pickle.dumps(user_options, self._pickle_version)
            except Exception:
                pickled_metadata = pickle.dumps(OptionsDictionary(), self._pickle_version)
                warnings.warn("Trying to record options which cannot be pickled "
                              "on system with name: %s. Use the 'options_excludes' "
                              "recording option on system objects to avoid attempting "
                              "to record options which cannot be pickled. Skipping "
                              "recording options for this system." % recording_requester.name,
                              RuntimeWarning)

            path = recording_requester.pathname
            if not path:
                path = 'root'

            scaling_factors = sqlite3.Binary(scaling_factors)
            pickled_metadata = sqlite3.Binary(pickled_metadata)

            with self.connection as c:
                c.execute("INSERT INTO system_metadata(id, scaling_factors, component_metadata) "
                          "VALUES(?,?,?)", (path, scaling_factors, pickled_metadata))

    def record_metadata_solver(self, recording_requester):
        """
        Record solver metadata.

        Parameters
        ----------
        recording_requester : Solver
            The Solver that would like to record its metadata.
        """
        if self.connection:
            path = recording_requester._system.pathname
            solver_class = type(recording_requester).__name__
            if not path:
                path = 'root'
            id = "{}.{}".format(path, solver_class)

            solver_options = pickle.dumps(recording_requester.options, self._pickle_version)

            with self.connection as c:
                c.execute("INSERT INTO solver_metadata(id, solver_options, solver_class) "
                          "VALUES(?,?,?)", (id, sqlite3.Binary(solver_options), solver_class))

    def shutdown(self):
        """
        Shut down the recorder.
        """
        # close database connection
        if self.connection:
            self.connection.close()
