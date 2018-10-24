"""
Class definition for SqliteRecorder, which provides dictionary backed by SQLite.
"""

from copy import deepcopy
from io import BytesIO

import os
import sqlite3

import json
import numpy as np

from six.moves import cPickle as pickle

from openmdao.recorders.case_recorder import CaseRecorder
from openmdao.utils.mpi import MPI
from openmdao.utils.record_util import values_to_array
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.general_utils import simple_warning
from openmdao.core.driver import Driver
from openmdao.core.system import System
from openmdao.core.problem import Problem
from openmdao.solvers.solver import Solver


"""
SQL case database version history.
----------------------------------
5 -- OpenMDAO 2.5
     Added source column (driver name, system/solver pathname) to global iterations table.
4 -- OpenMDAO 2.4
     Added variable settings metadata that contains scaling info.
3 -- OpenMDAO 2.4
     Storing most data as JSON rather than binary numpy arrays.
2 -- OpenMDAO 2.4, merged 20 July 2018.
     Added support for recording derivatives from driver, resulting in a new table.
1 -- Through OpenMDAO 2.3
     Original implementation.
"""
format_version = 5


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
    out = BytesIO()
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
    out = BytesIO(blob)
    out.seek(0)
    return np.load(out)


def convert_to_list(vals):
    """
    Recursively convert arrays, tuples, and sets to lists.

    Parameters
    ----------
    vals : numpy.array or list or tuple
        the object to be converted to a list

    Returns
    -------
    list :
        The converted list.
    """
    if isinstance(vals, np.ndarray):
        return convert_to_list(vals.tolist())
    elif isinstance(vals, (list, tuple, set)):
        return [convert_to_list(item) for item in vals]
    else:
        return vals


class SqliteRecorder(CaseRecorder):
    """
    Recorder that saves cases in a sqlite db.

    Attributes
    ----------
    _record_viewer_data : bool
        Flag indicating whether to record data needed to generate N2 diagram.
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

    def __init__(self, filepath, append=False, pickle_version=2, record_viewer_data=True):
        """
        Initialize the SqliteRecorder.

        Parameters
        ----------
        filepath : str
            Path to the recorder file.
        append : bool, optional
            Optional. If True, append to an existing case recorder file.
        pickle_version : int, optional
            The pickle protocol version to use when pickling metadata.
        record_viewer_data : bool, optional
            If True, record data needed for visualization.
        """
        if append:
            raise NotImplementedError("Append feature not implemented for SqliteRecorder")

        self.connection = None
        self._record_viewer_data = record_viewer_data

        self._abs2prom = {'input': {}, 'output': {}}
        self._prom2abs = {'input': {}, 'output': {}}
        self._abs2meta = {}
        self._pickle_version = pickle_version
        self._filepath = filepath
        self._database_initialized = False

        # default to record on all procs when running in parallel
        self._record_on_proc = True

        super(SqliteRecorder, self).__init__(record_viewer_data)

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
                c.execute("CREATE TABLE metadata(format_version INT, "
                          "abs2prom TEXT, prom2abs TEXT, abs2meta TEXT, var_settings TEXT)")
                c.execute("INSERT INTO metadata(format_version, abs2prom, prom2abs) "
                          "VALUES(?,?,?)", (format_version, None, None))

                # used to keep track of the order of the case records across all case tables
                c.execute("CREATE TABLE global_iterations(id INTEGER PRIMARY KEY, "
                          "record_type TEXT, rowid INT, source TEXT)")

                c.execute("CREATE TABLE driver_iterations(id INTEGER PRIMARY KEY, "
                          "counter INT, iteration_coordinate TEXT, timestamp REAL, "
                          "success INT, msg TEXT, inputs TEXT, outputs TEXT)")
                c.execute("CREATE TABLE driver_derivatives(id INTEGER PRIMARY KEY, "
                          "counter INT, iteration_coordinate TEXT, timestamp REAL, "
                          "success INT, msg TEXT, derivatives BLOB)")
                c.execute("CREATE INDEX driv_iter_ind on driver_iterations(iteration_coordinate)")

                c.execute("CREATE TABLE problem_cases(id INTEGER PRIMARY KEY, "
                          "counter INT, case_name TEXT, timestamp REAL, "
                          "success INT, msg TEXT, outputs TEXT)")
                c.execute("CREATE INDEX prob_name_ind on problem_cases(case_name)")

                c.execute("CREATE TABLE system_iterations(id INTEGER PRIMARY KEY, "
                          "counter INT, iteration_coordinate TEXT, timestamp REAL, "
                          "success INT, msg TEXT, inputs TEXT, outputs TEXT, residuals TEXT)")
                c.execute("CREATE INDEX sys_iter_ind on system_iterations(iteration_coordinate)")

                c.execute("CREATE TABLE solver_iterations(id INTEGER PRIMARY KEY, "
                          "counter INT, iteration_coordinate TEXT, timestamp REAL, "
                          "success INT, msg TEXT, abs_err REAL, rel_err REAL, "
                          "solver_inputs TEXT, solver_output TEXT, solver_residuals TEXT)")
                c.execute("CREATE INDEX solv_iter_ind on solver_iterations(iteration_coordinate)")

                c.execute("CREATE TABLE driver_metadata(id TEXT PRIMARY KEY, "
                          "model_viewer_data TEXT)")
                c.execute("CREATE TABLE system_metadata(id TEXT PRIMARY KEY, "
                          "scaling_factors BLOB, component_metadata BLOB)")
                c.execute("CREATE TABLE solver_metadata(id TEXT PRIMARY KEY, "
                          "solver_options BLOB, solver_class TEXT)")

        self._database_initialized = True

    def _cleanup_abs2meta(self):
        """
        Convert all abs2meta variable properties to a form that can be dumped as JSON.
        """
        for name in self._abs2meta:
            if 'lower' in self._abs2meta[name]:
                self._abs2meta[name]['lower'] = convert_to_list(self._abs2meta[name]['lower'])
            if 'upper' in self._abs2meta[name]:
                self._abs2meta[name]['upper'] = convert_to_list(self._abs2meta[name]['upper'])
            for prop in self._abs2meta[name]:
                val = self._abs2meta[name][prop]
                if isinstance(val, np.int8) or isinstance(val, np.int16) or\
                   isinstance(val, np.int32) or isinstance(val, np.int64):
                    self._abs2meta[name][prop] = val.item()
                elif isinstance(val, tuple):
                    self._abs2meta[name][prop] = [int(v) for v in val]

    def _cleanup_var_settings(self, var_settings):
        """
        Convert all var_settings variable properties to a form that can be dumped as JSON.

        Parameters
        ----------
        var_settings : dict
            Dictionary mapping absolute variable names to variable settings.

        Returns
        -------
        var_settings : dict
            Dictionary mapping absolute variable names to var settings that are JSON compatible.
        """
        # otherwise we trample on values that are used elsewhere
        var_settings = deepcopy(var_settings)
        for name in var_settings:
            for prop in var_settings[name]:
                val = var_settings[name][prop]
                if isinstance(val, np.int8) or isinstance(val, np.int16) or\
                   isinstance(val, np.int32) or isinstance(val, np.int64):
                    var_settings[name][prop] = val.item()
                elif isinstance(val, tuple):
                    var_settings[name][prop] = [int(v) for v in val]
                elif isinstance(val, np.ndarray):
                    var_settings[name][prop] = convert_to_list(var_settings[name][prop])

        return var_settings

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
        elif isinstance(recording_requester, Problem):
            system = recording_requester.model
        elif isinstance(recording_requester, Solver):
            system = recording_requester._system
        else:
            raise ValueError('Driver encountered a recording_requester it cannot handle'
                             ': {0}'.format(recording_requester))

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
                        self._abs2meta[name]['type'] = []
                        if name in states:
                            self._abs2meta[name]['explicit'] = False

                    if var_type not in self._abs2meta[name]['type']:
                        self._abs2meta[name]['type'].append(var_type)
                    self._abs2meta[name]['explicit'] = True

            for name in inputs:
                self._abs2meta[name] = system._var_allprocs_abs2meta[name].copy()
                self._abs2meta[name]['type'] = []
                self._abs2meta[name]['type'].append('input')
                self._abs2meta[name]['explicit'] = True
                if name in states:
                    self._abs2meta[name]['explicit'] = False

            self._cleanup_abs2meta()

            # store the updated abs2prom and prom2abs
            abs2prom = json.dumps(self._abs2prom)
            prom2abs = json.dumps(self._prom2abs)
            abs2meta = json.dumps(self._abs2meta)

            var_settings = {}
            var_settings.update(desvars)
            var_settings.update(objectives)
            var_settings.update(constraints)
            var_settings = self._cleanup_var_settings(var_settings)
            var_settings_json = json.dumps(var_settings)

            with self.connection as c:
                c.execute("UPDATE metadata SET abs2prom=?, prom2abs=?, abs2meta=?, var_settings=?",
                          (abs2prom, prom2abs, abs2meta, var_settings_json))

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

            # convert to list so this can be dumped as JSON
            for in_out in (inputs, outputs):
                if in_out is None:
                    continue
                for var in in_out:
                    in_out[var] = convert_to_list(in_out[var])

            outputs_text = json.dumps(outputs)
            inputs_text = json.dumps(inputs)

            with self.connection as c:
                c = c.cursor()  # need a real cursor for lastrowid

                c.execute("INSERT INTO driver_iterations(counter, iteration_coordinate, "
                          "timestamp, success, msg, inputs, outputs) VALUES(?,?,?,?,?,?,?)",
                          (self._counter, self._iteration_coordinate,
                           metadata['timestamp'], metadata['success'], metadata['msg'],
                           inputs_text, outputs_text))

                c.execute("INSERT INTO global_iterations(record_type, rowid, source) VALUES(?,?,?)",
                          ('driver', c.lastrowid, recording_requester._get_name()))

    def record_iteration_problem(self, recording_requester, data, metadata):
        """
        Record data and metadata from a Problem.

        Parameters
        ----------
        recording_requester : object
            Problem in need of recording.
        data : dict
            Dictionary containing desvars, objectives, and constraints.
        metadata : dict
            Dictionary containing execution metadata.
        """
        if self.connection:
            outputs = data['out']

            # convert to list so this can be dumped as JSON
            if outputs is not None:
                for var in outputs:
                    outputs[var] = convert_to_list(outputs[var])

            outputs_text = json.dumps(outputs)

            with self.connection as c:
                c = c.cursor()  # need a real cursor for lastrowid

                c.execute("INSERT INTO problem_cases(counter, case_name, "
                          "timestamp, success, msg, outputs) VALUES(?,?,?,?,?,?)",
                          (self._counter, metadata['name'],
                           metadata['timestamp'], metadata['success'], metadata['msg'],
                           outputs_text))

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

            # convert to list so this can be dumped as JSON
            for i_o_r in (inputs, outputs, residuals):
                if i_o_r is None:
                    continue
                for var in i_o_r:
                    i_o_r[var] = convert_to_list(i_o_r[var])

            outputs_text = json.dumps(outputs)
            inputs_text = json.dumps(inputs)
            residuals_text = json.dumps(residuals)

            with self.connection as c:
                c = c.cursor()  # need a real cursor for lastrowid

                c.execute("INSERT INTO system_iterations(counter, iteration_coordinate, "
                          "timestamp, success, msg, inputs , outputs , residuals ) "
                          "VALUES(?,?,?,?,?,?,?,?)",
                          (self._counter, self._iteration_coordinate,
                           metadata['timestamp'], metadata['success'], metadata['msg'],
                           inputs_text, outputs_text, residuals_text))

                # get the pathname of the source system
                source_system = recording_requester.pathname
                if source_system == '':
                    source_system = 'root'

                c.execute("INSERT INTO global_iterations(record_type, rowid, source) VALUES(?,?,?)",
                          ('system', c.lastrowid, source_system))

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

            # convert to list so this can be dumped as JSON
            for i_o_r in (inputs, outputs, residuals):
                if i_o_r is None:
                    continue
                for var in i_o_r:
                    i_o_r[var] = convert_to_list(i_o_r[var])

            outputs_text = json.dumps(outputs)
            inputs_text = json.dumps(inputs)
            residuals_text = json.dumps(residuals)

            with self.connection as c:
                c = c.cursor()  # need a real cursor for lastrowid

                c.execute("INSERT INTO solver_iterations(counter, iteration_coordinate, "
                          "timestamp, success, msg, abs_err, rel_err, "
                          "solver_inputs, solver_output, solver_residuals) "
                          "VALUES(?,?,?,?,?,?,?,?,?,?)",
                          (self._counter, self._iteration_coordinate,
                           metadata['timestamp'], metadata['success'], metadata['msg'],
                           abs, rel, inputs_text, outputs_text, residuals_text))

                # get the pathname of the source system
                source_system = recording_requester._system.pathname
                if source_system == '':
                    source_system = 'root'

                # get solver type from SOLVER class attribute to determine the solver pathname
                solver_type = recording_requester.SOLVER[0:2]
                if solver_type == 'NL':
                    source_solver = source_system + '.nonlinear_solver'
                elif solver_type == 'LS':
                    source_solver = source_system + '.nonlinear_solver.linesearch'
                else:
                    raise RuntimeError("Solver type '%s' not recognized during recording. "
                                       "Expecting NL or LS" % recording_requester.SOLVER)

                c.execute("INSERT INTO global_iterations(record_type, rowid, source) VALUES(?,?,?)",
                          ('solver', c.lastrowid, source_solver))

    def record_viewer_data(self, model_viewer_data, key='Driver'):
        """
        Record model viewer data.

        Parameters
        ----------
        model_viewer_data : dict
            Data required to visualize the model.
        key : str, optional
            The unique ID to use for this data in the table.
        """
        if self.connection:
            model_viewer_data = json.dumps(model_viewer_data)

            # Note: recorded to 'driver_metadata' table for legacy/compatibility reasons.
            try:
                with self.connection as c:
                    c.execute("INSERT INTO driver_metadata(id, model_viewer_data) "
                              "VALUES(?,?)", (key, model_viewer_data))
            except sqlite3.IntegrityError:
                print("Model viewer data has already has already been recorded for %s." % key)

    def record_metadata_system(self, recording_requester):
        """
        Record system metadata.

        Parameters
        ----------
        recording_requester : System
            The System that would like to record its metadata.
        """
        if self.connection:
            scaling_vecs, user_options = self._get_metadata_system(recording_requester)

            if scaling_vecs is None:
                return

            scaling_factors = pickle.dumps(scaling_vecs, self._pickle_version)

            # try to pickle the metadata, report if it failed
            try:
                pickled_metadata = pickle.dumps(user_options, self._pickle_version)
            except Exception:
                pickled_metadata = pickle.dumps(OptionsDictionary(), self._pickle_version)
                simple_warning("Trying to record options which cannot be pickled "
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
                # Because we can have a recorder attached to multiple Systems,
                #   and because we are now recording System metadata recursively,
                #   we can store System metadata multiple times. Need to ignore when that happens
                #   so we don't get database errors. So use OR IGNORE
                c.execute("INSERT OR IGNORE INTO system_metadata"
                          "(id, scaling_factors, component_metadata) "
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

    def record_derivatives_driver(self, recording_requester, data, metadata):
        """
        Record derivatives data from a Driver.

        Parameters
        ----------
        recording_requester : object
            Driver in need of recording.
        data : dict
            Dictionary containing derivatives keyed by 'of,wrt' to be recorded.
        metadata : dict
            Dictionary containing execution metadata.
        """
        if self.connection:

            data_array = values_to_array(data)
            data_blob = array_to_blob(data_array)

            with self.connection as c:
                c = c.cursor()  # need a real cursor for lastrowid

                c.execute("INSERT INTO driver_derivatives(counter, iteration_coordinate, "
                          "timestamp, success, msg, derivatives) VALUES(?,?,?,?,?,?)",
                          (self._counter, self._iteration_coordinate,
                           metadata['timestamp'], metadata['success'], metadata['msg'],
                           data_blob))

    def shutdown(self):
        """
        Shut down the recorder.
        """
        # close database connection
        if self.connection:
            self.connection.close()
