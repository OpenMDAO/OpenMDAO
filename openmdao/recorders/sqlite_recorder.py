"""
Class definition for SqliteRecorder, which provides dictionary backed by SQLite.
"""

import io
import sqlite3

import numpy as np
from six import iteritems
from six.moves import cPickle as pickle

from openmdao.core.driver import Driver
from openmdao.core.system import System
from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.solvers.solver import Solver, NonlinearSolver
from openmdao.recorders.recording_iteration_stack import \
    get_formatted_iteration_coordinate, recording_iteration_stack


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

    def __init__(self, out):
        """
        Initialize the SqliteRecorder.
        """
        super(SqliteRecorder, self).__init__()

        self.model_viewer_data = None

        # isolation_level=None causes autocommit
        self.con = sqlite3.connect(out, isolation_level=None)

        self.cursor = self.con.cursor()

        self.cursor.execute("CREATE TABLE metadata( format_version INT)")
        self.cursor.execute("INSERT INTO metadata(format_version) VALUES(?)", (format_version,))

        # used to keep track of the order of the case records across all three tables
        self.cursor.execute("CREATE TABLE global_iterations(id INTEGER PRIMARY KEY, "
                            "record_type TEXT, rowid INT)")
        self.cursor.execute("CREATE TABLE driver_iterations(id INTEGER PRIMARY KEY, counter INT,"
                            "iteration_coordinate TEXT, timestamp REAL, success INT, msg TEXT, "
                            "desvars BLOB, responses BLOB, objectives BLOB, constraints BLOB)")
        self.cursor.execute("CREATE TABLE system_iterations(id INTEGER PRIMARY KEY, counter INT, "
                            "iteration_coordinate TEXT,  timestamp REAL, success INT, msg TEXT, "
                            "inputs BLOB, outputs BLOB, residuals BLOB)")
        self.cursor.execute("CREATE TABLE solver_iterations(id INTEGER PRIMARY KEY, counter INT, "
                            "iteration_coordinate TEXT, timestamp REAL, success INT, msg TEXT, "
                            "abs_err REAL, rel_err REAL, solver_output BLOB, "
                            "solver_residuals BLOB)")

        self.cursor.execute("CREATE TABLE driver_metadata(id TEXT PRIMARY KEY, "
                            "model_viewer_data BLOB)")
        self.cursor.execute("CREATE TABLE system_metadata(id TEXT PRIMARY KEY,"
                            " scaling_factors BLOB)")
        self.cursor.execute("CREATE TABLE solver_metadata(id TEXT PRIMARY KEY, solver_options BLOB,"
                            " solver_class TEXT)")

    def record_iteration(self, object_requesting_recording, metadata, **kwargs):
        """
        Store the provided data in the sqlite file using the iteration coordinate for the key.

        Parameters
        ----------
        object_requesting_recording: <object>
            The item, a System, Solver, or Driver that wants to record an iteration.
        metadata : dict
            Dictionary containing execution metadata (e.g. iteration coordinate).
        **kwargs :
            Various keyword arguments needed for System or Solver recordings.
        """
        super(SqliteRecorder, self).record_iteration(object_requesting_recording, metadata)

        if isinstance(object_requesting_recording, Driver):
            self.record_iteration_driver(object_requesting_recording, metadata)

        elif isinstance(object_requesting_recording, System):
            self.record_iteration_system(object_requesting_recording, metadata)

        elif isinstance(object_requesting_recording, Solver):
            self.record_iteration_solver(object_requesting_recording, metadata, **kwargs)
        else:
            raise ValueError("Recorders must be attached to Drivers, Systems, or Solvers.")

    def record_iteration_driver(self, object_requesting_recording, metadata):
        """
        Record an iteration using the driver options.

        Parameters
        ----------
        object_requesting_recording: <Driver>
            The Driver object that wants to record an iteration.
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

        desvars_array = None
        responses_array = None
        objectives_array = None
        constraints_array = None

        # Just an example of the syntax for creating a numpy structured array
        # arr = np.zeros((1,), dtype=[('dv_x','(5,)f8'),('dv_y','(10,)f8')])

        # This returns a dict of names and values. Use this to build up the tuples of
        # used for the dtypes in the creation of the numpy structured array
        # we want to write to sqlite
        if self.options['record_desvars']:
            if self._filtered_driver:
                desvars_values = \
                    object_requesting_recording.get_design_var_values(self._filtered_driver['des'])
            else:
                desvars_values = object_requesting_recording.get_design_var_values()

            if desvars_values:
                dtype_tuples = []
                for name, value in iteritems(desvars_values):
                    tple = (name, '{}f8'.format(value.shape))
                    dtype_tuples.append(tple)

                desvars_array = np.zeros((1,), dtype=dtype_tuples)

                for name, value in iteritems(desvars_values):
                    desvars_array[name] = value

        if self.options['record_responses']:
            if self._filtered_driver:
                responses_values = \
                    object_requesting_recording.get_response_values(self._filtered_driver['res'])
            else:
                responses_values = object_requesting_recording.get_response_values()

            if responses_values:
                dtype_tuples = []
                for name, value in iteritems(responses_values):
                    tple = (name, '{}f8'.format(value.shape))
                    dtype_tuples.append(tple)

                responses_array = np.zeros((1,), dtype=dtype_tuples)

                for name, value in iteritems(responses_values):
                    responses_array[name] = value

        if self.options['record_objectives']:
            if self._filtered_driver:
                objectives_values = \
                    object_requesting_recording.get_objective_values(self._filtered_driver['obj'])
            else:
                objectives_values = object_requesting_recording.get_objective_values()

            if objectives_values:
                dtype_tuples = []
                for name, value in iteritems(objectives_values):
                    tple = (name, '{}f8'.format(value.shape))
                    dtype_tuples.append(tple)

                objectives_array = np.zeros((1,), dtype=dtype_tuples)

                for name, value in iteritems(objectives_values):
                    objectives_array[name] = value

        if self.options['record_constraints']:
            if self._filtered_driver:
                constraints_values = \
                    object_requesting_recording.get_constraint_values(self._filtered_driver['con'])
            else:
                constraints_values = object_requesting_recording.get_constraint_values()

            if constraints_values:
                dtype_tuples = []
                for name, value in iteritems(constraints_values):
                    tple = (name, '{}f8'.format(value.shape))
                    dtype_tuples.append(tple)

                constraints_array = np.zeros((1,), dtype=dtype_tuples)

                for name, value in iteritems(constraints_values):
                    constraints_array[name] = value

        desvars_blob = array_to_blob(desvars_array)
        responses_blob = array_to_blob(responses_array)
        objectives_blob = array_to_blob(objectives_array)
        constraints_blob = array_to_blob(constraints_array)

        iteration_coordinate = get_formatted_iteration_coordinate()

        self.cursor.execute("INSERT INTO driver_iterations(counter, iteration_coordinate, "
                            "timestamp, success, msg, desvars , responses , objectives , "
                            "constraints ) VALUES(?,?,?,?,?,?,?,?,?)",
                            (self._counter, iteration_coordinate,
                             metadata['timestamp'], metadata['success'],
                             metadata['msg'], desvars_blob,
                             responses_blob, objectives_blob,
                             constraints_blob))
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
        print("Recording STACK:", recording_iteration_stack)
        stack_top = recording_iteration_stack[-1][0]
        method = stack_top.split('.')[-1]
        print("METHOD: ", method)

        if method not in ['_apply_linear', '_apply_nonlinear', '_solve_linear',
                          '_solve_nonlinear']:
            raise ValueError("method must be one of: '_apply_linear, "
                             "_apply_nonlinear, _solve_linear, _solve_nonlinear'")

        if 'nonlinear' in method:
            inputs, outputs, residuals = object_requesting_recording.get_nonlinear_vectors()
        else:
            inputs, outputs, residuals = object_requesting_recording.get_linear_vectors()

        inputs_array = outputs_array = residuals_array = None

        # Inputs
        if self.options['record_inputs'] and inputs._names:
            ins = {}
            if 'i' in self._filtered_system:
                # use filtered inputs
                for inp in self._filtered_system['i']:
                    if inp in inputs._names:
                        ins[inp] = inputs._names[inp]
            else:
                # use all the inputs
                ins = inputs._names

            dtype_tuples = []
            for name, value in iteritems(ins):
                tple = (name, '{}f8'.format(value.shape))
                dtype_tuples.append(tple)

            inputs_array = np.zeros((1,), dtype=dtype_tuples)
            for name, value in iteritems(ins):
                inputs_array[name] = value

        # Outputs
        if self.options['record_outputs'] and outputs._names:
            outs = {}

            if 'o' in self._filtered_system:
                # use outputs from filtered list.
                for out in self._filtered_system['o']:
                    if out in outputs._names:
                        outs[out] = outputs._names[out]
            else:
                # use all the outputs
                outs = outputs._names

            dtype_tuples = []
            for name, value in iteritems(outs):
                tple = (name, '{}f8'.format(value.shape))
                dtype_tuples.append(tple)

            outputs_array = np.zeros((1,), dtype=dtype_tuples)
            for name, value in iteritems(outs):
                outputs_array[name] = value

        # Residuals
        if self.options['record_residuals'] and residuals._names:
            resids = {}

            if 'r' in self._filtered_system:
                # use filtered residuals
                for res in self._filtered_system['r']:
                    if res in residuals._names:
                        resids[res] = residuals._names[res]
            else:
                # use all the residuals
                resids = residuals._names

            dtype_tuples = []
            if resids:
                for name, value in iteritems(resids):
                    tple = (name, '{}f8'.format(value.shape))
                    dtype_tuples.append(tple)

                residuals_array = np.zeros((1,), dtype=dtype_tuples)
                for name, value in iteritems(resids):
                    residuals_array[name] = value

        inputs_blob = array_to_blob(inputs_array)
        outputs_blob = array_to_blob(outputs_array)
        residuals_blob = array_to_blob(residuals_array)

        iteration_coordinate = get_formatted_iteration_coordinate()

        self.cursor.execute("INSERT INTO system_iterations(counter, iteration_coordinate, "
                            "timestamp, success, msg, inputs , outputs , residuals ) "
                            "VALUES(?,?,?,?,?,?,?,?)",
                            (self._counter, iteration_coordinate,
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
        outputs_array = residuals_array = None

        # Go through the recording options of Solver to construct the entry to be inserted.
        if self.options['record_abs_error']:
            abs_error = kwargs.get('abs')
        else:
            abs_error = None

        if self.options['record_rel_error']:
            rel_error = kwargs.get('rel')
        else:
            rel_error = None

        if self.options['record_solver_output']:
            dtype_tuples = []

            if isinstance(object_requesting_recording, NonlinearSolver):
                outputs = object_requesting_recording._system._outputs
            else:  # it's a LinearSolver
                outputs = object_requesting_recording._system._vectors['output']['linear']

            outs = {}
            if 'out' in self._filtered_solver:
                for outp in outputs._names:
                    outs[outp] = outputs._names[outp]
            else:
                outs = outputs

            if outs:
                for name, value in iteritems(outs):
                    tple = (name, '{}f8'.format(value.shape))
                    dtype_tuples.append(tple)

                outputs_array = np.zeros((1,), dtype=dtype_tuples)

                for name, value in iteritems(outs):
                    outputs_array[name] = value

        if self.options['record_solver_residuals']:
            dtype_tuples = []

            if isinstance(object_requesting_recording, NonlinearSolver):
                residuals = object_requesting_recording._system._residuals
            else:  # it's a LinearSolver
                residuals = object_requesting_recording._system._vectors['residual']['linear']

            res = {}
            if 'res' in self._filtered_solver:
                for rez in residuals._names:
                    res[rez] = residuals._names[rez]
            else:
                res = residuals

            if res:
                for name, value in iteritems(res):
                    tple = (name, '{}f8'.format(value.shape))
                    dtype_tuples.append(tple)

                residuals_array = np.zeros((1,), dtype=dtype_tuples)
                for name, value in iteritems(res):
                    residuals_array[name] = value

        outputs_blob = array_to_blob(outputs_array)
        residuals_blob = array_to_blob(residuals_array)

        iteration_coordinate = get_formatted_iteration_coordinate()

        self.cursor.execute("INSERT INTO solver_iterations(counter, iteration_coordinate, "
                            "timestamp, success, msg, abs_err, rel_err, solver_output, "
                            "solver_residuals) VALUES(?,?,?,?,?,?,?,?,?)",
                            (self._counter, iteration_coordinate,
                             metadata['timestamp'],
                             metadata['success'], metadata['msg'],
                             abs_error, rel_error,
                             outputs_blob, residuals_blob))

        self.cursor.execute("INSERT INTO global_iterations(record_type, rowid) VALUES(?,?)",
                            ('solver', self.cursor.lastrowid))

    def record_metadata(self, object_requesting_recording):
        """
        Route the record_metadata call to the proper object.

        Parameters
        ----------
        object_requesting_recording: <object>
            The object that would like to record its metadata.
        """
        if self.options['record_metadata']:
            if isinstance(object_requesting_recording, Driver):
                self.record_metadata_driver(object_requesting_recording)
            elif isinstance(object_requesting_recording, System):
                self.record_metadata_system(object_requesting_recording)
            elif isinstance(object_requesting_recording, Solver):
                self.record_metadata_solver(object_requesting_recording)

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
        self.con.close()
