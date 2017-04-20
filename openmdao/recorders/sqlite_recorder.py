"""
Class definition for SqliteRecorder, which provides dictionary backed by SQLite.
"""
from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.core.driver import Driver
from openmdao.core.system import System
from openmdao.solvers.solver import Solver
from openmdao.utils.record_util import format_iteration_coordinate

from six import iteritems

import sqlite3
import numpy as np
import io
import cPickle

# this basically defines a special type of variable that can be written/read from a sqlite db.
#  the type is called "array"


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler).
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    """
    Utility function for numpy structured arrays.
    """
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

format_version = 1


class SqliteRecorder(BaseRecorder):
    """
    Recorder that saves cases in a sqlite db.

    Args
    ----
    sqlite_dict_args : dict
        Dictionary lf any additional arguments for the SQL db.

    Options
    -------
    options['record_metadata'] :  bool(True)
        Tells recorder whether to record variable attribute metadata.
    options['record_outputs'] :  bool(True)
        Tells recorder whether to record the outputs vector.
    options['record_inputs'] :  bool(False)
        Tells recorder whether to record the inputs vector.
    options['record_residuals'] :  bool(False)
        Tells recorder whether to record the residuals vector.
    options['record_derivs'] :  bool(True)
        Tells recorder whether to record derivatives that are requested by a `Driver`.
    options['includes'] :  list of strings
        Patterns for variables to include in recording.
    options['excludes'] :  list of strings
        Patterns for variables to exclude in recording (processed after includes).
    """

    def __init__(self, out):
        """
        Initialize.
        """
        super(SqliteRecorder, self).__init__()

        self.model_viewer_data = None

        self._counter = 0
        # isolation_level=None causes autocommit
        self.con = sqlite3.connect(out, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)
        # write the metadata
        self.con.execute("CREATE TABLE metadata( format_version INT)")
        self.con.execute("INSERT INTO metadata(format_version) VALUES(?)", (format_version,))

        self.con.execute("CREATE TABLE driver_iterations(id INTEGER PRIMARY KEY, "
                         "iteration_coordinate TEXT, timestamp REAL, success INT, msg TEXT, "
                         "desvars array, responses array, objectives array, constraints array)")
        self.con.execute("CREATE TABLE system_iterations(id INTEGER PRIMARY KEY, "
                         "iteration_coordinate TEXT,  timestamp REAL, success INT, msg TEXT, "
                         "system_values array)")
        self.con.execute("CREATE TABLE solver_iterations(id INTEGER PRIMARY KEY, "
                         "iteration_coordinate TEXT, timestamp REAL, success INT, msg TEXT, "
                         "solver_values array)")

        self.con.execute("CREATE TABLE driver_metadata(id TEXT PRIMARY KEY, "
                         "model_viewer_data BLOB)")
        self.con.execute("CREATE TABLE system_metadata(id TEXT PRIMARY KEY, "
                         "input_scale_factor REAL, output_scale_factor REAL, "
                         "residual_scale_factor REAL)")
        self.con.execute("CREATE TABLE solver_metadata(id TEXT PRIMARY KEY, solver_options BLOB,"
                         " solver_class TEXT)")

    def startup(self, object_requesting_recording):
        """
        Startup.
        """
        super(SqliteRecorder, self).startup(object_requesting_recording)
        self._counter = 0

    def record_iteration(self, object_requesting_recording, metadata):
        """
        Store the provided data in the sqlite file using the iteration coordinate for the key.
        """
        super(SqliteRecorder, self).record_iteration(object_requesting_recording, metadata)

        # Record an iteration from a Driver
        if isinstance(object_requesting_recording, Driver):
            self.record_iteration_driver(object_requesting_recording, metadata)

        elif isinstance(object_requesting_recording, System):
            self.record_iteration_system(object_requesting_recording, metadata)

        elif isinstance(object_requesting_recording, Solver):
            self.record_iteration_solver(object_requesting_recording, metadata)

        else:
            print ("YOU CAN'T ATTACH A RECORDER TO THIS OBJECT")
            # TODO_RECORDER is this the right way to handle this

    def record_iteration_driver(self, object_requesting_recording, metadata):
        """
        Record an iteration using the driver options.
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
            desvars_values = object_requesting_recording.get_design_var_values()
            if desvars_values:
                dtype_tuples = []
                for name, value in iteritems(desvars_values):
                    tple = (name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

                desvars_array = np.zeros((1,), dtype=dtype_tuples)

                for name, value in iteritems(desvars_values):
                    desvars_array[name] = value

        if self.options['record_responses']:
            responses_values = object_requesting_recording.get_response_values()
            if responses_values:
                dtype_tuples = []
                for name, value in iteritems(responses_values):
                    tple = (name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

                responses_array = np.zeros((1,), dtype=dtype_tuples)

                for name, value in iteritems(responses_values):
                    responses_array[name] = value

        if self.options['record_objectives']:
            objectives_values = object_requesting_recording.get_objective_values()
            if objectives_values:
                dtype_tuples = []
                for name, value in iteritems(objectives_values):
                    tple = (name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

                objectives_array = np.zeros((1,), dtype=dtype_tuples)

                for name, value in iteritems(objectives_values):
                    objectives_array[name] = value

        if self.options['record_constraints']:
            constraints_values = object_requesting_recording.get_constraint_values()
            if constraints_values:
                dtype_tuples = []
                for name, value in iteritems(constraints_values):
                    tple = (name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

                constraints_array = np.zeros((1,), dtype=dtype_tuples)

                for name, value in iteritems(constraints_values):
                    constraints_array[name] = value

        self.con.execute("INSERT INTO driver_iterations(iteration_coordinate, timestamp, "
                         "success, msg, desvars , responses , objectives , constraints ) "
                         "VALUES(?,?,?,?,?,?,?,?)", (format_iteration_coordinate(metadata['coord']),
                                                     metadata['timestamp'], metadata['success'],
                                                     metadata['msg'], desvars_array,
                                                     responses_array, objectives_array,
                                                     constraints_array))

    def record_iteration_system(self, object_requesting_recording, metadata):
        """
        Record an iteration using system options.
        """
        if isinstance(object_requesting_recording, System):
            dtype_tuples = []

            # go through the recording options of System to construct the entry to be inserted.
            if self.options['record_inputs']:
                inputs = object_requesting_recording._inputs
                if inputs:
                    for name, value in iteritems(design_vars):
                        tple = ('input.' + name, '({},)f8'.format(len(value)))
                        dtype_tuples.append(tple)

            if self.options['record_outputs']:
                outputs = object_requesting_recording._outputs
                if outputs:
                    for name, value in iteritems(ouputs):
                        tple = ('output.' + name, '({},)f8'.format(len(value)))
                        dtype_tuples.append(tple)

            if self.options['record_residuals']:
                residuals = object_requesting_recording._residuals
                if residuals:
                    for name, value in iteritems(residuals):
                        tple = ('residual.' + name, '({},)f8'.format(len(value)))
                        dtype_tuples.append(tple)

            if self.options['record_derivatives']:
                derivatives = object_requesting_recording._derivatves
                # TODO_RECORDERS: MAKE THIS EXIST?
                if derivatives:
                    for name, value in iteritems(derivatives):
                        tple = ('derivative.' + name, '({},)f8'.format(len(value)))
                        dtype_tuples.append(tple)

            # Create the mega array that we will write to the database
            # All of this needs to be looked into to be optimized !!
            system_values = np.zeros((1,), dtype=dtype_tuples)

            # Write the actual values to this array
            # Wish we didn't have to loop through this twice
            if self.options['record_inputs'] and inputs:
                for name, value in iteritems(inputs):
                    system_values['input.' + name] = value
            if self.options['record_outputs'] and outputs:
                for name, value in iteritems(outputs):
                    system_values['output.' + name] = value
            if self.options['record_residuals'] and residuals:
                for name, value in iteritems(residuals):
                    system_values['residual.' + name] = value
            if self.options['record_derivatives'] and derivatives:
                for name, value in iteritems(derivatives):
                    system_values['derivative.' + name] = value

            print("SYSTEM VALUES:", system_values)
            # Write this mega array to the database
            self.con.execute("INSERT INTO system_iterations(iteration_coordinate, timestamp, "
                             "success, msg, system_values) VALUES(?,?,?,?,?)",
                             (metadata['coord'], metadata['timestamp'], metadata['success'],
                              metadata['msg'], system_values))

    def record_iteration_solver(self, object_requesting_recording, metadata):
        """
        Record an iteration using solver options.
        """
        dtype_tuples = []

        # Go through the recording options of Solver to construct the entry to be inserted.
        if self.options['record_abs_error']:
            abs_errors = object_requesting_recording.get_abs_error()
            if abs_errors:
                for name, value in iteritems(abs_errors):
                    tple = ('abs_error.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

        if self.options['record_rel_error']:
            rel_errors = object_requesting_recording.get_rel_error()
            if rel_errors:
                for name, value in iteritems(rel_errors):
                    tple = ('rel_error.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

        if self.options['record_output']:
            outputs = object_requesting_recording.get_output()
            if outputs:
                for name, value in iteritems(outputs):
                    tple = ('output.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

        if self.options['record_solver_residuals']:
            residuals = object_requesting_recording.get_residuals()
            if residuals:
                for name, value in iteritems(residuals):
                    tple = ('residual.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

        # Create the mega array that we will write to the database
        # All of this needs to be looked into to be optimized !!
        solver_values = np.zeros((1,), dtype=dtype_tuples)

        # Write the actual values to this array
        # Wish we didn't have to loop through this twice
        if self.options['record_abs_error'] and abs_errors:
            for name, value in iteritems(abs_errors):
                solver_values['abs_error.' + name] = value
        if self.options['record_rel_error'] and rel_errors:
            for name, value in iteritems(rel_errors):
                solver_values['rel_error.' + name] = value
        if self.options['record_output'] and outputs:
            for name, value in iteritems(outputs):
                solver_values['output.' + name] = value
        if self.options['record_solver_residuals'] and residuals:
            for name, value in iteritems(residuals):
                solver_values['residual.' + name] = value

        if self.options['record_solver_residuals']:
            residuals = object_requesting_recording.get_residuals()

            for name, value in iteritems(residuals):
                tple = ('residual.' + name, '({},)f8'.format(len(value)))
                dtype_tuples.append(tple)
            solver_values = np.zeros((1,), dtype=dtype_tuples)
            for name, value in iteritems(residuals):
                solver_values['residual.' + name] = value

        print("SOLVER VALUES:", solver_values)
        # Write this mega array to the database
        self.con.execute("INSERT INTO solver_iterations(iteration_coordinate, timestamp, "
                         "success, msg, solver_values) "
                         "VALUES(?,?,?,?,?)", (metadata['coord'], metadata['timestamp'],
                                               metadata['success'], metadata['msg'],
                                               solver_values))

    def record_metadata(self, object_requesting_recording):
        """
        Route the record_metadata call to the proper object.
        """
        if self.options['record_metadata']:
            if isinstance(object_requesting_recording, Driver):
                self.record_metadata_driver(object_requesting_recording)
            elif isinstance(object_requesting_recording, System):
                self.record_metadata_system(object_requesting_recording)
            elif isinstance(object_requesting_recording, Solver):
                record_metadata_solver(object_requesting_recording)

    def record_metadata_driver(self, object_requesting_recording):
        """
        Record driver metadata.
        """
        driver_class = type(object_requesting_recording).__name__
        model_viewer_data = cPickle.dumps(object_requesting_recording._model_viewer_data,
                                          cPickle.HIGHEST_PROTOCOL)
        self.con.execute("INSERT INTO driver_metadata(id, model_viewer_data) "
                         "VALUES(?,:model_viewer_data)", (driver_class,
                                                          sqlite3.Binary(model_viewer_data)))

    def record_metadata_system(self, object_requesting_recording):
        """
        Record system metadata.
        """
        self.con.execute("INSERT INTO system_metadata(id, scaling_factor_input, "
                         "scaling_factor_output, scaling_factor_residual) VALUES(?,?,?,?)",
                         (object_requesting_recording.pathname,
                          object_requesting_recording._scaling_to_phys['input'],
                          object_requesting_recording._scaling_to_phys['output']),
                         object_requesting_recording._scaling_to_phys['residual'])

    def record_metadata_solver(self, object_requesting_recording):
        """
        Record solver metadata.
        """
        path = object_requesting_recording._system.pathname
        solver_class = type(object_requesting_recording).__name__
        id = "%s.%s".format(path, solver_class)

        solver_options = cPickle.dumps(object_requesting_recording.options,
                                       cPickle.HIGHEST_PROTOCOL)
        self.con.execute(
            "INSERT INTO solver_metadata(id, solver_options, solver_class) "
            "VALUES(?, :solver_options,?)", (id, sqlite3.Binary(solver_options), solver_class))

    def close(self):
        """
        Close `out`.
        """
        self.con.close()  # Not completely sure if it is this simple.
