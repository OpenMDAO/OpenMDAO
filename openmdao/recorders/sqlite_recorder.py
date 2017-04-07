"""
Class definition for SqliteRecorder, which provides dictionary backed by SQLite.
"""
from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.core.driver import Driver
from openmdao.core.system import System
from openmdao.solvers.solver import Solver
from openmdao.utils.record_util import format_iteration_coordinate
# from openmdao.devtools.partition_tree_n2 import get_model_viewer_data

from six import iteritems

import sqlite3
import numpy as np
import io


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
    options['record_unknowns'] :  bool(True)
        Tells recorder whether to record the unknowns vector.
    options['record_params'] :  bool(False)
        Tells recorder whether to record the params vector.
    options['record_resids'] :  bool(False)
        Tells recorder whether to record the ressiduals vector.
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
        self.con = sqlite3.connect(out, detect_types=sqlite3.PARSE_DECLTYPES,isolation_level=None)

        # Create the table for iterations for a driver
        # The primary key gets filled in automatically. It can be our "counter"
        # that John wants. It gets incremented by 1
        #   for each write of a record to this table
        self.con.execute("CREATE TABLE driver_iterations(id INTEGER PRIMARY KEY, iteration_coordinate TEXT, \
                         timestamp REAL, success INT, msg TEXT, driver_values array)")
        self.con.execute("CREATE TABLE system_iterations(id INTEGER PRIMARY KEY, iteration_coordinate TEXT,  \
                         timestamp REAL, success INT, msg TEXT, system_values array)")
        self.con.execute("CREATE TABLE solver_iterations(id INTEGER PRIMARY KEY, iteration_coordinate TEXT,  \
                         timestamp REAL, success INT, msg TEXT, solver_values array)")

    def startup(self):
        """
        Startup.
        """
        super(SqliteRecorder, self).startup()
        self._counter = 0
        # self.model_viewer_data = get_model_viewer_data(group)

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
        dtype_tuples = []

        # We will go through the recording options of Driver to construct the entry to be inserted.
        if self.options['record_desvars']:
            # Just an example of the syntax for creating a numpy structured array
            # arr = np.zeros((1,), dtype=[('dv_x','(5,)f8'),('dv_y','(10,)f8')])

            design_vars = object_requesting_recording.get_design_var_values()
            # This returns a dict of names and values. Use this to build up the tuples of
            # used for the dtypes in the creation of the numpy structured array
            # we want to write to sqlite
            if design_vars:
                for name, value in iteritems(design_vars):
                    tple = ('design_var.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

        if self.options['record_responses']:
            responses = object_requesting_recording.get_response_values()

            if responses:
                for name, value in iteritems(responses):
                    tple = ('response.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

        if self.options['record_objectives']:
            objectives = object_requesting_recording.get_objective_values()

            if objectives:
                for name, value in iteritems(objectives):
                    tple = ('objective.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

        if self.options['record_constraints']:
            constraints = object_requesting_recording.get_constraint_values()

            if constraints:
                for name, value in iteritems(constraints):
                    tple = ('constraint.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

        driver_values = np.zeros((1,), dtype=dtype_tuples)

        # Write the actual values to this array
        if self.options['record_desvars'] and design_vars:
            for name, value in iteritems(design_vars):
                driver_values['design_var.' + name] = value
        if self.options['record_responses'] and responses:
            for name, value in iteritems(responses):
                driver_values['response.' + name] = value
        if self.options['record_objectives'] and objectives:
            for name, value in iteritems(objectives):
                driver_values['objective.' + name] = value
        if self.options['record_constraints'] and constraints:
            for name, value in iteritems(constraints):
                driver_values['constraint.' + name] = value

        print("INSERT RECORD: DRIVER VALUES:", metadata['coord'], metadata['timestamp'],
              metadata['success'], metadata['msg'], driver_values)
        # Write this mega array to the database
        self.con.execute("INSERT INTO driver_iterations(iteration_coordinate, timestamp, "
                         "success, msg, driver_values) VALUES(?,?,?,?,?)",
                         (format_iteration_coordinate(metadata['coord']), metadata['timestamp'],
                          metadata['success'], metadata['msg'], driver_values))

    def record_iteration_system(self, object_requesting_recording, metadata):
        """
        Record an iteration using driver options.
        """
        if isinstance(object_requesting_recording, System):
            dtype_tuples = []

            # go through the recording options of Driver to construct the entry to be inserted.
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
        Record an iteration using driver options.
        """
        dtype_tuples = []

        # Go through the recording options of Driver to construct the entry to be inserted.
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

        print("DTYPE_TUPLES: ", dtype_tuples)

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

    def close(self):
<<<<<<< HEAD
        """Closes `out`"""

        print("closing sqlite_recorder")
        self.con.close() # Not completely sure if it is this simple.

        # if self._open_close_sqlitedict:
        #     # if self.out_metadata is not None:
        #     #     self.out_metadata.close()
        #     #     self.out_metadata = None
        #     if self.out_driver_iterations is not None:
        #         self.out_driver_iterations.close()
        #         self.out_driver_iterations = None
        #     # if self.out_derivs is not None:
        #     #     self.out_derivs.close()
        #     #     self.out_derivs = None
=======
        """
        Close `out`.
        """
        self.con.close()  # Not completely sure if it is this simple.
>>>>>>> keithfork/master
