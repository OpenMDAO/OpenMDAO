"""Class definition for SqliteRecorder, which provides dictionary backed by SQLite"""

from collections import OrderedDict
from sqlitedict import SqliteDict

import sqlite3
import numpy as np

from openmdao.recorders.base_recorder import BaseRecorder

from openmdao.core.driver import Driver
from openmdao.core.system import System
from openmdao.solvers.solver import Solver

# from openmdao.util.record_util import format_iteration_coordinate

# from openmdao.devtools.partition_tree_n2 import get_model_viewer_data

from six import iteritems



import sqlite3
import numpy as np
import io


# this basically defines a special type of variable that can be written and read from a sqlite data base.
#  the type is called "array"
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)


# from openmdao.core.mpi_wrap import MPI

format_version = 1

class SqliteRecorder(BaseRecorder):
    """ Recorder that saves cases in an SQLite dictionary.

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

    def __init__(self, out, **sqlite_dict_args):
        super(SqliteRecorder, self).__init__()

        self.model_viewer_data = None

        # self._counter = None # maybe later
        self._counter = 0 

        # if MPI and MPI.COMM_WORLD.rank > 0 :
        #     self._open_close_sqlitedict = False
        # else:
        #     self._open_close_sqlitedict = True


        self._open_close_sqlitedict = True

        # self.con = None # sqlite connection

        # might have to do autocommit. Need to investigate
        self.con = sqlite3.connect(out, detect_types=sqlite3.PARSE_DECLTYPES)
        # Create the table for iterations for a driver
        # The primary key gets filled in automatically. It can be our "counter" that John wants. It gets incremented by 1
        #   for each write of a record to this table
        self.con.execute("CREATE TABLE driver_iterations(id INTEGER PRIMARY KEY, iteration_coordinate TEXT, \
                         timestamp REAL, success INT, msg TEXT, driver_values array)")
        self.con.execute("CREATE TABLE system_iterations(id INTEGER PRIMARY KEY, iteration_coordinate TEXT,  \
                         timestamp REAL, success INT, msg TEXT, system_values array)")
        self.con.execute("CREATE TABLE solver_iterations(id INTEGER PRIMARY KEY, iteration_coordinate TEXT,  \
                         timestamp REAL, success INT, msg TEXT, solver_values array)")

        # if self._open_close_sqlitedict:
        #     sqlite_dict_args.setdefault('autocommit', True)
        #     # self.out_metadata = SqliteDict(filename=out, flag='n', tablename='metadata', **sqlite_dict_args)
        #     # self.out_metadata['format_version'] = format_version
            
        #     #self.con.close()

        #     # self.out_iterations = SqliteDict(filename=out, flag='w', tablename='system_iterations', **sqlite_dict_args)
        #     # self.out_driver_iterations = SqliteDict(filename=out, flag='w', tablename='driver_iterations', **sqlite_dict_args)
        #     # self.out_iterations = SqliteDict(filename=out, flag='w', tablename='solver_iterations', **sqlite_dict_args)
        #     # self.out_derivs = SqliteDict(filename=out, flag='w', tablename='derivs', **sqlite_dict_args)

        # else:
        #     self.out_metadata = None
        #     self.out_iterations = None
        #     self.out_derivs = None

    def startup(self):
        super(SqliteRecorder, self).startup()

        # Need this for use when recording the metadata
        # Can't do this in the record_metadata method because it only gets
        #   called for rank 0 when running in parallel and so the MPI gather
        #   that is called in that function won't work. All processes
        #   need to participate in that collective call

        self._counter = 0 

        # self.model_viewer_data = get_model_viewer_data(group)

    def record_metadata(self, group):
        """Stores the metadata of the given group in a sqlite file using
        the variable name for the key.

        Args
        ----
        group : `System`
            `System` containing vectors
        """

        if MPI and MPI.COMM_WORLD.rank > 0 :
            raise RuntimeError("not rank 0")
        else:
            # params = group.params.iteritems()
            # #resids = group.resids.iteritems()
            # unknowns = group.unknowns.iteritems()
            # self.out_metadata['Parameters'] = dict(params)
            # self.out_metadata['Unknowns'] = dict(unknowns)
            # self.out_metadata['system_metadata'] = group.metadata
            # self.out_metadata['model_viewer_data'] = self.model_viewer_data
            pass


    def record_iteration(self, object_requesting_recording, metadata):
        """
        Stores the provided data in the sqlite file using the iteration
        coordinate for the key.
        """
        super(SqliteRecorder, self).record_iteration(self, object_requesting_recording, metadata)

        # Record an iteration from a Driver
        if isinstance(object_requesting_recording, Driver):
            self.record_iteration_driver(object_requesting_recording, metadata)

        elif isinstance(object_requesting_recording, System):
            self.record_iteration_system(object_requesting_recording, metadata)

        elif isinstance(object_requesting_recording, Solver):
            self.record_iteration_solver(object_requesting_recording, metadata)

        else:
            print ("YOU CAN'T ATTACH A RECORDER TO THIS OBJECT")


    def record_iteration_driver(self, object_requesting_recording, metadata):
        dtype_tuples = []

        #We will go through the recording options of Driver to construct the entry to be inserted.
        if self.options['record_desvars']:
            # Just an example of the syntax for creating a numpy structured array
            # arr = np.zeros((1,), dtype=[('dv_x','(5,)f8'),('dv_y','(10,)f8')])

            design_vars = object_requesting_recording.get_design_var_values()
            # This returns a dict of names and values. Use this to build up the tuples of
            #   used for the dtypes in the creation of the numpy structured array we want to write to sqlite
            for name, value in iteritems(design_vars):
                tple = ('design_var.' + name, '({},)f8'.format(len(value)) )
                dtype_tuples.append(tple)

        if self.options['record_responses']:
            responses = object_requesting_recording.get_response_values()

            for name, value in iteritems(responses):
                tple = ('response.' + name, '({},)f8'.format(len(value)))
                dtype_tuples.append(tple)

        if self.options['record_objectives']:
            objectives = object_requesting_recording.get_objective_values()

            for name, value in iteritems(objectives):
                tple = ('objective.' + name, '({},)f8'.format(len(value)))
                dtype_tuples.append(tple)

        if self.options['record_constraints']:
            constraints = object_requesting_recording.get_constraint_values()

            for name, value in iteritems(constraints):
                tple = ('constraint.' + name, '({},)f8'.format(len(value)))
                dtype_tuples.append(tple)

        print("DTYPE_TUPLES: ", dtype_tuples)

        driver_values = np.zeros((1,), dtype=dtype_tuples)

        # Write the actual values to this array
        if self.options['record_desvars']:
            for name, value in iteritems(design_vars):
                driver_values['design_var.' + name] = value
        if self.options['record_responses']:
            for name, value in iteritems(responses):
                driver_values['response.' + name] = value
        if self.options['record_objectives']:
            for name, value in iteritems(objectives):
                driver_values['objective.' + name] = value
        if self.options['record_constraints']:
            for name, value in iteritems(constraints):
                driver_values['constraint.' + name] = value

        print("DRIVER VALUES:", driver_values)
        # Write this mega array to the database
        self.con.execute("INSERT INTO driver_iterations(iteration_coordinate, timestamp, success, msg, driver_values) VALUES(?,?,?,?,?)",
             (metadata['coord'], metadata['timestamp'], metadata['success'], metadata['msg'], driver_values))

    def record_iteration_system(self, object_requesting_recording, metadata):
        # Record an iteration from a System
        if isinstance(object_requesting_recording, System):
            dtype_tuples = []

            # We will go through the recording options of Driver to construct the entry to be inserted.
            if self.options['record_inputs']:
                inputs = object_requesting_recording.get_inputs()
                for name, value in iteritems(design_vars):
                    tple = ('input.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

            if self.options['record_outputs']:
                outputs = object_requesting_recording.get_outputs()
                for name, value in iteritems(ouputs):
                    tple = ('output.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

            if self.options['record_residuals']:
                residuals = object_requesting_recording.get_residuals()
                for name, value in iteritems(residuals):
                    tple = ('residual.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

            if self.options['record_derivatives']:
                derivatives = object_requesting_recording.get_derivatives()
                for name, value in iteritems(derivatives):
                    tple = ('derivative.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

            print("DTYPE_TUPLES: ", dtype_tuples)

            # Create the mega array that we will write to the database
            # All of this needs to be looked into to be optimized !!
            system_values = np.zeros((1,), dtype=dtype_tuples)

            # Write the actual values to this array
            # Wish we didn't have to loop through this twice
            if self.options['record_inputs']:
                for name, value in iteritems(inputs):
                    system_values['input.' + name] = value
            if self.options['record_outputs']:
                for name, value in iteritems(outputs):
                    system_values['output.' + name] = value
            if self.options['record_residuals']:
                for name, value in iteritems(residuals):
                    system_values['residual.' + name] = value
            if self.options['record_derivatives']:
                for name, value in iteritems(derivatives):
                    system_values['derivative.' + name] = value

            print("SYSTEM VALUES:", system_values)
            # Write this mega array to the database
            # self.con.execute("INSERT INTO system_iterations(iteration_coordinate,driver_values) VALUES(?,?)",
            #                  ("foobar1/1", system_values))

            self.con.execute("INSERT INTO system_iterations(iteration_coordinate, timestamp, success, msg, system_values) \
                VALUES(?,?,?,?,?)", (metadata['coord'], metadata['timestamp'], metadata['success'], metadata['msg'],
                system_values))

    def record_iteration_solver(self, object_requesting_recording, metadata):
            dtype_tuples = []

            # We will go through the recording options of Driver to construct the entry to be inserted.
            if self.options['record_abs_error']:
                abs_errors = object_requesting_recording.get_abs_error()

                for name, value in iteritems(abs_errors):
                    tple = ('abs_error.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

            if self.options['record_rel_error']:
                rel_errors = object_requesting_recording.get_rel_error()

                for name, value in iteritems(rel_errors):
                    tple = ('rel_error.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

            if self.options['record_output']:
                outputs = object_requesting_recording.get_output()

                for name, value in iteritems(outputs):
                    tple = ('output.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

            if self.options['record_solver_residuals']:
                residuals = object_requesting_recording.get_residuals()

                for name, value in iteritems(residuals):
                    tple = ('residual.' + name, '({},)f8'.format(len(value)))
                    dtype_tuples.append(tple)

            print("DTYPE_TUPLES: ", dtype_tuples)

            # Create the mega array that we will write to the database
            # All of this needs to be looked into to be optimized !!
            solver_values = np.zeros((1,), dtype=dtype_tuples)

            # Write the actual values to this array
            # Wish we didn't have to loop through this twice
            if self.options['record_abs_error']:
                for name, value in iteritems(abs_errors):
                    solver_values['abs_error.' + name] = value
            if self.options['record_rel_error']:
                for name, value in iteritems(rel_errors):
                    solver_values['rel_error.' + name] = value
            if self.options['record_output']:
                for name, value in iteritems(outputs):
                    solver_values['output.' + name] = value
            if self.options['record_solver_residuals']:
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
            self.con.execute("INSERT INTO solver_iterations(iteration_coordinate, timestamp, success, msg, solver_values) "
                "VALUES(?,?,?,?,?)", (metadata['coord'], metadata['timestamp'], metadata['success'], metadata['msg'],
                                      solver_values))


    # def record_derivatives(self, derivs, metadata):
    #     """Writes the derivatives that were calculated for the driver.
    #
    #     Args
    #     ----
    #     derivs : dict or ndarray depending on the optimizer
    #         Dictionary containing derivatives
    #
    #     metadata : dict, optional
    #         Dictionary containing execution metadata (e.g. iteration coordinate).
    #     """
    #
    #     data = OrderedDict()
    #     iteration_coordinate = metadata['coord']
    #     timestamp = metadata['timestamp']
    #
    #     group_name = format_iteration_coordinate(iteration_coordinate)
    #
    #     data['timestamp'] = timestamp
    #     data['success'] = metadata['success']
    #     data['msg'] = metadata['msg']
    #     data['Derivatives'] = derivs
    #
    #     self.out_derivs[group_name] = data

    def close(self):
        """Closes `out`"""


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
