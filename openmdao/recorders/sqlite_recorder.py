"""Class definition for SqliteRecorder, which provides dictionary backed by SQLite"""

from collections import OrderedDict
from sqlitedict import SqliteDict

import sqlite3
import numpy as np

from openmdao.recorders.base_recorder import BaseRecorder

from openmdao.core.driver import Driver

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
        self.con.execute("CREATE TABLE driver_iterations(id INTEGER PRIMARY KEY, iteration_coordinate TEXT, driver_values array)")

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

    def startup(self, group):
        super(SqliteRecorder, self).startup(group)

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
            params = group.params.iteritems()
            #resids = group.resids.iteritems()
            unknowns = group.unknowns.iteritems()
            self.out_metadata['Parameters'] = dict(params)
            self.out_metadata['Unknowns'] = dict(unknowns)
            self.out_metadata['system_metadata'] = group.metadata
            self.out_metadata['model_viewer_data'] = self.model_viewer_data



    # def record_iteration_the_blue_edition(self):
    #         # for recording Drivers
    #         design_vars=[], 
    #         response=[], 
    #         objectives=[], 
    #         constraints=[], 

    #         # For recording Systems
    #         inputs=[], 
    #         outputs=[], 
    #         residuals=[], 

    #         # For recording Solvers
    #         absolute_error=[], 
    #         relative_error=[], 
    #         full_outputs=[], 
    #         residuals=[], 

            # metadata

        # Not sure if we want to use the OrderDict method of aggregating 
        #   the data that we are going to stuff into sqlite
        data = OrderedDict()

        # This is just what we did in clippy. Need to re-think all this
        iteration_coordinate = metadata['coord']
        group_name = format_iteration_coordinate(iteration_coordinate)

        data['timestamp'] = metadata['timestamp']
        data['success'] = metadata['success']
        data['msg'] = metadata['msg']

        if design_vars and self.options['record_design_vars']:
            data['design_vars'] = design_vars

        if response and self.options['record_response']:
            data['response'] = response

        if objectives and self.options['record_objectives']:
            data['objectives'] = objectives

        # ... and so on for the rest of the possible

        self.out_iterations[group_name] = data






    # def record_iteration(self, params, unknowns, resids, metadata):
    def record_iteration(self, object_requesting_recording ):  ###### this is the one to look at Keith!
        """
        Stores the provided data in the sqlite file using the iteration
        coordinate for the key.

        Args
        ----
        params : dict
            Dictionary containing parameters. (p)

        unknowns : dict
            Dictionary containing outputs and states. (u)

        resids : dict
            Dictionary containing residuals. (r)

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

        self._counter += 1 # I don't think we need this because we can use the primary key feature of sqlite
        if isinstance(object_requesting_recording,Driver):

            # Just an example of the syntax for creating a numpy structured array
            # arr = np.zeros((1,), dtype=[('dv_x','(5,)f8'),('dv_y','(10,)f8')])

            design_vars = object_requesting_recording.get_design_var_values()
            # This returns a dict of names and values. Use this to build up the tuples of 
            #   used for the dtypes in the creation of the numpy structured array we want to write to sqlite
            dtype_tuples = []
            for name, value in iteritems(design_vars):
                tple = ('design_var.' + name, '({},)f8'.format(len(value)) )
                dtype_tuples.append(tple)

            # Do the same thing for response, objective and constraint using 
            #     get_response_values  -- even though Driver does not really support this!
            #     get_objective_values
            #     get_constraint_values
            # Just keep adding to the same dtype_tuples but you have to change the prepended string
            #     to indicate where the data came from, e.g. 'design_var.'


            # Create the mega array that we will write to the database
            # All of this needs to be looked into to be optimized !! 
            driver_values = np.zeros((1,), dtype=dtype_tuples)

            # Write the actual values to this array
            # Wish we didn't have to loop through this twice
            for name, value in iteritems(design_vars):
                driver_values['design_var.' + name] = value
            # Repeat for response, objective and constraint

            # Write this mega array to the database
            self.con.execute("INSERT INTO driver_iterations(iteration_coordinate,driver_values) VALUES(?,?)",
                 ("foobar1/1", driver_values))


        return ####### NOTICE THIS ########

        # # TODO_RECORDERS: Clean up the stuff below as needed

        # if MPI and MPI.COMM_WORLD.rank > 0 :
        #     raise RuntimeError("not rank 0")

        # data = OrderedDict()
        # iteration_coordinate = metadata['coord']
        # timestamp = metadata['timestamp']

        # group_name = format_iteration_coordinate(iteration_coordinate)

        # data['timestamp'] = timestamp
        # data['success'] = metadata['success']
        # data['msg'] = metadata['msg']

        # if self.options['record_params']:
        #     data['Parameters'] = self._filter_vector(params, 'p', iteration_coordinate)

        # if self.options['record_unknowns']:
        #     data['Unknowns'] = self._filter_vector(unknowns, 'u', iteration_coordinate)

        # if self.options['record_resids']:
        #     data['Residuals'] = self._filter_vector(resids, 'r', iteration_coordinate)

        # self.out_iterations[group_name] = data



    def record_iteration_drayton_nethod(self, object_requesting_recording, **kwargs):

        if isinstance(object_requesting_recording,System):
            self.record_system_iteration(kwargs['inputs'], kwargs['outputs'],kwargs['residuals'])

        if isinstance(object_requesting_recording,Driver):
            self.record_driver_iteration(kwargs['inputs'], kwargs['outputs'],kwargs['residuals'])


    def record_system_iteration(self, inputs, outputs, residuals):
            # blah, blah
            pass

    def record_driver_iteration(self, something):
            # blah, blah
            pass

    def record_derivatives(self, derivs, metadata):
        """Writes the derivatives that were calculated for the driver.

        Args
        ----
        derivs : dict or ndarray depending on the optimizer
            Dictionary containing derivatives

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

        data = OrderedDict()
        iteration_coordinate = metadata['coord']
        timestamp = metadata['timestamp']

        group_name = format_iteration_coordinate(iteration_coordinate)

        data['timestamp'] = timestamp
        data['success'] = metadata['success']
        data['msg'] = metadata['msg']
        data['Derivatives'] = derivs

        self.out_derivs[group_name] = data

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
