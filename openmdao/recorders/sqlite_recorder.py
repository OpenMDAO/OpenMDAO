"""Class definition for SqliteRecorder, which provides dictionary backed by SQLite"""

from collections import OrderedDict
from sqlitedict import SqliteDict
from openmdao.recorders.base_recorder import BaseRecorder
# from openmdao.util.record_util import format_iteration_coordinate

# from openmdao.devtools.partition_tree_n2 import get_model_viewer_data


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

        # if MPI and MPI.COMM_WORLD.rank > 0 :
        #     self._open_close_sqlitedict = False
        # else:
        #     self._open_close_sqlitedict = True


        self._open_close_sqlitedict = True


        if self._open_close_sqlitedict:
            sqlite_dict_args.setdefault('autocommit', True)
            # self.out_metadata = SqliteDict(filename=out, flag='n', tablename='metadata', **sqlite_dict_args)
            # self.out_metadata['format_version'] = format_version
            self.out_iterations = SqliteDict(filename=out, flag='w', tablename='iterations', **sqlite_dict_args)
            # self.out_derivs = SqliteDict(filename=out, flag='w', tablename='derivs', **sqlite_dict_args)

        else:
            self.out_metadata = None
            self.out_iterations = None
            self.out_derivs = None

    def startup(self):
        super(SqliteRecorder, self).startup()

        # Need this for use when recording the metadata
        # Can't do this in the record_metadata method because it only gets
        #   called for rank 0 when running in parallel and so the MPI gather
        #   that is called in that function won't work. All processes
        #   need to participate in that collective call


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

    # def record_iteration(self, params, unknowns, resids, metadata):
    def record_iteration(self, objective_values):
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
        data = OrderedDict()

        data['Objectives'] = objective_values
        self.out_iterations['qqqq'] = data # TODO_RECORDERS: Need a real iteration coordinate here

        return

        # TODO_RECORDERS: Clean up the stuff below as needed
        #
        # if MPI and MPI.COMM_WORLD.rank > 0 :
        #     raise RuntimeError("not rank 0")
        #
        # data = OrderedDict()
        # iteration_coordinate = metadata['coord']
        # timestamp = metadata['timestamp']
        #
        # group_name = format_iteration_coordinate(iteration_coordinate)
        #
        # data['timestamp'] = timestamp
        # data['success'] = metadata['success']
        # data['msg'] = metadata['msg']
        #
        # if self.options['record_params']:
        #     data['Parameters'] = self._filter_vector(params, 'p', iteration_coordinate)
        #
        # if self.options['record_unknowns']:
        #     data['Unknowns'] = self._filter_vector(unknowns, 'u', iteration_coordinate)
        #
        # if self.options['record_resids']:
        #     data['Residuals'] = self._filter_vector(resids, 'r', iteration_coordinate)
        #
        # self.out_iterations[group_name] = data

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

        if self._open_close_sqlitedict:
            # if self.out_metadata is not None:
            #     self.out_metadata.close()
            #     self.out_metadata = None
            if self.out_iterations is not None:
                self.out_iterations.close()
                self.out_iterations = None
            # if self.out_derivs is not None:
            #     self.out_derivs.close()
            #     self.out_derivs = None
