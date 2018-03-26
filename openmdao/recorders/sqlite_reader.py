"""
Definition of the SqliteCaseReader.
"""
from __future__ import print_function, absolute_import

import re
import sys
import sqlite3
import numpy as np

from collections import OrderedDict
from openmdao.recorders.base_case_reader import BaseCaseReader
from openmdao.recorders.case import DriverCase, SystemCase, SolverCase
from openmdao.recorders.cases import BaseCases
from openmdao.recorders.sqlite_recorder import blob_to_array
from openmdao.utils.record_util import is_valid_sqlite3_db

from six import PY2, PY3, iteritems

if PY2:
    import cPickle as pickle
if PY3:
    import pickle

_DEFAULT_OUT_STREAM = object()


class SqliteCaseReader(BaseCaseReader):
    """
    A CaseReader specific to files created with SqliteRecorder.

    Parameters
    ----------
    filename : str
        The path to the filename containing the recorded data.

    Attributes
    ----------
    format_version : int
        The version of the format assumed when loading the file.
    units : {'name': unit}
        Dictionary mapping absolute names to their units.
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    _column_widths : dict
        widths of the columns
    _align : str
        The Python formatting alignment used when writing values into columns
    _column_spacing: int
        Number of spaces between columns
    _indent_inc: int
        Number of spaces indented in levels of the hierarchy
    """

    def __init__(self, filename):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The path to the filename containing the recorded data.
        """
        super(SqliteCaseReader, self).__init__(filename)

        if filename is not None:
            if not is_valid_sqlite3_db(filename):
                raise IOError('File does not contain a valid '
                              'sqlite database ({0})'.format(filename))

        with sqlite3.connect(self.filename) as con:
            cur = con.cursor()
            cur.execute("SELECT format_version, abs2prom, prom2abs, abs2meta FROM metadata")
            row = cur.fetchone()
            self.format_version = row[0]
            self._abs2prom = None
            self._prom2abs = None
            self.abs2meta = None

            if PY2:
                self._abs2prom = pickle.loads(str(row[1])) if row[1] is not None else None
                self._prom2abs = pickle.loads(str(row[2])) if row[2] is not None else None
                self.abs2meta = pickle.loads(str(row[3])) if row[3] is not None else None
            if PY3:
                self._abs2prom = pickle.loads(row[1]) if row[1] is not None else None
                self._prom2abs = pickle.loads(row[2]) if row[2] is not None else None
                self.abs2meta = pickle.loads(row[3]) if row[3] is not None else None
        con.close()

        self._column_widths = {
            'value': 20,
            'resids': 20,
            'units': 10,
            'shape': 10,
            'lower': 20,
            'upper': 20,
            'ref': 20,
            'ref0': 20,
            'res_ref': 20,
        }
        self._align = ''
        self._column_spacing = 2
        self._indent_inc = 2

        self._load()

    def _load(self):
        """
        Load data from the sqlite database file.

        Load the metadata from the sqlite file, populating the
        `format_version`, `parameters`, and `unknowns` attributes of this
        CaseReader.

        The `iterations` table is read to load the keys which identify
        the individual cases/iterations from the recorded file.
        """
        self.driver_cases = DriverCases(self.filename)
        self.system_cases = SystemCases(self.filename)
        self.solver_cases = SolverCases(self.filename)

        self.driver_cases._prom2abs = self._prom2abs
        self.system_cases._prom2abs = self._prom2abs
        self.solver_cases._prom2abs = self._prom2abs

        if self.format_version in (1,):
            with sqlite3.connect(self.filename) as con:

                # Read in iterations from Drivers, Systems, and Solvers
                cur = con.cursor()
                cur.execute("SELECT iteration_coordinate FROM driver_iterations")
                rows = cur.fetchall()
                self.driver_cases._case_keys = [coord[0] for coord in rows]
                self.driver_cases.num_cases = len(self.driver_cases._case_keys)

                cur.execute("SELECT iteration_coordinate FROM system_iterations")
                rows = cur.fetchall()
                self.system_cases._case_keys = [coord[0] for coord in rows]
                self.system_cases.num_cases = len(self.system_cases._case_keys)

                cur.execute("SELECT iteration_coordinate FROM solver_iterations")
                rows = cur.fetchall()
                self.solver_cases._case_keys = [coord[0] for coord in rows]
                self.solver_cases.num_cases = len(self.solver_cases._case_keys)

                # Read in metadata for Drivers, Systems, and Solvers
                cur.execute("SELECT model_viewer_data FROM driver_metadata")
                for row in cur:
                    if PY2:
                        self.driver_metadata = pickle.loads(str(row[0]))
                    if PY3:
                        self.driver_metadata = pickle.loads(row[0])

                cur.execute("SELECT id, scaling_factors, user_metadata FROM system_metadata")
                for row in cur:
                    id = row[0]
                    self.system_metadata[id] = {}

                    if PY2:
                        self.system_metadata[id]['scaling_factors'] = pickle.loads(str(row[1]))
                        self.system_metadata[id]['user_metadata'] = pickle.loads(str(row[2]))
                    if PY3:
                        self.system_metadata[id]['scaling_factors'] = pickle.loads(row[1])
                        self.system_metadata[id]['user_metadata'] = pickle.loads(row[2])

                cur.execute("SELECT id, solver_options, solver_class FROM solver_metadata")
                for row in cur:
                    id = row[0]
                    if PY2:
                        solver_options = pickle.loads(str(row[1]))
                    if PY3:
                        solver_options = pickle.loads(row[1])
                    solver_class = row[2]
                    self.solver_metadata[id] = {
                        'solver_options': solver_options,
                        'solver_class': solver_class,
                    }
            con.close()
        else:
            raise ValueError('SQliteCaseReader encountered an unhandled '
                             'format version: {0}'.format(self.format_version))

    def get_child_cases(self, parent=None, recursive=False):
        """
        Allows one to iterate over the driver and solver cases.

        Parameters
        ----------
        parent : DriverCase or SolverCase or str
            Identifies which case's children to return. None indicates Root. Can pass a
            driver case, a solver case, or an iteration coordinate identifying a solver
            or driver case.
        recursive : bool
            If True, will enable iterating over all successors in case hierarchy
            rather than just the direct children.

        Returns
        -------
            Generator giving Driver and/or Solver cases in order.
        """
        iter_coord = ''
        if parent is not None:
            if parent is DriverCase or parent is SolverCase:
                iter_coord = parent.iteration_coordinate
            elif type(parent) is str:
                iter_coord = parent
            else:
                raise TypeError("parent parameter can only be DriverCase, SolverCase, or string")

        driver_iter = []
        solver_iter = []
        with sqlite3.connect(self.filename) as con:
            cur = con.cursor()
            cur.execute("SELECT iteration_coordinate FROM driver_iterations "
                        "ORDER BY counter ASC")
            driver_iter = cur.fetchall()

            cur.execute("SELECT iteration_coordinate, counter FROM solver_iterations "
                        "ORDER BY counter ASC")
            solver_iter = cur.fetchall()
        con.close()

        split_iter_coord = self._split_coordinate(iter_coord) if iter_coord is not ''\
            else []

        # grab an array of possible lengths of coordinates
        coord_lengths = [1]  # start with 1 because that is the length of driver iteration coords
        for s in solver_iter:
            s_len = len(self._split_coordinate(s[0]))
            if s_len not in coord_lengths:
                coord_lengths.append(s_len)
        coord_lengths = sorted(coord_lengths)

        # grab full set of cases to iterate over
        iter_set = self._find_child_cases(iter_coord, split_iter_coord, driver_iter,
                                          solver_iter, recursive, coord_lengths)

        # iterate over set of cases
        for iteration in iter_set:
            if iteration[1] is 'driver':
                yield self.driver_cases.get_case(iteration[0])
            else:
                yield self.solver_cases.get_case(iteration[0])

    def _find_child_cases(self, parent_iter_coord, split_parent_iter_coord, driver_iter,
                          solver_iter, recursive, coord_lengths):
        """
        Finds all child cases of a given parent.

        Parameters
        ----------
        parent_iter_coord : str
            Iteration coordinate of the parent case. If empty string, assumes root is parent.
        split_parent_iter_coord : [str]
            The split parent iteration coordinate.
        driver_iter : [(str)]
            The ordered list of driver iteration coordinates.
        solver_iter : [(str)]
            The ordered list of solver iteration coordinates.
        recursive : bool
            If True, will grab all successors recursively. Otherwise, will only return direct
            children.
        coord_lengths : [int]
            Sorted array of possible coordinate lengths. Used to determine if case is child
            of another case.

        Returns
        -------
            List of tuples of the form ('iteration_coordinate', 'type of iteration')
        """
        ret = []
        par_len = len(split_parent_iter_coord)
        par_len_idx = coord_lengths.index(par_len if par_len is not 0 else 1)
        expected_child_length = coord_lengths[par_len_idx + 1] if par_len_idx <\
            len(coord_lengths) - 1 else -1
        if parent_iter_coord is '':  # CASE: grabbing children of 'root'
            if len(driver_iter) > 0:  # grabbing all driver cases
                for d in driver_iter:
                    ret.append((d[0], 'driver'))
                    if recursive:
                        ret += self._find_child_cases(d[0], self._split_coordinate(d[0]),
                                                      driver_iter, solver_iter, recursive,
                                                      coord_lengths)
            elif len(solver_iter) > 0:  # grabbing first layer of solver iterations
                # find the iteration coordinate length of the first layer of solver iterations
                min_iter_len = (0, len(self._split_coordinate(solver_iter[0])))
                for idx, s in solver_iter:
                    length = len(self._split_coordinate(s[0]))
                    if length < min_iter_len[1]:
                        min_iter_len[0] = idx
                        min_iter_len[1] = length

                for s in solver_iter:
                    split_coord = self._split_coordinate(s[0])
                    if len(split_coord) is min_iter_len[1]:
                        ret.append((s[0], 'solver'))
                        if recursive:
                            ret += self._find_child_cases(s[0], split_coord, driver_iter,
                                                          solver_iter, recursive, coord_lengths)
        else:  # CASE: grabbing children of a case
            for s in solver_iter:
                if self._is_case_child(parent_iter_coord, s[0], expected_child_length):
                    ret.append((s[0], 'solver'))
                    if recursive:
                        ret += self._find_child_cases(s[0], self._split_coordinate(s[0]),
                                                      driver_iter, solver_iter, recursive,
                                                      coord_lengths)

        return ret

    def _is_case_child(self, parent_coordinate, coordinate, expected_child_length):
        """
        Tells if the given iteration coordinate indicates that the associated case is a
        child case of the parent.

        Parameters
        ----------
        parent_coordinate : str
            The iteration coordinate of the potential parent.
        coordinate : str
            Iteration coordinate of the case we want to test.
        expected_child_length : int
            Expected length of the split child iteration coordinate

        Returns
        -------
            True if the given coordinate indicates that the case is a child of the
            given parent case. False otherwise.
        """
        split_coord = self._split_coordinate(coordinate)
        if coordinate.startswith(parent_coordinate) and\
           len(split_coord) is expected_child_length:
            return True

        return False

    def _split_coordinate(self, coordinate):
        """
        Splits up an iteration coordinate string based on the iteration index.

        Parameters
        ----------
        coordinate : str
            The iteration coordinate to split.

        Returns
        -------
            coordinate as array of strings.
        """
        return re.compile('\|\\d+\|').split(coordinate)

    def list_outputs(self,
                     explicit=True, implicit=True,
                     values=True,
                     residuals=False,
                     residuals_tol=None,
                     units=False,
                     shape=False,
                     bounds=False,
                     scaling=False,
                     hierarchical=True,
                     print_arrays=False,
                     out_stream=_DEFAULT_OUT_STREAM):
        """
        Return and optionally log a list of output names and other optional information.

        If the model is parallel, only the local variables are returned to the process.
        Also optionally logs the information to a user defined output stream. If the model is
        parallel, the rank 0 process logs information about all variables across all processes.

        Parameters
        ----------
        explicit : bool, optional
            include outputs from explicit components. Default is True.
        implicit : bool, optional
            include outputs from implicit components. Default is True.
        values : bool, optional
            When True, display/return output values. Default is True.
        residuals : bool, optional
            When True, display/return residual values. Default is False.
        residuals_tol : float, optional
            If set, limits the output of list_outputs to only variables where
            the norm of the resids array is greater than the given 'residuals_tol'.
            Default is None.
        units : bool, optional
            When True, display/return units. Default is False.
        shape : bool, optional
            When True, display/return the shape of the value. Default is False.
        bounds : bool, optional
            When True, display/return bounds (lower and upper). Default is False.
        scaling : bool, optional
            When True, display/return scaling (ref, ref0, and res_ref). Default is False.
        hierarchical : bool, optional
            When True, human readable output shows variables in hierarchical format.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format  is affected
            by the values set with numpy.set_printoptions
            Default is False.
        out_stream : file-like
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list
            list of output names and other optional information about those outputs
        """
        meta = self.abs2meta
        expl_outputs = []
        impl_outputs = []
        final_system_case = self.system_cases.get_case(-1)
        outputs = final_system_case.outputs._values
        residuals = final_system_case.residuals._values
        if final_system_case is not None:
            for name in outputs.dtype.names:
                if residuals_tol and residuals is not None and\
                   np.linalg.norm(residuals[name]) < residuals_tol:
                    continue
                outs = {}
                if values:
                    outs['value'] = outputs[name]
                if residuals:
                    outs['resids'] = residuals[name]
                if units:
                    outs['units'] = meta[name]['units']
                if shape:
                    outs['shape'] = outputs[name].shape
                if bounds:
                    outs['lower'] = meta[name]['lower']
                    outs['upper'] = meta[name]['upper']
                if scaling:
                    outs['ref'] = meta[name]['ref']
                    outs['ref0'] = meta[name]['ref0']
                    outs['res_ref'] = meta[name]['res_ref']
                if meta[name]['type'] == 'Explicit':
                    expl_outputs.append((name, outs))
                else:
                    impl_outputs.append((name, outs))

        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if out_stream:
            if explicit:
                self._write_outputs('output', 'Explicit', expl_outputs, hierarchical, print_arrays,
                                    out_stream)

            if implicit:
                self._write_outputs('output', 'Implicit', impl_outputs, hierarchical, print_arrays,
                                    out_stream)

        if explicit and implicit:
            return expl_outputs + impl_outputs
        elif explicit:
            return expl_outputs
        elif implicit:
            return impl_outputs
        else:
            raise RuntimeError('You have excluded both Explicit and Implicit components.')

    def _write_outputs(self, in_or_out, comp_type, outputs, hierarchical, print_arrays,
                       out_stream):
        """
        Write table of variable names, values, residuals, and metadata to out_stream.

        The output values could actually represent input variables.
        In this context, outputs refers to the data that is being logged to an output stream.

        Parameters
        ----------
        in_or_out : str, 'input' or 'output'
            indicates whether the values passed in are from inputs or output variables.
        comp_type : str, 'Explicit' or 'Implicit'
            the type of component with the output values.
        outputs : list
            list of (name, dict of vals and metadata) tuples.
        hierarchical : bool
            When True, human readable output shows variables in hierarchical format.
        print_arrays : bool
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format  is affected
            by the values set with numpy.set_printoptions
            Default is False.
        out_stream : file-like object
            Where to send human readable output.
            Set to None to suppress.
        """
        if out_stream is None:
            return

        # Only local metadata but the most complete
        meta = self.abs2meta

        # Make a dict of outputs. Makes it easier to work with in this method
        dict_of_outputs = OrderedDict()
        for name, vals in outputs:
            dict_of_outputs[name] = vals

        count = len(dict_of_outputs)

        # Write header
        pathname = 'model'
        header_name = 'Input' if in_or_out == 'input' else 'Output'
        if in_or_out == 'input':
            header = "%d %s(s) in '%s'" % (count, header_name, pathname)
        else:
            header = "%d %s %s(s) in '%s'" % (count, comp_type, header_name, pathname)
        out_stream.write(header + '\n')
        out_stream.write('-' * len(header) + '\n' + '\n')

        if not count:
            return

        # Need an ordered list of possible output values for the two cases: inputs and outputs
        #  so that we do the column output in the correct order
        if in_or_out == 'input':
            out_types = ('value', 'units',)
        else:
            out_types = ('value', 'resids', 'units', 'shape', 'lower', 'upper', 'ref',
                         'ref0', 'res_ref')
        # Figure out which columns will be displayed
        # Look at any one of the outputs, they should all be the same
        outputs = dict_of_outputs[list(dict_of_outputs)[0]]
        column_names = []
        for out_type in out_types:
            if out_type in outputs:
                column_names.append(out_type)

        top_level_system_name = 'top'

        # Find with width of the first column in the table
        #    Need to look through all the possible varnames to find the max width
        max_varname_len = max(len(top_level_system_name), len('varname'))
        if hierarchical:
            for name, outs in iteritems(dict_of_outputs):
                for i, name_part in enumerate(name.split('.')):
                    total_len = (i + 1) * self._indent_inc + len(name_part)
                    max_varname_len = max(max_varname_len, total_len)
        else:
            for name, outs in iteritems(dict_of_outputs):
                max_varname_len = max(max_varname_len, len(name))

        # Determine the column widths of the data fields by finding the max width for all rows
        for column_name in column_names:
            self._column_widths[column_name] = len(column_name)  # has to be able to display name!
        for name in dict_of_outputs:
            if name in dict_of_outputs:
                for column_name in column_names:
                    if isinstance(dict_of_outputs[name][column_name], np.ndarray) and \
                            dict_of_outputs[name][column_name].size > 1:
                        out = '|{}|'.format(str(np.linalg.norm(dict_of_outputs[name][column_name])))
                    else:
                        out = str(dict_of_outputs[name][column_name])
                    self._column_widths[column_name] = max(self._column_widths[column_name],
                                                           len(str(out)))

        # Write out the column headers
        column_header = '{:{align}{width}}'.format('varname', align=self._align,
                                                   width=max_varname_len)
        column_dashes = max_varname_len * '-'
        for column_name in column_names:
            column_header += self._column_spacing * ' '
            column_header += '{:{align}{width}}'.format(column_name, align=self._align,
                                                        width=self._column_widths[column_name])
            column_dashes += self._column_spacing * ' ' + self._column_widths[column_name] * '-'
        out_stream.write(column_header + '\n')
        out_stream.write(column_dashes + '\n')

        # Write out the variable names and optional values and metadata
        if hierarchical:
            out_stream.write(top_level_system_name + '\n')

            cur_sys_names = []
            # _var_allprocs_abs_names has all the vars across all procs in execution order
            #   But not all the values need to be written since, at least for output vars,
            #      the output var lists are divided into explicit and implicit
            for varname in dict_of_outputs:
                # For hierarchical, need to display system levels in the rows above the
                #   actual row containing the var name and values. Want to make use
                #   of the hierarchies that have been written about this.
                existing_sys_names = []
                varname_sys_names = varname.split('.')[:-1]
                for i, sys_name in enumerate(varname_sys_names):
                    if varname_sys_names[:i + 1] != cur_sys_names[:i + 1]:
                        break
                    else:
                        existing_sys_names = cur_sys_names[:i + 1]

                # What parts of the hierarchy for this varname need to be written that
                #   were not already written above this
                remaining_sys_path_parts = varname_sys_names[len(existing_sys_names):]

                # Write the Systems in the var name path
                indent = len(existing_sys_names) * self._indent_inc
                for i, sys_name in enumerate(remaining_sys_path_parts):
                    indent += self._indent_inc
                    out_stream.write(indent * ' ' + sys_name + '\n')
                cur_sys_names = varname_sys_names

                indent += self._indent_inc
                row = '{:{align}{width}}'.format(indent * ' ' + varname.split('.')[-1],
                                                 align=self._align, width=max_varname_len)
                self._write_outputs_rows(out_stream, row, column_names, dict_of_outputs[varname],
                                         print_arrays)
        else:
            for name in dict_of_outputs:
                if name in dict_of_outputs:
                    row = '{:{align}{width}}'.format(name, align=self._align, width=max_varname_len)
                    self._write_outputs_rows(out_stream, row, column_names, dict_of_outputs[name],
                                             print_arrays)
        out_stream.write(2 * '\n')

    def _write_outputs_rows(self, out_stream, row, column_names, dict_of_outputs, print_arrays):
        """
        For one variable, write name, values, residuals, and metadata to out_stream.

        Parameters
        ----------
        out_stream : file-like object
            Where to send human readable output.
            Set to None to suppress.
        row : str
            The string containing the contents of the beginning of this row output.
            Contains the name of the System or varname, possibley indented to show hierarchy.

        column_names : list of str
            Indicates which columns will be written in this row.

        dict_of_outputs : dict
            Contains the values to be written in this row. Keys are columns names.

        print_arrays : bool
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format  is affected
            by the values set with numpy.set_printoptions
            Default is False.

        """
        if out_stream is None:
            return
        left_column_width = len(row)
        have_array_values = []  # keep track of which values are arrays
        for column_name in column_names:
            row += self._column_spacing * ' '
            if isinstance(dict_of_outputs[column_name], np.ndarray) and \
                    dict_of_outputs[column_name].size > 1:
                have_array_values.append(column_name)
                out = '|{}|'.format(str(np.linalg.norm(dict_of_outputs[column_name])))
            else:
                out = str(dict_of_outputs[column_name])
            row += '{:{align}{width}}'.format(out, align=self._align,
                                              width=self._column_widths[column_name])
        out_stream.write(row + '\n')
        if print_arrays:
            for column_name in have_array_values:
                out_stream.write("{}  {}:\n".format(left_column_width * ' ', column_name))
                out_str = str(dict_of_outputs[column_name])
                indented_lines = [(left_column_width + self._indent_inc) * ' ' +
                                  s for s in out_str.splitlines()]
                out_stream.write('\n'.join(indented_lines) + '\n')


class DriverCases(BaseCases):
    """
    Case specific to the entries that might be recorded in a Driver iteration.
    """

    def get_case(self, case_id):
        """
        Get a case from the database.

        Parameters
        ----------
        case_id : int or str
            The integer index or string-identifier of the case to be retrieved.

        Returns
        -------
            An instance of a Driver Case populated with data from the
            specified case/iteration.
        """
        iteration_coordinate = self.get_iteration_coordinate(case_id)

        with sqlite3.connect(self.filename) as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM driver_iterations WHERE "
                        "iteration_coordinate=:iteration_coordinate",
                        {"iteration_coordinate": iteration_coordinate})
            # Initialize the Case object from the iterations data
            row = cur.fetchone()
        con.close()

        idx, counter, iteration_coordinate, timestamp, success, msg, desvars_blob, responses_blob, \
            objectives_blob, constraints_blob, sysincludes_blob = row

        desvars_array = blob_to_array(desvars_blob)
        responses_array = blob_to_array(responses_blob)
        objectives_array = blob_to_array(objectives_blob)
        constraints_array = blob_to_array(constraints_blob)
        sysincludes_array = blob_to_array(sysincludes_blob)

        case = DriverCase(self.filename, counter, iteration_coordinate, timestamp, success, msg,
                          desvars_array, responses_array, objectives_array, constraints_array,
                          sysincludes_array, self._prom2abs)

        return case


class SystemCases(BaseCases):
    """
    Case specific to the entries that might be recorded in a System iteration.
    """

    def get_case(self, case_id):
        """
        Get a case from the database.

        Parameters
        ----------
        case_id : int or str
            The integer index or string-identifier of the case to be retrieved.

        Returns
        -------
            An instance of a System Case populated with data from the
            specified case/iteration.
        """
        iteration_coordinate = self.get_iteration_coordinate(case_id)

        with sqlite3.connect(self.filename) as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM system_iterations WHERE "
                        "iteration_coordinate=:iteration_coordinate",
                        {"iteration_coordinate": iteration_coordinate})
            # Initialize the Case object from the iterations data
            row = cur.fetchone()
        con.close()

        # inputs , outputs , residuals
        idx, counter, iteration_coordinate, timestamp, success, msg, inputs_blob,\
            outputs_blob, residuals_blob = row

        inputs_array = blob_to_array(inputs_blob)
        outputs_array = blob_to_array(outputs_blob)
        residuals_array = blob_to_array(residuals_blob)

        case = SystemCase(self.filename, counter, iteration_coordinate, timestamp, success, msg,
                          inputs_array, outputs_array, residuals_array, self._prom2abs)

        return case


class SolverCases(BaseCases):
    """
    Case specific to the entries that might be recorded in a Solver iteration.
    """

    def get_case(self, case_id):
        """
        Get a case from the database.

        Parameters
        ----------
        case_id : int or str
            The integer index or string-identifier of the case to be retrieved.

        Returns
        -------
            An instance of a solver Case populated with data from the
            specified case/iteration.
        """
        iteration_coordinate = self.get_iteration_coordinate(case_id)

        with sqlite3.connect(self.filename) as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM solver_iterations WHERE "
                        "iteration_coordinate=:iteration_coordinate",
                        {"iteration_coordinate": iteration_coordinate})
            # Initialize the Case object from the iterations data
            row = cur.fetchone()
        con.close()

        idx, counter, iteration_coordinate, timestamp, success, msg, abs_err, rel_err, \
            output_blob, residuals_blob = row

        output_array = blob_to_array(output_blob)
        residuals_array = blob_to_array(residuals_blob)

        case = SolverCase(self.filename, counter, iteration_coordinate, timestamp, success, msg,
                          abs_err, rel_err, output_array, residuals_array, self._prom2abs)

        return case
