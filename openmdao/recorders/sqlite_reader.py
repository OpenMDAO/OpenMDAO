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
from openmdao.recorders.case import DriverCase, SystemCase, SolverCase, PromotedToAbsoluteMap
from openmdao.recorders.cases import BaseCases
from openmdao.recorders.sqlite_recorder import blob_to_array
from openmdao.utils.record_util import is_valid_sqlite3_db
from openmdao.utils.write_outputs import write_outputs

from six import PY2, PY3

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
    output2meta : dict
        Dictionary mapping output variables to their metadata
    input2meta : dict
        Dictionary mapping input variables to their metadata
    _abs2meta : dict
        Dictionary mapping variables to their metadata
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    _coordinate_split_re : RegularExpression
        Regular expression used for splitting iteration coordinates.
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
        self._coordinate_split_re = re.compile('\|\\d+\|*')

        with sqlite3.connect(self.filename) as con:
            cur = con.cursor()
            cur.execute("SELECT format_version, abs2prom, prom2abs, abs2meta FROM metadata")
            row = cur.fetchone()
            self.format_version = row[0]
            self._abs2prom = None
            self._prom2abs = None
            self._abs2meta = None

            if PY2:
                self._abs2prom = pickle.loads(str(row[1])) if row[1] is not None else None
                self._prom2abs = pickle.loads(str(row[2])) if row[2] is not None else None
                self._abs2meta = pickle.loads(str(row[3])) if row[3] is not None else None
            if PY3:
                self._abs2prom = pickle.loads(row[1]) if row[1] is not None else None
                self._prom2abs = pickle.loads(row[2]) if row[2] is not None else None
                self._abs2meta = pickle.loads(row[3]) if row[3] is not None else None
        con.close()

        self.output2meta = PromotedToAbsoluteMap(self._abs2meta, self._prom2abs,
                                                 self._abs2prom, True)
        self.input2meta = PromotedToAbsoluteMap(self._abs2meta, self._prom2abs,
                                                self._abs2prom, False)

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
        self.driver_cases = DriverCases(self.filename, self._abs2prom,
                                        self._abs2meta, self._prom2abs)
        self.system_cases = SystemCases(self.filename, self._abs2prom,
                                        self._abs2meta, self._prom2abs)
        self.solver_cases = SolverCases(self.filename, self._abs2prom,
                                        self._abs2meta, self._prom2abs)

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

                cur.execute("SELECT id, scaling_factors, component_metadata FROM system_metadata")
                for row in cur:
                    id = row[0]
                    self.system_metadata[id] = {}

                    if PY2:
                        self.system_metadata[id]['scaling_factors'] = pickle.loads(str(row[1]))
                        self.system_metadata[id]['component_options'] = pickle.loads(str(row[2]))
                    if PY3:
                        self.system_metadata[id]['scaling_factors'] = pickle.loads(row[1])
                        self.system_metadata[id]['component_options'] = pickle.loads(row[2])

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

    def get_cases(self, parent=None, recursive=False):
        """
        Allow one to iterate over the driver and solver cases.

        Generator giving Driver and/or Solver cases in order.

        Parameters
        ----------
        parent : DriverCase or SolverCase or str, optional
            Identifies which case's children to return. None indicates Root. Can pass a
            driver case, a solver case, or an iteration coordinate identifying a solver
            or driver case. Defaults to None.
        recursive : bool, optional
            If True, will enable iterating over all successors in case hierarchy
            rather than just the direct children. Defaults to False.
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
        coord_lengths = [2]  # start with 2 because that is the length of driver iteration coords
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
        Find all children of a given parent case.

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
        list of tuples
            List of tuples of the form ('iteration_coordinate', 'type of iteration')
        """
        ret = []
        par_len = len(split_parent_iter_coord)
        par_len_idx = coord_lengths.index(par_len if par_len is not 0 else 2)
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
                min_iter_len = -1
                if len(coord_lengths) > 1:
                    min_iter_len = coord_lengths[1]

                for s in solver_iter:
                    split_coord = self._split_coordinate(s[0])
                    if len(split_coord) is min_iter_len:
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
        Tell if the given case is a child case of the parent.

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
        bool
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
        Split up an iteration coordinate string based on the iteration index.

        Parameters
        ----------
        coordinate : str
            The iteration coordinate to split.

        Returns
        -------
        list
            coordinate as list of strings.
        """
        return self._coordinate_split_re.split(coordinate)

    def list_inputs(self,
                    case=None,
                    values=True,
                    units=False,
                    hierarchical=True,
                    print_arrays=False,
                    out_stream=_DEFAULT_OUT_STREAM):
        """
        Return and optionally log a list of input names and other optional information.

        Also optionally logs the information to a user defined output stream.

        Parameters
        ----------
        case : Case, optional
            The case whose inputs will be listed. If None, gives all inputs. Defaults to None.
        values : bool, optional
            When True, display/return input values. Default is True.
        units : bool, optional
            When True, display/return units. Default is False.
        hierarchical : bool, optional
            When True, human readable output shows variables in hierarchical format.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format is affected
            by the values set with numpy.set_printoptions
            Default is False.
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list
            list of input names and other optional information about those inputs
        """
        meta = self._abs2meta
        if case is None:
            sys_vars = self._get_all_sysvars(False)
        else:
            sys_vars = self._get_case_sysvars(case, False)
        inputs = []

        if sys_vars is not None and len(sys_vars) > 0:
            for name in sys_vars:
                outs = {}
                if values:
                    outs['value'] = sys_vars[name]['value']
                if units:
                    outs['units'] = meta[name]['units']
                inputs.append((name, outs))

        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if out_stream:
            if sys_vars is None:
                out_stream.write('WARNING: No system cases recorded. Make sure the recorder ' +
                                 'is attached to a system object\n')
            elif len(sys_vars) is 0:
                out_stream.write('WARNING: Inputs not recorded. Make sure your recording ' +
                                 'settings have record_inputs set to True\n')

            self._write_outputs('input', None, inputs, hierarchical, print_arrays, out_stream)

        return inputs

    def list_outputs(self,
                     case=None,
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

        Also optionally logs the information to a user defined output stream.

        Parameters
        ----------
        case : Case, optional
            The case whose outputs will be listed. If None, gives all outputs. Defaults to None.
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
        meta = self._abs2meta
        expl_outputs = []
        impl_outputs = []
        sys_vars = self._get_all_sysvars()

        if case is None:
            sys_vars = self._get_all_sysvars()
        else:
            sys_vars = self._get_case_sysvars(case)

        if sys_vars is not None and len(sys_vars) > 0:
            for name in sys_vars:
                if residuals_tol and residuals_vars is not None and\
                   sys_vars[name]['residuals'] is not 'Not Recorded' and\
                   np.linalg.norm(sys_vars[name]['residuals']) < residuals_tol:
                    continue
                outs = {}
                if values:
                    outs['value'] = sys_vars[name]['value']
                if residuals:
                    outs['resids'] = sys_vars[name]['residuals']
                if units:
                    outs['units'] = meta[name]['units']
                if shape:
                    outs['shape'] = sys_vars[name]['value'].shape
                if bounds:
                    outs['lower'] = meta[name]['lower']
                    outs['upper'] = meta[name]['upper']
                if scaling:
                    outs['ref'] = meta[name]['ref']
                    outs['ref0'] = meta[name]['ref0']
                    outs['res_ref'] = meta[name]['res_ref']
                if meta[name]['explicit']:
                    expl_outputs.append((name, outs))
                else:
                    impl_outputs.append((name, outs))

        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if out_stream:
            if sys_vars is None:
                out_stream.write('WARNING: No system cases recorded. Make sure the recorder ' +
                                 'is attached to a system object\n')
            elif len(sys_vars) is 0:
                out_stream.write('WARNING: Outputs not recorded. Make sure your recording ' +
                                 'settings have record_outputs set to True\n')

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

    def _get_case_sysvars(self, case, get_outputs=True):
        """
        Get the set of output or input variables and their values for a given case.

        Parameters
        ----------
        case : Case
            The case whose variables will be returned.
        get_outputs : bool, optional
            indicates if the returned set should contain outputs. If false, returns inputs.

        Returns
        -------
        dictionary
            dictionary of global variable names to their values. None if no system iterations
            were recorded.
        """
        variables = {}
        if get_outputs and case.outputs is None:
            return variables

        outputs = case.outputs._values if case.outputs is not None else None
        residuals = case.residuals._values if case.residuals is not None else None
        inputs = case.inputs._values if case.inputs is not None else None
        if get_outputs:
            for var_name in outputs.dtype.names:
                variables[var_name] = {'value': outputs[var_name]}
                if residuals is not None and var_name in residuals.dtype.names:
                    variables[var_name]['residuals'] = residuals[var_name]
                else:
                    variables[var_name]['residuals'] = 'Not Recorded'
        elif inputs is not None:
            for var_name in inputs.dtype.names:
                if var_name not in variables:
                    variables[var_name] = {'value': inputs[var_name]}

        return variables

    def _get_all_sysvars(self, get_outputs=True):
        """
        Get the set of output or input variables and their values.

        Parameters
        ----------
        get_outputs : bool, optional
            indicates if the returned set should contain outputs. If false, returns inputs.

        Returns
        -------
        dictionary
            dictionary of global variable names to their values. None if no system iterations
            were recorded.
        """
        coords = self.system_cases._case_keys

        # store the iteration coordinates without iteration numbers.
        # coord_map intializes each iter_key to False, indicating we haven't
        # grabbed values from this system
        coord_map = {}
        for c in coords:
            split_iter = self._split_coordinate(c)
            iter_key = ':'.join(split_iter)
            coord_map[iter_key] = False

        # didn't record any system iterations, return None
        if len(coord_map) is 0:
            return None

        variables = {}
        iteration_num = -1
        # iterate over cases from end to start, unless we've grabbed values from
        # every system
        while not self._has_all_values(coord_map):
            iteration = self.system_cases._case_keys[iteration_num]
            iteration_num -= 1
            split_iter = self._split_coordinate(iteration)
            iter_key = ':'.join(split_iter)

            # if coord_map[iter_key] is False, we haven't grabbed variable values
            # from this system
            if not coord_map[iter_key]:
                coord_map[iter_key] = True
                case = self.system_cases.get_case(iteration)
                if get_outputs and case.outputs is None:
                    continue
                if not get_outputs and case.inputs is None:
                    continue
                outputs = case.outputs._values if case.outputs is not None else None
                residuals = case.residuals._values if case.residuals is not None else None
                inputs = case.inputs._values if case.inputs is not None else None
                if get_outputs:
                    for var_name in outputs.dtype.names:
                        if var_name not in variables:
                            variables[var_name] = {'value': outputs[var_name]}
                            if residuals is not None and var_name in residuals.dtype.names:
                                variables[var_name]['residuals'] = residuals[var_name]
                            else:
                                variables[var_name]['residuals'] = 'Not Recorded'
                elif inputs is not None:
                    for var_name in inputs.dtype.names:
                        if var_name not in variables:
                            variables[var_name] = {'value': inputs[var_name]}

        return variables

    def _has_all_values(self, coord_map):
        """
        Tell if all variables from every recorded system have been iterated over.

        Parameters
        ----------
        coord_map : dict
            maps stripped iteration coordinates to a bool indicating whether or not the system(s)
            associated with that iteration coordinate have been iterated over.

        Returns
        -------
        bool
            True if coord_map is True for each key, False otherwise.
        """
        for coord in coord_map:
            if not coord_map[coord]:
                return False
        return True

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
        meta = self._abs2meta

        # Make a dict of outputs. Makes it easier to work with in this method
        dict_of_outputs = OrderedDict()
        for name, vals in outputs:
            dict_of_outputs[name] = vals

        allprocs_abs_names = {
            'input': dict_of_outputs.keys(),
            'output': dict_of_outputs.keys()
        }

        write_outputs(in_or_out, comp_type, dict_of_outputs, hierarchical, print_arrays, out_stream,
                      'model', allprocs_abs_names)


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

        idx, counter, iteration_coordinate, timestamp, success, msg, inputs_blob, \
            outputs_blob, = row

        inputs_array = blob_to_array(inputs_blob)
        outputs_array = blob_to_array(outputs_blob)

        case = DriverCase(self.filename, counter, iteration_coordinate, timestamp, success, msg,
                          inputs_array, outputs_array,
                          self._prom2abs, self._abs2prom, self._abs2meta)

        return case


class SystemCases(BaseCases):
    """
    Case specific to the entries that might be recorded in a System iteration.
    """

    def __init__(self, filename, abs2prom, abs2meta, prom2abs):
        super(SystemCases, self).__init__(filename, abs2prom, abs2meta, prom2abs)
        self._cases = {}
        with sqlite3.connect(self.filename) as con:
            cur = con.cursor()
            cur.execute("SELECT * FROM system_iterations")
            rows = cur.fetchall()
            for row in rows:
                idx, counter, iteration_coordinate, timestamp, success, msg, inputs_blob,\
                    outputs_blob, residuals_blob = row
                inputs_array = blob_to_array(inputs_blob)
                outputs_array = blob_to_array(outputs_blob)
                residuals_array = blob_to_array(residuals_blob)

                case = SystemCase(self.filename, counter, iteration_coordinate, timestamp,
                                  success, msg, inputs_array, outputs_array, residuals_array,
                                  self._prom2abs, self._abs2prom, self._abs2meta)
                self._cases[iteration_coordinate] = case

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
        if iteration_coordinate in self._cases:
            return self._cases[iteration_coordinate]

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
                          inputs_array, outputs_array, residuals_array,
                          self._prom2abs, self._abs2prom, self._abs2meta)

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
            input_blob, output_blob, residuals_blob = row

        input_array = blob_to_array(input_blob)
        output_array = blob_to_array(output_blob)
        residuals_array = blob_to_array(residuals_blob)

        case = SolverCase(self.filename, counter, iteration_coordinate, timestamp, success, msg,
                          abs_err, rel_err, input_array, output_array, residuals_array,
                          self._prom2abs, self._abs2prom, self._abs2meta)

        return case
