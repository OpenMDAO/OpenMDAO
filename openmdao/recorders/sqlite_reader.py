"""
Definition of the SqliteCaseReader.
"""
import sqlite3
from collections import OrderedDict

import sys
import numpy as np

from openmdao.recorders.base_case_reader import BaseCaseReader
from openmdao.recorders.case import Case
from openmdao.core.constants import _DEFAULT_OUT_STREAM
from openmdao.utils.variable_table import write_source_table
from openmdao.utils.record_util import check_valid_sqlite3_db, get_source_system
from openmdao.utils.om_warnings import issue_warning, CaseRecorderWarning

from openmdao.recorders.sqlite_recorder import format_version, META_KEY_SEP

from openmdao.utils.notebook_utils import notebook, display, HTML
from openmdao.visualization.tables.table_builder import generate_table

import pickle
import zlib
import re
from json import loads as json_loads
from io import TextIOBase


class SqliteCaseReader(BaseCaseReader):
    """
    A CaseReader specific to files created with SqliteRecorder.

    Parameters
    ----------
    filename : str
        The path to the filename containing the recorded data.
    pre_load : bool
        If True, load all the data into memory during initialization.
    metadata_filename : str
        The path to the filename containing the recorded metadata, if separate.

    Attributes
    ----------
    problem_metadata : dict
        Metadata about the problem, including the system hierachy and connections.
    solver_metadata : dict
        The solver options for each solver in the recorded model.
    _system_options : dict
        Metadata about each system in the recorded model, including options and scaling factors.
    _format_version : int
        The version of the format assumed when loading the file.
    _filename : str
        The path to the filename containing the recorded data.
    _abs2meta : dict
        Dictionary mapping variables to their metadata
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    _conns : dict
        Dictionary of all model connections.
    _auto_ivc_map : dict
        Dictionary that maps all auto_ivc sources to either an absolute input name for single
        connections or a promoted input name for multiple connections. This is for output display.
    _driver_cases : DriverCases
        Helper object for accessing cases from the driver_iterations table.
    _system_cases : SystemCases
        Helper object for accessing cases from the system_iterations table.
    _solver_cases : SolverCases
        Helper object for accessing cases from the solver_iterations table.
    _problem_cases : ProblemCases
        Helper object for accessing cases from the problem_cases table.
    _global_iterations : list
        List of iteration cases and the table and row in which they are found.
    """

    def __init__(self, filename, pre_load=False, metadata_filename=None):
        """Initialize."""
        super().__init__(filename, pre_load)

        check_valid_sqlite3_db(filename)

        if metadata_filename:
            check_valid_sqlite3_db(metadata_filename)

        # initialize private attributes
        self._filename = filename
        self._abs2prom = None
        self._prom2abs = None
        self._abs2meta = None
        self._conns = None
        self._auto_ivc_map = {}
        self._global_iterations = None

        with sqlite3.connect(filename) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()

            # get the global iterations table, and save it as an attribute
            self._global_iterations = self._get_global_iterations(cur)

            # If separate metadata not specified, check the current db
            # to make sure it's there
            if metadata_filename is None:
                cur.execute("SELECT count(name) FROM sqlite_master "
                            "WHERE type='table' AND name='metadata'")

                # If not, take a guess at the filename:
                if cur.fetchone()[0] == 0:
                    metadata_filename = re.sub(r'^(.*)_(\d+)', r'\1_meta', filename)
                    check_valid_sqlite3_db(metadata_filename)
                else:
                    metadata_filename = filename

        con.close()

        # collect metadata from database
        with sqlite3.connect(metadata_filename) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()

            # collect data from the metadata table. this includes:
            #   format_version
            #   openmdao_version
            #   VOI metadata, which is added to problem_metadata
            #   var name maps and metadata for all vars, which are saved as private attributes
            self._collect_metadata(cur)

            # collect data from the driver_metadata table. this includes:
            #   model viewer data, which is added to problem_metadata
            self._collect_driver_metadata(cur)

            # collect data from the system_metadata table. this includes:
            #   component metadata and scaling factors for each system,
            #   which is added to _system_options
            self._collect_system_metadata(cur)

            # collect data from the solver_metadata table. this includes:
            #   solver class and options for each solver, which is saved as an attribute
            self._collect_solver_metadata(cur)

        con.close()

        # create helper objects for accessing cases from the three iteration tables and
        # the problem cases table
        var_info = self.problem_metadata['variables']
        self._driver_cases = DriverCases(filename, self._format_version, self._global_iterations,
                                         self._prom2abs, self._abs2prom, self._abs2meta,
                                         self._conns, self._auto_ivc_map, var_info)
        self._system_cases = SystemCases(filename, self._format_version, self._global_iterations,
                                         self._prom2abs, self._abs2prom, self._abs2meta,
                                         self._conns, self._auto_ivc_map, var_info)
        self._solver_cases = SolverCases(filename, self._format_version, self._global_iterations,
                                         self._prom2abs, self._abs2prom, self._abs2meta,
                                         self._conns, self._auto_ivc_map, var_info)
        if self._format_version >= 2:
            self._problem_cases = ProblemCases(filename,
                                               self._format_version,
                                               self._global_iterations,
                                               self._prom2abs, self._abs2prom, self._abs2meta,
                                               self._conns, self._auto_ivc_map, var_info)

        # if requested, load all the iteration data into memory
        if pre_load:
            self._load_cases()

    def _collect_metadata(self, cur):
        """
        Load data from the metadata table.

        Populates the `format_version` attribute and the `variables` data in
        the `problem_metadata` attribute of this CaseReader. Also saves the
        variable name maps and variable metadata to private attributes.

        Parameters
        ----------
        cur : sqlite3.Cursor
            Database cursor to use for reading the data.
        """
        cur.execute('select * from metadata')

        row = cur.fetchone()

        self._format_version = version = row['format_version']

        if version >= 13:
            self._openmdao_version = row['openmdao_version']

        if version not in range(1, format_version + 1):
            raise ValueError('SqliteCaseReader encountered an unhandled '
                             'format version: {0}'.format(self._format_version))

        if version >= 11:
            # Auto-IVC
            if version >= 14:
                self._conns = json_loads(zlib.decompress(row['conns']).decode('ascii'))
            else:
                self._conns = json_loads(row['conns'])

        # add metadata for VOIs (des vars, objective, constraints) to problem metadata
        if version >= 4:
            if version >= 14:
                self.problem_metadata['variables'] = \
                    json_loads(zlib.decompress(row['var_settings']).decode('ascii'))
            else:
                self.problem_metadata['variables'] = json_loads(row['var_settings'])
        else:
            self.problem_metadata['variables'] = None

        # get variable name maps and metadata for all variables
        if version >= 3:
            if version >= 14:
                self._abs2prom = json_loads(zlib.decompress(row['abs2prom']).decode('ascii'))
                self._prom2abs = json_loads(zlib.decompress(row['prom2abs']).decode('ascii'))
                self._abs2meta = json_loads(zlib.decompress(row['abs2meta']).decode('ascii'))
            else:
                self._abs2prom = json_loads(row['abs2prom'])
                self._prom2abs = json_loads(row['prom2abs'])
                self._abs2meta = json_loads(row['abs2meta'])

            # need to convert bounds to numpy arrays
            for meta in self._abs2meta.values():
                if 'lower' in meta and meta['lower'] is not None:
                    meta['lower'] = np.resize(np.array(meta['lower']), meta['shape'])
                if 'upper' in meta and meta['upper'] is not None:
                    meta['upper'] = np.resize(np.array(meta['upper']), meta['shape'])

            # Map source names to input display text.
            if version >= 11:
                self._auto_ivc_map = auto_ivc_map = {}
                abs2prom_in = self._abs2prom['input']
                for target, src in self._conns.items():
                    if src.startswith('_auto_ivc.'):
                        if src not in auto_ivc_map:
                            auto_ivc_map[src] = []
                        auto_ivc_map[src].append(target)
                for output, input_list in auto_ivc_map.items():
                    if len(input_list) > 1:
                        for input_name in input_list:
                            # If this recorder is on a component, we might have only a subset of
                            # the metadata dictionary, but one of them will be in there.
                            if input_name in abs2prom_in:
                                auto_ivc_map[output] = abs2prom_in[input_name]
                                break
                    else:
                        auto_ivc_map[output] = abs2prom_in[input_list[0]]

        elif version in (1, 2):
            abs2prom = row['abs2prom']
            prom2abs = row['prom2abs']
            abs2meta = row['abs2meta']

            try:
                self._abs2prom = pickle.loads(abs2prom)
                self._prom2abs = pickle.loads(prom2abs)
                self._abs2meta = pickle.loads(abs2meta)
            except TypeError:
                # Reading in a python 2 pickle recorded pre-OpenMDAO 2.4.
                self._abs2prom = pickle.loads(abs2prom.encode())
                self._prom2abs = pickle.loads(prom2abs.encode())
                self._abs2meta = pickle.loads(abs2meta.encode())

        self.problem_metadata['abs2prom'] = self._abs2prom

    def _collect_driver_metadata(self, cur):
        """
        Load data from the driver_metadata table.

        Populates the `problem_metadata` attribute of this CaseReader.

        Parameters
        ----------
        cur : sqlite3.Cursor
            Database cursor to use for reading the data.
        """
        cur.execute("SELECT model_viewer_data FROM driver_metadata")
        row = cur.fetchone()

        if row is not None:
            if self._format_version >= 3:
                driver_metadata = json_loads(row[0])
            elif self._format_version in (1, 2):
                driver_metadata = pickle.loads(row[0])

            self.problem_metadata.update(driver_metadata)

    def _collect_system_metadata(self, cur):
        """
        Load data from the system table.

        Populates the `system_options` attribute of this CaseReader.

        Parameters
        ----------
        cur : sqlite3.Cursor
            Database cursor to use for reading the data.
        """
        cur.execute("SELECT id, scaling_factors, component_metadata FROM system_metadata")
        for row in cur:
            id = row[0]
            self._system_options[id] = {}

            if self._format_version >= 14:
                self._system_options[id]['scaling_factors'] = \
                    pickle.loads(zlib.decompress(row[1]))
                self._system_options[id]['component_options'] = \
                    pickle.loads(zlib.decompress(row[2]))
            else:
                self._system_options[id]['scaling_factors'] = pickle.loads(row[1])
                self._system_options[id]['component_options'] = pickle.loads(row[2])

    def _collect_solver_metadata(self, cur):
        """
        Load data from the solver_metadata table.

        Populates the `solver_metadata` attribute of this CaseReader.

        Parameters
        ----------
        cur : sqlite3.Cursor
            Database cursor to use for reading the data.
        """
        cur.execute("SELECT id, solver_options, solver_class FROM solver_metadata")
        for row in cur:
            id = row[0]
            if self._format_version >= 14:
                solver_options = pickle.loads(zlib.decompress(row[1]))
            else:
                solver_options = pickle.loads(row[1])
            solver_class = row[2]
            self.solver_metadata[id] = {
                'solver_options': solver_options,
                'solver_class': solver_class,
            }

    def _get_global_iterations(self, cur):
        """
        Get the global iterations table.

        Parameters
        ----------
        cur : sqlite3.Cursor
            Database cursor to use for reading the data.

        Returns
        -------
        list
            List of global iterations and the table and row where the associated case is found.
        """
        cur.execute('select * from global_iterations')
        return cur.fetchall()

    def _load_cases(self):
        """
        Load all driver, solver, and system cases into memory.
        """
        self._driver_cases._load_cases()
        self._solver_cases._load_cases()
        self._system_cases._load_cases()
        if self._format_version >= 2:
            self._problem_cases._load_cases()

    def list_sources(self, out_stream=_DEFAULT_OUT_STREAM):
        """
        List of all the different recording sources for which there is recorded data.

        Parameters
        ----------
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list
            One or more of: `problem`, `driver`, `<system hierarchy location>`,
                            `<solver hierarchy location>`
        """
        sources = []

        if self._driver_cases.count() > 0:
            sources.extend(self._driver_cases.list_sources())
        if self._solver_cases.count() > 0:
            sources.extend(self._solver_cases.list_sources())
        if self._system_cases.count() > 0:
            sources.extend(self._system_cases.list_sources())
        if self._format_version >= 2 and self._problem_cases.count() > 0:
            sources.extend(self._problem_cases.list_sources())

        if out_stream:
            if notebook and out_stream is _DEFAULT_OUT_STREAM:
                display(HTML(str(generate_table([[s] for s in sources], headers=['Sources'],
                                                tablefmt='html'))))
            else:
                if out_stream is _DEFAULT_OUT_STREAM:
                    out_stream = sys.stdout
                elif not isinstance(out_stream, TextIOBase):
                    raise TypeError("Invalid output stream specified for 'out_stream'.")
                for source in sources:
                    out_stream.write('{}\n'.format(source))

        return sources

    def list_source_vars(self, source, out_stream=_DEFAULT_OUT_STREAM):
        """
        List of all inputs and outputs recorded by the specified source.

        Parameters
        ----------
        source : {'problem', 'driver', <system hierarchy location>, <solver hierarchy location>}
            Identifies the source for which to return information.
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        dict
            {'inputs':[key list], 'outputs':[key list], 'residuals':[key list]}. No recurse.
        """
        dct = {
            'inputs': [],
            'outputs': [],
            'residuals': [],
        }

        case = None

        if source == 'problem':
            if self._problem_cases.count() > 0:
                case = self._problem_cases.get_case(0)
        elif source == 'driver':
            if self._driver_cases.count() > 0:
                case = self._driver_cases.get_case(0)
        elif source in self._system_cases.list_sources():
            source_cases = self._system_cases.list_cases(source)
            case = self._system_cases.get_case(source_cases[0])
        elif source in self._solver_cases.list_sources():
            source_cases = self._solver_cases.list_cases(source)
            case = self._solver_cases.get_case(source_cases[0])
        else:
            raise RuntimeError('Source not found: %s' % source)

        if case is None:
            raise RuntimeError('No cases recorded for %s' % source)

        if case.inputs:
            dct['inputs'] = list(case.inputs)
        if case.outputs:
            dct['outputs'] = list(case.outputs)
        if case.residuals:
            dct['residuals'] = list(case.residuals)

        if out_stream:
            write_source_table(dct, out_stream)

        return dct

    def systems(self, tree=None, path=None, paths=[]):
        """
        List pathnames of systems in the system hierarchy.

        Parameters
        ----------
        tree : dict
            Nested dictionary of system information.
        path : str or None
            Pathname of root system (None for the root model).
        paths : list
            List to which pathnames are appended.

        Returns
        -------
        list
            List of pathnames of systems.
        """
        if tree is None:
            tree = self.problem_metadata['tree']

        path = '.'.join([path, tree['name']]) if path else tree['name']
        paths.append(path)

        if 'children' in tree:
            for child in tree['children']:
                if child['type'] == 'subsystem':
                    self.systems(child, path, paths)

        return paths

    def list_model_options(self, run_number=0, system=None, out_stream=_DEFAULT_OUT_STREAM):
        """
        List model options for the specified run.

        Parameters
        ----------
        run_number : int
            Run_driver or run_model iteration to inspect.
        system : str or None
            Pathname of system (None for all systems).
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        dict
            {system: {key: val}}.
        """
        dct = {}

        if not self._system_options:
            issue_warning("System options not recorded.", category=CaseRecorderWarning)
            return dct

        if out_stream is _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        num_header = None

        # need to handle edge case for v11 recording
        if self._format_version < 12:
            SEP = '_'
        else:
            SEP = META_KEY_SEP

        for key in self._system_options:
            if key.find(SEP) > 0:
                name, num = key.rsplit(SEP, 1)
            else:
                name = key
                num = 0

            if (system is None or system == name) and (run_number == int(num)):

                if out_stream:
                    if num_header != num:
                        out_stream.write(f"Run Number: {num}\n")
                        num_header = num

                    out_stream.write(f"    Subsystem : {name}\n")

                dct[name] = {}

                comp_options = self._system_options[key]['component_options']

                for opt, val in comp_options.items():
                    dct[name][opt] = val

                    if out_stream:
                        out_stream.write(f"        {opt}: {val}\n")

        return dct

    def list_solver_options(self, run_number=0, solver=None, out_stream=_DEFAULT_OUT_STREAM):
        """
        List solver options for the specified run.

        Parameters
        ----------
        run_number : int
            Run_driver or run_model iteration to inspect.
        solver : str or None
            Pathname of solver (None for all solvers).
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        dict
            {solver: {key: val}}.
        """
        dct = {}

        if not self.solver_metadata:
            issue_warning("Solver options not recorded.", category=CaseRecorderWarning)
            return dct

        if out_stream is _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        num_header = None

        for key in self.solver_metadata:
            if key.find(META_KEY_SEP) > 0:
                name, num = key.rsplit(META_KEY_SEP, 1)
            else:
                name = key
                num = 0

            if (solver is None or solver == name) and (run_number == int(num)):

                if out_stream:
                    if num_header != num:
                        out_stream.write(f"Run Number: {num}\n")
                        num_header = num

                    out_stream.write(f"    Solver: {name}\n")

                dct[name] = {}

                comp_options = self.solver_metadata[key]['solver_options']

                for opt, val in comp_options.items():
                    dct[name][opt] = val

                    if out_stream:
                        out_stream.write(f"        {opt}: {val}\n")

        return dct

    def list_cases(self, source=None, recurse=True, flat=True, out_stream=_DEFAULT_OUT_STREAM):
        """
        Iterate over Driver, Solver and System cases in order.

        Parameters
        ----------
        source : 'problem', 'driver', component pathname, solver pathname, case_name
            If not None, only cases originating from the specified source or case are returned.
        recurse : bool, optional
            If True, will enable iterating over all successors in case hierarchy.
        flat : bool, optional
            If False and there are child cases, then a nested ordered dictionary
            is returned rather than an iterator.
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        iterator or dict
            An iterator or a nested dictionary of identified cases.
        """
        # if source was not specified, return all cases
        if source is None:
            if flat:
                source = ''
            else:
                if self._driver_cases.count() > 0:
                    source = 'driver'
                elif 'root' in self._system_cases.list_sources():
                    source = 'root'
                else:
                    # if there are no driver or model cases, then we need
                    # another starting point to build the nested dict.
                    raise RuntimeError("A nested dictionary of all cases was requested, but "
                                       "neither the driver or the model was recorded. Please "
                                       "specify another source (system or solver) for the cases "
                                       "you want to see.")

        if not isinstance(source, str):
            raise TypeError("Source parameter must be a string, %s is type %s." %
                            (source, type(source).__name__))

        if not source:
            cases = self._list_cases_recurse_flat(out_stream=None)

        elif source == 'problem':
            if self._format_version >= 2:
                cases = self._problem_cases.list_cases()
            else:
                raise RuntimeError('No problem cases recorded (data format = %d).' %
                                   self._format_version)

        else:
            # figure out which table has cases from the source
            if source == 'driver':
                case_table = self._driver_cases
            elif source in self._system_cases.list_sources():
                case_table = self._system_cases
            elif source in self._solver_cases.list_sources():
                case_table = self._solver_cases
            else:
                case_table = None

            if case_table is not None:
                if not recurse:
                    # return list of cases from the source alone
                    cases = case_table.list_cases(source)
                elif flat:
                    # return list of cases from the source plus child cases
                    cases = []
                    source_cases = case_table.get_cases(source)
                    for case in source_cases:
                        cases += self._list_cases_recurse_flat(case.name, out_stream=None)
                else:
                    # return nested dict of cases from the source and child cases
                    cases = OrderedDict()
                    source_cases = case_table.get_cases(source)
                    for case in source_cases:
                        cases.update(self._list_cases_recurse_nested(case.name))
                    return cases
            elif '|' in source:
                # source is a coordinate
                if recurse:
                    if flat:
                        cases = self._list_cases_recurse_flat(source, out_stream=None)
                    else:
                        return self._list_cases_recurse_nested(source)
            else:
                raise RuntimeError('Source not found: %s' % source)

        if out_stream:
            if not source:
                for source, subcases in self.source_cases_table.items():
                    if subcases:
                        write_source_table({source: subcases}, out_stream)
                del self.source_cases_table
            else:
                write_source_table({source: cases}, out_stream)

        return cases

    def _list_cases_recurse_flat(self, coord=None, out_stream=_DEFAULT_OUT_STREAM):
        """
        Iterate recursively over Driver, Solver and System cases in order.

        Parameters
        ----------
        coord : an iteration coordinate
            Identifies the parent of the cases to return.
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        dict
            A nested dictionary of identified cases.
        """
        solver_cases = self._solver_cases.list_cases()
        system_cases = self._system_cases.list_cases()
        driver_cases = self._driver_cases.list_cases()
        if self._format_version >= 2:
            problem_cases = self._problem_cases.list_cases()
        global_iters = self._global_iterations

        if not coord:
            # will return all cases
            coord = ''
            parent_case_counter = len(global_iters)
        elif coord in driver_cases:
            parent_case_counter = self._driver_cases.get_case(coord).counter
        elif coord in system_cases:
            parent_case_counter = self._system_cases.get_case(coord).counter
        elif coord in solver_cases:
            parent_case_counter = self._solver_cases.get_case(coord).counter
        elif coord in problem_cases:
            parent_case_counter = self._problem_cases.get_case(coord).counter
        else:
            raise RuntimeError('Case not found for coordinate:', coord)

        cases = []

        self.source_cases_table = {'solver': [], 'system': [], 'driver': [], 'problem': []}
        source_cases = []

        # return all cases in the global iteration table that precede the given case
        # and whose coordinate is prefixed by the given coordinate
        current_table = None
        current_cases = []

        for i in range(0, parent_case_counter):
            global_iter = global_iters[i]
            table, row = global_iter[1], global_iter[2]
            if table == 'solver':
                case_coord = solver_cases[row - 1]
            elif table == 'system':
                case_coord = system_cases[row - 1]
            elif table == 'driver':
                case_coord = driver_cases[row - 1]
            elif table == 'problem':
                case_coord = problem_cases[row - 1]
            else:
                raise RuntimeError('Unexpected table name in global iterations:', table)

            if case_coord.startswith(coord):
                cases.append(case_coord)
                self.source_cases_table[table].append(case_coord)

                if out_stream:
                    if not current_cases:
                        current_table = table
                        current_cases = {table: [case_coord]}
                    elif table == current_table:
                        current_cases[table].append(case_coord)
                    else:
                        source_cases.append(current_cases)
                        current_table = table
                        current_cases = {table: [case_coord]}

        if out_stream:
            if current_cases:
                source_cases.append(current_cases)
            write_source_table(source_cases, out_stream)

        return cases

    def _list_cases_recurse_nested(self, coord=None):
        """
        Iterate recursively over Driver, Solver and System cases in order.

        Parameters
        ----------
        coord : an iteration coordinate
            Identifies the parent of the cases to return.

        Returns
        -------
        dict
            A nested dictionary of identified cases.
        """
        solver_cases = self._solver_cases.list_cases()
        system_cases = self._system_cases.list_cases()
        driver_cases = self._driver_cases.list_cases()
        global_iters = self._global_iterations

        if coord in driver_cases:
            parent_case = self._driver_cases.get_case(coord)
        elif coord in system_cases:
            parent_case = self._system_cases.get_case(coord)
        elif coord in solver_cases:
            parent_case = self._solver_cases.get_case(coord)
        else:
            raise RuntimeError('Case not found for coordinate:', coord)

        cases = OrderedDict()
        children = OrderedDict()
        cases[parent_case.name] = children

        # return all cases in the global iteration table that precede the given case
        # and whose coordinate is prefixed by the given coordinate
        for i in range(0, parent_case.counter - 1):
            global_iter = global_iters[i]
            table, row = global_iter[1], global_iter[2]
            if table == 'solver':
                case_coord = solver_cases[row - 1]
                if case_coord.startswith(coord):
                    parent_coord = '|'.join(case_coord.split('|')[:-2])
                    if parent_coord == coord:
                        children.update(self._list_cases_recurse_nested(case_coord))
            elif table == 'system':
                case_coord = system_cases[row - 1]
                if case_coord.startswith(coord):
                    parent_coord = '|'.join(case_coord.split('|')[:-2])
                    if parent_coord == coord:
                        children.update(self._list_cases_recurse_nested(case_coord))

        return cases

    def get_cases(self, source=None, recurse=True, flat=True):
        """
        Iterate over the cases.

        Parameters
        ----------
        source : 'problem', 'driver', component pathname, solver pathname, case_name
            Identifies which cases to return.
        recurse : bool, optional
            If True, will enable iterating over all successors in case hierarchy.
        flat : bool, optional
            If False and there are child cases, then a nested ordered dictionary
            is returned rather than an iterator.

        Returns
        -------
        list or dict
            The cases identified by source.
        """
        case_ids = self.list_cases(source, recurse, flat, out_stream=None)
        if isinstance(case_ids, list):
            return [self.get_case(case_id) for case_id in case_ids]
        else:
            return self._get_cases_nested(case_ids, OrderedDict())

    def _get_cases_nested(self, case_ids, cases):
        """
        Populate a nested dictionary of cases matching the provided dictionary of case IDs.

        Parameters
        ----------
        case_ids : OrderedDict
            The nested dictionary of case IDs.
        cases : OrderedDict
            The nested dictionary of cases.

        Returns
        -------
        OrderedDict
            The nested dictionary of cases with cases added from case_ids.
        """
        for case_id in case_ids:
            case = self.get_case(case_id)
            children = case_ids[case_id]
            if len(children.keys()) > 0:
                cases[case] = self._get_cases_nested(children, OrderedDict())
            else:
                cases[case] = OrderedDict()

        return cases

    def get_case(self, case_id, recurse=False):
        """
        Get case identified by case_id.

        Parameters
        ----------
        case_id : str or int
            The unique identifier of the case to return or an index into all cases.
        recurse : bool, optional
            If True, will return all successors to the case as well.

        Returns
        -------
        dict
            The case identified by case_id.
        """
        if isinstance(case_id, int):
            # it's a global index rather than a coordinate
            global_iters = self._global_iterations
            if case_id > len(global_iters) - 1:
                raise IndexError("Invalid index into available cases:", case_id)
            global_iter = global_iters[case_id]
            table, row = global_iter[1], global_iter[2]
            if table == 'solver':
                solver_cases = self._solver_cases.list_cases()
                case_id = solver_cases[row - 1]
            elif table == 'system':
                system_cases = self._system_cases.list_cases()
                case_id = system_cases[row - 1]
            elif table == 'driver':
                driver_cases = self._driver_cases.list_cases()
                case_id = driver_cases[row - 1]

        if recurse:
            return self.get_cases(case_id, recurse=True)

        tables = [self._driver_cases, self._system_cases, self._solver_cases]
        if self._format_version >= 2:
            tables.append(self._problem_cases)

        for table in tables:
            case = table.get_case(case_id)
            if case:
                return case

        raise RuntimeError('Case not found:', case_id)


class CaseTable(object):
    """
    Base class for wrapping case tables in a recording database.

    Parameters
    ----------
    fname : str
        The name of the recording file from which to instantiate the case reader.
    ver : int
        The version of the format assumed when loading the file.
    table : str
        The name of the table in the database.
    index : str
        The name of the case index column in the table.
    giter : list of tuple
        The global iterations table.
    prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    abs2meta : dict
        Dictionary mapping absolute variable names to variable metadata.
    conns : dict
        Dictionary of all model connections.
    auto_ivc_map : dict
        Dictionary that maps all auto_ivc sources to either an absolute input name for single
        connections or a promoted input name for multiple connections. This is for output
        display.
    var_info : dict
        Dictionary with information about variables (scaling, indices, execution order).

    Attributes
    ----------
    _filename : str
        The name of the recording file from which to instantiate the case reader.
    _format_version : int
        The version of the format assumed when loading the file.
    _table_name : str
        The name of the table in the database.
    _index_name : str
        The name of the case index column in the table.
    _global_iterations : list
        List of iteration cases and the table and row in which they are found.
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    _abs2meta : dict
        Dictionary mapping absolute variable names to variable metadata.
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    _conns : dict
        Dictionary of all model connections.
    _var_info : dict
        Dictionary with information about variables (scaling, indices, execution order).
    _sources : list
        List of sources of cases in the table.
    _keys : list
        List of keys of cases in the table.
    _cases : dict
        Dictionary mapping keys to cases that have already been loaded.
    _auto_ivc_map : dict
        Dictionary that maps all auto_ivc sources to either an absolute input name for single
        connections or a promoted input name for multiple connections. This is for output display.
    _global_iterations : list
        List of iteration cases and the table and row in which they are found.
    """

    def __init__(self, fname, ver, table, index, giter, prom2abs, abs2prom, abs2meta, conns,
                 auto_ivc_map, var_info):
        """
        Initialize.
        """
        self._filename = fname
        self._format_version = ver
        self._table_name = table
        self._index_name = index
        self._global_iterations = giter
        self._prom2abs = prom2abs
        self._abs2prom = abs2prom
        self._abs2meta = abs2meta
        self._conns = conns
        self._auto_ivc_map = auto_ivc_map
        self._var_info = var_info

        # cached keys/cases
        self._sources = None
        self._keys = None
        self._cases = {}

    def count(self):
        """
        Get the number of cases recorded in the table.

        Returns
        -------
        int
            The number of cases recorded in the table.
        """
        with sqlite3.connect(self._filename) as con:
            cur = con.cursor()
            cur.execute(f"SELECT count(*) FROM {self._table_name}")  # nosec: trusted input
            rows = cur.fetchall()

        con.close()

        return rows[0][0]

    def list_cases(self, source=None):
        """
        Get list of case IDs for cases in the table.

        Parameters
        ----------
        source : str, optional
            A source of cases or the iteration coordinate of a case.
            If not None, only cases originating from the specified source or case are returned.

        Returns
        -------
        list
            The cases from the table from the specified source or parent case.
        """
        if not self._keys:
            with sqlite3.connect(self._filename) as con:
                cur = con.cursor()
                cur.execute(f"SELECT {self._index_name} FROM {self._table_name}"
                            " ORDER BY id ASC")  # nosec trusted input
                rows = cur.fetchall()

            con.close()

            # cache case list for future use
            self._keys = [row[0] for row in rows]

        if not source:
            # return all cases
            return self._keys
        elif '|' in source:
            # source is a coordinate
            return [key for key in self._keys if key.startswith(source)]
        else:
            # source is a system or solver
            return [key for key in self._keys if self._get_source(key) == source]

    def get_cases(self, source=None, recurse=False, flat=False):
        """
        Get list of case names for cases in the table.

        Parameters
        ----------
        source : str, optional
            If not None, only cases that have the specified source will be returned.
        recurse : bool, optional
            If True, will enable iterating over all successors in case hierarchy.
        flat : bool, optional
            If False and there are child cases, then a nested ordered dictionary
            is returned rather than an iterator.

        Returns
        -------
        list or dict
            The cases from the table that have the specified source.
        """
        if self._keys is None:
            self.list_cases()

        if not source:
            # return all cases
            return [self.get_case(key) for key in self._keys]
        elif '|' in source:
            # source is a coordinate
            if recurse and not flat:
                cases = OrderedDict()
                for key in self._keys:
                    if len(key) > len(source) and key.startswith(source):
                        cases[key] = self.get_cases(key, recurse, flat)
                return cases
            else:
                return list([self.get_case(key) for key in self._keys if key.startswith(source)])
        else:
            # source is a system or solver
            if recurse:
                if flat:
                    # return all cases under the source system
                    source_sys = source.replace('.nonlinear_solver', '')
                    return list([self.get_case(key) for key in self._keys
                                 if get_source_system(key).startswith(source_sys)])
                else:
                    cases = OrderedDict()
                    for key in self._keys:
                        case_source = self._get_source(key)
                        if case_source == source:
                            cases[key] = self.get_cases(key, recurse, flat)
                    return cases
            else:
                return [self.get_case(key) for key in self._keys
                        if self._get_source(key) == source]

    def get_case(self, case_id, cache=False):
        """
        Get a case from the database.

        Parameters
        ----------
        case_id : str or int
            The string-identifier of the case to be retrieved or the index of the case.
        cache : bool
            If True, case will be cached for faster access by key.

        Returns
        -------
        Case
            The specified case from the table.
        """
        # check to see if we've already cached this case
        if isinstance(case_id, int):
            case_id = self._get_iteration_coordinate(case_id)

        # if we've already cached this case, return the cached instance
        if case_id in self._cases:
            return self._cases[case_id]

        # we don't have it, so fetch it
        with sqlite3.connect(self._filename) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(f"SELECT * FROM {self._table_name} "  # nosec: trusted input
                        f"WHERE {self._index_name}=?", (case_id, ))
            row = cur.fetchone()

        con.close()

        # if found, extract the data and optionally cache the Case
        if row is not None:
            if self._format_version >= 5:
                source = self._get_row_source(row['id'])

                # check for situations where parsing the iter coord doesn't work correctly
                iter_source = self._get_source(row[self._index_name])
                if iter_source != source:
                    msg = f'Mismatched source for {row["id"]}: {row[self._index_name]} = ' \
                          f'{iter_source} vs {source}'
                    issue_warning(msg, category=CaseRecorderWarning)
            else:
                source = self._get_source(row[self._index_name])

            case = Case(source, row, self._prom2abs, self._abs2prom, self._abs2meta,
                        self._conns, self._auto_ivc_map, self._var_info, self._format_version)

            # cache it if requested
            if cache:
                self._cases[case_id] = case

            return case
        else:
            return None

    def _get_iteration_coordinate(self, case_idx):
        """
        Return the iteration coordinate for the indexed case (handles negative indices, etc.).

        Parameters
        ----------
        case_idx : int
            The case number that we want the iteration coordinate for.

        Returns
        -------
        iteration_coordinate : str
            The iteration coordinate.
        """
        # if keys have not been cached yet, get them now
        if self._keys is None:
            self.list_cases()

        return self._keys[case_idx]

    def cases(self, cache=False):
        """
        Iterate over all cases, optionally caching them into memory.

        Parameters
        ----------
        cache : bool
            If True, cases will be cached for faster access by key.

        Yields
        ------
        case
        """
        with sqlite3.connect(self._filename) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(f"SELECT * FROM {self._table_name} ORDER BY id ASC")  # nosec: trusted input
            # rows = cur.fetchall()
            for row in cur:
                case_id = row[self._index_name]
                source = self._get_source(case_id)
                case = Case(source, row, self._prom2abs, self._abs2prom, self._abs2meta,
                            self._conns, self._auto_ivc_map, self._var_info, self._format_version)
                if cache:
                    self._cases[case_id] = case
                yield case

        con.close()

    def _load_cases(self):
        """
        Load all cases into memory.
        """
        for case in self.cases(cache=True):
            pass

    def list_sources(self):
        """
        Get the list of sources that recorded data in this table.

        Returns
        -------
        list
            List of sources.
        """
        if self._sources is None:
            if self._format_version >= 5:
                table = self._table_name.split('_')[0]  # remove "_iterations" from table name
                sources = set()
                for global_iter in self._global_iterations:
                    record_type, source = global_iter[1], global_iter[3]
                    if record_type == table:
                        if not source.startswith('root'):
                            sources.add('root.' + source)
                        else:
                            sources.add(source)
                self._sources = sources
            else:
                self._sources = set([self._get_source(case) for case in self.list_cases()])

        return self._sources

    def _get_source(self, iteration_coordinate):
        """
        Get the source of the iteration.

        Parameters
        ----------
        iteration_coordinate : str
            The full unique identifier for this iteration.

        Returns
        -------
        str
            The source of the iteration.
        """
        return get_source_system(iteration_coordinate)

    def _get_row_source(self, row_id):
        """
        Get the source of the case at the specified row of this table.

        Parameters
        ----------
        row_id : int
            The row_id of the case in the table.

        Returns
        -------
        str
            The source of the case.
        """
        table = self._table_name.split('_')[0]  # remove "_iterations" from table name

        for global_iter in self._global_iterations:
            record_type, row, source = global_iter[1], global_iter[2], global_iter[3]
            if record_type == table and row == row_id:
                return source

        return None

    def _get_first(self, source):
        """
        Get the first case from the specified source.

        Parameters
        ----------
        source : str
            The source.

        Returns
        -------
        Case
            The first case from the specified source.
        """
        for case in self.cases():
            if case.source == source:
                return case

        return None


class DriverCases(CaseTable):
    """
    Cases specific to the entries that might be recorded in a Driver iteration.

    Parameters
    ----------
    filename : str
        The name of the recording file from which to instantiate the case reader.
    format_version : int
        The version of the format assumed when loading the file.
    giter : list of tuple
        The global iterations table.
    prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    abs2meta : dict
        Dictionary mapping absolute variable names to variable metadata.
    conns : dict
        Dictionary of all model connections.
    auto_ivc_map : dict
        Dictionary that maps all auto_ivc sources to either an absolute input name for single
        connections or a promoted input name for multiple connections. This is for output
        display.
    var_info : dict
        Dictionary with information about variables (scaling, indices, execution order).
    """

    def __init__(self, filename, format_version, giter, prom2abs, abs2prom, abs2meta, conns,
                 auto_ivc_map, var_info):
        """
        Initialize.
        """
        super().__init__(filename, format_version,
                         'driver_iterations', 'iteration_coordinate', giter,
                         prom2abs, abs2prom, abs2meta, conns, auto_ivc_map,
                         var_info)
        self._var_info = var_info

    def cases(self, cache=False):
        """
        Iterate over all cases, optionally caching them into memory.

        Override base class to add derivatives from the derivatives table.

        Parameters
        ----------
        cache : bool
            If True, cases will be cached for faster access by key.

        Yields
        ------
        case
        """
        with sqlite3.connect(self._filename) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(f"SELECT * FROM {self._table_name} ORDER BY id ASC")  # nosec: trusted input
            rows = cur.fetchall()

            for row in rows:
                if self._format_version > 1:
                    # fetch associated derivative data, if available
                    cur.execute("SELECT * FROM driver_derivatives WHERE iteration_coordinate=?",
                                (row['iteration_coordinate'], ))
                    derivs_row = cur.fetchone()

                    if derivs_row:
                        # convert row to a regular dict and add jacobian
                        row = dict(zip(row.keys(), row))
                        row['jacobian'] = derivs_row['derivatives']

                case = Case('driver', row, self._prom2abs, self._abs2prom, self._abs2meta,
                            self._conns, self._auto_ivc_map, self._var_info, self._format_version)

                if cache:
                    self._cases[case.name] = case

                yield case

        con.close()

    def get_case(self, case_id, cache=False):
        """
        Get a case from the database.

        Parameters
        ----------
        case_id : int or str
            The integer index or string-identifier of the case to be retrieved.
        cache : bool
            If True, cache the case so it does not have to be fetched on next access.

        Returns
        -------
        Case
            The specified case from the driver_iterations and driver_derivatives tables.
        """
        # check to see if we've already cached this case
        if isinstance(case_id, int):
            case_id = self._get_iteration_coordinate(case_id)

        # return cached case if present, else fetch it
        if case_id in self._cases:
            return self._cases[case_id]

        # Get an unscaled case if does not already exist in _cases
        with sqlite3.connect(self._filename) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()

            # fetch driver iteration data
            cur.execute("SELECT * FROM driver_iterations WHERE "
                        "iteration_coordinate=:iteration_coordinate",
                        {"iteration_coordinate": case_id})
            row = cur.fetchone()

            # fetch associated derivative data, if available
            if row and self._format_version > 1:
                cur.execute("SELECT * FROM driver_derivatives WHERE "
                            "iteration_coordinate=:iteration_coordinate",
                            {"iteration_coordinate": case_id})
                derivs_row = cur.fetchone()

                if derivs_row:
                    # convert row to a regular dict and add jacobian
                    row = dict(zip(row.keys(), row))
                    row['jacobian'] = derivs_row['derivatives']
        con.close()

        # if found, create Case object (and cache it if requested) else return None
        if row:
            case = Case('driver', row, self._prom2abs, self._abs2prom, self._abs2meta,
                        self._conns, self._auto_ivc_map, self._var_info, self._format_version)
            if cache:
                self._cases[case_id] = case
            return case
        else:
            return None

    def list_sources(self):
        """
        Get the list of sources that recorded data in this table (just the driver).

        Returns
        -------
        list
            List of sources.
        """
        return ['driver']

    def _get_source(self, iteration_coordinate):
        """
        Get the source of the iteration.

        Parameters
        ----------
        iteration_coordinate : str
            The full unique identifier for this iteration.

        Returns
        -------
        str
            'driver' (all cases in this table come from the driver).
        """
        return 'driver'

    def _get_row_source(self, row_id):
        """
        Get the source of the case at the specified row of this table.

        Parameters
        ----------
        row_id : int
            The row_id of the case in the table.

        Returns
        -------
        str
            The source of the case.
        """
        return 'driver'


class SystemCases(CaseTable):
    """
    Cases specific to the entries that might be recorded in a System iteration.

    Parameters
    ----------
    filename : str
        The name of the recording file from which to instantiate the case reader.
    format_version : int
        The version of the format assumed when loading the file.
    giter : list of tuple
        The global iterations table.
    prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    abs2meta : dict
        Dictionary mapping absolute variable names to variable metadata.
    conns : dict
        Dictionary of all model connections.
    auto_ivc_map : dict
        Dictionary that maps all auto_ivc sources to either an absolute input name for single
        connections or a promoted input name for multiple connections. This is for output
        display.
    var_info : dict
        Dictionary with information about variables (scaling, indices, execution order).
    """

    def __init__(self, filename, format_version, giter, prom2abs, abs2prom, abs2meta, conns,
                 auto_ivc_map, var_info):
        """
        Initialize.
        """
        super().__init__(filename, format_version,
                         'system_iterations', 'iteration_coordinate', giter,
                         prom2abs, abs2prom, abs2meta, conns, auto_ivc_map,
                         var_info)


class SolverCases(CaseTable):
    """
    Cases specific to the entries that might be recorded in a Solver iteration.

    Parameters
    ----------
    filename : str
        The name of the recording file from which to instantiate the case reader.
    format_version : int
        The version of the format assumed when loading the file.
    giter : list of tuple
        The global iterations table.
    prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    abs2meta : dict
        Dictionary mapping absolute variable names to variable metadata.
    conns : dict
        Dictionary of all model connections.
    auto_ivc_map : dict
        Dictionary that maps all auto_ivc sources to either an absolute input name for single
        connections or a promoted input name for multiple connections. This is for output
        display.
    var_info : dict
        Dictionary with information about variables (scaling, indices, execution order).
    """

    def __init__(self, filename, format_version, giter, prom2abs, abs2prom, abs2meta, conns,
                 auto_ivc_map, var_info):
        """
        Initialize.
        """
        super().__init__(filename, format_version,
                         'solver_iterations', 'iteration_coordinate', giter,
                         prom2abs, abs2prom, abs2meta, conns, auto_ivc_map,
                         var_info)

    def _get_source(self, iteration_coordinate):
        """
        Get pathname of solver that is the source of the iteration.

        Parameters
        ----------
        iteration_coordinate : str
            The full unique identifier for this iteration.

        Returns
        -------
        str
            The pathname of the solver that is the source of the iteration.
        """
        source_system = get_source_system(iteration_coordinate)

        system_solve = source_system.split('.')[-1] + '._solve_nonlinear'
        system_coord_len = iteration_coordinate.index(system_solve) + len(system_solve)
        system_coord_nodes = len(iteration_coordinate[:system_coord_len].split('|')) + 1
        iter_coord_nodes = len(iteration_coordinate.split('|'))

        if iter_coord_nodes == system_coord_nodes + 2:
            return source_system + '.nonlinear_solver'
        elif iter_coord_nodes == system_coord_nodes + 4:
            return source_system + '.nonlinear_solver.linesearch'
        else:
            raise RuntimeError("Can't parse solver iteration coordinate: %s" % iteration_coordinate)


class ProblemCases(CaseTable):
    """
    Cases specific to the entries that might be recorded in a Driver iteration.

    Parameters
    ----------
    filename : str
        The name of the recording file from which to instantiate the case reader.
    format_version : int
        The version of the format assumed when loading the file.
    giter : list of tuple
        The global iterations table.
    prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    abs2meta : dict
        Dictionary mapping absolute variable names to variable metadata.
    conns : dict
        Dictionary of all model connections.
    auto_ivc_map : dict
        Dictionary that maps all auto_ivc sources to either an absolute input name for single
        connections or a promoted input name for multiple connections. This is for output
        display.
    var_info : dict
        Dictionary with information about variables (scaling, indices, execution order).
    """

    def __init__(self, filename, format_version, giter, prom2abs, abs2prom, abs2meta, conns,
                 auto_ivc_map, var_info):
        """
        Initialize.
        """
        super().__init__(filename, format_version,
                         'problem_cases', 'case_name', giter,
                         prom2abs, abs2prom, abs2meta, conns, auto_ivc_map,
                         var_info)

    def list_sources(self):
        """
        Get the list of sources that recorded data in this table (just the problem).

        Returns
        -------
        list
            List of sources.
        """
        return ['problem']

    def _get_source(self, iteration_coordinate):
        """
        Get the source of the iteration.

        Parameters
        ----------
        iteration_coordinate : str
            The full unique identifier for this iteration.

        Returns
        -------
        str
            'problem' (all cases in this table come from the problem).
        """
        return 'problem'

    def _get_row_source(self, row_id):
        """
        Get the source of the case at the specified row of this table.

        Parameters
        ----------
        row_id : int
            The row_id of the case in the table.

        Returns
        -------
        str
            The source of the case.
        """
        return 'problem'
