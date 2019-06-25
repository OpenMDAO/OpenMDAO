"""Define the Problem class and a FakeComm class for non-MPI users."""

from __future__ import division, print_function

import sys
import os

from collections import defaultdict, namedtuple
from fnmatch import fnmatchcase
from itertools import product

from six import iteritems, iterkeys, itervalues
from six.moves import range, cStringIO

import numpy as np
import scipy.sparse as sparse

from openmdao.core.component import Component
from openmdao.core.driver import Driver
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group
from openmdao.core.group import System
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.total_jac import _TotalJacInfo
from openmdao.approximation_schemes.complex_step import ComplexStep
from openmdao.approximation_schemes.finite_difference import FiniteDifference
from openmdao.solvers.solver import SolverInfo
from openmdao.error_checking.check_config import _default_checks, _all_checks
from openmdao.recorders.recording_iteration_stack import _RecIteration
from openmdao.recorders.recording_manager import RecordingManager, record_viewer_data
from openmdao.utils.record_util import create_local_meta
from openmdao.utils.general_utils import warn_deprecation, ContainsAll, pad_name, simple_warning
from openmdao.utils.mpi import FakeComm
from openmdao.utils.mpi import MPI
from openmdao.utils.name_maps import prom_name2abs_name
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.units import get_conversion
from openmdao.utils import coloring as coloring_mod
from openmdao.utils.name_maps import abs_key2rel_key
from openmdao.vectors.default_vector import DefaultVector
from openmdao.utils.logger_utils import get_logger
import openmdao.utils.coloring as coloring_mod

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.utils.name_maps import rel_key2abs_key, rel_name2abs_name

# Use this as a special value to be able to tell if the caller set a value for the optional
#   out_stream argument. We run into problems running testflo if we use a default of sys.stdout.
_DEFAULT_OUT_STREAM = object()

ErrorTuple = namedtuple('ErrorTuple', ['forward', 'reverse', 'forward_reverse'])
MagnitudeTuple = namedtuple('MagnitudeTuple', ['forward', 'reverse', 'fd'])

_contains_all = ContainsAll()
_undefined = object()


CITATION = """@article{openmdao_2019,
    Author={Justin S. Gray and John T. Hwang and Joaquim R. R. A.
            Martins and Kenneth T. Moore and Bret A. Naylor},
    Title="{OpenMDAO: An Open-Source Framework for Multidisciplinary
            Design, Analysis, and Optimization}",
    Journal="{Structural and Multidisciplinary Optimization}",
    Year={2019},
    Publisher={Springer},
    pdf={http://openmdao.org/pubs/openmdao_overview_2019.pdf},
    note= {In Press}
    }"""


class Problem(object):
    """
    Top-level container for the systems and drivers.

    Attributes
    ----------
    model : <System>
        Pointer to the top-level <System> object (root node in the tree).
    comm : MPI.Comm or <FakeComm>
        The global communicator.
    driver : <Driver>
        Slot for the driver. The default driver is `Driver`, which just runs
        the model once.
    _mode : 'fwd' or 'rev'
        Derivatives calculation mode, 'fwd' for forward, and 'rev' for
        reverse (adjoint).
    _orig_mode : 'fwd', 'rev', or 'auto'
        Derivatives calculation mode assigned by the user.  If set to 'auto', _mode will be
        automatically assigned to 'fwd' or 'rev' based on relative sizes of design variables vs.
        responses.
    _solver_print_cache : list
        Allows solver iprints to be set to requested values after setup calls.
    _initial_condition_cache : dict
        Any initial conditions that are set at the problem level via setitem are cached here
        until they can be processed.
    _setup_status : int
        Current status of the setup in _model.
        0 -- Newly initialized problem or newly added model.
        1 -- The `setup` method has been called, but vectors not initialized.
        2 -- The `final_setup` has been run, everything ready to run.
    cite : str
        Listing of relevant citations that should be referenced when
        publishing work that uses this class.
    options : <OptionsDictionary>
        Dictionary with general options for the problem.
    recording_options : <OptionsDictionary>
        Dictionary with problem recording options.
    _rec_mgr : <RecordingManager>
        Object that manages all recorders added to this problem.
    _remote_var_set : set
        Set of variables (absolute names) that require remote data transfer to reach all procs.
    _check : bool
        If True, call check_config at the end of final_setup.
    _recording_iter : _RecIteration
        Manages recording of iterations.
    _filtered_vars_to_record : dict
        Dictionary of lists of design vars, constraints, etc. to record.
    _logger : object or None
        Object for logging config checks if _check is True.
    _force_alloc_complex : bool
        Force allocation of imaginary part in nonlinear vectors. OpenMDAO can generally
        detect when you need to do this, but in some cases (e.g., complex step is used
        after a reconfiguration) you may need to set this to True.
    _color_dir_hash : str
        Hash used to detect collisions of coloring files.
    __get_remote : bool
        Flag used to determine when __getitem__ will retrieve remote variables.
    """

    _post_setup_func = None

    def __init__(self, model=None, driver=None, comm=None, root=None, **options):
        """
        Initialize attributes.

        Parameters
        ----------
        model : <System> or None
            The top-level <System>. If not specified, an empty <Group> will be created.
        driver : <Driver> or None
            The driver for the problem. If not specified, a simple "Run Once" driver will be used.
        comm : MPI.Comm or <FakeComm> or None
            The global communicator.
        root : <System> or None
            Deprecated kwarg for `model`.
        **options : named args
            All remaining named args are converted to options.
        """
        # used by the get_val function when get_remote is True
        self.__get_remote = False

        self.cite = CITATION

        if comm is None:
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            except ImportError:
                comm = FakeComm()

        if root is not None:
            if model is not None:
                raise ValueError("Cannot specify both 'root' and 'model'. "
                                 "'root' has been deprecated, please use 'model'.")

            warn_deprecation("The 'root' argument provides backwards compatibility "
                             "with OpenMDAO <= 1.x ; use 'model' instead.")

            model = root

        if model is None:
            self.model = Group()
        elif isinstance(model, System):
            self.model = model
        else:
            raise TypeError("The value provided for 'model' is not a valid System.")

        if driver is None:
            self.driver = Driver()
        elif isinstance(driver, Driver):
            self.driver = driver
        else:
            raise TypeError("The value provided for 'driver' is not a valid Driver.")

        self.comm = comm

        self._solver_print_cache = []

        self._mode = None  # mode is assigned in setup()

        self._initial_condition_cache = {}

        # Status of the setup of _model.
        # 0 -- Newly initialized problem or newly added model.
        # 1 -- The `setup` method has been called, but vectors not initialized.
        # 2 -- The `final_setup` has been run, everything ready to run.
        self._setup_status = 0

        self._rec_mgr = RecordingManager()

        # General options
        self.options = OptionsDictionary()
        self.options.declare('coloring_dir', types=str,
                             default=os.path.join(os.getcwd(), 'coloring_files'),
                             desc='Directory containing coloring files (if any) for this Problem.')
        self.options.update(options)

        # Case recording options
        self.recording_options = OptionsDictionary()

        self.recording_options.declare('record_desvars', types=bool, default=True,
                                       desc='Set to True to record design variables at the '
                                            'problem level')
        self.recording_options.declare('record_objectives', types=bool, default=True,
                                       desc='Set to True to record objectives at the problem level')
        self.recording_options.declare('record_constraints', types=bool, default=True,
                                       desc='Set to True to record constraints at the '
                                            'problem level')
        self.recording_options.declare('includes', types=list, default=['*'],
                                       desc='Patterns for variables to include in recording')
        self.recording_options.declare('excludes', types=list, default=[],
                                       desc='Patterns for vars to exclude in recording '
                                            '(processed post-includes)')

    def _get_var_abs_name(self, name):
        if name in self.model._var_allprocs_abs2meta:
            return name
        elif name in self.model._var_allprocs_prom2abs_list['output']:
            return self.model._var_allprocs_prom2abs_list['output'][name][0]
        elif name in self.model._var_allprocs_prom2abs_list['input']:
            abs_names = self.model._var_allprocs_prom2abs_list['input'][name]
            if len(abs_names) == 1:
                return abs_names[0]
            else:
                raise KeyError("Using promoted name `{}' is ambiguous and matches unconnected "
                               "inputs %s. Use absolute name to disambiguate.".format(name,
                                                                                      abs_names))

        raise KeyError("Variable '{}' not found.".format(name))

    def is_local(self, name):
        """
        Return True if the named variable or system is local to the current process.

        Parameters
        ----------
        name : str
            Name of a variable or system.

        Returns
        -------
        bool
            True if the named system or variable is local to this process.
        """
        if self._setup_status < 1:
            raise RuntimeError("is_local('{}') was called before setup() completed.".format(name))

        try:
            abs_name = self._get_var_abs_name(name)
        except KeyError:
            sub = self.model._get_subsystem(name)
            if sub is None:  # either the sub is remote or there is no sub by that name
                # TODO: raise exception if sub does not exist
                return False
            else:
                # if system has been set up, _var_sizes will be initialized
                return sub._var_sizes is not None

        # variable exists, but may be remote
        return abs_name in self.model._var_abs2meta

    def __getitem__(self, name):
        """
        Get an output/input variable.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.

        Returns
        -------
        float or ndarray or any python object
            the requested output/input variable.
        """
        # Caching only needed if vectors aren't allocated yet.
        proms = self.model._var_allprocs_prom2abs_list
        meta = self.model._var_abs2meta

        val = _undefined
        abs_name = None

        if self._setup_status == 1:

            # We have set and cached already
            if name in self._initial_condition_cache:
                return self._initial_condition_cache[name]

            # Vector not setup, so we need to pull values from saved metadata request.
            else:
                if name in meta:
                    if isinstance(self.model, Group) and name in self.model._conn_abs_in2out:
                        src_name = self.model._conn_abs_in2out[name]
                        val = meta[src_name]['value']
                    else:
                        val = meta[name]['value']

                elif name in proms['output']:
                    abs_name = prom_name2abs_name(self.model, name, 'output')
                    if abs_name in meta:
                        val = meta[abs_name]['value']

                elif name in proms['input']:
                    abs_name = proms['input'][name][0]
                    conn = self.model._conn_abs_in2out
                    if abs_name in meta:
                        if isinstance(self.model, Group) and abs_name in conn:
                            src_name = self.model._conn_abs_in2out[abs_name]
                            # So, if the inputs and outputs are promoted to the same name, then we
                            # allow getitem, but if they aren't, then we raise an error due to non
                            # uniqueness.
                            if name not in proms['output']:
                                # This triggers a check for unconnected non-unique inputs, and
                                # raises the same error as vector access.
                                abs_name = prom_name2abs_name(self.model, name, 'input')
                            val = meta[src_name]['value']
                        else:
                            # This triggers a check for unconnected non-unique inputs, and
                            # raises the same error as vector access.
                            abs_name = prom_name2abs_name(self.model, name, 'input')
                            val = meta[abs_name]['value']

                if val is not _undefined:
                    # Need to cache the "get" in case the user calls in-place numpy operations.
                    self._initial_condition_cache[name] = val

        else:
            if name in proms['output']:
                name = proms['output'][name][0]
            elif name in proms['input']:
                name = prom_name2abs_name(self.model, name, 'input')

            if name in meta:   # local var
                if name in self.model._outputs._views:
                    val = self.model._outputs[name]
                else:
                    val = self.model._inputs[name]

            elif name in self.model._discrete_outputs:
                val = self.model._discrete_outputs[name]

            elif name in self.model._discrete_inputs:
                val = self.model._discrete_inputs[name]

        if self.model.comm.size > 1:
            allprocs_meta = self.model._var_allprocs_abs2meta
            # check for remote var
            if name in allprocs_meta:
                abs_name = name
            elif name in proms['output']:
                abs_name = proms['output'][name][0]
            elif name in proms['input']:
                abs_name = proms['input'][name][0]

            if abs_name in self._remote_var_set:
                if self.__get_remote:
                    if abs_name in allprocs_meta and allprocs_meta[abs_name]['distributed']:
                        raise RuntimeError("Retrieval of the full distributed variable '%s' is not "
                                           "supported." % abs_name)
                    loc_val = val
                    owner = self.model._owning_rank[abs_name]
                    if owner != self.model.comm.rank:
                        val = None
                    else:
                        owner = self.model._owning_rank[abs_name]
                        if val is _undefined:
                            val = None
                    new_val = self.model.comm.bcast(val, root=owner)
                    if loc_val is not None and loc_val is not _undefined:
                        val = loc_val
                    else:
                        val = new_val
                elif val is _undefined:
                    raise RuntimeError(
                        "Variable '{}' is not local to rank {}. You can retrieve values from "
                        "other processes using "
                        "`problem.get_val(<name>, get_remote=True)`.".format(name, self.comm.rank))

        if val is _undefined:
            raise KeyError('Variable name "{}" not found.'.format(name))

        return val

    def get_val(self, name, units=None, indices=None, get_remote=False):
        """
        Get an output/input variable.

        Function is used if you want to specify display units.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.
        units : str, optional
            Units to convert to before upon return.
        indices : int or list of ints or tuple of ints or int ndarray or Iterable or None, optional
            Indices or slice to return.
        get_remote : bool
            If True, retrieve the value even if it is on a remote process.  Note that if the
            variable is remote on ANY process, this function must be called on EVERY process
            in the Problem's MPI communicator.

        Returns
        -------
        float or ndarray
            The requested output/input variable.
        """
        if get_remote:
            self.__get_remote = True
            try:
                val = self[name]
            finally:
                self.__get_remote = False
        else:
            val = self[name]

        if indices is not None:
            val = val[indices]

        if units is not None:
            base_units = self._get_units(name)

            if base_units is None:
                msg = "Can't express variable '{}' with units of 'None' in units of '{}'."
                raise TypeError(msg.format(name, units))

            try:
                scale, offset = get_conversion(base_units, units)
            except TypeError:
                msg = "Can't express variable '{}' with units of '{}' in units of '{}'."
                raise TypeError(msg.format(name, base_units, units))

            val = (val + offset) * scale

        return val

    def _get_units(self, name):
        """
        Get the units for a variable name.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.

        Returns
        -------
        str
            Unit string.
        """
        meta = self.model._var_allprocs_abs2meta
        if name in meta:
            return meta[name]['units']

        proms = self.model._var_allprocs_prom2abs_list
        if self._setup_status >= 1:

            if name in proms['output']:
                abs_name = prom_name2abs_name(self.model, name, 'output')
                return meta[abs_name]['units']
            elif name in proms['input']:
                # This triggers a check for unconnected non-unique inputs, and
                # raises the same error as vector access.
                abs_name = prom_name2abs_name(self.model, name, 'input')
                return meta[abs_name]['units']

        raise KeyError('Variable name "{}" not found.'.format(name))

    def __setitem__(self, name, value):
        """
        Set an output/input variable.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.
        value : float or ndarray or any python object
            value to set this variable to.
        """
        # Caching only needed if vectors aren't allocated yet.
        if self._setup_status == 1:
            self._initial_condition_cache[name] = value
        else:
            all_proms = self.model._var_allprocs_prom2abs_list

            if name in all_proms['output']:
                abs_name = all_proms['output'][name][0]
            elif name in all_proms['input']:
                abs_name = prom_name2abs_name(self.model, name, 'input')
            else:
                abs_name = name

            if abs_name in self.model._outputs._views:
                self.model._outputs[abs_name] = value
            elif abs_name in self.model._inputs._views:
                self.model._inputs[abs_name] = value
            elif abs_name in self.model._discrete_outputs:
                self.model._discrete_outputs[abs_name] = value
            elif abs_name in self.model._discrete_inputs:
                self.model._discrete_inputs[abs_name] = value
            else:
                # might be a remote var.  If so, just do nothing on this proc
                if abs_name in self.model._var_allprocs_abs2meta:
                    print("Variable '{}' is remote on rank {}.  "
                          "Local assignment ignored.".format(name, self.comm.rank))
                else:
                    raise KeyError('Variable name "{}" not found.'.format(name))

    def set_val(self, name, value, units=None, indices=None):
        """
        Set an output/input variable.

        Function is used if you want to set a value using a different unit.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.
        value : float or ndarray or list
            Value to set this variable to.
        units : str, optional
            Units that value is defined in.
        indices : int or list of ints or tuple of ints or int ndarray or Iterable or None, optional
            Indices or slice to set to specified value.
        """
        if units is not None:
            base_units = self._get_units(name)

            if base_units is None:
                msg = "Can't set variable '{}' with units of 'None' to value with units of '{}'."
                raise TypeError(msg.format(name, units))

            try:
                scale, offset = get_conversion(units, base_units)
            except TypeError:
                msg = "Can't set variable '{}' with units of '{}' to value with units of '{}'."
                raise TypeError(msg.format(name, base_units, units))

            value = (value + offset) * scale

        if indices is not None:
            self[name][indices] = value
        else:
            self[name] = value

    def _set_initial_conditions(self):
        """
        Set all initial conditions that have been saved in cache after setup.
        """
        for name, value in iteritems(self._initial_condition_cache):
            self[name] = value

        # Clean up cache
        self._initial_condition_cache = {}

    @property
    def root(self):
        """
        Provide 'root' property for backwards compatibility.

        Returns
        -------
        <Group>
            reference to the 'model' property.
        """
        warn_deprecation("The 'root' property provides backwards compatibility "
                         "with OpenMDAO <= 1.x ; use 'model' instead.")
        return self.model

    @root.setter
    def root(self, model):
        """
        Provide for setting the 'root' property for backwards compatibility.

        Parameters
        ----------
        model : <Group>
            reference to a <Group> to be assigned to the 'model' property.
        """
        warn_deprecation("The 'root' property provides backwards compatibility "
                         "with OpenMDAO <= 1.x ; use 'model' instead.")
        self.model = model

    def run_model(self, case_prefix=None, reset_iter_counts=True):
        """
        Run the model by calling the root system's solve_nonlinear.

        Parameters
        ----------
        case_prefix : str or None
            Prefix to prepend to coordinates when recording.

        reset_iter_counts : bool
            If True and model has been run previously, reset all iteration counters.
        """
        if self._mode is None:
            raise RuntimeError("The `setup` method must be called before `run_model`.")

        if case_prefix:
            if not isinstance(case_prefix, str):
                raise TypeError("The 'case_prefix' argument should be a string.")
            self._recording_iter.prefix = case_prefix
        else:
            self._recording_iter.prefix = None

        if self.model.iter_count > 0 and reset_iter_counts:
            self.driver.iter_count = 0
            self.model._reset_iter_counts()

        self.final_setup()
        self.model._clear_iprint()
        self.model.run_solve_nonlinear()

    def run_driver(self, case_prefix=None, reset_iter_counts=True):
        """
        Run the driver on the model.

        Parameters
        ----------
        case_prefix : str or None
            Prefix to prepend to coordinates when recording.

        reset_iter_counts : bool
            If True and model has been run previously, reset all iteration counters.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        if self._mode is None:
            raise RuntimeError("The `setup` method must be called before `run_driver`.")

        if case_prefix:
            if not isinstance(case_prefix, str):
                raise TypeError("The 'case_prefix' argument should be a string.")
            self._recording_iter.prefix = case_prefix
        else:
            self._recording_iter.prefix = None

        if self.model.iter_count > 0 and reset_iter_counts:
            self.driver.iter_count = 0
            self.model._reset_iter_counts()

        self.final_setup()
        self.model._clear_iprint()
        return self.driver.run()

    def run_once(self):
        """
        Backward compatible call for run_model.
        """
        warn_deprecation("The 'run_once' method provides backwards compatibility with "
                         "OpenMDAO <= 1.x ; use 'run_model' instead.")

        self.run_model()

    def run(self):
        """
        Backward compatible call for run_driver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        warn_deprecation("The 'run' method provides backwards compatibility with "
                         "OpenMDAO <= 1.x ; use 'run_driver' instead.")

        return self.run_driver()

    def _setup_recording(self):
        """
        Set up case recording.
        """
        self._filtered_vars_to_record = self.driver._get_vars_to_record(self.recording_options)

        self._rec_mgr.startup(self)

    def add_recorder(self, recorder):
        """
        Add a recorder to the problem.

        Parameters
        ----------
        recorder : CaseRecorder
           A recorder instance.
        """
        self._rec_mgr.append(recorder)

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        # shut down all recorders
        self._rec_mgr.shutdown()

        # clean up driver and model resources
        self.driver.cleanup()
        for system in self.model.system_iter(include_self=True, recurse=True):
            system.cleanup()

    def record_iteration(self, case_name):
        """
        Record the variables at the Problem level.

        Parameters
        ----------
        case_name : str
            Name used to identify this Problem case.
        """
        if not self._rec_mgr._recorders:
            return

        # Get the data to record (collective calls that get across all ranks)
        opts = self.recording_options
        filt = self._filtered_vars_to_record

        model = self.model
        driver = self.driver

        if opts['record_desvars']:
            des_vars = driver.get_design_var_values()
        else:
            des_vars = {}

        if opts['record_objectives']:
            obj_vars = driver.get_objective_values()
        else:
            obj_vars = {}

        if opts['record_constraints']:
            con_vars = driver.get_constraint_values()
        else:
            con_vars = {}

        des_vars = {name: des_vars[name] for name in filt['des']}
        obj_vars = {name: obj_vars[name] for name in filt['obj']}
        con_vars = {name: con_vars[name] for name in filt['con']}

        names = model._outputs._names
        views = model._outputs._views
        sys_vars = {name: views[name] for name in names if name in filt['sys']}

        if MPI:
            des_vars = driver._gather_vars(model, des_vars)
            obj_vars = driver._gather_vars(model, obj_vars)
            con_vars = driver._gather_vars(model, con_vars)
            sys_vars = driver._gather_vars(model, sys_vars)

        outs = {}
        if not MPI or model.comm.rank == 0:
            outs.update(des_vars)
            outs.update(obj_vars)
            outs.update(con_vars)
            outs.update(sys_vars)

        data = {
            'out': outs,
        }

        metadata = create_local_meta(case_name)

        self._rec_mgr.record_iteration(self, data, metadata)

    def setup(self, vector_class=None, check=False, logger=None, mode='auto',
              force_alloc_complex=False, distributed_vector_class=PETScVector,
              local_vector_class=DefaultVector, derivatives=True):
        """
        Set up the model hierarchy.

        When `setup` is called, the model hierarchy is assembled, the processors are allocated
        (for MPI), and variables and connections are all assigned. This method traverses down
        the model hierarchy to call `setup` on each subsystem, and then traverses up the model
        hierarchy to call `configure` on each subsystem.

        Parameters
        ----------
        vector_class : type
            Reference to an actual <Vector> class; not an instance. This is deprecated. Use
            distributed_vector_class instead.
        check : boolean
            whether to run config check after setup is complete.
        logger : object
            Object for logging config checks if check is True.
        mode : string
            Derivatives calculation mode, 'fwd' for forward, and 'rev' for
            reverse (adjoint). Default is 'auto', which will pick 'fwd' or 'rev' based on
            the direction resulting in the smallest number of linear solves required to
            compute derivatives.
        force_alloc_complex : bool
            Force allocation of imaginary part in nonlinear vectors. OpenMDAO can generally
            detect when you need to do this, but in some cases (e.g., complex step is used
            after a reconfiguration) you may need to set this to True.
        distributed_vector_class : type
            Reference to the <Vector> class or factory function used to instantiate vectors
            and associated transfers involved in interprocess communication.
        local_vector_class : type
            Reference to the <Vector> class or factory function used to instantiate vectors
            and associated transfers involved in intraprocess communication.
        derivatives : bool
            If True, perform any memory allocations necessary for derivative computation.

        Returns
        -------
        self : <Problem>
            this enables the user to instantiate and setup in one line.
        """
        model = self.model
        model.force_alloc_complex = force_alloc_complex
        comm = self.comm

        if vector_class is not None:
            warn_deprecation("'vector_class' has been deprecated. Use "
                             "'distributed_vector_class' and/or 'local_vector_class' instead.")
            distributed_vector_class = vector_class

        # PETScVector is required for MPI
        if comm.size > 1:
            if PETScVector is None:
                raise ValueError("Attempting to run in parallel under MPI but PETScVector could not"
                                 "be imported.")
            elif distributed_vector_class is not PETScVector:
                raise ValueError("The `distributed_vector_class` argument must be `PETScVector` "
                                 "when running in parallel under MPI but '%s' was specified."
                                 % distributed_vector_class.__name__)

        if mode not in ['fwd', 'rev', 'auto']:
            msg = "Unsupported mode: '%s'. Use either 'fwd' or 'rev'." % mode
            raise ValueError(msg)

        self._mode = self._orig_mode = mode

        # this will be shared by all Solvers in the model
        model._solver_info = SolverInfo()
        self._recording_iter = _RecIteration()
        model._recording_iter = self._recording_iter

        model_comm = self.driver._setup_comm(comm)

        model._setup(model_comm, 'full', mode, distributed_vector_class, local_vector_class,
                     derivatives, self.options)

        # get set of all vars that we may need to bcast later
        self._remote_var_set = remote_var_set = set()
        if model_comm.size > 1:
            for type_ in ('input', 'output'):
                remote_discrete = set()
                sizes = self.model._var_sizes['nonlinear'][type_]
                for i, vname in enumerate(self.model._var_allprocs_abs_names[type_]):
                    if not np.all(sizes[:, i]):
                        remote_var_set.add(vname)

                if self.model._var_allprocs_discrete[type_]:
                    local = list(self.model._var_discrete[type_])
                    byrank = self.comm.gather(local, root=0)
                    if model_comm.rank == 0:
                        full = set()
                        for names in byrank:
                            full.update(names)

                        for names in byrank:
                            diff = full.difference(names)
                            remote_discrete.update(diff)

                        model_comm.bcast(remote_discrete, root=0)
                    else:
                        remote_discrete = model_comm.bcast(None, root=0)

                    remote_var_set.update(remote_discrete)

        # Cache all args for final setup.
        self._check = check
        self._logger = logger
        self._force_alloc_complex = force_alloc_complex

        self._setup_status = 1

        return self

    def final_setup(self):
        """
        Perform final setup phase on problem in preparation for run.

        This is the second phase of setup, and is done automatically at the start of `run_driver`
        and `run_model`. At the beginning of final_setup, we have a model hierarchy with defined
        variables, solvers, case_recorders, and derivative settings. During this phase, the vectors
        are created and populated, the drivers and solvers are initialized, and the recorders are
        started, and the rest of the framework is prepared for execution.
        """
        driver = self.driver

        response_size, desvar_size = driver._update_voi_meta(self.model)

        # update mode if it's been set to 'auto'
        if self._orig_mode == 'auto':
            mode = 'rev' if response_size < desvar_size else 'fwd'
            self._mode = mode
        else:
            mode = self._orig_mode

        if self._setup_status < 2:
            self.model._final_setup(self.comm, 'full',
                                    force_alloc_complex=self._force_alloc_complex)

        driver._setup_driver(self)

        coloring = driver._coloring_info['coloring']
        if coloring is coloring_mod._STD_COLORING_FNAME:
            coloring = driver._get_static_coloring()
        if (coloring and coloring is not coloring_mod._DYN_COLORING and
                coloring_mod._use_total_sparsity):
            # if we're using simultaneous total derivatives then our effective size is less
            # than the full size
            if coloring._fwd and coloring._rev:
                pass  # we're doing both!
            elif mode == 'fwd' and coloring._fwd:
                desvar_size = coloring.total_solves()
            elif mode == 'rev' and coloring._rev:
                response_size = coloring.total_solves()

        if ((mode == 'fwd' and desvar_size > response_size) or
                (mode == 'rev' and response_size > desvar_size)):
            simple_warning("Inefficient choice of derivative mode.  You chose '%s' for a "
                           "problem with %d design variables and %d response variables "
                           "(objectives and nonlinear constraints)." %
                           (mode, desvar_size, response_size), RuntimeWarning)

        # we only want to set up recording once, after problem setup
        if self._setup_status == 1:
            driver._setup_recording()
            self._setup_recording()
            record_viewer_data(self)

        # Now that setup has been called, we can set the iprints.
        for items in self._solver_print_cache:
            self.set_solver_print(level=items[0], depth=items[1], type_=items[2])

        if self._check and self.comm.rank == 0:
            self.check_config(self._logger)

        if self._setup_status < 2:
            self._setup_status = 2
            self._set_initial_conditions()

        # check for post-setup hook
        if Problem._post_setup_func is not None:
            Problem._post_setup_func(self)

    def check_partials(self, out_stream=_DEFAULT_OUT_STREAM, includes=None, excludes=None,
                       compact_print=False, abs_err_tol=1e-6, rel_err_tol=1e-6,
                       method='fd', step=None, form='forward', step_calc='abs',
                       force_dense=True, show_only_incorrect=False):
        """
        Check partial derivatives comprehensively for all components in your model.

        Parameters
        ----------
        out_stream : file-like object
            Where to send human readable output. By default it goes to stdout.
            Set to None to suppress.
        includes : None or list_like
            List of glob patterns for pathnames to include in the check. Default is None, which
            includes all components in the model.
        excludes : None or list_like
            List of glob patterns for pathnames to exclude from the check. Default is None, which
            excludes nothing.
        compact_print : bool
            Set to True to just print the essentials, one line per unknown-param pair.
        abs_err_tol : float
            Threshold value for absolute error.  Errors about this value will have a '*' displayed
            next to them in output, making them easy to search for. Default is 1.0E-6.
        rel_err_tol : float
            Threshold value for relative error.  Errors about this value will have a '*' displayed
            next to them in output, making them easy to search for. Note at times there may be a
            significant relative error due to a minor absolute error.  Default is 1.0E-6.
        method : str
            Method, 'fd' for finite difference or 'cs' for complex step. Default is 'fd'.
        step : float
            Step size for approximation. Default is None, which means 1e-6 for 'fd' and 1e-40 for
            'cs'.
        form : string
            Form for finite difference, can be 'forward', 'backward', or 'central'. Default
            'forward'.
        step_calc : string
            Step type for finite difference, can be 'abs' for absolute', or 'rel' for relative.
            Default is 'abs'.
        force_dense : bool
            If True, analytic derivatives will be coerced into arrays. Default is True.
        show_only_incorrect : bool, optional
            Set to True if output should print only the subjacs found to be incorrect.

        Returns
        -------
        dict of dicts of dicts
            First key:
                is the component name;
            Second key:
                is the (output, input) tuple of strings;
            Third key:
                is one of ['rel error', 'abs error', 'magnitude', 'J_fd', 'J_fwd', 'J_rev'];

            For 'rel error', 'abs error', 'magnitude' the value is: A tuple containing norms for
                forward - fd, adjoint - fd, forward - adjoint.
            For 'J_fd', 'J_fwd', 'J_rev' the value is: A numpy array representing the computed
                Jacobian for the three different methods of computation.
        """
        if self._setup_status < 2:
            self.final_setup()

        model = self.model

        if not model._use_derivatives:
            raise RuntimeError("Can't check partials.  Derivative support has been turned off.")

        # TODO: Once we're tracking iteration counts, run the model if it has not been run before.

        includes = [includes] if isinstance(includes, str) else includes
        excludes = [excludes] if isinstance(excludes, str) else excludes

        comps = []
        for comp in model.system_iter(typ=Component, include_self=True):
            if isinstance(comp, IndepVarComp):
                continue

            name = comp.pathname

            # Process includes
            if includes is not None:
                for pattern in includes:
                    if fnmatchcase(name, pattern):
                        break
                else:
                    continue

            # Process excludes
            if excludes is not None:
                match = False
                for pattern in excludes:
                    if fnmatchcase(name, pattern):
                        match = True
                        break
                if match:
                    continue

            comps.append(comp)

        self.set_solver_print(level=0)

        # This is a defaultdict of (defaultdict of dicts).
        partials_data = defaultdict(lambda: defaultdict(dict))

        # Caching current point to restore after setups.
        input_cache = model._inputs._clone()
        output_cache = model._outputs._clone()

        # Keep track of derivative keys that are declared dependent so that we don't print them
        # unless they are in error.
        indep_key = {}

        # Analytic Jacobians
        for mode in ('fwd', 'rev'):
            model._inputs.set_vec(input_cache)
            model._outputs.set_vec(output_cache)
            # Make sure we're in a valid state
            model.run_apply_nonlinear()

            jac_key = 'J_' + mode

            for comp in comps:

                # Only really need to linearize once.
                if mode == 'fwd':
                    comp.run_linearize()

                explicit = isinstance(comp, ExplicitComponent)
                matrix_free = comp.matrix_free
                c_name = comp.pathname
                indep_key[c_name] = set()

                # TODO: Check deprecated deriv_options.

                with comp._unscaled_context():

                    of_list, wrt_list = \
                        comp._get_potential_partials_lists(include_wrt_outputs=not explicit)

                    # Matrix-free components need to calculate their Jacobian by matrix-vector
                    # product.
                    if matrix_free:

                        dstate = comp._vectors['output']['linear']
                        if mode == 'fwd':
                            dinputs = comp._vectors['input']['linear']
                            doutputs = comp._vectors['residual']['linear']
                            in_list = wrt_list
                            out_list = of_list
                        else:
                            dinputs = comp._vectors['residual']['linear']
                            doutputs = comp._vectors['input']['linear']
                            in_list = of_list
                            out_list = wrt_list

                        for inp in in_list:
                            inp_abs = rel_name2abs_name(comp, inp)

                            try:
                                flat_view = dinputs._views_flat[inp_abs]
                            except KeyError:
                                # Implicit state
                                flat_view = dstate._views_flat[inp_abs]

                            n_in = len(flat_view)
                            for idx in range(n_in):

                                dinputs.set_const(0.0)
                                dstate.set_const(0.0)

                                # Dictionary access returns a scalar for 1d input, and we
                                # need a vector for clean code, so use _views_flat.
                                flat_view[idx] = 1.0

                                # Matrix Vector Product
                                comp._apply_linear(None, ['linear'], _contains_all, mode)

                                for out in out_list:
                                    out_abs = rel_name2abs_name(comp, out)

                                    try:
                                        derivs = doutputs._views_flat[out_abs]
                                    except KeyError:
                                        # Implicit state
                                        derivs = dstate._views_flat[out_abs]

                                    if mode == 'fwd':
                                        key = out, inp
                                        deriv = partials_data[c_name][key]

                                        # Allocate first time
                                        if jac_key not in deriv:
                                            shape = (len(derivs), n_in)
                                            deriv[jac_key] = np.zeros(shape)

                                        deriv[jac_key][:, idx] = derivs

                                    else:
                                        key = inp, out
                                        deriv = partials_data[c_name][key]

                                        # Allocate first time
                                        if jac_key not in deriv:
                                            shape = (n_in, len(derivs))
                                            deriv[jac_key] = np.zeros(shape)

                                        deriv[jac_key][idx, :] = derivs

                    # These components already have a Jacobian with calculated derivatives.
                    else:
                        subjacs = comp._jacobian._subjacs_info

                        for rel_key in product(of_list, wrt_list):
                            abs_key = rel_key2abs_key(comp, rel_key)
                            of, wrt = abs_key

                            # No need to calculate partials; they are already stored
                            try:
                                deriv_value = subjacs[abs_key]['value']
                                rows = subjacs[abs_key]['rows']
                            except KeyError:
                                deriv_value = rows = None

                            # Testing for pairs that are not dependent so that we suppress printing
                            # them unless the fd is non zero. Note: subjacs_info is empty for
                            # undeclared partials, which is the default behavior now.
                            try:
                                if not subjacs[abs_key]['dependent']:
                                    indep_key[c_name].add(rel_key)
                            except KeyError:
                                indep_key[c_name].add(rel_key)

                            if deriv_value is None:
                                # Missing derivatives are assumed 0.
                                in_size = comp._var_abs2meta[wrt]['size']
                                out_size = comp._var_abs2meta[of]['size']
                                deriv_value = np.zeros((out_size, in_size))

                            if force_dense:
                                if rows is not None:
                                    try:
                                        in_size = comp._var_abs2meta[wrt]['size']
                                    except KeyError:
                                        in_size = comp._var_abs2meta[wrt]['size']
                                    out_size = comp._var_abs2meta[of]['size']
                                    tmp_value = np.zeros((out_size, in_size))
                                    # if a scalar value is provided (in declare_partials),
                                    # expand to the correct size array value for zipping
                                    if deriv_value.size == 1:
                                        deriv_value *= np.ones(rows.size)
                                    for i, j, val in zip(rows, subjacs[abs_key]['cols'],
                                                         deriv_value):
                                        tmp_value[i, j] += val
                                    deriv_value = tmp_value

                                elif sparse.issparse(deriv_value):
                                    deriv_value = deriv_value.todense()

                            partials_data[c_name][rel_key][jac_key] = deriv_value.copy()

        model._inputs.set_vec(input_cache)
        model._outputs.set_vec(output_cache)
        model.run_apply_nonlinear()

        # Finite Difference to calculate Jacobian
        jac_key = 'J_fd'
        alloc_complex = model._outputs._alloc_complex
        all_fd_options = {}
        comps_could_not_cs = set()
        requested_method = method
        for comp in comps:

            c_name = comp.pathname
            all_fd_options[c_name] = {}
            explicit = isinstance(comp, ExplicitComponent)

            approximations = {'fd': FiniteDifference(),
                              'cs': ComplexStep()}

            of, wrt = comp._get_potential_partials_lists(include_wrt_outputs=not explicit)

            # Load up approximation objects with the requested settings.
            local_opts = comp._get_check_partial_options()
            for rel_key in product(of, wrt):
                abs_key = rel_key2abs_key(comp, rel_key)
                local_wrt = rel_key[1]

                # Determine if fd or cs.
                method = requested_method
                if local_wrt in local_opts:
                    local_method = local_opts[local_wrt]['method']
                    if local_method:
                        method = local_method

                # We can't use CS if we haven't allocated a complex vector, so we fall back on fd.
                if method == 'cs' and not alloc_complex:
                    comps_could_not_cs.add(c_name)
                    method = 'fd'

                fd_options = {'order': None,
                              'method': method}

                if method == 'cs':
                    defaults = ComplexStep.DEFAULT_OPTIONS

                    fd_options['form'] = None
                    fd_options['step_calc'] = None

                elif method == 'fd':
                    defaults = FiniteDifference.DEFAULT_OPTIONS

                    fd_options['form'] = form
                    fd_options['step_calc'] = step_calc

                if step:
                    fd_options['step'] = step
                else:
                    fd_options['step'] = defaults['step']

                # Precedence: component options > global options > defaults
                if local_wrt in local_opts:
                    for name in ['form', 'step', 'step_calc', 'directional']:
                        value = local_opts[local_wrt][name]
                        if value is not None:
                            fd_options[name] = value

                all_fd_options[c_name][local_wrt] = fd_options

                approximations[fd_options['method']].add_approximation(abs_key, fd_options)

            approx_jac = {}
            for approximation in itervalues(approximations):
                # Perform the FD here.
                approximation.compute_approximations(comp, jac=approx_jac)

            for abs_key, partial in iteritems(approx_jac):
                rel_key = abs_key2rel_key(comp, abs_key)
                partials_data[c_name][rel_key][jac_key] = partial

                # If this is a directional derivative, convert the analytic to a directional one.
                wrt = rel_key[1]
                if wrt in local_opts and local_opts[wrt]['directional']:
                    deriv = partials_data[c_name][rel_key]
                    for key in ['J_fwd', 'J_rev']:
                        deriv[key] = np.atleast_2d(np.sum(deriv[key], axis=1)).T

        # Conversion of defaultdict to dicts
        partials_data = {comp_name: dict(outer) for comp_name, outer in iteritems(partials_data)}

        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if len(comps_could_not_cs) > 0:
            msg = "The following components requested complex step, but force_alloc_complex " + \
                  "has not been set to True, so finite difference was used: "
            msg += str(list(comps_could_not_cs))
            msg += "\nTo enable complex step, specify 'force_alloc_complex=True' when calling " + \
                   "setup on the problem, e.g. 'problem.setup(force_alloc_complex=True)'"
            simple_warning(msg)

        _assemble_derivative_data(partials_data, rel_err_tol, abs_err_tol, out_stream,
                                  compact_print, comps, all_fd_options, indep_key=indep_key,
                                  all_comps_provide_jacs=not self.model.matrix_free,
                                  show_only_incorrect=show_only_incorrect)

        return partials_data

    def check_totals(self, of=None, wrt=None, out_stream=_DEFAULT_OUT_STREAM, compact_print=False,
                     driver_scaling=False, abs_err_tol=1e-6, rel_err_tol=1e-6,
                     method='fd', step=None, form=None, step_calc='abs'):
        """
        Check total derivatives for the model vs. finite difference.

        Parameters
        ----------
        of : list of variable name strings or None
            Variables whose derivatives will be computed. Default is None, which
            uses the driver's objectives and constraints.
        wrt : list of variable name strings or None
            Variables with respect to which the derivatives will be computed.
            Default is None, which uses the driver's desvars.
        out_stream : file-like object
            Where to send human readable output. By default it goes to stdout.
            Set to None to suppress.
        compact_print : bool
            Set to True to just print the essentials, one line per unknown-param pair.
        driver_scaling : bool
            When True, return derivatives that are scaled according to either the adder and scaler
            or the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is False, which is unscaled.
        abs_err_tol : float
            Threshold value for absolute error.  Errors about this value will have a '*' displayed
            next to them in output, making them easy to search for. Default is 1.0E-6.
        rel_err_tol : float
            Threshold value for relative error.  Errors about this value will have a '*' displayed
            next to them in output, making them easy to search for. Note at times there may be a
            significant relative error due to a minor absolute error.  Default is 1.0E-6.
        method : str
            Method, 'fd' for finite difference or 'cs' for complex step. Default is 'fd'
        step : float
            Step size for approximation. Default is None, which means 1e-6 for 'fd' and 1e-40 for
            'cs'.
        form : string
            Form for finite difference, can be 'forward', 'backward', or 'central'. Default
            None, which defaults to 'forward' for FD.
        step_calc : string
            Step type for finite difference, can be 'abs' for absolute', or 'rel' for relative.
            Default is 'abs'.

        Returns
        -------
        Dict of Dicts of Tuples of Floats

            First key:
                is the (output, input) tuple of strings;
            Second key:
                is one of ['rel error', 'abs error', 'magnitude', 'fdstep'];

            For 'rel error', 'abs error', 'magnitude' the value is: A tuple containing norms for
                forward - fd, adjoint - fd, forward - adjoint.
        """
        if self._setup_status < 2:
            raise RuntimeError("run_model must be called before total derivatives can be checked.")

        model = self.model

        if method == 'cs' and not model._outputs._alloc_complex:
            msg = "\nTo enable complex step, specify 'force_alloc_complex=True' when calling " + \
                  "setup on the problem, e.g. 'problem.setup(force_alloc_complex=True)'"
            raise RuntimeError(msg)

        # TODO: Once we're tracking iteration counts, run the model if it has not been run before.

        # Calculate Total Derivatives
        total_info = _TotalJacInfo(self, of, wrt, False, return_format='flat_dict',
                                   driver_scaling=driver_scaling)
        Jcalc = total_info.compute_totals()

        if step is None:
            if method == 'cs':
                step = ComplexStep.DEFAULT_OPTIONS['step']
            else:
                step = FiniteDifference.DEFAULT_OPTIONS['step']

        # Approximate FD
        fd_args = {
            'step': step,
            'form': form,
            'step_calc': step_calc,
        }
        approx = model._owns_approx_jac
        old_jac = model._jacobian

        model.approx_totals(method=method, step=step, form=form,
                            step_calc=step_calc if method is 'fd' else None)
        total_info = _TotalJacInfo(self, of, wrt, False, return_format='flat_dict', approx=True,
                                   driver_scaling=driver_scaling)
        Jfd = total_info.compute_totals_approx(initialize=True)

        # reset the _owns_approx_jac flag after approximation is complete.
        if not approx:
            model._jacobian = old_jac
            model._owns_approx_jac = False

        # Assemble and Return all metrics.
        data = {}
        data[''] = {}
        for key, val in iteritems(Jcalc):
            data[''][key] = {}
            data[''][key]['J_fwd'] = val
            data[''][key]['J_fd'] = Jfd[key]
        fd_args['method'] = 'fd'

        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        _assemble_derivative_data(data, rel_err_tol, abs_err_tol, out_stream, compact_print,
                                  [model], {'': fd_args}, totals=True)
        return data['']

    def compute_totals(self, of=None, wrt=None, return_format='flat_dict', debug_print=False,
                       driver_scaling=False):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        Parameters
        ----------
        of : list of variable name strings or None
            Variables whose derivatives will be computed. Default is None, which
            uses the driver's objectives and constraints.
        wrt : list of variable name strings or None
            Variables with respect to which the derivatives will be computed.
            Default is None, which uses the driver's desvars.
        return_format : string
            Format to return the derivatives. Can be either 'dict' or 'flat_dict'.
            Default is a 'flat_dict', which returns them in a dictionary whose keys are
            tuples of form (of, wrt).
        debug_print : bool
            Set to True to print out some debug information during linear solve.
        driver_scaling : bool
            When True, return derivatives that are scaled according to either the adder and scaler
            or the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is False, which is unscaled.

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        if self._setup_status < 2:
            self.final_setup()

        if self.model._owns_approx_jac:
            total_info = _TotalJacInfo(self, of, wrt, False, return_format,
                                       approx=True, driver_scaling=driver_scaling)
            return total_info.compute_totals_approx(initialize=True)
        else:
            total_info = _TotalJacInfo(self, of, wrt, False, return_format,
                                       debug_print=debug_print, driver_scaling=driver_scaling)
            return total_info.compute_totals()

    def set_solver_print(self, level=2, depth=1e99, type_='all'):
        """
        Control printing for solvers and subsolvers in the model.

        Parameters
        ----------
        level : int
            iprint level. Set to 2 to print residuals each iteration; set to 1
            to print just the iteration totals; set to 0 to disable all printing
            except for failures, and set to -1 to disable all printing including failures.
        depth : int
            How deep to recurse. For example, you can set this to 0 if you only want
            to print the top level linear and nonlinear solver messages. Default
            prints everything.
        type_ : str
            Type of solver to set: 'LN' for linear, 'NL' for nonlinear, or 'all' for all.
        """
        if (level, depth, type_) not in self._solver_print_cache:
            self._solver_print_cache.append((level, depth, type_))

        self.model._set_solver_print(level=level, depth=depth, type_=type_)

    def list_problem_vars(self,
                          show_promoted_name=True,
                          print_arrays=False,
                          desvar_opts=[],
                          cons_opts=[],
                          objs_opts=[],
                          ):
        """
        Print all design variables and responses (objectives and constraints).

        Parameters
        ----------
        show_promoted_name : bool
            If True, then show the promoted names of the variables.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format is affected
            by the values set with numpy.set_printoptions
            Default is False.
        desvar_opts : list of str
            List of optional columns to be displayed in the desvars table.
            Allowed values are:
            ['lower', 'upper', 'ref', 'ref0', 'indices', 'adder', 'scaler', 'parallel_deriv_color',
            'vectorize_derivs', 'cache_linear_solution']
        cons_opts : list of str
            List of optional columns to be displayed in the cons table.
            Allowed values are:
            ['lower', 'upper', 'equals', 'ref', 'ref0', 'indices', 'index', 'adder', 'scaler',
            'linear', 'parallel_deriv_color', 'vectorize_derivs',
            'cache_linear_solution']
        objs_opts : list of str
            List of optional columns to be displayed in the objs table.
            Allowed values are:
            ['ref', 'ref0', 'indices', 'adder', 'scaler',
            'parallel_deriv_color', 'vectorize_derivs', 'cache_linear_solution']

        """
        default_col_names = ['name', 'value', 'size']

        # Design vars
        desvars = self.model.get_design_vars()
        header = "Design Variables"
        col_names = default_col_names + desvar_opts
        self._write_var_info_table(header, col_names, desvars,
                                   show_promoted_name=show_promoted_name,
                                   print_arrays=print_arrays,
                                   col_spacing=2)

        # Constraints
        cons = self.model.get_constraints()
        header = "Constraints"
        col_names = default_col_names + cons_opts
        self._write_var_info_table(header, col_names, cons, show_promoted_name=show_promoted_name,
                                   print_arrays=print_arrays,
                                   col_spacing=2)

        objs = self.model.get_objectives()
        header = "Objectives"
        col_names = default_col_names + objs_opts
        self._write_var_info_table(header, col_names, objs, show_promoted_name=show_promoted_name,
                                   print_arrays=print_arrays,
                                   col_spacing=2)

    def _write_var_info_table(self, header, col_names, vars, print_arrays=False,
                              show_promoted_name=True, col_spacing=1):
        """
        Write a table of information for the data in vars.

        Parameters
        ----------
        header : str
            The header line for the table.
        col_names : list of str
            List of column labels.
        vars : OrderedDict
            Keys are variable names and values are metadata for the variables.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format is affected
            by the values set with numpy.set_printoptions
            Default is False.
        show_promoted_name : bool
            If True, then show the promoted names of the variables.
        col_spacing : int
            Number of spaces between columns in the table.
        """
        abs2prom = self.model._var_abs2prom

        # Get the values for all the elements in the tables
        rows = []
        for name, meta in iteritems(vars):
            row = {}
            for col_name in col_names:
                if col_name == 'name':
                    if show_promoted_name:
                        row[col_name] = name
                    else:
                        if name in abs2prom['input']:
                            row[col_name] = abs2prom['input'][name]
                        else:
                            row[col_name] = abs2prom['output'][name]
                elif col_name == 'value':
                    row[col_name] = self[name]
                else:
                    row[col_name] = meta[col_name]
            rows.append(row)

        col_space = ' ' * col_spacing
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        # loop through the rows finding the max widths
        max_width = {}
        for col_name in col_names:
            max_width[col_name] = len(col_name)
        for row in rows:
            for col_name in col_names:
                cell = row[col_name]
                if isinstance(cell, np.ndarray) and cell.size > 1:
                    out = '|{}|'.format(str(np.linalg.norm(cell)))
                else:
                    out = str(cell)
                max_width[col_name] = max(len(out), max_width[col_name])

        # print col headers
        header_div = ''
        header_col_names = ''
        for col_name in col_names:
            header_div += '-' * max_width[col_name] + col_space
            header_col_names += pad_name(col_name, max_width[col_name], quotes=False) + col_space
        print(header_col_names)
        print(header_div[:-1])

        # print rows with var info
        for row in rows:
            have_array_values = []  # keep track of which values are arrays
            row_string = ''
            for col_name in col_names:
                cell = row[col_name]
                if isinstance(cell, np.ndarray) and cell.size > 1:
                    out = '|{}|'.format(str(np.linalg.norm(cell)))
                    have_array_values.append(col_name)
                else:
                    out = str(cell)
                row_string += pad_name(out, max_width[col_name], quotes=False) + col_space
            print(row_string)

            if print_arrays:
                left_column_width = max_width['name']
                for col_name in have_array_values:
                    print("{}{}:".format((left_column_width + col_spacing) * ' ', col_name))
                    cell = row[col_name]
                    out_str = str(cell)
                    indented_lines = [(left_column_width + col_spacing) * ' ' +
                                      s for s in out_str.splitlines()]
                    print('\n'.join(indented_lines) + '\n')

        print()

    def load_case(self, case):
        """
        Pull all input and output variables from a case into the model.

        Parameters
        ----------
        case : Case object
            A Case from a CaseRecorder file.
        """
        inputs = case.inputs if case.inputs is not None else None
        if inputs:
            for name in inputs.absolute_names():
                if name not in self.model._var_abs_names['input']:
                    raise KeyError("Input variable, '{}', recorded in the case is not "
                                   "found in the model".format(name))
                self[name] = inputs[name]

        outputs = case.outputs if case.outputs is not None else None
        if outputs:
            for name in outputs.absolute_names():
                if name not in self.model._var_abs_names['output']:
                    raise KeyError("Output variable, '{}', recorded in the case is not "
                                   "found in the model".format(name))
                self[name] = outputs[name]

        return

    def check_config(self, logger=None, checks=None):
        """
        Perform optional error checks on a Problem.

        Parameters
        ----------
        logger : object
            Logging object.
        checks : list of str or None
            List of specific checks to be performed.
        """
        logger = logger if logger else get_logger('check_config', use_format=True)

        if checks is None:
            checks = sorted(_default_checks)

        for c in checks:
            if c not in _all_checks:
                print("WARNING: '%s' is not a recognized check." % c)
                continue
            _all_checks[c](self, logger)


def _assemble_derivative_data(derivative_data, rel_error_tol, abs_error_tol, out_stream,
                              compact_print, system_list, global_options, totals=False,
                              indep_key=None, all_comps_provide_jacs=False,
                              show_only_incorrect=False):
    """
    Compute the relative and absolute errors in the given derivatives and print to the out_stream.

    Parameters
    ----------
    derivative_data : dict
        Dictionary containing derivative information keyed by system name.
    rel_error_tol : float
        Relative error tolerance.
    abs_error_tol : float
        Absolute error tolerance.
    out_stream : file-like object
            Where to send human readable output.
            Set to None to suppress.
    compact_print : bool
        If results should be printed verbosely or in a table.
    system_list : Iterable
        The systems (in the proper order) that were checked.0
    global_options : dict
        Dictionary containing the options for the approximation.
    totals : bool
        Set to True if we are doing check_totals to skip a bunch of stuff.
    indep_key : dict of sets, optional
        Keyed by component name, contains the of/wrt keys that are declared not dependent.
    all_comps_provide_jacs : bool, optional
        Set to True if all components provide a Jacobian (are not matrix-free).
    show_only_incorrect : bool, optional
        Set to True if output should print only the subjacs found to be incorrect.
    """
    nan = float('nan')
    suppress_output = out_stream is None

    if compact_print:
        if totals:
            deriv_line = "{0} wrt {1} | {2:.4e} | {3:.4e} | {4:.4e} | {5:.4e}"
        else:
            if not all_comps_provide_jacs:
                deriv_line = "{0} wrt {1} | {2:.4e} | {3} | {4:.4e} | {5:.4e} | {6} | {7}" \
                             " | {8:.4e} | {9} | {10}"
            else:
                deriv_line = "{0} wrt {1} | {2:.4e} | {3:.4e} | {4:.4e} | {5:.4e}"

    # Keep track of the worst subjac in terms of relative error for fwd and rev
    if not suppress_output and compact_print and not totals:
        worst_subjac_rel_err = 0.0
        worst_subjac = None

    if not suppress_output and not totals and show_only_incorrect:
        out_stream.write('\n** Only writing information about components with '
                         'incorrect Jacobians **\n\n')

    for system in system_list:

        sys_name = system.pathname
        sys_class_name = type(system).__name__

        # Match header to appropriate type.
        if isinstance(system, Component):
            sys_type = 'Component'
        elif isinstance(system, Group):
            sys_type = 'Group'
        else:
            sys_type = type(system).__name__

        if sys_name not in derivative_data:
            msg = "No derivative data found for %s '%s'." % (sys_type, sys_name)
            simple_warning(msg)
            continue

        derivatives = derivative_data[sys_name]

        if totals:
            sys_name = 'Full Model'

        # Sorted keys ensures deterministic ordering
        sorted_keys = sorted(iterkeys(derivatives))

        if not suppress_output:
            # Need to capture the output of a component's derivative
            # info so that it can be used if that component is the
            # worst subjac. That info is printed at the bottom of all the output
            out_buffer = cStringIO()
            num_bad_jacs = 0  # Keep track of number of bad derivative values for each component
            if out_stream:
                header_str = '-' * (len(sys_name) + len(sys_type) + len(sys_class_name) + 5) + '\n'
                out_buffer.write(header_str)
                out_buffer.write("{}: {} '{}'".format(sys_type, sys_class_name, sys_name) + '\n')
                out_buffer.write(header_str)

            if compact_print:
                # Error Header
                if totals:
                    header = "{0} wrt {1} | {2} | {3} | {4} | {5}"\
                        .format(
                            pad_name('<output>', 30, quotes=True),
                            pad_name('<variable>', 30, quotes=True),
                            pad_name('calc mag.'),
                            pad_name('check mag.'),
                            pad_name('a(cal-chk)'),
                            pad_name('r(cal-chk)'),
                        )
                else:
                    max_width_of = len("'<output>'")
                    max_width_wrt = len("'<variable>'")
                    for of, wrt in sorted_keys:
                        max_width_of = max(max_width_of, len(of) + 2)  # 2 to include quotes
                        max_width_wrt = max(max_width_wrt, len(wrt) + 2)

                    if not all_comps_provide_jacs:
                        header = \
                            "{0} wrt {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10}" \
                            .format(
                                pad_name('<output>', max_width_of, quotes=True),
                                pad_name('<variable>', max_width_wrt, quotes=True),
                                pad_name('fwd mag.'),
                                pad_name('rev mag.'),
                                pad_name('check mag.'),
                                pad_name('a(fwd-chk)'),
                                pad_name('a(rev-chk)'),
                                pad_name('a(fwd-rev)'),
                                pad_name('r(fwd-chk)'),
                                pad_name('r(rev-chk)'),
                                pad_name('r(fwd-rev)')
                            )
                    else:
                        header = "{0} wrt {1} | {2} | {3} | {4} | {5}"\
                            .format(
                                pad_name('<output>', max_width_of, quotes=True),
                                pad_name('<variable>', max_width_wrt, quotes=True),
                                pad_name('fwd mag.'),
                                pad_name('check mag.'),
                                pad_name('a(fwd-chk)'),
                                pad_name('r(fwd-chk)'),
                            )
                if out_stream:
                    out_buffer.write(header + '\n')
                    out_buffer.write('-' * len(header) + '\n' + '\n')

        for of, wrt in sorted_keys:

            if totals:
                fd_opts = global_options['']
            else:
                fd_opts = global_options[sys_name][wrt]

            derivative_info = derivatives[of, wrt]
            forward = derivative_info['J_fwd']
            if not totals:
                reverse = derivative_info.get('J_rev')
            fd = derivative_info['J_fd']

            fwd_error = np.linalg.norm(forward - fd)
            if totals:
                rev_error = fwd_rev_error = None
            else:
                rev_error = np.linalg.norm(reverse - fd)
                fwd_rev_error = np.linalg.norm(forward - reverse)

            fwd_norm = np.linalg.norm(forward)
            if totals:
                rev_norm = None
            else:
                rev_norm = np.linalg.norm(reverse)
            fd_norm = np.linalg.norm(fd)

            derivative_info['abs error'] = abs_err = ErrorTuple(fwd_error, rev_error, fwd_rev_error)
            derivative_info['magnitude'] = magnitude = MagnitudeTuple(fwd_norm, rev_norm, fd_norm)

            if fd_norm == 0.:
                if fwd_norm == 0.:
                    derivative_info['rel error'] = rel_err = ErrorTuple(nan, nan, nan)

                else:
                    # If fd_norm is zero, let's use fwd_norm as the divisor for relative
                    # check. That way we don't accidentally squelch a legitimate problem.
                    derivative_info['rel error'] = rel_err = ErrorTuple(fwd_error / fwd_norm,
                                                                        rev_error / fwd_norm,
                                                                        fwd_rev_error / fwd_norm)

            else:
                if totals:
                    derivative_info['rel error'] = rel_err = ErrorTuple(fwd_error / fd_norm,
                                                                        nan,
                                                                        nan)
                else:
                    derivative_info['rel error'] = rel_err = ErrorTuple(fwd_error / fd_norm,
                                                                        rev_error / fd_norm,
                                                                        fwd_rev_error / fd_norm)

            # Skip printing the dependent keys if the derivatives are fine.
            if not compact_print and indep_key is not None:
                rel_key = (of, wrt)
                if rel_key in indep_key[sys_name] and fd_norm < abs_error_tol:
                    del derivative_data[sys_name][rel_key]
                    continue

            if not suppress_output:
                directional = fd_opts.get('directional')

                if compact_print:
                    if totals:
                        if out_stream:
                            out_stream.write(deriv_line.format(
                                pad_name(of, 30, quotes=True),
                                pad_name(wrt, 30, quotes=True),
                                magnitude.forward,
                                magnitude.fd,
                                abs_err.forward,
                                rel_err.forward,
                            ) + '\n')
                    else:
                        error_string = ''
                        for error in abs_err:
                            if not np.isnan(error) and error >= abs_error_tol:
                                error_string += ' >ABS_TOL'
                                break

                        # See if this component has the greater
                        # error in the derivative computation
                        # compared to the other components so far
                        is_worst_subjac = False
                        for i, error in enumerate(rel_err):
                            if not np.isnan(error):
                                #  only 1st and 2d errs
                                if i < 2 and error > worst_subjac_rel_err:
                                    worst_subjac_rel_err = error
                                    worst_subjac = (sys_type, sys_class_name, sys_name)
                                    is_worst_subjac = True
                            if not np.isnan(error) and error >= rel_error_tol:
                                error_string += ' >REL_TOL'
                                break

                        if error_string:  # Any error string indicates that at least one of the
                            #  derivative calcs is greater than the rel tolerance
                            num_bad_jacs += 1

                        if out_stream:
                            if directional:
                                wrt = "(d)'%s'" % wrt
                                wrt_padded = pad_name(wrt, max_width_wrt, quotes=False)
                            else:
                                wrt_padded = pad_name(wrt, max_width_wrt, quotes=True)
                            if not all_comps_provide_jacs:
                                deriv_info_line = \
                                    deriv_line.format(
                                        pad_name(of, max_width_of, quotes=True),
                                        wrt_padded,
                                        magnitude.forward,
                                        _format_if_not_matrix_free(
                                            system.matrix_free, magnitude.reverse),
                                        magnitude.fd,
                                        abs_err.forward,
                                        _format_if_not_matrix_free(system.matrix_free,
                                                                   abs_err.reverse),
                                        _format_if_not_matrix_free(
                                            system.matrix_free, abs_err.forward_reverse),
                                        rel_err.forward,
                                        _format_if_not_matrix_free(system.matrix_free,
                                                                   rel_err.reverse),
                                        _format_if_not_matrix_free(
                                            system.matrix_free, rel_err.forward_reverse),
                                    )
                            else:
                                deriv_info_line = \
                                    deriv_line.format(
                                        pad_name(of, max_width_of, quotes=True),
                                        wrt_padded,
                                        magnitude.forward,
                                        magnitude.fd,
                                        abs_err.forward,
                                        rel_err.forward,
                                    )
                            if not show_only_incorrect or error_string:
                                out_buffer.write(deriv_info_line + error_string + '\n')

                            if is_worst_subjac:
                                worst_subjac_line = deriv_info_line
                else:  # not compact print

                    fd_desc = "{}:{}".format(fd_opts['method'], fd_opts['form'])

                    # Magnitudes
                    if out_stream:
                        if directional:
                            out_buffer.write("  {}: '{}' wrt (d)'{}'\n".format(sys_name, of, wrt))
                        else:
                            out_buffer.write("  {}: '{}' wrt '{}'\n".format(sys_name, of, wrt))
                        out_buffer.write('    Forward Magnitude : {:.6e}\n'.format(
                            magnitude.forward))
                    if not totals and system.matrix_free:
                        txt = '    Reverse Magnitude : {:.6e}'
                        if out_stream:
                            out_buffer.write(txt.format(magnitude.reverse) + '\n')
                    if out_stream:
                        out_buffer.write('         Fd Magnitude : {:.6e} ({})\n'.format(
                            magnitude.fd, fd_desc))
                    # Absolute Errors
                    if totals or not system.matrix_free:
                        error_descs = ('(Jfor  - Jfd) ', )
                    else:
                        error_descs = ('(Jfor  - Jfd) ', '(Jrev  - Jfd) ', '(Jfor  - Jrev)')
                    for error, desc in zip(abs_err, error_descs):
                        error_str = _format_error(error, abs_error_tol)
                        if error_str.endswith('*'):
                            num_bad_jacs += 1
                        if out_stream:
                            out_buffer.write('    Absolute Error {}: {}\n'.format(desc, error_str))
                    if out_stream:
                        out_buffer.write('\n')

                    # Relative Errors
                    for error, desc in zip(rel_err, error_descs):
                        error_str = _format_error(error, rel_error_tol)
                        if error_str.endswith('*'):
                            num_bad_jacs += 1
                        if out_stream:
                            out_buffer.write('    Relative Error {}: {}\n'.format(desc, error_str))

                    if out_stream:
                        if MPI and MPI.COMM_WORLD.size > 1:
                            out_buffer.write('    MPI Rank {}\n'.format(MPI.COMM_WORLD.rank))
                        out_buffer.write('\n')

                    # Raw Derivatives
                    if out_stream:
                        if directional:
                            out_buffer.write('    Directional Forward Derivative (Jfor)\n')
                        else:
                            out_buffer.write('    Raw Forward Derivative (Jfor)\n')
                        out_buffer.write(str(forward) + '\n')
                        out_buffer.write('\n')

                    if not totals and system.matrix_free:
                        if out_stream:
                            if directional:
                                out_buffer.write('    Directional Reverse Derivative (Jrev)\n')
                            else:
                                out_buffer.write('    Raw Reverse Derivative (Jrev)\n')
                            out_buffer.write(str(reverse) + '\n')
                            out_buffer.write('\n')

                    if out_stream:
                        if directional:
                            out_buffer.write('    Directional FD Derivative (Jfd)\n')
                        else:
                            out_buffer.write('    Raw FD Derivative (Jfd)\n')
                        out_buffer.write(str(fd) + '\n')
                        out_buffer.write('\n')

                    if out_stream:
                        out_buffer.write(' -' * 30 + '\n')

                # End of if compact print if/else
            # End of if not suppress_output
        # End of for of, wrt in sorted_keys

        if not show_only_incorrect or num_bad_jacs:
            if out_stream and not suppress_output:
                out_stream.write(out_buffer.getvalue())

    # End of for system in system_list

    if not suppress_output and compact_print and not totals:
        if worst_subjac:
            worst_subjac_header = \
                "Sub Jacobian with Largest Relative Error: {1} '{2}'".format(*worst_subjac)
            out_stream.write('\n' + '#' * len(worst_subjac_header) + '\n')
            out_stream.write("{}\n".format(worst_subjac_header))
            out_stream.write('#' * len(worst_subjac_header) + '\n')
            out_stream.write(header + '\n')
            out_stream.write('-' * len(header) + '\n')
            out_stream.write(worst_subjac_line + '\n')


def _format_if_not_matrix_free(matrix_free, val):
    """
    Return string to represent deriv check value in compact display.

    Parameters
    ----------
    matrix_free : bool
        If True, then the associated Component is matrix-free.
    val : float
        The deriv check value.

    Returns
    -------
    str
        String which is the actual value if matrix-free, otherwise 'n/a'
    """
    if matrix_free:
        return '{0:.4e}'.format(val)
    else:
        return pad_name('n/a')


def _format_error(error, tol):
    """
    Format the error, flagging if necessary.

    Parameters
    ----------
    error : float
        The absolute or relative error.
    tol : float
        Tolerance above which errors are flagged

    Returns
    -------
    str
        Formatted and possibly flagged error.
    """
    if np.isnan(error) or error < tol:
        return '{:.6e}'.format(error)
    return '{:.6e} *'.format(error)
