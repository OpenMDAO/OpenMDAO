"""Define the Problem class and a FakeComm class for non-MPI users."""

import sys
import pprint
import os
import logging
import weakref
import time

from collections import defaultdict, namedtuple, OrderedDict
from fnmatch import fnmatchcase
from itertools import product

from io import StringIO

import numpy as np
import scipy.sparse as sparse

from openmdao.core.component import Component
from openmdao.core.driver import Driver, record_iteration
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group, System
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.total_jac import _TotalJacInfo
from openmdao.core.constants import _DEFAULT_OUT_STREAM, _UNDEFINED, INT_DTYPE
from openmdao.approximation_schemes.complex_step import ComplexStep
from openmdao.approximation_schemes.finite_difference import FiniteDifference
from openmdao.solvers.solver import SolverInfo
from openmdao.error_checking.check_config import _default_checks, _all_checks
from openmdao.recorders.recording_iteration_stack import _RecIteration
from openmdao.recorders.recording_manager import RecordingManager, record_viewer_data, \
    record_system_options
from openmdao.utils.record_util import create_local_meta
from openmdao.utils.general_utils import ContainsAll, pad_name, simple_warning, warn_deprecation, \
    _is_slicer_op, _slice_indices
from openmdao.utils.mpi import FakeComm
from openmdao.utils.mpi import MPI
from openmdao.utils.name_maps import prom_name2abs_name, name2abs_names
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.units import convert_units
from openmdao.utils import coloring as coloring_mod
from openmdao.core.constants import _SetupStatus
from openmdao.utils.name_maps import abs_key2rel_key
from openmdao.vectors.vector import _full_slice
from openmdao.vectors.default_vector import DefaultVector
from openmdao.utils.logger_utils import get_logger, TestLogger
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.hooks import _setup_hooks

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.utils.name_maps import rel_key2abs_key, rel_name2abs_name


ErrorTuple = namedtuple('ErrorTuple', ['forward', 'reverse', 'forward_reverse'])
MagnitudeTuple = namedtuple('MagnitudeTuple', ['forward', 'reverse', 'fd'])

_contains_all = ContainsAll()


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
    _initial_condition_cache : dict
        Any initial conditions that are set at the problem level via setitem are cached here
        until they can be processed.
    cite : str
        Listing of relevant citations that should be referenced when
        publishing work that uses this class.
    options : <OptionsDictionary>
        Dictionary with general options for the problem.
    recording_options : <OptionsDictionary>
        Dictionary with problem recording options.
    _rec_mgr : <RecordingManager>
        Object that manages all recorders added to this problem.
    _check : bool
        If True, call check_config at the end of final_setup.
    _filtered_vars_to_record : dict
        Dictionary of lists of design vars, constraints, etc. to record.
    _logger : object or None
        Object for logging config checks if _check is True.
    _name : str
        Problem name.
    _system_options_recorded : bool
        A flag to indicate whether the system options for all the systems have been recorded
    _metadata : dict
        Problem level metadata.
    _run_counter : int
        The number of times run_driver or run_model has been called.
    """

    def __init__(self, model=None, driver=None, comm=None, name=None, **options):
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
        name : str
            Problem name. Can be used to specify a Problem instance when multiple Problems
            exist.
        **options : named args
            All remaining named args are converted to options.
        """
        self.cite = CITATION
        self._name = name

        if comm is None:
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            except ImportError:
                comm = FakeComm()

        if model is None:
            self.model = Group()
        elif isinstance(model, System):
            self.model = model
        else:
            raise TypeError(self.msginfo +
                            ": The value provided for 'model' is not a valid System.")

        if driver is None:
            self.driver = Driver()
        elif isinstance(driver, Driver):
            self.driver = driver
        else:
            raise TypeError(self.msginfo +
                            ": The value provided for 'driver' is not a valid Driver.")

        self.comm = comm

        self._mode = None  # mode is assigned in setup()

        self._initial_condition_cache = {}

        self._metadata = None
        self._run_counter = -1
        self._system_options_recorded = False
        self._rec_mgr = RecordingManager()

        # General options
        self.options = OptionsDictionary(parent_name=type(self).__name__)
        self.options.declare('coloring_dir', types=str,
                             default=os.path.join(os.getcwd(), 'coloring_files'),
                             desc='Directory containing coloring files (if any) for this Problem.')
        self.options.update(options)

        # Case recording options
        self.recording_options = OptionsDictionary(parent_name=type(self).__name__)

        self.recording_options.declare('record_desvars', types=bool, default=True,
                                       desc='Set to True to record design variables at the '
                                            'problem level')
        self.recording_options.declare('record_objectives', types=bool, default=True,
                                       desc='Set to True to record objectives at the problem level')
        self.recording_options.declare('record_constraints', types=bool, default=True,
                                       desc='Set to True to record constraints at the '
                                            'problem level')
        self.recording_options.declare('record_responses', types=bool, default=False,
                                       desc='Set True to record constraints and objectives at the '
                                            'problem level.')
        self.recording_options.declare('record_inputs', types=bool, default=False,
                                       desc='Set True to record inputs at the '
                                            'problem level.')
        self.recording_options.declare('record_outputs', types=bool, default=True,
                                       desc='Set True to record outputs at the '
                                            'problem level.')
        self.recording_options.declare('record_residuals', types=bool, default=False,
                                       desc='Set True to record residuals at the '
                                            'problem level.')
        self.recording_options.declare('record_derivatives', types=bool, default=False,
                                       desc='Set to True to record derivatives for the problem '
                                            'level')
        self.recording_options.declare('record_abs_error', types=bool, default=True,
                                       desc='Set to True to record absolute error of '
                                            'model nonlinear solver')
        self.recording_options.declare('record_rel_error', types=bool, default=True,
                                       desc='Set to True to record relative error of model \
                                       nonlinear solver')
        self.recording_options.declare('includes', types=list, default=['*'],
                                       desc='Patterns for variables to include in recording. \
                                       Uses fnmatch wildcards')
        self.recording_options.declare('excludes', types=list, default=[],
                                       desc='Patterns for vars to exclude in recording '
                                            '(processed post-includes). Uses fnmatch wildcards')

        _setup_hooks(self)

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
                raise KeyError("{}: Using promoted name `{}' is ambiguous and matches unconnected "
                               "inputs %s. Use absolute name to disambiguate.".format(self.msginfo,
                                                                                      name,
                                                                                      abs_names))

        raise KeyError('{}: Variable "{}" not found.'.format(self.msginfo, name))

    @property
    def msginfo(self):
        """
        Return info to prepend to messages.

        Returns
        -------
        str
            Info to prepend to messages.
        """
        if self._name is None:
            return type(self).__name__
        return '{} {}'.format(type(self).__name__, self._name)

    def _get_inst_id(self):
        return self._name

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
        if self._metadata is None:
            raise RuntimeError("{}: is_local('{}') was called before setup() "
                               "completed.".format(self.msginfo, name))

        try:
            abs_name = self._get_var_abs_name(name)
        except KeyError:
            sub = self.model._get_subsystem(name)
            return sub is not None and sub._is_local

        # variable exists, but may be remote
        return abs_name in self.model._var_abs2meta['input'] or \
            abs_name in self.model._var_abs2meta['output']

    def _get_cached_val(self, name, get_remote=False):
        # We have set and cached already
        if name in self._initial_condition_cache:
            return self._initial_condition_cache[name]

        # Vector not setup, so we need to pull values from saved metadata request.
        else:
            proms = self.model._var_allprocs_prom2abs_list
            meta = self.model._var_abs2meta
            try:
                conns = self.model._conn_abs_in2out
            except AttributeError:
                conns = {}

            abs_names = name2abs_names(self.model, name)
            if not abs_names:
                raise KeyError('{}: Variable "{}" not found.'.format(self.model.msginfo, name))

            abs_name = abs_names[0]
            vars_to_gather = self._metadata['vars_to_gather']

            io = 'output' if abs_name in meta['output'] else 'input'
            if abs_name in meta[io]:
                if abs_name in conns:
                    val = meta['output'][conns[abs_name]]['value']
                else:
                    val = meta[io][abs_name]['value']

            if get_remote and abs_name in vars_to_gather:
                owner = vars_to_gather[abs_name]
                if self.model.comm.rank == owner:
                    self.model.comm.bcast(val, root=owner)
                else:
                    val = self.model.comm.bcast(None, root=owner)

            if val is not _UNDEFINED:
                # Need to cache the "get" in case the user calls in-place numpy operations.
                self._initial_condition_cache[name] = val

            return val

    @property
    def _recording_iter(self):
        return self._metadata['recording_iter']

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
        return self.get_val(name, get_remote=None)

    def get_val(self, name, units=None, indices=None, get_remote=False):
        """
        Get an output/input variable.

        Function is used if you want to specify display units.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.
        units : str, optional
            Units to convert to before return.
        indices : int or list of ints or tuple of ints or int ndarray or Iterable or None, optional
            Indices or slice to return.
        get_remote : bool or None
            If True, retrieve the value even if it is on a remote process.  Note that if the
            variable is remote on ANY process, this function must be called on EVERY process
            in the Problem's MPI communicator.
            If False, only retrieve the value if it is on the current process, or only the part
            of the value that's on the current process for a distributed variable.
            If None and the variable is remote or distributed, a RuntimeError will be raised.

        Returns
        -------
        object
            The value of the requested output/input variable.
        """
        if self._metadata['setup_status'] == _SetupStatus.POST_SETUP:
            val = self._get_cached_val(name, get_remote=get_remote)
            if val is not _UNDEFINED:
                if indices is not None:
                    val = val[indices]
                if units is not None:
                    val = self.model.convert2units(name, val, units)
        else:
            val = self.model.get_val(name, units=units, indices=indices, get_remote=get_remote,
                                     from_src=True)

        if val is _UNDEFINED:
            if get_remote:
                raise KeyError('{}: Variable name "{}" not found.'.format(self.msginfo, name))
            else:
                raise RuntimeError(f"{self.model.msginfo}: Variable '{name}' is not local to "
                                   f"rank {self.comm.rank}. You can retrieve values from "
                                   "other processes using `get_val(<name>, get_remote=True)`.")

        return val

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
        self.set_val(name, value)

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
        model = self.model
        if self._metadata is not None:
            conns = model._conn_global_abs_in2out
        else:
            raise RuntimeError(f"{self.msginfo}: '{name}' Cannot call set_val before setup.")

        all_meta = model._var_allprocs_abs2meta
        loc_meta = model._var_abs2meta
        n_proms = 0  # if nonzero, name given was promoted input name w/o a matching prom output

        try:
            ginputs = model._group_inputs
        except AttributeError:
            ginputs = {}  # could happen if top level system is not a Group

        abs_names = name2abs_names(model, name)
        if abs_names:
            n_proms = len(abs_names)  # for output this will never be > 1
            if n_proms > 1 and name in ginputs:
                abs_name = ginputs[name][0].get('use_tgt', abs_names[0])
            else:
                abs_name = abs_names[0]
        else:
            raise KeyError(f'{model.msginfo}: Variable "{name}" not found.')

        if abs_name in conns:
            src = conns[abs_name]
            if abs_name not in model._var_allprocs_discrete['input']:
                value = np.asarray(value)
                tmeta = all_meta['input'][abs_name]
                tunits = tmeta['units']
                sunits = all_meta['output'][src]['units']
                if abs_name in loc_meta['input']:
                    tlocmeta = loc_meta['input'][abs_name]
                else:
                    tlocmeta = None

                gunits = ginputs[name][0].get('units') if name in ginputs else None
                if n_proms > 1:  # promoted input name was used
                    if gunits is None:
                        tunit_list = [all_meta['input'][n]['units'] for n in abs_names]
                        tu0 = tunit_list[0]
                        for tu in tunit_list:
                            if tu != tu0:
                                model._show_ambiguity_msg(name, ('units',), abs_names)

                if units is None:
                    # avoids double unit conversion
                    if self._metadata['setup_status'] > _SetupStatus.POST_SETUP:
                        ivalue = value
                        if sunits is not None:
                            if gunits is not None and gunits != tunits:
                                value = model.convert_from_units(src, value, gunits)
                            else:
                                value = model.convert_from_units(src, value, tunits)
                else:
                    if gunits is None:
                        ivalue = model.convert_from_units(abs_name, value, units)
                    else:
                        ivalue = model.convert_units(name, value, units, gunits)
                    if self._metadata['setup_status'] == _SetupStatus.POST_SETUP:
                        value = ivalue
                    else:
                        value = model.convert_from_units(src, value, units)
        else:
            src = abs_name
            if units is not None:
                value = model.convert_from_units(abs_name, value, units)

        # Caching only needed if vectors aren't allocated yet.
        if self._metadata['setup_status'] == _SetupStatus.POST_SETUP:
            if indices is not None:
                self._get_cached_val(name)
                try:
                    if _is_slicer_op(indices):
                        self._initial_condition_cache[name] = value[indices]
                    else:
                        self._initial_condition_cache[name][indices] = value
                except IndexError:
                    self._initial_condition_cache[name][indices] = value
                except Exception as err:
                    raise RuntimeError(f"Failed to set value of '{name}': {str(err)}.")
            else:
                self._initial_condition_cache[name] = value
        else:
            myrank = model.comm.rank

            if indices is None:
                indices = _full_slice

            if model._outputs._contains_abs(abs_name):
                model._outputs.set_var(abs_name, value, indices)
            elif abs_name in conns:  # input name given. Set value into output
                if model._outputs._contains_abs(src):  # src is local
                    if (model._outputs._abs_get_val(src).size == 0 and
                            src.rsplit('.', 1)[0] == '_auto_ivc' and
                            all_meta['output'][src]['distributed']):
                        pass  # special case, auto_ivc dist var with 0 local size
                    elif tmeta['has_src_indices']:
                        if tlocmeta:  # target is local
                            src_indices = tlocmeta['src_indices']
                            flat = False
                            if name in model._var_prom2inds:
                                sshape, inds, flat = model._var_prom2inds[name]
                                if inds is not None:
                                    if _is_slicer_op(inds):
                                        inds = _slice_indices(inds, np.prod(sshape), sshape).ravel()
                                        value = value.ravel()
                                        flat = True
                                src_indices = inds
                            if src_indices is None:
                                model._outputs.set_var(src, value, None, flat)
                            else:
                                if tmeta['distributed']:
                                    ssizes = model._var_sizes['nonlinear']['output']
                                    sidx = model._var_allprocs_abs2idx['nonlinear'][src]
                                    ssize = ssizes[myrank, sidx]
                                    start = np.sum(ssizes[:myrank, sidx])
                                    end = start + ssize
                                    if np.any(src_indices < start) or np.any(src_indices >= end):
                                        raise RuntimeError(f"{model.msginfo}: Can't set {name}: "
                                                           "src_indices refer "
                                                           "to out-of-process array entries.")
                                    if start > 0:
                                        src_indices = src_indices - start
                                model._outputs.set_var(src, value, src_indices[indices], flat)
                        else:
                            raise RuntimeError(f"{model.msginfo}: Can't set {abs_name}: remote"
                                               " connected inputs with src_indices currently not"
                                               " supported.")
                    else:
                        value = np.asarray(value)
                        model._outputs.set_var(src, value, indices)
                elif src in model._discrete_outputs:
                    model._discrete_outputs[src] = value
                # also set the input
                # TODO: maybe remove this if inputs are removed from case recording
                if n_proms < 2:
                    if model._inputs._contains_abs(abs_name):
                        model._inputs.set_var(abs_name, ivalue, indices)
                    elif abs_name in model._discrete_inputs:
                        model._discrete_inputs[abs_name] = value
                    else:
                        # must be a remote var. so, just do nothing on this proc. We can't get here
                        # unless abs_name is found in connections, so the variable must exist.
                        if abs_name in model._var_allprocs_abs2meta:
                            print(f"Variable '{name}' is remote on rank {self.comm.rank}.  "
                                  "Local assignment ignored.")
            elif abs_name in model._discrete_outputs:
                model._discrete_outputs[abs_name] = value
            elif model._inputs._contains_abs(abs_name):   # could happen if model is a component
                model._inputs.set_var(abs_name, value, indices)
            elif abs_name in model._discrete_inputs:   # could happen if model is a component
                model._discrete_inputs[abs_name] = value

    def _set_initial_conditions(self):
        """
        Set all initial conditions that have been saved in cache after setup.
        """
        for name, value in self._initial_condition_cache.items():
            self.set_val(name, value)

        # Clean up cache
        self._initial_condition_cache = OrderedDict()

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
            raise RuntimeError(self.msginfo +
                               ": The `setup` method must be called before `run_model`.")

        if case_prefix:
            if not isinstance(case_prefix, str):
                raise TypeError(self.msginfo + ": The 'case_prefix' argument should be a string.")
            self._recording_iter.prefix = case_prefix
        else:
            self._recording_iter.prefix = None

        if self.model.iter_count > 0 and reset_iter_counts:
            self.driver.iter_count = 0
            self.model._reset_iter_counts()

        self._run_counter += 1

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
            raise RuntimeError(self.msginfo +
                               ": The `setup` method must be called before `run_driver`.")

        if case_prefix:
            if not isinstance(case_prefix, str):
                raise TypeError(self.msginfo + ": The 'case_prefix' argument should be a string.")
            self._recording_iter.prefix = case_prefix
        else:
            self._recording_iter.prefix = None

        if self.model.iter_count > 0 and reset_iter_counts:
            self.driver.iter_count = 0
            self.model._reset_iter_counts()

        self._run_counter += 1

        self.final_setup()
        self.model._clear_iprint()
        return self.driver.run()

    def compute_jacvec_product(self, of, wrt, mode, seed):
        """
        Given a seed and 'of' and 'wrt' variables, compute the total jacobian vector product.

        Parameters
        ----------
        of : list of str
            Variables whose derivatives will be computed.
        wrt : list of str
            Derivatives will be computed with respect to these variables.
        mode : str
            Derivative direction ('fwd' or 'rev').
        seed : dict or list
            Either a dict keyed by 'wrt' varnames (fwd) or 'of' varnames (rev), containing
            dresidual (fwd) or doutput (rev) values, OR a list of dresidual or doutput
            values that matches the corresponding 'wrt' (fwd) or 'of' (rev) varname list.

        Returns
        -------
        dict
            The total jacobian vector product, keyed by variable name.
        """
        if mode == 'fwd':
            if len(wrt) != len(seed):
                raise RuntimeError(self.msginfo +
                                   ": seed and 'wrt' list must be the same length in fwd mode.")
            lnames, rnames = of, wrt
            lkind, rkind = 'output', 'residual'
        else:  # rev
            if len(of) != len(seed):
                raise RuntimeError(self.msginfo +
                                   ": seed and 'of' list must be the same length in rev mode.")
            lnames, rnames = wrt, of
            lkind, rkind = 'residual', 'output'

        rvec = self.model._vectors[rkind]['linear']
        lvec = self.model._vectors[lkind]['linear']

        rvec.set_val(0.)

        conns = self.model._conn_global_abs_in2out

        # set seed values into dresids (fwd) or doutputs (rev)
        # seed may have keys that are inputs and must be converted into auto_ivcs
        try:
            seed[rnames[0]]
        except (IndexError, TypeError):
            for i, name in enumerate(rnames):
                if name in conns:
                    rvec[conns[name]] = seed[i]
                else:
                    rvec[name] = seed[i]
        else:
            for name in rnames:
                if name in conns:
                    rvec[conns[name]] = seed[name]
                else:
                    rvec[name] = seed[name]

        # We apply a -1 here because the derivative of the output is minus the derivative of
        # the residual in openmdao.
        data = rvec.asarray()
        data *= -1.

        self.model.run_solve_linear(['linear'], mode)

        if mode == 'fwd':
            return {n: lvec[n].copy() for n in lnames}
        else:
            # may need to convert some lnames to auto_ivc names
            return {n: lvec[conns[n] if n in conns else n].copy() for n in lnames}

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

    def record(self, case_name):
        """
        Record the variables at the Problem level.

        Must be called after `final_setup` has been called. This can either
        happen automatically through `run_driver` or `run_model`, or it can be
        called manually.

        Parameters
        ----------
        case_name : str
            Name used to identify this Problem case.
        """
        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            raise RuntimeError(f"{self.msginfo}: Problem.record() cannot be called before "
                               "`Problem.run_model()`, `Problem.run_driver()`, or "
                               "`Problem.final_setup()`.")
        else:
            record_iteration(self, self, case_name)

    def record_iteration(self, case_name):
        """
        Record the variables at the Problem level.

        Parameters
        ----------
        case_name : str
            Name used to identify this Problem case.
        """
        warn_deprecation("'Problem.record_iteration' has been deprecated. "
                         "Use 'Problem.record' instead.")

        record_iteration(self, self, case_name)

    def _get_recorder_metadata(self, case_name):
        """
        Return metadata from the latest iteration for use in the recorder.

        Parameters
        ----------
        case_name : str
            Name of current case.

        Returns
        -------
        dict
            Metadata dictionary for the recorder.
        """
        return create_local_meta(case_name)

    def setup(self, check=False, logger=None, mode='auto', force_alloc_complex=False,
              distributed_vector_class=PETScVector, local_vector_class=DefaultVector,
              derivatives=True):
        """
        Set up the model hierarchy.

        When `setup` is called, the model hierarchy is assembled, the processors are allocated
        (for MPI), and variables and connections are all assigned. This method traverses down
        the model hierarchy to call `setup` on each subsystem, and then traverses up the model
        hierarchy to call `configure` on each subsystem.

        Parameters
        ----------
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
        comm = self.comm

        # PETScVector is required for MPI
        if comm.size > 1:
            if PETScVector is None:
                raise ValueError(self.msginfo +
                                 ": Attempting to run in parallel under MPI but PETScVector "
                                 "could not be imported.")
            elif distributed_vector_class is not PETScVector:
                raise ValueError("%s: The `distributed_vector_class` argument must be "
                                 "`PETScVector` when running in parallel under MPI but '%s' was "
                                 "specified." % (self.msginfo, distributed_vector_class.__name__))

        if mode not in ['fwd', 'rev', 'auto']:
            msg = "%s: Unsupported mode: '%s'. Use either 'fwd' or 'rev'." % (self.msginfo, mode)
            raise ValueError(msg)

        self._mode = self._orig_mode = mode

        model_comm = self.driver._setup_comm(comm)

        # this metadata will be shared by all Systems/Solvers in the system tree
        self._metadata = {
            'coloring_dir': self.options['coloring_dir'],  # directory for coloring files
            'recording_iter': _RecIteration(),  # manager of recorder iterations
            'local_vector_class': local_vector_class,
            'distributed_vector_class': distributed_vector_class,
            'solver_info': SolverInfo(),
            'use_derivatives': derivatives,
            'force_alloc_complex': force_alloc_complex,
            'vars_to_gather': {},  # vars that are remote somewhere. does not include distrib vars
            'prom2abs': {'input': {}, 'output': {}},  # includes ALL promotes including buried ones
            'static_mode': False,  # used to determine where various 'static'
                                   # and 'dynamic' data structures are stored.
                                   # Dynamic ones are added during System
                                   # setup/configure. They are wiped out and re-created during
                                   # each Problem setup.  Static ones are added outside of
                                   # Problem setup and they are never wiped out or re-created.
            'config_info': None,  # used during config to determine if additional updates required
            'parallel_groups': [],  # list of pathnames of parallel groups in this model (all procs)
            'setup_status': _SetupStatus.PRE_SETUP,
            'vec_names': None,  # names of all nonlinear and linear vectors
            'lin_vec_names': None,  # names of linear vectors
            'model_ref': weakref.ref(model),  # ref to the model (needed to get out-of-scope
                                              # src data for inputs)
        }
        model._setup(model_comm, mode, self._metadata)

        # set static mode back to True in all systems in this Problem
        self._metadata['static_mode'] = True

        # Cache all args for final setup.
        self._check = check
        self._logger = logger

        self._metadata['setup_status'] = _SetupStatus.POST_SETUP

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

        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            self.model._final_setup(self.comm)

        driver._setup_driver(self)

        info = driver._coloring_info
        coloring = info['coloring']
        if coloring is None and info['static'] is not None:
            coloring = driver._get_static_coloring()

        if coloring and coloring_mod._use_total_sparsity:
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

        if self._metadata['setup_status'] == _SetupStatus.PRE_SETUP and \
                hasattr(self.model, '_order_set') and self.model._order_set:
            raise RuntimeError("%s: Cannot call set_order without calling "
                               "setup after" % (self.msginfo))

        # we only want to set up recording once, after problem setup
        if self._metadata['setup_status'] == _SetupStatus.POST_SETUP:
            driver._setup_recording()
            self._setup_recording()
            record_viewer_data(self)
            record_system_options(self)

        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            self._metadata['setup_status'] = _SetupStatus.POST_FINAL_SETUP
            self._set_initial_conditions()

        if self._check:
            if self._check is True:
                checks = _default_checks
            else:
                checks = self._check
            if self.comm.rank == 0:
                logger = self._logger
            else:
                logger = TestLogger()
            self.check_config(logger, checks=checks)

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
            Set to True to just print the essentials, one line per input-output pair.
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
        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            self.final_setup()

        model = self.model

        if not model._use_derivatives:
            raise RuntimeError(self.msginfo +
                               ": Can't check partials.  Derivative support has been turned off.")

        # TODO: Once we're tracking iteration counts, run the model if it has not been run before.

        includes = [includes] if isinstance(includes, str) else includes
        excludes = [excludes] if isinstance(excludes, str) else excludes

        comps = []
        for comp in model.system_iter(typ=Component, include_self=True):
            if comp._no_check_partials:
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
        input_cache = model._inputs.asarray(copy=True)
        output_cache = model._outputs.asarray(copy=True)

        # Keep track of derivative keys that are declared dependent so that we don't print them
        # unless they are in error.
        indep_key = {}

        # Directional derivative directions for matrix free comps.
        mfree_directions = {}

        # Analytic Jacobians
        print_reverse = False
        for mode in ('fwd', 'rev'):
            model._inputs.set_val(input_cache)
            model._outputs.set_val(output_cache)
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
                if mode == 'fwd':
                    indep_key[c_name] = set()

                with comp._unscaled_context():

                    imp = not explicit
                    of_list, wrt_list = comp._get_potential_partials_lists(include_wrt_outputs=imp)

                    # Matrix-free components need to calculate their Jacobian by matrix-vector
                    # product.
                    if matrix_free:
                        print_reverse = True
                        local_opts = comp._get_check_partial_options(include_wrt_outputs=imp)

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
                            if mode == 'fwd':
                                directional = inp in local_opts and local_opts[inp]['directional']
                            else:
                                directional = c_name in mfree_directions

                            try:
                                flat_view = dinputs._abs_get_val(inp_abs)
                            except KeyError:
                                # Implicit state
                                flat_view = dstate._abs_get_val(inp_abs)

                            if directional:
                                n_in = 1
                                if c_name not in mfree_directions:
                                    mfree_directions[c_name] = {}

                                if inp in mfree_directions[c_name]:
                                    perturb = mfree_directions[c_name][inp]
                                else:
                                    perturb = 2.0 * np.random.random(len(flat_view)) - 1.0
                                    mfree_directions[c_name][inp] = perturb

                            else:
                                n_in = len(flat_view)
                                perturb = 1.0

                            for idx in range(n_in):

                                dinputs.set_val(0.0)
                                dstate.set_val(0.0)

                                # Dictionary access returns a scalar for 1d input, and we
                                # need a vector for clean code, so use _views_flat.
                                if directional:
                                    flat_view[:] = perturb
                                else:
                                    flat_view[idx] = perturb

                                # Matrix Vector Product
                                comp._apply_linear(None, ['linear'], _contains_all, mode)

                                for out in out_list:
                                    out_abs = rel_name2abs_name(comp, out)

                                    try:
                                        derivs = doutputs._abs_get_val(out_abs)
                                    except KeyError:
                                        # Implicit state
                                        derivs = dstate._abs_get_val(out_abs)

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

                                        if directional:
                                            # Dot product test for adjoint validity.
                                            m = mfree_directions[c_name][out]
                                            d = mfree_directions[c_name][inp]
                                            mhat = derivs
                                            dhat = partials_data[c_name][inp, out]['J_fwd'][:, idx]

                                            deriv['directional_fwd_rev'] = mhat.dot(m) - dhat.dot(d)

                                        # Allocate first time
                                        if jac_key not in deriv:
                                            shape = (n_in, len(derivs))
                                            deriv[jac_key] = np.zeros(shape)

                                        deriv[jac_key][idx, :] = derivs

                    # These components already have a Jacobian with calculated derivatives.
                    else:

                        if mode == 'rev':
                            # Skip reverse mode because it is not different than forward.
                            continue

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

                            if wrt in comp._var_abs2meta['input']:
                                wrt_meta = comp._var_abs2meta['input'][wrt]
                            else:
                                wrt_meta = comp._var_abs2meta['output'][wrt]

                            if deriv_value is None:
                                # Missing derivatives are assumed 0.
                                in_size = wrt_meta['size']
                                out_size = comp._var_abs2meta['output'][of]['size']
                                deriv_value = np.zeros((out_size, in_size))

                            if force_dense:
                                if rows is not None:
                                    try:
                                        in_size = wrt_meta['size']
                                    except KeyError:
                                        in_size = wrt_meta['size']
                                    out_size = comp._var_abs2meta['output'][of]['size']
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

        model._inputs.set_val(input_cache)
        model._outputs.set_val(output_cache)
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

                if step and requested_method == method:
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
                if c_name in mfree_directions:
                    vector = mfree_directions[c_name].get(local_wrt)
                else:
                    vector = None

                approximations[fd_options['method']].add_approximation(abs_key, self.model,
                                                                       fd_options, vector=vector)

            approx_jac = {}
            for approximation in approximations.values():
                # Perform the FD here.
                approximation.compute_approximations(comp, jac=approx_jac)

            for abs_key, partial in approx_jac.items():
                rel_key = abs_key2rel_key(comp, abs_key)
                partials_data[c_name][rel_key][jac_key] = partial

                # If this is a directional derivative, convert the analytic to a directional one.
                wrt = rel_key[1]
                if wrt in local_opts and local_opts[wrt]['directional']:
                    deriv = partials_data[c_name][rel_key]
                    deriv['J_fwd'] = np.atleast_2d(np.sum(deriv['J_fwd'], axis=1)).T
                    if comp.matrix_free:
                        deriv['J_rev'] = np.atleast_2d(np.sum(deriv['J_rev'], axis=0)).T

                        # Dot product test for adjoint validity.
                        m = mfree_directions[c_name][rel_key[0]].flatten()
                        d = mfree_directions[c_name][wrt].flatten()
                        mhat = partial.flatten()
                        dhat = deriv['J_rev'].flatten()

                        deriv['directional_fd_rev'] = dhat.dot(d) - mhat.dot(m)

        # Conversion of defaultdict to dicts
        partials_data = {comp_name: dict(outer) for comp_name, outer in partials_data.items()}

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
                                  print_reverse=print_reverse,
                                  show_only_incorrect=show_only_incorrect)

        return partials_data

    def check_totals(self, of=None, wrt=None, out_stream=_DEFAULT_OUT_STREAM, compact_print=False,
                     driver_scaling=False, abs_err_tol=1e-6, rel_err_tol=1e-6,
                     method='fd', step=None, form=None, step_calc='abs', show_progress=False):
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
            Set to True to just print the essentials, one line per input-output pair.
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
        show_progress : bool
            True to show progress of check_totals

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
        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            raise RuntimeError(self.msginfo + ": run_model must be called before total "
                               "derivatives can be checked.")

        model = self.model
        lcons = []

        if method == 'cs' and not model._outputs._alloc_complex:
            msg = "\n" + self.msginfo + ": To enable complex step, specify "\
                  "'force_alloc_complex=True' when calling " + \
                  "setup on the problem, e.g. 'problem.setup(force_alloc_complex=True)'"
            raise RuntimeError(msg)

        # TODO: Once we're tracking iteration counts, run the model if it has not been run before.

        if wrt is None:
            wrt = list(self.driver._designvars)
            if not wrt:
                raise RuntimeError("Driver is not providing any design variables "
                                   "for compute_totals.")

        if of is None:
            of = list(self.driver._objs)
            of.extend(self.driver._cons)
            if not of:
                raise RuntimeError("Driver is not providing any response variables "
                                   "for compute_totals.")
            lcons = [n for n, meta in self.driver._cons.items()
                     if ('linear' in meta and meta['linear'])]

        # Calculate Total Derivatives
        if model._owns_approx_jac:
            # Support this, even though it is a bit silly (though you could compare fd with cs.)
            total_info = _TotalJacInfo(self, of, wrt, False, return_format='flat_dict',
                                       approx=True, driver_scaling=driver_scaling)
            Jcalc = total_info.compute_totals_approx(initialize=True)

        else:
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
        approx_of = model._owns_approx_of
        approx_wrt = model._owns_approx_wrt
        old_jac = model._jacobian
        old_subjacs = model._subjacs_info.copy()

        model.approx_totals(method=method, step=step, form=form,
                            step_calc=step_calc if method == 'fd' else None)
        total_info = _TotalJacInfo(self, of, wrt, False, return_format='flat_dict', approx=True,
                                   driver_scaling=driver_scaling)
        if show_progress:
            Jfd = total_info.compute_totals_approx(initialize=True, progress_out_stream=out_stream)
        else:
            Jfd = total_info.compute_totals_approx(initialize=True)
        # reset the _owns_approx_jac flag after approximation is complete.
        if not approx:
            model._jacobian = old_jac
            model._owns_approx_jac = False
            model._owns_approx_of = approx_of
            model._owns_approx_wrt = approx_wrt
            model._subjacs_info = old_subjacs

        # Assemble and Return all metrics.
        data = {}
        data[''] = {}
        resp = self.driver._responses
        # TODO key should not be fwd when exact computed in rev mode or auto
        for key, val in Jcalc.items():
            data[''][key] = {}
            data[''][key]['J_fwd'] = val
            data[''][key]['J_fd'] = Jfd[key]

            # Display whether indices were declared when response was added.
            of = key[0]
            if of in resp and resp[of]['indices'] is not None:
                data[''][key]['indices'] = len(resp[of]['indices'])

        fd_args['method'] = method

        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        _assemble_derivative_data(data, rel_err_tol, abs_err_tol, out_stream, compact_print,
                                  [model], {'': fd_args}, totals=True, lcons=lcons)
        return data['']

    def compute_totals(self, of=None, wrt=None, return_format='flat_dict', debug_print=False,
                       driver_scaling=False, use_abs_names=False, get_remote=True):
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
            Format to return the derivatives. Can be 'dict', 'flat_dict', or 'array'.
            Default is a 'flat_dict', which returns them in a dictionary whose keys are
            tuples of form (of, wrt).
        debug_print : bool
            Set to True to print out some debug information during linear solve.
        driver_scaling : bool
            When True, return derivatives that are scaled according to either the adder and scaler
            or the ref and ref0 values that were specified when add_design_var, add_objective, and
            add_constraint were called on the model. Default is False, which is unscaled.
        use_abs_names : bool
            Set to True when passing in absolute names to skip some translation steps.
        get_remote : bool
            If True, the default, the full distributed total jacobian will be retrieved.

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        if self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            self.final_setup()

        if wrt is None:
            wrt = list(self.driver._designvars)
            if not wrt:
                raise RuntimeError("Driver is not providing any design variables "
                                   "for compute_totals.")

        if of is None:
            of = list(self.driver._objs)
            of.extend(self.driver._cons)
            if not of:
                raise RuntimeError("Driver is not providing any response variables "
                                   "for compute_totals.")

        if self.model._owns_approx_jac:
            total_info = _TotalJacInfo(self, of, wrt, use_abs_names, return_format,
                                       approx=True, driver_scaling=driver_scaling)
            return total_info.compute_totals_approx(initialize=True)
        else:
            total_info = _TotalJacInfo(self, of, wrt, use_abs_names, return_format,
                                       debug_print=debug_print, driver_scaling=driver_scaling,
                                       get_remote=get_remote)
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
        self.model.set_solver_print(level=level, depth=depth, type_=type_)

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
            'vectorize_derivs', 'cache_linear_solution', 'units']
        cons_opts : list of str
            List of optional columns to be displayed in the cons table.
            Allowed values are:
            ['lower', 'upper', 'equals', 'ref', 'ref0', 'indices', 'index', 'adder', 'scaler',
            'linear', 'parallel_deriv_color', 'vectorize_derivs',
            'cache_linear_solution', 'units']
        objs_opts : list of str
            List of optional columns to be displayed in the objs table.
            Allowed values are:
            ['ref', 'ref0', 'indices', 'adder', 'scaler', 'units',
            'parallel_deriv_color', 'vectorize_derivs', 'cache_linear_solution']

        """
        default_col_names = ['name', 'value', 'size']

        # Design vars
        desvars = self.driver._designvars
        vals = self.driver.get_design_var_values(get_remote=True)
        header = "Design Variables"
        col_names = default_col_names + desvar_opts
        self._write_var_info_table(header, col_names, desvars, vals,
                                   show_promoted_name=show_promoted_name,
                                   print_arrays=print_arrays,
                                   col_spacing=2)

        # Constraints
        cons = self.driver._cons
        vals = self.driver.get_constraint_values()
        header = "Constraints"
        col_names = default_col_names + cons_opts
        self._write_var_info_table(header, col_names, cons, vals,
                                   show_promoted_name=show_promoted_name,
                                   print_arrays=print_arrays,
                                   col_spacing=2)

        objs = self.driver._objs
        vals = self.driver.get_objective_values()
        header = "Objectives"
        col_names = default_col_names + objs_opts
        self._write_var_info_table(header, col_names, objs, vals,
                                   show_promoted_name=show_promoted_name,
                                   print_arrays=print_arrays,
                                   col_spacing=2)

    def _write_var_info_table(self, header, col_names, meta, vals, print_arrays=False,
                              show_promoted_name=True, col_spacing=1):
        """
        Write a table of information for the problem variable in meta and vals.

        Parameters
        ----------
        header : str
            The header line for the table.
        col_names : list of str
            List of column labels.
        meta : OrderedDict
            Dictionary of metadata for each problem variable.
        vals : OrderedDict
            Dictionary of values for each problem variable.
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
        for name, meta in meta.items():

            row = {}
            for col_name in col_names:
                if col_name == 'name':
                    if show_promoted_name:
                        row[col_name] = name
                    else:
                        if name in abs2prom['input']:
                            row[col_name] = abs2prom['input'][name]
                        elif name in abs2prom['output']:
                            row[col_name] = abs2prom['output'][name]
                        else:
                            # Promoted auto_ivc name. Keep it promoted
                            row[col_name] = name

                elif col_name == 'value':
                    row[col_name] = vals[name]
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
                    out_str = pprint.pformat(cell)
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
                if name not in self.model._var_abs2prom['input']:
                    raise KeyError("{}: Input variable, '{}', recorded in the case is not "
                                   "found in the model".format(self.msginfo, name))
                self[name] = inputs[name]

        outputs = case.outputs if case.outputs is not None else None
        if outputs:
            for name in outputs.absolute_names():
                if name not in self.model._var_abs2prom['output']:
                    raise KeyError("{}: Output variable, '{}', recorded in the case is not "
                                   "found in the model".format(self.msginfo, name))
                self[name] = outputs[name]

    def check_config(self, logger=None, checks=None, out_file='openmdao_checks.out'):
        """
        Perform optional error checks on a Problem.

        Parameters
        ----------
        logger : object
            Logging object.
        checks : list of str or None
            List of specific checks to be performed.
        out_file : str or None
            If not None, output will be written to this file in addition to stdout.
        """
        if logger is None:
            logger = get_logger('check_config', out_file=out_file, use_format=True)

        if checks is None:
            checks = sorted(_default_checks)
        elif checks == 'all':
            checks = sorted(_all_checks)

        for c in checks:
            if c not in _all_checks:
                print("WARNING: '%s' is not a recognized check.  Available checks are: %s" %
                      (c, sorted(_all_checks)))
                continue
            logger.info('checking %s' % c)
            _all_checks[c](self, logger)

    def set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        if self._metadata is None or \
           self._metadata['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            raise RuntimeError(f"{self.msginfo}: set_complex_step_mode cannot be called before "
                               "`Problem.run_model()`, `Problem.run_driver()`, or "
                               "`Problem.final_setup()`.")

        if active and not self._metadata['force_alloc_complex']:
            raise RuntimeError(f"{self.msginfo}: To enable complex step, specify "
                               "'force_alloc_complex=True' when calling setup on the problem, "
                               "e.g. 'problem.setup(force_alloc_complex=True)'")

        self.model._set_complex_step_mode(active)


def _assemble_derivative_data(derivative_data, rel_error_tol, abs_error_tol, out_stream,
                              compact_print, system_list, global_options, totals=False,
                              indep_key=None, print_reverse=False,
                              show_only_incorrect=False, lcons=None):
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
    print_reverse : bool, optional
        Set to True if compact_print results need to include columns for reverse mode.
    show_only_incorrect : bool, optional
        Set to True if output should print only the subjacs found to be incorrect.
    lcons : list or None
        For total derivatives only, list of outputs that are actually linear constraints.
    """
    nan = float('nan')
    suppress_output = out_stream is None

    if compact_print:
        if print_reverse:
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
        matrix_free = system.matrix_free

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
        sorted_keys = sorted(derivatives.keys())

        if not suppress_output:
            # Need to capture the output of a component's derivative
            # info so that it can be used if that component is the
            # worst subjac. That info is printed at the bottom of all the output
            out_buffer = StringIO()
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

                    if print_reverse:
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
                                pad_name('calc mag.'),
                                pad_name('check mag.'),
                                pad_name('a(cal-chk)'),
                                pad_name('r(cal-chk)'),
                            )

                if out_stream:
                    out_buffer.write(header + '\n')
                    out_buffer.write('-' * len(header) + '\n' + '\n')

        for of, wrt in sorted_keys:

            if totals:
                fd_opts = global_options['']
            else:
                fd_opts = global_options[sys_name][wrt]

            directional = fd_opts.get('directional')
            do_rev = not totals and matrix_free and not directional
            do_rev_dp = not totals and matrix_free and directional

            derivative_info = derivatives[of, wrt]
            # TODO total derivs may have been computed in rev mode, not fwd
            forward = derivative_info['J_fwd']
            fd = derivative_info['J_fd']
            if do_rev:
                reverse = derivative_info.get('J_rev')

            fwd_error = np.linalg.norm(forward - fd)
            if do_rev_dp:
                fwd_rev_error = derivative_info['directional_fwd_rev']
                rev_error = derivative_info['directional_fd_rev']
            elif do_rev:
                rev_error = np.linalg.norm(reverse - fd)
                fwd_rev_error = np.linalg.norm(forward - reverse)
            else:
                rev_error = fwd_rev_error = None

            fwd_norm = np.linalg.norm(forward)
            fd_norm = np.linalg.norm(fd)
            if do_rev:
                rev_norm = np.linalg.norm(reverse)
            else:
                rev_norm = None

            derivative_info['abs error'] = abs_err = ErrorTuple(fwd_error, rev_error, fwd_rev_error)
            derivative_info['magnitude'] = magnitude = MagnitudeTuple(fwd_norm, rev_norm, fd_norm)

            if fd_norm == 0.:
                if fwd_norm == 0.:
                    derivative_info['rel error'] = rel_err = ErrorTuple(nan, nan, nan)

                else:
                    # If fd_norm is zero, let's use fwd_norm as the divisor for relative
                    # check. That way we don't accidentally squelch a legitimate problem.
                    if do_rev or do_rev_dp:
                        rel_err = ErrorTuple(fwd_error / fwd_norm,
                                             rev_error / fwd_norm,
                                             fwd_rev_error / fwd_norm)
                        derivative_info['rel error'] = rel_err
                    else:
                        derivative_info['rel error'] = rel_err = ErrorTuple(fwd_error / fwd_norm,
                                                                            None,
                                                                            None)

            else:
                if do_rev or do_rev_dp:
                    derivative_info['rel error'] = rel_err = ErrorTuple(fwd_error / fd_norm,
                                                                        rev_error / fd_norm,
                                                                        fwd_rev_error / fd_norm)
                else:
                    derivative_info['rel error'] = rel_err = ErrorTuple(fwd_error / fd_norm,
                                                                        None,
                                                                        None)

            # Skip printing the dependent keys if the derivatives are fine.
            if not compact_print and indep_key is not None:
                rel_key = (of, wrt)
                if rel_key in indep_key[sys_name] and fd_norm < abs_error_tol:
                    del derivative_data[sys_name][rel_key]
                    continue

            # Informative output for responses that were declared with an index.
            indices = derivative_info.get('indices')
            if indices is not None:
                of = '{} (index size: {})'.format(of, indices)

            if not suppress_output:

                if compact_print:
                    if totals:
                        if out_stream:
                            out_buffer.write(deriv_line.format(
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
                            if error is None:
                                continue
                            if not np.isnan(error) and error >= abs_error_tol:
                                error_string += ' >ABS_TOL'
                                break

                        # See if this component has the greater
                        # error in the derivative computation
                        # compared to the other components so far
                        is_worst_subjac = False
                        for i, error in enumerate(rel_err):
                            if error is None:
                                continue
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
                            if print_reverse:
                                deriv_info_line = \
                                    deriv_line.format(
                                        pad_name(of, max_width_of, quotes=True),
                                        wrt_padded,
                                        magnitude.forward,
                                        _format_if_not_matrix_free(matrix_free and not directional,
                                                                   magnitude.reverse),
                                        magnitude.fd,
                                        abs_err.forward,
                                        _format_if_not_matrix_free(matrix_free,
                                                                   abs_err.reverse),
                                        _format_if_not_matrix_free(matrix_free,
                                                                   abs_err.forward_reverse),
                                        rel_err.forward,
                                        _format_if_not_matrix_free(matrix_free,
                                                                   rel_err.reverse),
                                        _format_if_not_matrix_free(matrix_free,
                                                                   rel_err.forward_reverse),
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
                            out_buffer.write(f"  {sys_name}: '{of}' wrt (d)'{wrt}'")
                        else:
                            out_buffer.write(f"  {sys_name}: '{of}' wrt '{wrt}'")
                        if lcons and of in lcons:
                            out_buffer.write(" (Linear constraint)")

                        out_buffer.write('\n')
                        if do_rev or do_rev_dp:
                            out_buffer.write('     Forward')
                        else:
                            out_buffer.write('    Analytic')
                        out_buffer.write(' Magnitude : {:.6e}\n'.format(magnitude.forward))
                    if do_rev:
                        txt = '     Reverse Magnitude : {:.6e}'
                        if out_stream:
                            out_buffer.write(txt.format(magnitude.reverse) + '\n')
                    if out_stream:
                        out_buffer.write('          Fd Magnitude : {:.6e} ({})\n'.format(
                            magnitude.fd, fd_desc))

                    # Absolute Errors
                    if do_rev:
                        error_descs = ('(Jfor - Jfd) ', '(Jrev - Jfd) ', '(Jfor - Jrev)')
                    elif do_rev_dp:
                        error_descs = ('(Jfor - Jfd) ', '(Jrev - Jfd  Dot Product Test) ',
                                       '(Jrev - Jfor Dot Product Test) ')
                    else:
                        error_descs = ('(Jan - Jfd) ', )

                    for error, desc in zip(abs_err, error_descs):
                        error_str = _format_error(error, abs_error_tol)
                        if error_str.endswith('*'):
                            num_bad_jacs += 1
                        if out_stream:
                            out_buffer.write('    Absolute Error {}: {}\n'.format(desc, error_str))
                    if out_stream:
                        out_buffer.write('\n')

                    # Relative Errors
                    if do_rev:
                        if fd_norm == 0.:
                            error_descs = ('(Jfor - Jfd) / Jfor ', '(Jrev - Jfd) / Jfor ',
                                           '(Jfor - Jrev) / Jfor')
                        else:
                            error_descs = ('(Jfor - Jfd) / Jfd ', '(Jrev - Jfd) / Jfd ',
                                           '(Jfor - Jrev) / Jfd')
                    elif do_rev_dp:
                        if fd_norm == 0.:
                            error_descs = ('(Jfor - Jfd) / Jfor ',
                                           '(Jrev - Jfd  Dot Product Test) / Jfor ',
                                           '(Jrev - Jfor Dot Product Test) / Jfor ')
                        else:
                            error_descs = ('(Jfor - Jfd) / Jfd ',
                                           '(Jrev - Jfd  Dot Product Test) / Jfd ',
                                           '(Jrev - Jfor Dot Product Test) / Jfd ')
                    else:
                        if fd_norm == 0.:
                            error_descs = ('(Jan - Jfd) / Jan ', )
                        else:
                            error_descs = ('(Jan - Jfd) / Jfd ', )

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
                        if do_rev_dp:
                            out_buffer.write('    Directional Forward Derivative (Jfor)\n')
                        else:
                            if not totals and matrix_free:
                                out_buffer.write('    Raw Forward')
                            else:
                                out_buffer.write('    Raw Analytic')
                            out_buffer.write(' Derivative (Jfor)\n')
                        out_buffer.write(str(forward) + '\n')
                        out_buffer.write('\n')

                    if not totals and matrix_free:
                        if out_stream:
                            if not directional:
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


class Slicer(object):
    """
    Helper class that can be used with the indices argument for Problem set_val and get_val.
    """

    def __getitem__(self, val):
        """
        Pass through indices or slice.

        Parameters
        ----------
        val : int or slice object or tuples of slice objects
            Indices or slice to return.

        Returns
        -------
        indices : int or slice object or tuples of slice objects
            Indices or slice to return.
        """
        return val


# instance of the Slicer class to be used by users for the set_val and get_val methods of Problem
slicer = Slicer()
