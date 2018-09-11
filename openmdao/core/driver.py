"""Define a base class for all Drivers in OpenMDAO."""
from __future__ import print_function

import os
import json
from collections import OrderedDict
import pprint
import sys
import warnings

from six import iteritems, itervalues, string_types

import numpy as np

from openmdao.core.total_jac import _TotalJacInfo
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.record_util import create_local_meta, check_path
from openmdao.utils.mpi import MPI
from openmdao.recorders.recording_iteration_stack import recording_iteration
from openmdao.utils.options_dictionary import OptionsDictionary
import openmdao.utils.coloring as coloring_mod


def _is_debug_print_opts_valid(opts):
    """
    Check validity of debug_print option for Driver.

    Parameters
    ----------
    opts : list
        The value of the debug_print option set by the user.

    Returns
    -------
    bool
        True if the option is valid. Otherwise, False.
    """
    if not isinstance(opts, list):
        return False
    _valid_opts = ['desvars', 'nl_cons', 'ln_cons', 'objs', 'totals']
    for opt in opts:
        if opt not in _valid_opts:
            return False
    return True


class Driver(object):
    """
    Top-level container for the systems and drivers.

    Attributes
    ----------
    fail : bool
        Reports whether the driver ran successfully.
    iter_count : int
        Keep track of iterations for case recording.
    options : <OptionsDictionary>
        Dictionary with general pyoptsparse options.
    recording_options : <OptionsDictionary>
        Dictionary with driver recording options.
    debug_print : <OptionsDictionary>
        Dictionary with debugging printing options.
    cite : str
        Listing of relevant citataions that should be referenced when
        publishing work that uses this class.
    _problem : <Problem>
        Pointer to the containing problem.
    supports : <OptionsDictionary>
        Provides a consistant way for drivers to declare what features they support.
    _designvars : dict
        Contains all design variable info.
    _cons : dict
        Contains all constraint info.
    _objs : dict
        Contains all objective info.
    _responses : dict
        Contains all response info.
    _rec_mgr : <RecordingManager>
        Object that manages all recorders added to this driver.
    _vars_to_record: dict
        Dict of lists of var names indicating what to record
    _model_viewer_data : dict
        Structure of model, used to make n2 diagram.
    _remote_dvs : dict
        Dict of design variables that are remote on at least one proc. Values are
        (owning rank, size).
    _remote_cons : dict
        Dict of constraints that are remote on at least one proc. Values are
        (owning rank, size).
    _remote_objs : dict
        Dict of objectives that are remote on at least one proc. Values are
        (owning rank, size).
    _remote_responses : dict
        A combined dict containing entries from _remote_cons and _remote_objs.
    _simul_coloring_info : tuple of dicts
        A data structure describing coloring for simultaneous derivs.
    _total_jac_sparsity : dict, str, or None
        Specifies sparsity of sub-jacobians of the total jacobian. Only used by pyOptSparseDriver.
    _res_jacs : dict
        Dict of sparse subjacobians for use with certain optimizers, e.g. pyOptSparseDriver.
    _total_jac : _TotalJacInfo or None
        Cached total jacobian handling object.
    """

    def __init__(self, **kwargs):
        """
        Initialize the driver.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        self._rec_mgr = RecordingManager()
        self._vars_to_record = {
            'desvarnames': set(),
            'responsenames': set(),
            'objectivenames': set(),
            'constraintnames': set(),
            'sysinclnames': set(),
        }

        self._problem = None
        self._designvars = None
        self._cons = None
        self._objs = None
        self._responses = None

        # Driver options
        self.options = OptionsDictionary()

        self.options.declare('debug_print', types=list, is_valid=_is_debug_print_opts_valid,
                             desc="List of what type of Driver variables to print at each "
                                  "iteration. Valid items in list are 'desvars', 'ln_cons', "
                                  "'nl_cons', 'objs', 'totals'",
                             default=[])

        # Case recording options
        self.recording_options = OptionsDictionary()

        self.recording_options.declare('record_metadata', types=bool, default=True,
                                       desc='Record Driver metadata')
        self.recording_options.declare('record_model_metadata', types=bool, default=True,
                                       desc='Record metadata for all Systems in the model')
        self.recording_options.declare('record_desvars', types=bool, default=True,
                                       desc='Set to True to record design variables at the '
                                            'driver level')
        self.recording_options.declare('record_responses', types=bool, default=False,
                                       desc='Set to True to record responses at the driver level')
        self.recording_options.declare('record_objectives', types=bool, default=True,
                                       desc='Set to True to record objectives at the driver level')
        self.recording_options.declare('record_constraints', types=bool, default=True,
                                       desc='Set to True to record constraints at the '
                                            'driver level')
        self.recording_options.declare('includes', types=list, default=[],
                                       desc='Patterns for variables to include in recording')
        self.recording_options.declare('excludes', types=list, default=[],
                                       desc='Patterns for vars to exclude in recording '
                                            '(processed post-includes)')
        self.recording_options.declare('record_derivatives', types=bool, default=False,
                                       desc='Set to True to record derivatives at the driver '
                                            'level')
        self.recording_options.declare('record_inputs', types=bool, default=True,
                                       desc='Set to True to record inputs at the driver level')
        self.recording_options.declare('record_n2_data', types=bool, default=True,
                                       desc='Set to True to record metadata required for '
                                       'N^2 viewing')

        # What the driver supports.
        self.supports = OptionsDictionary()
        self.supports.declare('inequality_constraints', types=bool, default=False)
        self.supports.declare('equality_constraints', types=bool, default=False)
        self.supports.declare('linear_constraints', types=bool, default=False)
        self.supports.declare('two_sided_constraints', types=bool, default=False)
        self.supports.declare('multiple_objectives', types=bool, default=False)
        self.supports.declare('integer_design_vars', types=bool, default=False)
        self.supports.declare('gradients', types=bool, default=False)
        self.supports.declare('active_set', types=bool, default=False)
        self.supports.declare('simultaneous_derivatives', types=bool, default=False)
        self.supports.declare('total_jac_sparsity', types=bool, default=False)
        # TODO, support these in OpenMDAO
        self.supports.declare('integer_design_vars', types=bool, default=False)

        self.iter_count = 0
        self._model_viewer_data = None
        self.cite = ""

        self._simul_coloring_info = None
        self._total_jac_sparsity = None
        self._res_jacs = {}
        self._total_jac = None

        self.fail = False

        self._declare_options()
        self.options.update(kwargs)

    def add_recorder(self, recorder):
        """
        Add a recorder to the driver.

        Parameters
        ----------
        recorder : BaseRecorder
           A recorder instance.
        """
        self._rec_mgr.append(recorder)

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        # shut down all recorders
        self._rec_mgr.shutdown()

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.

        This is optionally implemented by subclasses of Driver.
        """
        pass

    def _setup_comm(self, comm):
        """
        Perform any driver-specific setup of communicators for the model.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The communicator for the Problem.

        Returns
        -------
        MPI.Comm or <FakeComm> or None
            The communicator for the Problem model.
        """
        return comm

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        self._problem = problem
        model = problem.model
        mode = problem._mode

        self._total_jac = None

        self._has_scaling = (
            np.any([r['scaler'] is not None for r in itervalues(self._responses)]) or
            np.any([dv['scaler'] is not None for dv in itervalues(self._designvars)])
        )

        con_set = set()
        obj_set = set()
        dv_set = set()

        self._remote_dvs = dv_dict = {}
        self._remote_cons = con_dict = {}
        self._remote_objs = obj_dict = {}

        # Now determine if later we'll need to allgather cons, objs, or desvars.
        if model.comm.size > 1 and model._subsystems_allprocs:
            local_out_vars = set(model._outputs._views)
            remote_dvs = set(self._designvars) - local_out_vars
            remote_cons = set(self._cons) - local_out_vars
            remote_objs = set(self._objs) - local_out_vars
            all_remote_vois = model.comm.allgather(
                (remote_dvs, remote_cons, remote_objs))
            for rem_dvs, rem_cons, rem_objs in all_remote_vois:
                con_set.update(rem_cons)
                obj_set.update(rem_objs)
                dv_set.update(rem_dvs)

            # If we have remote VOIs, pick an owning rank for each and use that
            # to bcast to others later
            owning_ranks = model._owning_rank
            sizes = model._var_sizes['nonlinear']['output']
            for i, vname in enumerate(model._var_allprocs_abs_names['output']):
                owner = owning_ranks[vname]
                if vname in dv_set:
                    dv_dict[vname] = (owner, sizes[owner, i])
                if vname in con_set:
                    con_dict[vname] = (owner, sizes[owner, i])
                if vname in obj_set:
                    obj_dict[vname] = (owner, sizes[owner, i])

        self._remote_responses = self._remote_cons.copy()
        self._remote_responses.update(self._remote_objs)

        # set up case recording
        self._setup_recording()

        # set up simultaneous deriv coloring
        if (coloring_mod._use_sparsity and self._simul_coloring_info and
                self.supports['simultaneous_derivatives']):
            self._setup_simul_coloring()

    def _setup_recording(self):
        """
        Set up case recording.
        """
        problem = self._problem
        model = problem.model

        mydesvars = myobjectives = myconstraints = myresponses = set()
        myinputs = set()
        mysystem_outputs = set()

        incl = self.recording_options['includes']
        excl = self.recording_options['excludes']

        rec_desvars = self.recording_options['record_desvars']
        rec_objectives = self.recording_options['record_objectives']
        rec_constraints = self.recording_options['record_constraints']
        rec_responses = self.recording_options['record_responses']
        rec_inputs = self.recording_options['record_inputs']

        all_desvars = {n for n in self._designvars
                       if check_path(n, incl, excl, True)}
        all_objectives = {n for n in self._objs
                          if check_path(n, incl, excl, True)}
        all_constraints = {n for n in self._cons
                           if check_path(n, incl, excl, True)}
        if rec_desvars:
            mydesvars = all_desvars

        if rec_objectives:
            myobjectives = all_objectives

        if rec_constraints:
            myconstraints = all_constraints

        if rec_responses:
            myresponses = {n for n in self._responses
                           if check_path(n, incl, excl, True)}

        # get the includes that were requested for this Driver recording
        if incl:
            # The my* variables are sets

            # First gather all of the desired outputs
            # The following might only be the local vars if MPI
            mysystem_outputs = {n for n in model._outputs
                                if check_path(n, incl, excl)}

            # If MPI, and on rank 0, need to gather up all the variables
            #    even those not local to rank 0
            if MPI:
                all_vars = model.comm.gather(mysystem_outputs, root=0)
                if MPI.COMM_WORLD.rank == 0:
                    mysystem_outputs = all_vars[-1]
                    for d in all_vars[:-1]:
                        mysystem_outputs.update(d)

            # de-duplicate mysystem_outputs
            mysystem_outputs = mysystem_outputs.difference(all_desvars, all_objectives,
                                                           all_constraints)

        if rec_inputs:
            prob = self._problem
            root = prob.model
            myinputs = {n for n in root._inputs
                        if check_path(n, incl, excl)}

            if MPI:
                all_vars = root.comm.gather(myinputs, root=0)
                if MPI.COMM_WORLD.rank == 0:
                    myinputs = all_vars[-1]
                    for d in all_vars[:-1]:
                        myinputs.update(d)

        if MPI:  # filter based on who owns the variables
            # TODO Eventually, we think we can get rid of this next check. But to be safe,
            #       we are leaving it in there.
            if not model.is_active():
                raise RuntimeError("RecordingManager.startup should never be called when "
                                   "running in parallel on an inactive System")
            rrank = problem.comm.rank
            rowned = model._owning_rank
            mydesvars = [n for n in mydesvars if rrank == rowned[n]]
            myresponses = [n for n in myresponses if rrank == rowned[n]]
            myobjectives = [n for n in myobjectives if rrank == rowned[n]]
            myconstraints = [n for n in myconstraints if rrank == rowned[n]]
            mysystem_outputs = [n for n in mysystem_outputs if rrank == rowned[n]]
            myinputs = [n for n in myinputs if rrank == rowned[n]]

        self._filtered_vars_to_record = {
            'des': mydesvars,
            'obj': myobjectives,
            'con': myconstraints,
            'res': myresponses,
            'sys': mysystem_outputs,
            'in': myinputs
        }

        self._rec_mgr.startup(self)
        if self.recording_options['record_metadata']:
            if self.recording_options['record_n2_data']:
                if self._rec_mgr._recorders:
                    from openmdao.devtools.problem_viewer.problem_viewer import _get_viewer_data
                    self._model_viewer_data = _get_viewer_data(problem)
            self._rec_mgr.record_metadata(self)

        # Also record the system metadata to the recorders attached to this Driver
        if self.recording_options['record_model_metadata']:
            for sub in model.system_iter(recurse=True, include_self=True):
                self._rec_mgr.record_metadata(sub)

    def _get_voi_val(self, name, meta, remote_vois, unscaled=False, ignore_indices=False):
        """
        Get the value of a variable of interest (objective, constraint, or design var).

        This will retrieve the value if the VOI is remote.

        Parameters
        ----------
        name : str
            Name of the variable of interest.
        meta : dict
            Metadata for the variable of interest.
        remote_vois : dict
            Dict containing (owning_rank, size) for all remote vois of a particular
            type (design var, constraint, or objective).
        unscaled : bool
            Set to True if unscaled (physical) design variables are desired.
        ignore_indices : bool
            Set to True if the full array is desired, not just those indicated by indices.

        Returns
        -------
        float or ndarray
            The value of the named variable of interest.
        """
        model = self._problem.model
        comm = model.comm
        vec = model._outputs._views_flat
        indices = meta['indices']

        if name in remote_vois:
            owner, size = remote_vois[name]
            if owner == comm.rank:
                if indices is None or ignore_indices:
                    val = vec[name].copy()
                else:
                    val = vec[name][indices]
            else:
                if not (indices is None or ignore_indices):
                    size = len(indices)
                val = np.empty(size)

            comm.Bcast(val, root=owner)
        else:
            if indices is None or ignore_indices:
                val = vec[name].copy()
            else:
                val = vec[name][indices]

        if self._has_scaling and not unscaled:
            # Scale design variable values
            adder = meta['adder']
            if adder is not None:
                val += adder

            scaler = meta['scaler']
            if scaler is not None:
                val *= scaler

        return val

    def get_design_var_values(self, filter=None, unscaled=False, ignore_indices=False):
        """
        Return the design variable values.

        This is called to gather the initial design variable state.

        Parameters
        ----------
        filter : list
            List of desvar names used by recorders.
        unscaled : bool
            Set to True if unscaled (physical) design variables are desired.
        ignore_indices : bool
            Set to True if the full array is desired, not just those indicated by indices.

        Returns
        -------
        dict
           Dictionary containing values of each design variable.
        """
        if filter:
            dvs = filter
        else:
            # use all the designvars
            dvs = self._designvars

        return {n: self._get_voi_val(n, self._designvars[n], self._remote_dvs, unscaled=unscaled,
                                     ignore_indices=ignore_indices) for n in dvs}

    def set_design_var(self, name, value):
        """
        Set the value of a design variable.

        Parameters
        ----------
        name : str
            Global pathname of the design variable.
        value : float or ndarray
            Value for the design variable.
        """
        if (name in self._remote_dvs and
                self._problem.model._owning_rank[name] != self._problem.comm.rank):
            return

        meta = self._designvars[name]
        indices = meta['indices']
        if indices is None:
            indices = slice(None)

        desvar = self._problem.model._outputs._views_flat[name]
        desvar[indices] = value

        if self._has_scaling:
            # Scale design variable values
            scaler = meta['scaler']
            if scaler is not None:
                desvar[indices] *= 1.0 / scaler

            adder = meta['adder']
            if adder is not None:
                desvar[indices] -= adder

    def get_response_values(self, filter=None):
        """
        Return response values.

        Parameters
        ----------
        filter : list
            List of response names used by recorders.

        Returns
        -------
        dict
           Dictionary containing values of each response.
        """
        if filter:
            resps = filter
        else:
            resps = self._responses

        return {n: self._get_voi_val(n, self._responses[n], self._remote_objs) for n in resps}

    def get_objective_values(self, unscaled=False, filter=None, ignore_indices=False):
        """
        Return objective values.

        Parameters
        ----------
        unscaled : bool
            Set to True if unscaled (physical) design variables are desired.
        filter : list
            List of objective names used by recorders.
        ignore_indices : bool
            Set to True if the full array is desired, not just those indicated by indices.

        Returns
        -------
        dict
           Dictionary containing values of each objective.
        """
        if filter:
            objs = filter
        else:
            objs = self._objs

        return {n: self._get_voi_val(n, self._objs[n], self._remote_objs, unscaled=unscaled,
                                     ignore_indices=ignore_indices)
                for n in objs}

    def get_constraint_values(self, ctype='all', lintype='all', unscaled=False, filter=None,
                              ignore_indices=False):
        """
        Return constraint values.

        Parameters
        ----------
        ctype : string
            Default is 'all'. Optionally return just the inequality constraints
            with 'ineq' or the equality constraints with 'eq'.
        lintype : string
            Default is 'all'. Optionally return just the linear constraints
            with 'linear' or the nonlinear constraints with 'nonlinear'.
        unscaled : bool
            Set to True if unscaled (physical) design variables are desired.
        filter : list
            List of constraint names used by recorders.
        ignore_indices : bool
            Set to True if the full array is desired, not just those indicated by indices.

        Returns
        -------
        dict
           Dictionary containing values of each constraint.
        """
        if filter is not None:
            cons = filter
        else:
            cons = self._cons

        con_dict = {}
        for name in cons:
            meta = self._cons[name]

            if lintype == 'linear' and not meta['linear']:
                continue

            if lintype == 'nonlinear' and meta['linear']:
                continue

            if ctype == 'eq' and meta['equals'] is None:
                continue

            if ctype == 'ineq' and meta['equals'] is not None:
                continue

            con_dict[name] = self._get_voi_val(name, meta, self._remote_cons, unscaled=unscaled,
                                               ignore_indices=ignore_indices)

        return con_dict

    def _get_ordered_nl_responses(self):
        """
        Return the names of nonlinear responses in the order used by the driver.

        Default order is objectives followed by nonlinear constraints.  This is used for
        simultaneous derivative coloring and sparsity determination.

        Returns
        -------
        list of str
            The nonlinear response names in order.
        """
        order = list(self._objs)
        order.extend(n for n, meta in iteritems(self._cons)
                     if not ('linear' in meta and meta['linear']))
        return order

    def _update_voi_meta(self, model):
        """
        Collect response and design var metadata from the model and size desvars and responses.

        Parameters
        ----------
        model : System
            The System that represents the entire model.

        Returns
        -------
        int
            Total size of responses, with linear constraints excluded.
        int
            Total size of design vars.
        """
        self._objs = objs = OrderedDict()
        self._cons = cons = OrderedDict()

        self._responses = resps = model.get_responses(recurse=True)
        for name, data in iteritems(resps):
            if data['type'] == 'con':
                cons[name] = data
            else:
                objs[name] = data

        response_size = sum(resps[n]['size'] for n in self._get_ordered_nl_responses())

        # Gather up the information for design vars.
        self._designvars = designvars = model.get_design_vars(recurse=True)
        desvar_size = sum(data['size'] for data in itervalues(designvars))

        return response_size, desvar_size

    def run(self):
        """
        Execute this driver.

        The base `Driver` just runs the model. All other drivers overload
        this method.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            failure_flag, _, _ = self._problem.model._solve_nonlinear()

        self.iter_count += 1
        return failure_flag

    def _compute_totals(self, of=None, wrt=None, return_format='flat_dict', global_names=True):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        All derivatives are returned using driver scaling.

        Parameters
        ----------
        of : list of variable name strings or None
            Variables whose derivatives will be computed. Default is None, which
            uses the driver's objectives and constraints.
        wrt : list of variable name strings or None
            Variables with respect to which the derivatives will be computed.
            Default is None, which uses the driver's desvars.
        return_format : string
            Format to return the derivatives. Default is a 'flat_dict', which
            returns them in a dictionary whose keys are tuples of form (of, wrt). For
            the scipy optimizer, 'array' is also supported.
        global_names : bool
            Set to True when passing in global names to skip some translation steps.

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        total_jac = self._total_jac
        debug_print = 'totals' in self.options['debug_print'] and (not MPI or
                                                                   MPI.COMM_WORLD.rank == 0)

        if debug_print:
            header = 'Driver total derivatives for iteration: ' + str(self.iter_count)
            print(header)
            print(len(header) * '-' + '\n')

        if self._problem.model._owns_approx_jac:
            recording_iteration.stack.append(('_compute_totals_approx', 0))

            try:
                if total_jac is None:
                    total_jac = _TotalJacInfo(self._problem, of, wrt, global_names,
                                              return_format, approx=True, debug_print=debug_print)
                    self._total_jac = total_jac
                    totals = total_jac.compute_totals_approx(initialize=True)
                else:
                    totals = total_jac.compute_totals_approx()
            finally:
                recording_iteration.stack.pop()

        else:
            if total_jac is None:
                total_jac = _TotalJacInfo(self._problem, of, wrt, global_names, return_format,
                                          debug_print=debug_print)

            # don't cache linear constraint jacobian
            if not total_jac.has_lin_cons:
                self._total_jac = total_jac

            recording_iteration.stack.append(('_compute_totals', 0))

            try:
                totals = total_jac.compute_totals()
            finally:
                recording_iteration.stack.pop()

        if self._rec_mgr._recorders and self.recording_options['record_derivatives']:
            metadata = create_local_meta(self._get_name())
            total_jac.record_derivatives(self, metadata)

        return totals

    def record_iteration(self):
        """
        Record an iteration of the current Driver.
        """
        if not self._rec_mgr._recorders:
            return

        # Get the data to record (collective calls that get across all ranks)
        opts = self.recording_options
        filt = self._filtered_vars_to_record

        if opts['record_desvars']:
            des_vars = self.get_design_var_values(unscaled=True, ignore_indices=True)
        else:
            des_vars = {}

        if opts['record_objectives']:
            obj_vars = self.get_objective_values(unscaled=True, ignore_indices=True)
        else:
            obj_vars = {}

        if opts['record_constraints']:
            con_vars = self.get_constraint_values(unscaled=True, ignore_indices=True)
        else:
            con_vars = {}

        if opts['record_responses']:
            # res_vars = self.get_response_values()  # not really working yet
            res_vars = {}
        else:
            res_vars = {}

        des_vars = {name: des_vars[name] for name in filt['des']}
        obj_vars = {name: obj_vars[name] for name in filt['obj']}
        con_vars = {name: con_vars[name] for name in filt['con']}
        # res_vars = {name: res_vars[name] for name in filt['res']}

        model = self._problem.model

        names = model._outputs._names
        views = model._outputs._views
        sys_vars = {name: views[name] for name in names if name in filt['sys']}

        if self.recording_options['record_inputs']:
            names = model._inputs._names
            views = model._inputs._views
            in_vars = {name: views[name] for name in names if name in filt['in']}
        else:
            in_vars = {}

        if MPI:
            des_vars = self._gather_vars(model, des_vars)
            res_vars = self._gather_vars(model, res_vars)
            obj_vars = self._gather_vars(model, obj_vars)
            con_vars = self._gather_vars(model, con_vars)
            sys_vars = self._gather_vars(model, sys_vars)
            in_vars = self._gather_vars(model, in_vars)

        outs = {}
        if not MPI or model.comm.rank == 0:
            outs.update(des_vars)
            outs.update(res_vars)
            outs.update(obj_vars)
            outs.update(con_vars)
            outs.update(sys_vars)

        data = {
            'out': outs,
            'in': in_vars
        }

        metadata = create_local_meta(self._get_name())

        self._rec_mgr.record_iteration(self, data, metadata)

    def _gather_vars(self, root, local_vars):
        """
        Gather and return only variables listed in `local_vars` from the `root` System.

        Parameters
        ----------
        root : <System>
            the root System for the Problem
        local_vars : dict
            local variable names and values

        Returns
        -------
        dct : dict
            variable names and values.
        """
        # if trace:
        #     debug("gathering vars for recording in %s" % root.pathname)
        all_vars = root.comm.gather(local_vars, root=0)
        # if trace:
        #     debug("DONE gathering rec vars for %s" % root.pathname)

        if root.comm.rank == 0:
            dct = all_vars[-1]
            for d in all_vars[:-1]:
                dct.update(d)
            return dct

    def _get_name(self):
        """
        Get name of current Driver.

        Returns
        -------
        str
            Name of current Driver.
        """
        return "Driver"

    def set_simul_deriv_color(self, simul_info):
        """
        Set the coloring (and possibly the sub-jac sparsity) for simultaneous total derivatives.

        Parameters
        ----------
        simul_info : str or dict

            ::

                # Information about simultaneous coloring for design vars and responses.  If a
                # string, then simul_info is assumed to be the name of a file that contains the
                # coloring information in JSON format.  If a dict, the structure looks like this:

                {
                "fwd": [
                    # First, a list of column index lists, each index list representing columns
                    # having the same color, except for the very first index list, which contains
                    # indices of all columns that are not colored.
                    [
                        [i1, i2, i3, ...]    # list of non-colored columns
                        [ia, ib, ...]    # list of columns in first color
                        [ic, id, ...]    # list of columns in second color
                           ...           # remaining color lists, one list of columns per color
                    ],

                    # Next is a list of lists, one for each column, containing the nonzero rows for
                    # that column.  If a column is not colored, then it will have a None entry
                    # instead of a list.
                    [
                        [r1, rn, ...]   # list of nonzero rows for column 0
                        None,           # column 1 is not colored
                        [ra, rb, ...]   # list of nonzero rows for column 2
                            ...
                    ],
                ],
                # This example is not a bidirectional coloring, so the opposite direction, "rev"
                # in this case, has an empty row index list.  It could also be removed entirely.
                "rev": [[[]], []],
                "sparsity":
                    # The sparsity entry can be absent, indicating that no sparsity structure is
                    # specified, or it can be a nested dictionary where the outer keys are response
                    # names, the inner keys are design variable names, and the value is a tuple of
                    # the form (row_list, col_list, shape).
                    {
                        resp1_name: {
                            dv1_name: (rows, cols, shape),  # for sub-jac d_resp1/d_dv1
                            dv2_name: (rows, cols, shape),
                              ...
                        },
                        resp2_name: {
                            ...
                        }
                        ...
                    }
                }

        """
        if self.supports['simultaneous_derivatives']:
            self._simul_coloring_info = simul_info
        else:
            raise RuntimeError("Driver '%s' does not support simultaneous derivatives." %
                               self._get_name())

    def set_total_jac_sparsity(self, sparsity):
        """
        Set the sparsity of sub-jacobians of the total jacobian.

        Note: This currently will have no effect if you are not using the pyOptSparseDriver.

        Parameters
        ----------
        sparsity : str or dict

            ::

                # Sparsity is a nested dictionary where the outer keys are response
                # names, the inner keys are design variable names, and the value is a tuple of
                # the form (row_list, col_list, shape).
                {
                    resp1: {
                        dv1: (rows, cols, shape),  # for sub-jac d_resp1/d_dv1
                        dv2: (rows, cols, shape),
                          ...
                    },
                    resp2: {
                        ...
                    }
                    ...
                }
        """
        if self.supports['total_jac_sparsity']:
            self._total_jac_sparsity = sparsity
        else:
            raise RuntimeError("Driver '%s' does not support setting of total jacobian sparsity." %
                               self._get_name())

    def _setup_simul_coloring(self):
        """
        Set up metadata for simultaneous derivative solution.
        """
        # command line simul_coloring uses this env var to turn pre-existing coloring off
        if not coloring_mod._use_sparsity:
            return

        if isinstance(self._simul_coloring_info, string_types):
            with open(self._simul_coloring_info, 'r') as f:
                self._simul_coloring_info = coloring_mod._json2coloring(json.load(f))

        if 'rev' in self._simul_coloring_info and self._problem._orig_mode not in ('rev', 'auto'):
            revcol = self._simul_coloring_info['rev'][0][0]
            if revcol:
                raise RuntimeError("Simultaneous coloring does reverse solves but mode has "
                                   "been set to '%s'" % self._problem._orig_mode)
        if 'fwd' in self._simul_coloring_info and self._problem._orig_mode not in ('fwd', 'auto'):
            fwdcol = self._simul_coloring_info['fwd'][0][0]
            if fwdcol:
                raise RuntimeError("Simultaneous coloring does forward solves but mode has "
                                   "been set to '%s'" % self._problem._orig_mode)

        # simul_coloring_info can contain data for either fwd, rev, or both, along with optional
        # sparsity patterns
        if 'sparsity' in self._simul_coloring_info:
            sparsity = self._simul_coloring_info['sparsity']
            del self._simul_coloring_info['sparsity']
        else:
            sparsity = None

        if sparsity is not None and self._total_jac_sparsity is not None:
            raise RuntimeError("Total jac sparsity was set in both _simul_coloring_info"
                               " and _total_jac_sparsity.")
        self._total_jac_sparsity = sparsity

    def _pre_run_model_debug_print(self):
        """
        Optionally print some debugging information before the model runs.
        """
        debug_opt = self.options['debug_print']
        if not debug_opt or debug_opt == ['totals']:
            return

        if not MPI or MPI.COMM_WORLD.rank == 0:
            header = 'Driver debug print for iter coord: {}'.format(
                recording_iteration.get_formatted_iteration_coordinate())
            print(header)
            print(len(header) * '-')

        if 'desvars' in debug_opt:
            desvar_vals = self.get_design_var_values(unscaled=True, ignore_indices=True)
            if not MPI or MPI.COMM_WORLD.rank == 0:
                print("Design Vars")
                if desvar_vals:
                    pprint.pprint(desvar_vals)
                else:
                    print("None")
                print()

        sys.stdout.flush()

    def _post_run_model_debug_print(self):
        """
        Optionally print some debugging information after the model runs.
        """
        if 'nl_cons' in self.options['debug_print']:
            cons = self.get_constraint_values(lintype='nonlinear', unscaled=True)
            if not MPI or MPI.COMM_WORLD.rank == 0:
                print("Nonlinear constraints")
                if cons:
                    pprint.pprint(cons)
                else:
                    print("None")
                print()

        if 'ln_cons' in self.options['debug_print']:
            cons = self.get_constraint_values(lintype='linear', unscaled=True)
            if not MPI or MPI.COMM_WORLD.rank == 0:
                print("Linear constraints")
                if cons:
                    pprint.pprint(cons)
                else:
                    print("None")
                print()

        if 'objs' in self.options['debug_print']:
            objs = self.get_objective_values(unscaled=True)
            if not MPI or MPI.COMM_WORLD.rank == 0:
                print("Objectives")
                if objs:
                    pprint.pprint(objs)
                else:
                    print("None")
                print()

        sys.stdout.flush()


class RecordingDebugging(Recording):
    """
    A class that acts as a context manager.

    Handles doing the case recording and also the Driver
    debugging printing.
    """

    def __enter__(self):
        """
        Do things before the code inside the 'with RecordingDebugging' block.

        Returns
        -------
        self : object
            self
        """
        super(RecordingDebugging, self).__enter__()
        self.recording_requester._pre_run_model_debug_print()
        return self

    def __exit__(self, *args):
        """
        Do things after the code inside the 'with RecordingDebugging' block.

        Parameters
        ----------
        *args : array
            Solver recording requires extra args.
        """
        self.recording_requester._post_run_model_debug_print()
        super(RecordingDebugging, self).__exit__()
