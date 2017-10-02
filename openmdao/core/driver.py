"""Define a base class for all Drivers in OpenMDAO."""
from __future__ import print_function
from collections import OrderedDict

from six import iteritems

import numpy as np

from openmdao.recorders.recording_manager import RecordingManager
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.record_util import create_local_meta
from openmdao.utils.options_dictionary import OptionsDictionary


class Driver(object):
    """
    Top-level container for the systems and drivers.

    Attributes
    ----------
    fail : bool
        Reports whether the driver ran successfully.
    iter_count : int
        Keep track of iterations for case recording.
    metadata : list
        List of metadata
    options : <OptionsDictionary>
        Dictionary with general pyoptsparse options.
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
    """

    def __init__(self):
        """
        Initialize the driver.
        """
        self._rec_mgr = RecordingManager()

        self._problem = None
        self._designvars = None
        self._cons = None
        self._objs = None
        self._responses = None
        self.options = OptionsDictionary()

        # What the driver supports.
        self.supports = OptionsDictionary()
        self.supports.declare('inequality_constraints', type_=bool, default=False)
        self.supports.declare('equality_constraints', type_=bool, default=False)
        self.supports.declare('linear_constraints', type_=bool, default=False)
        self.supports.declare('two_sided_constraints', type_=bool, default=False)
        self.supports.declare('multiple_objectives', type_=bool, default=False)
        self.supports.declare('integer_design_vars', type_=bool, default=False)
        self.supports.declare('gradients', type_=bool, default=False)
        self.supports.declare('active_set', type_=bool, default=False)

        self.iter_count = 0
        self.metadata = None
        self._model_viewer_data = None

        # TODO, support these in Openmdao blue
        self.supports.declare('integer_design_vars', type_=bool, default=False)

        self.fail = False

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
        self._rec_mgr.close()

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

        self._objs = objs = OrderedDict()
        self._cons = cons = OrderedDict()
        self._responses = model.get_responses(recurse=True)
        for name, data in iteritems(self._responses):
            if data['type'] == 'con':
                cons[name] = data
            else:
                objs[name] = data

        # Gather up the information for design vars.
        self._designvars = model.get_design_vars(recurse=True)

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
            all_remote_vois = model.comm.allgather((remote_dvs, remote_cons, remote_objs))
            for rem_dvs, rem_cons, rem_objs in all_remote_vois:
                con_set.update(rem_cons)
                obj_set.update(rem_objs)
                dv_set.update(rem_dvs)

            # If we have remote VOIs, pick an owning rank for each and use that
            # to bcast to others later
            owning_ranks = model._owning_rank['output']
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

        self._rec_mgr.startup(self)
        if (self._rec_mgr._recorders):
            from openmdao.devtools.problem_viewer.problem_viewer import _get_viewer_data
            self._model_viewer_data = _get_viewer_data(problem)
        self._rec_mgr.record_metadata(self)

    def _get_voi_val(self, name, meta, remote_vois):
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
                if indices is None:
                    val = vec[name].copy()
                else:
                    val = vec[name][indices]
            else:
                if indices is not None:
                    size = len(indices)
                val = np.empty(size)
            comm.Bcast(val, root=owner)
        else:
            if indices is None:
                val = vec[name].copy()
            else:
                val = vec[name][indices]

        # Scale design variable values
        adder = meta['adder']
        if adder is not None:
            val += adder

        scaler = meta['scaler']
        if scaler is not None:
            val *= scaler

        return val

    def get_design_var_values(self, filter=None):
        """
        Return the design variable values.

        This is called to gather the initial design variable state.

        Parameters
        ----------
        filter : list
            List of desvar names used by recorders.

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

        return {n: self._get_voi_val(n, self._designvars[n], self._remote_dvs) for n in dvs}

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
                self._problem.model._owning_rank['output'][name] != self._problem.comm.rank):
            return

        meta = self._designvars[name]
        indices = meta['indices']
        if indices is None:
            indices = slice(None)

        desvar = self._problem.model._outputs._views_flat[name]
        desvar[indices] = value

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
        # TODO: finish this method when we have a driver that requires it.
        return {}

    def get_objective_values(self, filter=None):
        """
        Return objective values.

        Parameters
        ----------
        filter : list
            List of objective names used by recorders.

        Returns
        -------
        dict
           Dictionary containing values of each objective.
        """
        if filter:
            objs = filter
        else:
            objs = self._objs

        return {n: self._get_voi_val(n, self._objs[n], self._remote_objs) for n in objs}

    def get_constraint_values(self, ctype='all', lintype='all', filter=None):
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

        filter : list
            List of constraint names used by recorders.

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

            con_dict[name] = self._get_voi_val(name, meta, self._remote_cons)

        return con_dict

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
        with Recording(self._get_name(), self.iter_count, self) as rec:
            failure_flag = self._problem.model._solve_nonlinear()

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
        prob = self._problem

        # Compute the derivatives in dict format...
        if prob.model._owns_approx_jac:
            derivs = prob._compute_totals_approx(of=of, wrt=wrt, return_format='dict',
                                                 global_names=global_names)
        else:
            derivs = prob._compute_totals(of=of, wrt=wrt, return_format='dict',
                                          global_names=global_names)

        # ... then convert to whatever the driver needs.
        if return_format == 'dict':

            for okey, oval in iteritems(derivs):
                for ikey, val in iteritems(oval):

                    imeta = self._designvars[ikey]
                    ometa = self._responses[okey]

                    iscaler = imeta['scaler']
                    oscaler = ometa['scaler']

                    # Scale response side
                    if oscaler is not None:
                        val[:] = (oscaler * val.T).T

                    # Scale design var side
                    if iscaler is not None:
                        val *= 1.0 / iscaler

        elif return_format == 'array':

            # Use sizes pre-computed in derivs for ease
            osize = 0
            isize = 0
            do_wrt = True
            islices = {}
            oslices = {}
            for okey, oval in iteritems(derivs):
                if do_wrt:
                    for ikey, val in iteritems(oval):
                        istart = isize
                        isize += val.shape[1]
                        islices[ikey] = slice(istart, isize)
                    do_wrt = False
                ostart = osize
                osize += oval[ikey].shape[0]
                oslices[okey] = slice(ostart, osize)

            new_derivs = np.zeros((osize, isize))

            relevant = prob.model._relevant

            # Apply driver ref/ref0 and position subjac into array jacobian.
            for okey, oval in iteritems(derivs):
                oscaler = self._responses[okey]['scaler']
                for ikey, val in iteritems(oval):
                    if okey in relevant[ikey] or ikey in relevant[okey]:
                        iscaler = self._designvars[ikey]['scaler']

                        # Scale response side
                        if oscaler is not None:
                            val[:] = (oscaler * val.T).T

                        # Scale design var side
                        if iscaler is not None:
                            val *= 1.0 / iscaler

                        new_derivs[oslices[okey], islices[ikey]] = val

            derivs = new_derivs

        else:
            msg = "Derivative scaling by the driver only supports the 'dict' format at present."
            raise RuntimeError(msg)

        return derivs

    def get_req_procs(self, model):
        """
        Return min and max MPI processes usable by this Driver for the model.

        This should be overridden by Drivers that can use more processes than
        the model uses, e.g., DOEDriver.

        Parameters
        ----------
        model : <System>
            Top level <System> that contains the entire model.

        Returns
        -------
        tuple : (int, int or None)
            A tuple of the form (min_procs, max_procs), indicating the min
            and max processors usable by this `Driver` and the given model.
            max_procs can be None, indicating all available procs can be used.
        """
        return model.get_req_procs()

    def record_iteration(self):
        """
        Record an iteration of the current Driver.
        """
        metadata = create_local_meta(self._get_name())
        self._rec_mgr.record_iteration(self, metadata)

    def _get_name(self):
        """
        Get name of current Driver.

        Returns
        -------
        str
            Name of current Driver.
        """
        return "Driver"
