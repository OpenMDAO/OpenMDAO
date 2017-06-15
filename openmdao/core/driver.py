"""Define a base class for all Drivers in OpenMDAO."""
import numpy as np

from openmdao.devtools.problem_viewer.problem_viewer import _get_viewer_data
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.utils.record_util import create_local_meta, update_local_meta

from openmdao.utils.options_dictionary import OptionsDictionary
from six import iteritems


class Driver(object):
    """
    Top-level container for the systems and drivers.

    Attributes
    ----------
    fail : bool
        Reports whether the driver ran successfully.
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
        object that manages all recorders added to this driver
    _model_viewer_data : dict
        structure of model, used to make n2 diagram.
    iter_count : int
        keep track of iterations for case recording
    metadata : list
        list of metadata
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

        # Gather up the information for design vars.
        self._designvars = model.get_design_vars(recurse=True)

        self._responses = model.get_responses(recurse=True)
        self._objs = model.get_objectives(recurse=True)
        self._cons = model.get_constraints(recurse=True)

        self._rec_mgr.startup(self)
        if (self._rec_mgr._recorders):
            self._model_viewer_data = _get_viewer_data(problem)
        self._rec_mgr.record_metadata(self)

    def get_design_var_values(self, filter=None):
        """
        Return the design variable values.

        This is called to gather the initial design variable state.

        Parameters
        ----------
        filter : list
            List of desvar names used by recorders to
            filter by includes/excludes.

        Returns
        -------
        dict
           Dictionary containing values of each design variable.
        """
        designvars = {}

        if filter:
            # pull out designvars of those names into filtered dict.
            for inc in filter:
                designvars[inc] = self._designvars[inc]

        else:
            # use all the designvars
            designvars = self._designvars

        vec = self._problem.model._outputs._views_flat
        dv_dict = {}
        for name, meta in iteritems(designvars):
            scaler = meta['scaler']
            adder = meta['adder']
            indices = meta['indices']
            if indices is None:
                val = vec[name].copy()
            else:
                val = vec[name][indices]

            # Scale design variable values
            if adder is not None:
                val += adder
            if scaler is not None:
                val *= scaler

            dv_dict[name] = val

        return dv_dict

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
        meta = self._designvars[name]
        scaler = meta['scaler']
        adder = meta['adder']
        indices = meta['indices']
        if indices is None:
            indices = slice(None)

        desvar = self._problem.model._outputs._views_flat[name]
        desvar[indices] = value

        # Scale design variable values
        if scaler is not None:
            desvar[indices] *= 1.0 / scaler
        if adder is not None:
            desvar[indices] -= adder

    def get_response_values(self, filter=None):
        """
        Return response values.

        Parameters
        ----------
        filter : list
            List of response names used by recorders to
            filter by includes/excludes.

        Returns
        -------
        dict
           Dictionary containing values of each response.
        """
        # TODO: finish this method when we have a driver that requires it.
        pass

    def get_objective_values(self, filter=None):
        """
        Return objective values.

        Parameters
        ----------
        filter : list
            List of objective names used by recorders to
            filter by includes/excludes.

        Returns
        -------
        dict
           Dictionary containing values of each objective.
        """
        objectives = {}

        if filter:
            # pull out objectives of those names into filtered dict.
            for inc in filter:
                objectives[inc] = self._objs[inc]

        else:
            # use all the objectives
            objectives = self._objs

        vec = self._problem.model._outputs._views_flat
        obj_dict = {}
        for name, meta in iteritems(objectives):
            scaler = meta['scaler']
            adder = meta['adder']
            indices = meta['indices']
            if indices is None:
                val = vec[name].copy()
            else:
                val = vec[name][indices]

            # Scale objectives
            if adder is not None:
                val += adder
            if scaler is not None:
                val *= scaler

            obj_dict[name] = val

        return obj_dict

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
            List of objective names used by recorders to
            filter by includes/excludes.

        Returns
        -------
        dict
           Dictionary containing values of each constraint.
        """
        constraints = {}

        if filter is not None:
            # pull out objectives of those names into filtered dict.
            for inc in filter:
                constraints[inc] = self._cons[inc]

        else:
            # use all the objectives
            constraints = self._cons

        vec = self._problem.model._outputs._views_flat
        con_dict = {}

        for name, meta in iteritems(constraints):

            if lintype == 'linear' and meta['linear'] is False:
                continue

            if lintype == 'nonlinear' and meta['linear']:
                continue

            if ctype == 'eq' and meta['equals'] is None:
                continue

            if ctype == 'ineq' and meta['equals'] is not None:
                continue

            scaler = meta['scaler']
            adder = meta['adder']
            indices = meta['indices']

            if indices is None:
                val = vec[name].copy()
            else:
                val = vec[name][indices]

            # Scale objectives
            if adder is not None:
                val += adder
            if scaler is not None:
                val *= scaler

            # TODO: Need to get the allgathered values? Like:
            # cons[name] = self._get_distrib_var(name, meta, 'constraint')
            con_dict[name] = val
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
        # Metadata Setup
        self.iter_count += 1
        metadata = self.metadata = create_local_meta(None, 'Driver')
        update_local_meta(metadata, (self.iter_count,))

        from openmdao.recorders.base_recorder import push_recording_iteration_stack, \
            print_recording_iteration_stack, pop_recording_iteration_stack, \
            iter_get_norm_on_call_stack
        push_recording_iteration_stack('Driver', self.iter_count)

        failure_flag = self._problem.model._solve_nonlinear()
        self._rec_mgr.record_iteration(self, metadata)

        print_recording_iteration_stack()
        pop_recording_iteration_stack()

        return failure_flag

    def _compute_total_derivs(self, of=None, wrt=None, return_format='flat_dict',
                              global_names=True):
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

        if return_format == 'dict':

            derivs = prob._compute_total_derivs(of=of, wrt=wrt, return_format=return_format,
                                                global_names=global_names)

            for okey, oval in iteritems(derivs):
                for ikey, val in iteritems(oval):

                    imeta = self._designvars[ikey]
                    ometa = self._responses[okey]

                    iscaler = imeta['scaler']
                    iadder = imeta['adder']
                    oscaler = ometa['scaler']
                    oadder = ometa['adder']

                    # Scale response side
                    if oadder is not None:
                        val += oadder
                    if oscaler is not None:
                        val *= oscaler

                    # Scale design var side
                    if iscaler is not None:
                        val *= 1.0 / iscaler
                    if iadder is not None:
                        val -= iadder

        elif return_format == 'array':

            # Compute the derivatives in dict format, and then convert to array.
            derivs = prob._compute_total_derivs(of=of, wrt=wrt, return_format='dict',
                                                global_names=global_names)

            # Use sizes pre-computed in derivs for ease
            osize = 0
            isize = 0
            do_wrt = True
            Jslices = {}
            for okey, oval in iteritems(derivs):
                if do_wrt:
                    for ikey, val in iteritems(oval):
                        istart = isize
                        isize += val.shape[1]
                        Jslices[ikey] = slice(istart, isize)

                do_wrt = False
                ostart = osize
                osize += oval[ikey].shape[0]
                Jslices[okey] = slice(ostart, osize)

            new_derivs = np.zeros((osize, isize))

            # Apply driver ref/ref0 and position subjac into array jacobian.
            for okey, oval in iteritems(derivs):
                for ikey, val in iteritems(oval):

                    imeta = self._designvars[ikey]
                    ometa = self._responses[okey]

                    iscaler = imeta['scaler']
                    iadder = imeta['adder']
                    oscaler = ometa['scaler']
                    oadder = ometa['adder']

                    # Scale response side
                    if oadder is not None:
                        val += oadder
                    if oscaler is not None:
                        val *= oscaler

                    # Scale design var side
                    if iscaler is not None:
                        val *= 1.0 / iscaler
                    if iadder is not None:
                        val -= iadder

                    new_derivs[Jslices[okey], Jslices[ikey]] = val

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
