"""Define a base class for all Drivers in OpenMDAO."""

from six import iteritems

from openmdao.utils.generalized_dict import OptionsDictionary


class Driver(object):
    """
    Top-level container for the systems and drivers.

    Attributes
    ----------
    options : <OptionsDictionary>
        Dictionary with general pyoptsparse options.
    problem : <Problem>
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
    """

    def __init__(self):
        """Initialize the driver."""
        self.problem = None
        self._designvars = None
        self._cons = None
        self._objs = None
        self._responses = None
        self.options = OptionsDictionary()

        # What the driver supports.
        # Note Driver based class supports setting up problems that use every
        # feature, but it doesn't do anything except run the model. This is
        # primarilly for generic testing.
        self.supports = OptionsDictionary()#read_only=True)
        self.supports.declare('inequality_constraints', type_=bool, value=True)
        self.supports.declare('equality_constraints', type_=bool, value=True)
        self.supports.declare('linear_constraints', type_=bool, value=True)
        self.supports.declare('two_sided_constraints', type_=bool, value=True)
        self.supports.declare('multiple_objectives', type_=bool, value=True)
        self.supports.declare('integer_design_vars', type_=bool, value=True)
        self.supports.declare('gradients', type_=bool, value=True)
        self.supports.declare('active_set', type_=bool, value=True)

        # TODO, support these in Openmdao blue
        # self.supports.declare('linear_constraints', True)
        # self.supports.declare('integer_design_vars', True)

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <`Problem`>
            Pointer to the containing problem.
        """
        self.problem = problem
        model = problem.model

        # Gather up the information for design vars.
        self._designvars = model.get_design_vars(recurse=True)

        self._responses = model.get_responses(recurse=True)
        self._objs = model.get_objectives(recurse=True)
        self._cons = model.get_constraints(recurse=True)

    def get_design_var_values(self):
        """
        Return the design variable values.

        This is called to gather the initial design variable state.

        Returns
        -------
        dict
           Dictionary containing values of each design variable.
        """
        dvs = self._designvars
        vec = self.problem.model._outputs
        dv_dict = {}
        for name in dvs:

            # TODO: use dv scaling

            dv_dict[name] = vec[name]
        return dv_dict

    def set_design_var(self, name, value):
        """
        Sets the value of a design variable.

        Parameters
        ----------
        name : str
            Global pathname of the design variable.
        value : float or ndarray
            Value for the design variable.
        """
        self.problem.model._outputs[name] = value

    def get_response_values(self):
        """
        Return response values.

        Returns
        -------
        dict
           Dictionary containing values of each response.
        """
        # TODO: finish this method when we have a driver that requires is.
        pass

    def get_objective_values(self):
        """
        Return objective values.

        Returns
        -------
        dict
           Dictionary containing values of each objective.
        """
        objs = self._objs
        vec = self.problem.model._outputs
        obj_dict = {}
        for name in objs:
            obj_dict[name] = vec[name]

        return obj_dict

    def get_constraint_values(self, ctype='all', lintype='all'):
        """
        Return constraint values.

        Args
        ----
        ctype : string
            Default is 'all'. Optionally return just the inequality constraints
            with 'ineq' or the equality constraints with 'eq'.

        lintype : string
            Default is 'all'. Optionally return just the linear constraints
            with 'linear' or the nonlinear constraints with 'nonlinear'.

        Returns
        -------
        dict
           Dictionary containing values of each constraint.
        """
        cons = self._cons
        vec = self.problem.model._outputs
        con_dict = {}

        for name, meta in iteritems(self._cons):

            if lintype == 'linear' and meta['linear'] is False:
                continue

            if lintype == 'nonlinear' and meta['linear']:
                continue

            if ctype == 'eq' and meta['equals'] is None:
                continue

            if ctype == 'ineq' and meta['equals'] is not None:
                continue

            # TODO: Need to get the allgathered values? Like:
            # cons[name] = self._get_distrib_var(name, meta, 'constraint')
            con_dict[name] = vec[name]

        return con_dict

    def get_total_derivatives(self, return_format='dict'):
        """
        Return the derivatives.

        These derivatives are of the responses with respect to the design vars.

        Parameters
        ----------
        return_format : string
            Format for the derivatives. Default is 'string'.
        """
        pass

    def run(self):
        """
        Execute this driver.

        The base `Driver` just runs the model. All other drivers overload
        this method.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        return self.problem.model._solve_nonlinear()
