"""Define a base class for all Drivers in OpenMDAO."""


class Driver(object):
    """Top-level container for the systems and drivers.

    Attributes
    ----------
    problem : <Problem>
        Pointer to the containing problem.
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

    def _setup_driver(self, problem):
        """Prepare the driver for execution.

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

        # Error checking is only possible now.
        for name in self._designvars:
            if name not in model._outputs:
                msg = "Output not found for design variable '{0}'."
                raise RuntimeError(msg.format(name))


        self._responses = model.get_responses(recurse=True)
        self._objs = model.get_objectives(recurse=True)
        self._cons = model.get_constraints(recurse=True)

    def get_design_var_values(self):
        """Return design variable values.

        Returns
        -------
        dict
           Dictionary containing values of each design variable.
        """
        dvs = self._designvars
        vec = self.problem.model._outputs
        dv_dict = {}
        for name in dvs:
            dv_dict[name] = vec[name]
        return dv_dict

    def get_response_values(self):
        """Return response values.

        Returns
        -------
        dict
           Dictionary containing values of each response.
        """
        pass

    def get_objective_values(self):
        """Return objective values.

        Returns
        -------
        dict
           Dictionary containing values of each objective.
        """
        pass

    def get_constraint_values(self):
        """Return constraint values.

        Returns
        -------
        dict
           Dictionary containing values of each constraint.
        """
        pass

    def get_total_derivatives(self, return_format='dict'):
        """Return the derivatives.

        These derivatives are of the responses with respect to the design vars.

        Parameters
        ----------
        return_format : string
            Format for the derivatives. Default is 'string'.
        """
        pass

    def run(self):
        """Execute this driver.

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
