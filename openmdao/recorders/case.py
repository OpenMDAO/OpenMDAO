"""
A Case class.
"""

class Case(object):
    """
    Case wraps the data from a single iteration of a recording to make it more easily accessible.

    Parameters
    ----------
    filename : str
        The filename from which the Case was constructed.
    case_id : str
        The identifier string associated with the Case.
    case_dict : dict
        A dictionary containing fields for the parameters, unknowns,
        derivatives, and residuals in the Case, as well as a Case
        timestamp, success flag, and string message.

    Attributes
    ----------
    filename : str
        The file from which the case was loaded
    case_id : str
        The identifier of the case/iteration in the case recorder
    timestamp : str
        Time of execution of the case
    success : str
        Success flag for the case
    msg : str
        Message associated with the case
    parameters : dict
        Parameters in the case.  Keyed by parameter path name, values are
        float or dict.
    unknowns : dict
        Unknowns in the case.  Keyed by variable path name, values are
        float or dict.
    derivs : dict
        Derivatives in the case.  Different optimizers store the derivatives
        in different ways.
    resids : dict
        Residuals in the case.  Keyed by variable path name, values are
        float or dict.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg ):
        """
        Initialize.
        """
        self.filename = filename
        self.counter = counter
        self.iteration_coordinate = iteration_coordinate

        self.timestamp = timestamp
        self.success = success
        self.msg = msg

class DriverCase(Case):
    """
    Case wraps the data from a single iteration of a recording to make it more easily accessible.

    Parameters
    ----------
    filename : str
        The filename from which the Case was constructed.
    case_id : str
        The identifier string associated with the Case.
    case_dict : dict
        A dictionary containing fields for the parameters, unknowns,
        derivatives, and residuals in the Case, as well as a Case
        timestamp, success flag, and string message.

    Attributes
    ----------
    filename : str
        The file from which the case was loaded
    case_id : str
        The identifier of the case/iteration in the case recorder
    timestamp : str
        Time of execution of the case
    success : str
        Success flag for the case
    msg : str
        Message associated with the case
    parameters : dict
        Parameters in the case.  Keyed by parameter path name, values are
        float or dict.
    unknowns : dict
        Unknowns in the case.  Keyed by variable path name, values are
        float or dict.
    derivs : dict
        Derivatives in the case.  Different optimizers store the derivatives
        in different ways.
    resids : dict
        Residuals in the case.  Keyed by variable path name, values are
        float or dict.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg, desvars, responses, objectives, constraints ):
        """
        Initialize.
        """
        super(DriverCase, self).__init__(filename, counter, iteration_coordinate, timestamp, success, msg )

        self.desvars = desvars[0] if desvars.dtype.names else None
        self.responses = responses[0] if responses.dtype.names else None
        self.objectives = objectives[0] if objectives.dtype.names else None
        self.constraints = constraints[0] if constraints.dtype.names else None

class SystemCase(Case):
    """
    Case wraps the data from a single iteration of a recording to make it more easily accessible.

    Parameters
    ----------
    filename : str
        The filename from which the Case was constructed.
    case_id : str
        The identifier string associated with the Case.
    case_dict : dict
        A dictionary containing fields for the parameters, unknowns,
        derivatives, and residuals in the Case, as well as a Case
        timestamp, success flag, and string message.

    Attributes
    ----------
    filename : str
        The file from which the case was loaded
    case_id : str
        The identifier of the case/iteration in the case recorder
    timestamp : str
        Time of execution of the case
    success : str
        Success flag for the case
    msg : str
        Message associated with the case
    parameters : dict
        Parameters in the case.  Keyed by parameter path name, values are
        float or dict.
    unknowns : dict
        Unknowns in the case.  Keyed by variable path name, values are
        float or dict.
    derivs : dict
        Derivatives in the case.  Different optimizers store the derivatives
        in different ways.
    resids : dict
        Residuals in the case.  Keyed by variable path name, values are
        float or dict.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg, inputs , outputs , residuals ):
        """
        Initialize.
        """
        super(SystemCase, self).__init__(filename, counter, iteration_coordinate, timestamp, success, msg)

        self.inputs = inputs[0] if inputs.dtype.names else None
        self.outputs = outputs[0] if outputs.dtype.names else None
        self.residuals = residuals[0] if residuals.dtype.names else None


class SolverCase(Case):
    """
    Case wraps the data from a single iteration of a recording to make it more easily accessible.

    Parameters
    ----------
    filename : str
        The filename from which the Case was constructed.
    case_id : str
        The identifier string associated with the Case.
    case_dict : dict
        A dictionary containing fields for the parameters, unknowns,
        derivatives, and residuals in the Case, as well as a Case
        timestamp, success flag, and string message.

    Attributes
    ----------
    filename : str
        The file from which the case was loaded
    case_id : str
        The identifier of the case/iteration in the case recorder
    timestamp : str
        Time of execution of the case
    success : str
        Success flag for the case
    msg : str
        Message associated with the case
    parameters : dict
        Parameters in the case.  Keyed by parameter path name, values are
        float or dict.
    unknowns : dict
        Unknowns in the case.  Keyed by variable path name, values are
        float or dict.
    derivs : dict
        Derivatives in the case.  Different optimizers store the derivatives
        in different ways.
    resids : dict
        Residuals in the case.  Keyed by variable path name, values are
        float or dict.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg,
                    abs_err, rel_err, outputs, residuals ):
        """
        Initialize.
        """
        super(SolverCase, self).__init__(filename, counter, iteration_coordinate, timestamp, success, msg)

        self.abs_err = abs_err
        self.rel_err = rel_err
        self.outputs = outputs[0] if outputs.dtype.names else None
        self.residuals = residuals[0] if residuals.dtype.names else None