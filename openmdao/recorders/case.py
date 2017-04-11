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

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg, desvars_array, responses_array, objectives_array, constraints_array):
        """
        Initialize.
        """
        self.filename = filename
        self.counter = counter
        self.iteration_coordinate = iteration_coordinate

        self.timestamp = timestamp
        self.success = success
        self.msg = msg

        self.desvars = desvars_array[0] if desvars_array else None
        self.responses = responses_array[0] if responses_array else None
        self.objectives = objectives_array[0] if objectives_array else None
        self.constraints = constraints_array[0] if constraints_array else None

    def __getitem__(self, item):
        """
        Access an unknown of the given name.

        This is intended to be a convenient shorthand for case.unknowns[item].
        """
        if self.unknowns is None:
            raise ValueError('No unknowns are available'
                             ' in file {0}'.format(self.filename))
        return self.unknowns[item]
