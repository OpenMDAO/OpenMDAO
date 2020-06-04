"""Exception raised by grid interpolators when they go out of bounds."""


class OutOfBoundsError(Exception):
    """
    Handles error when interpolated values are requested outside of the domain of the input data.

    Attributes
    ----------
    idx : int
        index of the variable that is out of bounds.
    value : double
        value of the variable that is out of bounds.
    lower : double
        lower bounds of the variable that is out of bounds.
    upper : double
        upper bounds of the variable that is out of bounds.
    """

    def __init__(self, message, idx, value, lower, upper):
        """
        Initialize instance of OutOfBoundsError class.

        Parameters
        ----------
        message : str
            description of error.
        idx : int
            index of the variable that is out of bounds.
        value : double
            value of the variable that is out of bounds.
        lower : double
            lower bounds of the variable that is out of bounds.
        upper : double
            upper bounds of the variable that is out of bounds.
        """
        super(OutOfBoundsError, self).__init__(message)
        self.idx = idx
        self.value = value
        self.lower = lower
        self.upper = upper
