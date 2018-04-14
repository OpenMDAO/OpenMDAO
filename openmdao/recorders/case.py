"""
A Case class.
"""


class Case(object):
    """
    Case wraps the data from a single iteration of a recording to make it more easily accessible.

    Attributes
    ----------
    filename : str
        The file from which the case was loaded.
    counter : int
        The global execution counter.
    iteration_coordinate : str
        The string that holds the full unique identifier for this iteration.
    timestamp : float
        Time of execution of the case.
    success : str
        Success flag for the case.
    msg : str
        Message associated with the case.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The filename from which the Case was constructed.
        counter : int
            The global execution counter.
        iteration_coordinate : str
            The string that holds the full unique identifier for this iteration.
        timestamp : float
            Time of execution of the case.
        success : str
            Success flag for the case.
        msg : str
            Message associated with the case.
        """
        self.filename = filename
        self.counter = counter
        self.iteration_coordinate = iteration_coordinate

        self.timestamp = timestamp
        self.success = success
        self.msg = msg


class DriverCase(Case):
    """
    Wrap data from a single iteration of a Driver recording to make it more easily accessible.

    Attributes
    ----------
    desvars : PromotedToAbsoluteMap
        Driver design variables that have been read in from the recording file.
    responses : PromotedToAbsoluteMap
        Driver responses that have been read in from the recording file.
    objectives : PromotedToAbsoluteMap
        Driver objectives that have been read in from the recording file.
    constraints : PromotedToAbsoluteMap
        Driver constraints that have been read in from the recording file.
    sysincludes : PromotedToAbsoluteMap
        Driver sysincludes that have been read in from the recording file.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg, desvars,
                 responses, objectives, constraints, sysincludes, prom2abs):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The filename from which the DriverCase was constructed.
        counter : int
            The global execution counter.
        iteration_coordinate : str
            The string that holds the full unique identifier for the desired iteration.
        timestamp : float
            Time of execution of the case.
        success : str
            Success flag for the case.
        msg : str
            Message associated with the case.
        desvars : array
            Driver design variables to read in from the recording file.
        responses : array
            Driver responses to read in from the recording file.
        objectives : array
            Driver objectives to read in from the recording file.
        constraints : array
            Driver constraints to read in from the recording file.
        sysincludes : array
            Driver sysincludes to read in from the recording file.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        """
        super(DriverCase, self).__init__(filename, counter, iteration_coordinate,
                                         timestamp, success, msg)

        self.desvars = PromotedToAbsoluteMap(desvars[0], prom2abs)\
            if desvars.dtype.names else None
        self.responses = PromotedToAbsoluteMap(responses[0], prom2abs)\
            if responses.dtype.names else None
        self.objectives = PromotedToAbsoluteMap(objectives[0], prom2abs)\
            if objectives.dtype.names else None
        self.constraints = PromotedToAbsoluteMap(constraints[0], prom2abs)\
            if constraints.dtype.names else None
        self.sysincludes = PromotedToAbsoluteMap(sysincludes[0], prom2abs)\
            if sysincludes.dtype.names else None


class SystemCase(Case):
    """
    Wraps data from a single iteration of a System recording to make it more accessible.

    Attributes
    ----------
    inputs : PromotedToAbsoluteMap
        System inputs that have been read in from the recording file.
    outputs : PromotedToAbsoluteMap
        System outputs that have been read in from the recording file.
    residuals : PromotedToAbsoluteMap
        System residuals that have been read in from the recording file.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg, inputs,
                 outputs, residuals, prom2abs):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The filename from which the SystemCase was constructed.
        counter : int
            The global execution counter.
        iteration_coordinate : str
            The string that holds the full unique identifier for the desired iteration.
        timestamp : float
            Time of execution of the case
        success : str
            Success flag for the case
        msg : str
            Message associated with the case
        inputs : array
            System inputs to read in from the recording file.
        outputs : array
            System outputs to read in from the recording file.
        residuals : array
            System residuals to read in from the recording file.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        """
        super(SystemCase, self).__init__(filename, counter, iteration_coordinate,
                                         timestamp, success, msg)

        self.inputs = PromotedToAbsoluteMap(inputs[0], prom2abs, False) if inputs.dtype.names\
            else None
        self.outputs = PromotedToAbsoluteMap(outputs[0], prom2abs) if outputs.dtype.names\
            else None
        self.residuals = PromotedToAbsoluteMap(residuals[0], prom2abs) if residuals.dtype.names\
            else None


class SolverCase(Case):
    """
    Wraps data from a single iteration of a System recording to make it more accessible.

    Attributes
    ----------
    abs_err : array
        Solver absolute error that has been read in from the recording file.
    rel_err : array
        Solver relative error that has been read in from the recording file.
    outputs : PromotedToAbsoluteMap
        Solver outputs that have been read in from the recording file.
    residuals : PromotedToAbsoluteMap
        Solver residuals that have been read in from the recording file.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg,
                 abs_err, rel_err, outputs, residuals, prom2abs):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The filename from which the SystemCase was constructed.
        counter : int
            The global execution counter.
        iteration_coordinate : str
            The iteration coordinate, in a specific format.
        timestamp : float
            Time of execution of the case
        success : str
            Success flag for the case
        msg : str
            Message associated with the case
        abs_err : array
            Solver absolute error to read in from the recording file.
        rel_err : array
            Solver relative error to read in from the recording file.
        outputs : array
            Solver outputs to read in from the recording file.
        residuals : array
            Solver residuals to read in from the recording file.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        """
        super(SolverCase, self).__init__(filename, counter, iteration_coordinate, timestamp,
                                         success, msg)

        self.abs_err = abs_err
        self.rel_err = rel_err
        self.outputs = PromotedToAbsoluteMap(outputs[0], prom2abs) if outputs.dtype.names else None
        self.residuals = PromotedToAbsoluteMap(residuals[0], prom2abs) if residuals.dtype.names\
            else None


class PromotedToAbsoluteMap:
    """
    Enables access of values through promoted variable names by mapping to the absolute name.

    Attributes
    ----------
    _values : array
        Array of values accessible via absolute variable name.
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    _is_output : bool
        True if this should map using output variable names, False for input variable names.
    """

    def __init__(self, values, prom2abs, output=True):
        """
        Initialize.

        Parameters
        ----------
        values : array
            Array of values accessible via absolute variable name.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        output : bool
            True if this should map using output variable names, False for input variable names.
        """
        self._values = values
        self._prom2abs = prom2abs
        self._is_output = output

    def __getitem__(self, key):
        """
        Use the variable name to get the corresponding value.

        Parameters
        ----------
        key : string
            variable name.

        Returns
        -------
        array :
            An array entry value that corresponds to the given variable name.
        """
        var_names = list(self._values.keys()) if isinstance(self._values, dict)\
            else self._values.dtype.names

        # user trying to access via absolute name rather than promoted
        if '.' in key:
            if key in var_names:
                return self._values[key]

        # outputs only have one option in _prom2abs
        if self._is_output:
            return self._values[self._prom2abs['output'][key][0]]

        # inputs may have multiple options, so we try until we succeed
        for k in self._prom2abs['input'][key]:
            if k in var_names:
                return self._values[k]

        raise ValueError("no field of name " + key)
