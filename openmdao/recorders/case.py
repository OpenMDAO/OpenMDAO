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
    prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
    abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    meta : dict
        Dicitonary mapping absolute variable names to variable metadata.
    inputs : PromotedToAbsoluteMap
        Map of inputs to values recorded.
    outputs : PromotedToAbsoluteMap
        Map of outputs to values recorded.
    residuals : PromotedToAbsoluteMap
        Map of outputs to residuals recorded.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg,
                 prom2abs, abs2prom, meta, inputs, outputs, residuals=None):
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
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        meta : dict
            Dicitonary mapping absolute variable names to variable metadata.
        inputs : array
            Inputs to read in from the recording file.
        outputs : array
            Outputs to read in from the recording file.
        residuals : array, optional
            Residuals to read in from the recording file.

        """
        self.filename = filename
        self.counter = counter
        self.iteration_coordinate = iteration_coordinate

        self.timestamp = timestamp
        self.success = success
        self.msg = msg
        self.inputs = None
        self.outputs = None
        self.residuals = None
        self.meta = meta
        self.prom2abs = prom2abs
        self.abs2prom = abs2prom

        if inputs is not None and inputs.dtype.names:
            self.inputs = PromotedToAbsoluteMap(inputs[0], prom2abs, abs2prom, False)
        if outputs is not None and outputs.dtype.names:
            self.outputs = PromotedToAbsoluteMap(outputs[0], prom2abs, abs2prom)
        if residuals is not None and residuals.dtype.names:
            self.residuals = PromotedToAbsoluteMap(residuals[0], prom2abs, abs2prom)

    def get_desvars(self):
        """
        Get the design variables and their values.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of variables to their values.
        """
        return self._get_variables_of_type('desvar')

    def get_objectives(self):
        """
        Get the objective variables and their values.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of variables to their values.
        """
        return self._get_variables_of_type('objective')

    def get_constraints(self):
        """
        Get the constraint variables and their values.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of variables to their values.
        """
        return self._get_variables_of_type('constraint')

    def get_responses(self):
        """
        Get the response variables and their values.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of variables to their values.
        """
        return self._get_variables_of_type('response')

    def _get_variables_of_type(self, var_type):
        """
        Get the variables of a given type and their values.

        Parameters
        ----------
        var_type : str
            String indicating which value for 'type' should be accepted for a variable
            to be included in the returned map.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of variables to their values.
        """
        if self.outputs is None:
            return PromotedToAbsoluteMap({}, self.prom2abs, self.abs2prom)

        ret_vars = {}
        for var in self.outputs._values.dtype.names:
            if var_type in self.meta[var]['type']:
                ret_vars[var] = self.outputs._values[var]
        return PromotedToAbsoluteMap(ret_vars, self.prom2abs, self.abs2prom)


class DriverCase(Case):
    """
    Wrap data from a single iteration of a Driver recording to make it more easily accessible.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success,
                 msg, inputs, outputs, prom2abs, abs2prom, meta):
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
        inputs : array
            Driver inputs to read in from the recording file.
        outputs : array
            Driver outputs to read in from the recording file.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        meta : dict
            Dicitonary mapping absolute variable names to variable metadata.
        """
        super(DriverCase, self).__init__(filename, counter, iteration_coordinate,
                                         timestamp, success, msg, prom2abs,
                                         abs2prom, meta, inputs, outputs)


class SystemCase(Case):
    """
    Wraps data from a single iteration of a System recording to make it more accessible.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg, inputs,
                 outputs, residuals, prom2abs, abs2prom, meta):
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
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        meta : dict
            Dicitonary mapping absolute variable names to variable metadata.
        """
        super(SystemCase, self).__init__(filename, counter, iteration_coordinate,
                                         timestamp, success, msg, prom2abs,
                                         abs2prom, meta, inputs, outputs,
                                         residuals)


class SolverCase(Case):
    """
    Wraps data from a single iteration of a System recording to make it more accessible.

    Attributes
    ----------
    abs_err : array
        Solver absolute error that has been read in from the recording file.
    rel_err : array
        Solver relative error that has been read in from the recording file.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg,
                 abs_err, rel_err, inputs, outputs, residuals, prom2abs, abs2prom, meta):
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
        inputs : array
            Solver inputs to read in from the recording file.
        outputs : array
            Solver outputs to read in from the recording file.
        residuals : array
            Solver residuals to read in from the recording file.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        meta : dict
            Dicitonary mapping absolute variable names to variable metadata.
        """
        super(SolverCase, self).__init__(filename, counter, iteration_coordinate, timestamp,
                                         success, msg, prom2abs, abs2prom, meta,
                                         inputs, outputs, residuals)

        self.abs_err = abs_err
        self.rel_err = rel_err


class PromotedToAbsoluteMap:
    """
    Enables access of values through promoted variable names by mapping to the absolute name.

    Attributes
    ----------
    keys : array
        Array of this map's keys as promoted variable names.
    _values : array
        Array of values accessible via absolute variable name.
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    _is_output : bool
        True if this should map using output variable names, False for input variable names.
    _iteration_index : int
        integer indicating the next key to return when iterating.
    """

    def __init__(self, values, prom2abs, abs2prom, output=True):
        """
        Initialize.

        Parameters
        ----------
        values : array
            Array of values accessible via absolute variable name.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        output : bool
            True if this should map using output variable names, False for input variable names.
        """
        self._values = values
        self._prom2abs = prom2abs
        self._abs2prom = abs2prom
        self._is_output = output
        self._iteration_index = 0

        if self._values is None:
            self.keys = []
        else:
            var_names = list(self._values.keys()) if isinstance(self._values, dict)\
                else self._values.dtype.names
            io = 'output' if output else 'input'
            self.keys = []
            for n in var_names:
                if n in self._abs2prom[io]:
                    self.keys.append(self._abs2prom[io][n])

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

    def __iter__(self):
        """
        Initialize iteration.

        Returns
        -------
        Iterator
            Object which can iterate over map keys.
        """
        self._iteration_index = 0
        return self

    def next(self):
        """
        Get the next key iteration.

        Returns
        -------
        string
            Next key in the keys array.
        """
        if self._iteration_index >= len(self.keys):
            raise StopIteration
        self._iteration_index += 1
        return self.keys[self._iteration_index - 1]

    def __next__(self):
        """
        Get the next key iteration.

        Returns
        -------
        string
            Next key in the keys array.
        """
        return self.next()
