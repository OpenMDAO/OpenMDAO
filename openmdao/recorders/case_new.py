"""
A Case class.
"""

import re
import itertools

_DEFAULT_OUT_STREAM = object()


class Case(object):
    """
    Case wraps the data from a single iteration of a recording to make it more easily accessible.

    Attributes
    ----------
        source : str
            The unique id of the system/solver/driver/problem that did the recording.
        iteration_coordinate : str
            The full unique identifier for this iteration.
        timestamp : float
            Time of execution of the case.
        success : str
            Success flag for the case.
        msg : str
            Message associated with the case.
        outputs : PromotedToAbsoluteMap
            Map of outputs to values recorded.
        inputs : PromotedToAbsoluteMap or None
            Map of inputs to values recorded (None if not recorded).
        residuals : PromotedToAbsoluteMap or None
            Map of outputs to residuals recorded (None if not recorded).
        jacobian : PromotedToAbsoluteMap or None
            Map of (output, input) to derivatives recorded (None if not recorded).
        parent : str
            The full unique identifier for the parent this iteration.
        children : list
            The full unique identifiers for children of this iteration.
        abs_tol : float or None
            Absolute tolerance (None if not recorded).
        rel_tol : float or None
            Relative tolerance (None if not recorded).
    """

    def __init__(self, source, iteration_coordinate, timestamp, success, msg,
                 outputs, inputs=None, residuals=None, jacobian=None,
                 parent=None, children=None, abs_tol=None, rel_tol=None,
                 prom2abs=None, abs2prom=None):
        """
        Initialize.

        Parameters
        ----------
        source : str
            The unique id of the system/solver/driver/problem that did the recording.
        iteration_coordinate : str
            The full unique identifier for this iteration.
        timestamp : float
            Time of execution of the case.
        success : str
            Success flag for the case.
        msg : str
            Message associated with the case.
        inputs : array
            Inputs as read from the recording file.
        outputs : array
            Outputs as read from the recording file.
        residuals : PromotedToAbsoluteMap or None
            Map of outputs to residuals recorded (None if not recorded).
        jacobian : PromotedToAbsoluteMap or None
            Map of (output, input) to derivatives recorded (None if not recorded).
        parent : str
            The full unique identifier for the parent this iteration.
        children : list
            The full unique identifiers for children of this iteration.
        abs_tol : float or None
            Absolute tolerance for solver. (Solver cases only, None if not recorded).
        rel_tol : float or None
            Relative tolerance for solver. (Solver cases only, None if not recorded).
        """
        self.source = source
        self.iteration_coordinate = iteration_coordinate
        self.timestamp = timestamp
        self.success = success
        self.msg = msg

        self.jacobian = jacobian
        self.parent = parent
        self.children = children
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

        if inputs is not None and inputs.dtype.names:
            self.inputs = PromotedToAbsoluteMap(inputs[0], prom2abs, abs2prom, output=False)
        if outputs is not None and outputs.dtype.names:
            self.outputs = PromotedToAbsoluteMap(outputs[0], prom2abs, abs2prom)
        if residuals is not None and residuals.dtype.names:
            self.residuals = PromotedToAbsoluteMap(residuals[0], prom2abs, abs2prom)

    def get_desvars(self, scaled=True, use_indices=True):
        """
        Get the values of the design variables, as seen by the driver, for this case.

        Parameters
        ----------
        scaled : bool
            The unique id of the system/solver/driver/problem that did the recording.
        use_indices : bool
            The full unique identifier for this iteration.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of variables to their values.
        """
        return self._get_variables_of_type('desvar')

    def get_objectives(self, scaled=True, use_indices=True):
        """
        Get the values of the objectives, as seen by the driver, for this case.

        Parameters
        ----------
        scaled : bool
            The unique id of the system/solver/driver/problem that did the recording.
        use_indices : bool
            The full unique identifier for this iteration.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of variables to their values.
        """
        return self._get_variables_of_type('objective')

    def get_constraints(self, scaled=True, use_indices=True):
        """
        Get the values of the constraints, as seen by the driver, for this case.

        Parameters
        ----------
        scaled : bool
            The unique id of the system/solver/driver/problem that did the recording.
        use_indices : bool
            The full unique identifier for this iteration.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of variables to their values.
        """
        return self._get_variables_of_type('constraint')

    def get_responses(self, scaled=True, use_indices=True):
        """
        Get the values of the responses, as seen by the driver, for this case.

        Parameters
        ----------
        scaled : bool
            The unique id of the system/solver/driver/problem that did the recording.
        use_indices : bool
            The full unique identifier for this iteration.

        Returns
        -------
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
        for var in self.outputs.absolute_names():
            if var_type in self.meta[var]['type']:
                ret_vars[var] = self.outputs[var]

        return PromotedToAbsoluteMap(ret_vars, self.prom2abs, self.abs2prom)

    def list_inputs(self,
                    values=True,
                    units=False,
                    hierarchical=True,
                    print_arrays=False,
                    out_stream=_DEFAULT_OUT_STREAM):
        """
        Return and optionally log a list of input names and other optional information.

        If the model is parallel, only the local variables are returned to the process.
        Also optionally logs the information to a user defined output stream. If the model is
        parallel, the rank 0 process logs information about all variables across all processes.

        Parameters
        ----------
        values : bool, optional
            When True, display/return input values. Default is True.
        units : bool, optional
            When True, display/return units. Default is False.
        hierarchical : bool, optional
            When True, human readable output shows variables in hierarchical format.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format is affected
            by the values set with numpy.set_printoptions
            Default is False.
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list
            list of input names and other optional information about those inputs
        """
        pass

    def list_outputs(self,
                     explicit=True, implicit=True,
                     values=True,
                     prom_name=False,
                     residuals=False,
                     residuals_tol=None,
                     units=False,
                     shape=False,
                     bounds=False,
                     scaling=False,
                     hierarchical=True,
                     print_arrays=False,
                     out_stream=_DEFAULT_OUT_STREAM):
        """
        Return and optionally log a list of output names and other optional information.

        If the model is parallel, only the local variables are returned to the process.
        Also optionally logs the information to a user defined output stream. If the model is
        parallel, the rank 0 process logs information about all variables across all processes.

        Parameters
        ----------
        explicit : bool, optional
            include outputs from explicit components. Default is True.
        implicit : bool, optional
            include outputs from implicit components. Default is True.
        values : bool, optional
            When True, display/return output values. Default is True.
        prom_name : bool, optional
            When True, display/return the promoted name of the variable.
            Default is False.
        residuals : bool, optional
            When True, display/return residual values. Default is False.
        residuals_tol : float, optional
            If set, limits the output of list_outputs to only variables where
            the norm of the resids array is greater than the given 'residuals_tol'.
            Default is None.
        units : bool, optional
            When True, display/return units. Default is False.
        shape : bool, optional
            When True, display/return the shape of the value. Default is False.
        bounds : bool, optional
            When True, display/return bounds (lower and upper). Default is False.
        scaling : bool, optional
            When True, display/return scaling (ref, ref0, and res_ref). Default is False.
        hierarchical : bool, optional
            When True, human readable output shows variables in hierarchical format.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format  is affected
            by the values set with numpy.set_printoptions
            Default is False.
        out_stream : file-like
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list
            list of output names and other optional information about those outputs
        """
        pass


class DriverCase(Case):
    """
    Wrap data from a single iteration of a Driver recording to make it more easily accessible.

    Attributes
    ----------
    _var_settings : dict
        Dictionary mapping absolute variable names to variable settings.
    """

    def __init__(self, source, iteration_coordinate, timestamp, success, msg,
                 outputs, inputs, prom2abs, abs2prom, var_settings):
        """
        Initialize.

        Parameters
        ----------
        source : str
            The unique id of the system/solver/driver/problem that did the recording.
        iteration_coordinate : str
            The full unique identifier for this iteration.
        timestamp : float
            Time of execution of the case.
        success : str
            Success flag for the case.
        msg : str
            Message associated with the case.
        inputs : array
            Inputs as read from the recording file.
        outputs : array
            Outputs as read from the recording file.
        var_settings : dict
            Dictionary mapping absolute variable names to variable settings.
        """
        super(DriverCase, self).__init__(iteration_coordinate, timestamp, success, msg,
                                         outputs, inputs, prom2abs=prom2abs, abs2prom=abs2prom)
        self._var_settings = var_settings

    def scale(self):
        """
        Scale the outputs array using _var_settings.
        """
        for name in self.outputs.absolute_names():
            if name in self._var_settings:
                # physical to scaled
                if self._var_settings[name]['adder'] is not None:
                    self.outputs[name] += self._var_settings[name]['adder']
                if self._var_settings[name]['scaler'] is not None:
                    self.outputs[name] *= self._var_settings[name]['scaler']


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
            Dictionary mapping absolute variable names to variable metadata.
        """
        super(SystemCase, self).__init__(filename, counter, iteration_coordinate,
                                         timestamp, success, msg, prom2abs,
                                         abs2prom, meta, inputs, outputs,
                                         residuals=residuals)


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
            Dictionary mapping absolute variable names to variable metadata.
        """
        super(SolverCase, self).__init__(filename, counter, iteration_coordinate, timestamp,
                                         success, msg, prom2abs, abs2prom, meta,
                                         inputs, outputs, residuals=residuals)

        self.abs_err = abs_err
        self.rel_err = rel_err


class ProblemCase(Case):
    """
    Wraps data from a single case of a Problem recording to make it more accessible.
    """

    def __init__(self, filename, counter, case_name, timestamp, success, msg,
                 outputs, prom2abs, abs2prom, meta):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The filename from which the SystemCase was constructed.
        counter : int
            The global execution counter.
        case_name : str
            Name used to identify this Problem case.
        timestamp : float
            Time of execution of the case
        success : str
            Success flag for the case
        msg : str
            Message associated with the case
        outputs : array
            Solver outputs to read in from the recording file.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        meta : dict
            Dictionary mapping absolute variable names to variable metadata.
        """
        super(ProblemCase, self).__init__(filename, counter, case_name, timestamp,
                                          success, msg, prom2abs, abs2prom, meta,
                                          None, outputs)


class PromotedToAbsoluteMap(dict):
    """
    A dictionary that enables accessing values via absolute or promoted variable names.

    Attributes
    ----------
    _values : array or dict
        Array or dict of values accessible via absolute variable name.
    _keys : array
        Absolute variable names that map to the values in the _values array.
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    _is_output : bool
        True if this should map using output variable names, False for input variable names.
    """

    def __init__(self, values, prom2abs, abs2prom, output=True):
        """
        Initialize.

        Parameters
        ----------
        values : array or dict
            Numpy structured array or dictionary of values.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        output : bool
            True if this should map using output variable names, False for input variable names.
        """
        super(PromotedToAbsoluteMap, self).__init__()

        self._is_output = output

        self._prom2abs = prom2abs
        self._abs2prom = abs2prom

        if output:
            prom2abs = self._prom2abs['output']
            abs2prom = self._abs2prom['output']
        else:
            prom2abs = self._prom2abs['input']
            abs2prom = self._abs2prom['input']

        if isinstance(values, dict):
            # dict of values, keyed on either absolute or promoted names
            self._values = {}
            for key in values.keys():
                if isinstance(key, tuple) or ',' in key:
                    # derivative keys can be either (of, wrt) or 'of,wrt'
                    abs_keys, prom_key = self._deriv_keys(key)
                    for abs_key in abs_keys:
                        self._values[abs_key] = values[key]
                    super(PromotedToAbsoluteMap, self).__setitem__(prom_key, values[key])
                else:
                    if key in abs2prom:
                        # key is absolute name
                        self._values[key] = values[key]
                        prom_key = abs2prom[key]
                        super(PromotedToAbsoluteMap, self).__setitem__(prom_key, values[key])
                    elif key in prom2abs:
                        # key is promoted name
                        for abs_key in prom2abs[key]:
                            self._values[abs_key] = values[key]
                        super(PromotedToAbsoluteMap, self).__setitem__(key, values[key])
            self._keys = self._values.keys()
        else:
            # numpy structured array, which will always use absolute names
            self._values = values
            self._keys = values.dtype.names
            for key in self._keys:
                if key in abs2prom:
                    prom_key = abs2prom[key]
                    super(PromotedToAbsoluteMap, self).__setitem__(prom_key, values[key])
                elif ',' in key:
                    # derivative keys will be a string in the form of 'of,wrt'
                    abs_keys, prom_key = self._deriv_keys(key)
                    super(PromotedToAbsoluteMap, self).__setitem__(prom_key, values[key])

    def _deriv_keys(self, key):
        """
        Get the absolute and promoted name versions of the provided derivative key.

        Parameters
        ----------
        key : tuple or string
            derivative key as either (of, wrt) or 'of,wrt'.

        Returns
        -------
        list of tuples:
            list of (of, wrt) mapping the provided key to absolute names.
        tuple :
            (of, wrt) mapping the provided key to promoted names.
        """
        prom2abs = self._prom2abs['output']
        abs2prom = self._abs2prom['output']

        # derivative could be tuple or string, using absolute or promoted names
        if isinstance(key, tuple):
            of, wrt = key
        else:
            of, wrt = re.sub('[( )]', '', key).split(',')

        # if promoted, will map to all connected absolute names
        abs_of = [of] if of in abs2prom else prom2abs[of]
        abs_wrt = [wrt] if wrt in abs2prom else prom2abs[wrt]
        abs_keys = ['%s,%s' % (o, w) for o, w in itertools.product(abs_of, abs_wrt)]

        prom_of = of if of in prom2abs else abs2prom[of]
        prom_wrt = wrt if wrt in prom2abs else abs2prom[wrt]
        prom_key = (prom_of, prom_wrt)

        return abs_keys, prom_key

    def __getitem__(self, key):
        """
        Use the variable name to get the corresponding value.

        Parameters
        ----------
        key : string
            Absolute or promoted variable name.

        Returns
        -------
        array :
            An array entry value that corresponds to the given variable name.
        """
        if key in self._keys:
            # absolute name
            return self._values[key]

        elif key in self:
            # promoted name
            return super(PromotedToAbsoluteMap, self).__getitem__(key)

        elif isinstance(key, tuple) or ',' in key:
            # derivative keys can be either (of, wrt) or 'of,wrt'
            abs_keys, prom_key = self._deriv_keys(key)
            return super(PromotedToAbsoluteMap, self).__getitem__(prom_key)

        raise KeyError(key)

    def __setitem__(self, key, value):
        """
        Set the value for the given key, which may use absolute or promoted names.

        Parameters
        ----------
        key : string
            Absolute or promoted variable name.
        value : any
            value for variable
        """
        if self._is_output:
            prom2abs = self._prom2abs['output']
            abs2prom = self._abs2prom['output']
        else:
            prom2abs = self._prom2abs['input']
            abs2prom = self._abs2prom['input']

        if isinstance(key, tuple) or ',' in key:
            # derivative keys can be either (of, wrt) or 'of,wrt'
            abs_keys, prom_key = self._deriv_keys(key)

            for abs_key in abs_keys:
                self._values[abs_key] = value

            super(PromotedToAbsoluteMap, self).__setitem__(prom_key, value)

        elif key in self._keys:
            # absolute name
            self._values[key] = value
            super(PromotedToAbsoluteMap, self).__setitem__(abs2prom[key], value)
        else:
            # promoted name, propagate to all connected absolute names
            for abs_key in prom2abs[key]:
                if abs_key in self._keys:
                    self._values[abs_key] = value
            super(PromotedToAbsoluteMap, self).__setitem__(key, value)

    def absolute_names(self):
        """
        Yield absolute names for variables contained in this dictionary.

        Similar to keys() but with absolute variable names instead of promoted names.

        Yields
        ------
        str
            absolute names for variables contained in this dictionary.
        """
        for key in self._keys:
            if ',' in key:
                # return derivative keys as tuples instead of strings
                of, wrt = re.sub('[( )]', '', key).split(',')
                yield (of, wrt)
            else:
                yield key


class DriverDerivativesCase(object):
    """
    Wrap data from a derivative calculation in a Driver recording to make it more accessible.

    Attributes
    ----------
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
        Dictionary mapping absolute variable names to variable metadata.
    totals : PromotedToAbsoluteMap
        Map of inputs to values recorded.
    """

    def __init__(self, iteration_coordinate, timestamp, success, msg, totals,
                 prom2abs, abs2prom, meta):
        """
        Initialize.

        Parameters
        ----------
        iteration_coordinate : str
            The string that holds the full unique identifier for this iteration.
        timestamp : float
            Time of execution of the case.
        success : str
            Success flag for the case.
        msg : str
            Message associated with the case.
        totals : array
            Derivatives to read in from the recording file.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        meta : dict
            Dictionary mapping absolute variable names to variable metadata.
        """
        self.iteration_coordinate = iteration_coordinate

        self.timestamp = timestamp
        self.success = success
        self.msg = msg
        self.meta = meta
        self.prom2abs = prom2abs
        self.abs2prom = abs2prom

        if totals is not None and totals.dtype.names:
            self.totals = PromotedToAbsoluteMap(totals[0], prom2abs, abs2prom, output=True)

    def get_derivatives(self):
        """
        Get the derivatives and their values.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of derivatives to their values.
        """
        ret_vars = {}
        for key in self.totals.absolute_names():
            ret_vars[key] = self.totals[key]
        return PromotedToAbsoluteMap(ret_vars, self.prom2abs, self.abs2prom, output=True)
