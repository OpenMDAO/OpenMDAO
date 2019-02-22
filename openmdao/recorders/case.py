"""
A Case class.
"""

import sys
import re
import itertools

from collections import OrderedDict

import numpy as np

from openmdao.utils.record_util import json_to_np_array
from openmdao.recorders.sqlite_recorder import blob_to_array

from openmdao.utils.write_outputs import write_outputs

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
    parent : str
        The iteration coordinate of the parent case for this iteration if any, else None.
    counter : int
        The unique sequential index of this case in the recording.
    timestamp : float
        Time of execution of the case.
    success : str
        Success flag for the case.
    msg : str
        Message associated with the case.
    outputs : PromotedToAbsoluteMap
        Map of outputs to values recorded.
    inputs : PromotedToAbsoluteMap or None
        Map of inputs to values recorded, None if not recorded.
    residuals : PromotedToAbsoluteMap or None
        Map of outputs to residuals recorded, None if not recorded.
    jacobian : PromotedToAbsoluteMap or None
        Map of (output, input) to derivatives recorded, None if not recorded.
    parent : str
        The full unique identifier for the parent this iteration.
    abs_err : float or None
        Absolute tolerance (None if not recorded).
    rel_err : float or None
        Relative tolerance (None if not recorded).
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names of all variables to absolute names.
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names of all variables to promoted names.
    _abs2meta : dict
        Dictionary mapping absolute names of all variables to variable metadata.
    _voi_meta : dict
        Dictionary mapping absolute names of variables of interest to variable metadata.
    _format_version : int
        A version number specifying the format of array data, if not numpy arrays.
    """

    def __init__(self, source, data, prom2abs, abs2prom, abs2meta, voi_meta, data_format=None):
        """
        Initialize.

        Parameters
        ----------
        source : str
            The unique id of the system/solver/driver/problem that did the recording.
        data : dict-like
            Dictionary of data for a case
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names of all variables to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names of all variables to promoted names.
        abs2meta : dict
            Dictionary mapping absolute names of all variables to variable metadata.
        voi_meta : dict
            Dictionary mapping absolute names of variables of interest to variable metadata.
        data_format : int
            A version number specifying the format of array data, if not numpy arrays.
        """
        self.source = source
        self._format_version = data_format

        if 'iteration_coordinate' in data.keys():
            self.iteration_coordinate = data['iteration_coordinate']
            parts = self.iteration_coordinate.split('|')
            if len(parts) > 2:
                self.parent = '|'.join(parts[:-2])
            else:
                self.parent = None
        elif 'case_name' in data.keys():
            self.iteration_coordinate = data['case_name']  # problem cases
            self.parent = None
        else:
            self.iteration_coordinate = None
            self.parent = None

        self.counter = data['counter']
        self.timestamp = data['timestamp']
        self.success = data['success']
        self.msg = data['msg']

        # for a solver case
        self.abs_err = data['abs_err'] if 'abs_err' in data.keys() else None
        self.rel_err = data['abs_err'] if 'rel_err' in data.keys() else None

        # rename solver keys
        if 'solver_inputs' in data.keys():
            if not isinstance(data, dict):
                data = dict(zip(data.keys(), data))
            data['inputs'] = data.pop('solver_inputs')
            data['outputs'] = data.pop('solver_output')
            data['residuals'] = data.pop('solver_residuals')

        # default properties to None
        self.inputs = None
        self.outputs = None
        self.residuals = None
        self.jacobian = None

        if 'inputs' in data.keys():
            if data_format >= 3:
                inputs = json_to_np_array(data['inputs'], abs2meta)
            elif data_format in (1, 2):
                inputs = blob_to_array(data['inputs'])
                if type(inputs) is np.ndarray and not inputs.shape:
                    inputs = None
            else:
                inputs = data['inputs']
            if inputs is not None:
                self.inputs = PromotedToAbsoluteMap(inputs[0], prom2abs, abs2prom, output=False)

        if 'outputs' in data.keys():
            if data_format >= 3:
                outputs = json_to_np_array(data['outputs'], abs2meta)
            elif self._format_version in (1, 2):
                outputs = blob_to_array(data['outputs'])
                if type(outputs) is np.ndarray and not outputs.shape:
                    outputs = None
            else:
                outputs = data['outputs']
            if outputs is not None:
                self.outputs = PromotedToAbsoluteMap(outputs[0], prom2abs, abs2prom)

        if 'residuals' in data.keys():
            if data_format >= 3:
                residuals = json_to_np_array(data['residuals'], abs2meta)
            elif data_format in (1, 2):
                residuals = blob_to_array(data['residuals'])
                if type(residuals) is np.ndarray and not residuals.shape:
                    residuals = None
            else:
                residuals = data['residuals']
            if residuals is not None:
                self.residuals = PromotedToAbsoluteMap(residuals[0], prom2abs, abs2prom)

        if 'jacobian' in data.keys():
            if data_format >= 2:
                jacobian = blob_to_array(data['jacobian'])
                if type(jacobian) is np.ndarray and not jacobian.shape:
                    jacobian = None
            else:
                jacobian = data['jacobian']
            if jacobian is not None:
                self.jacobian = PromotedToAbsoluteMap(jacobian[0], prom2abs, abs2prom, output=True)

        # save var name & meta dict references for use by self._get_variables_of_type()
        self._prom2abs = prom2abs
        self._abs2prom = abs2prom
        self._abs2meta = abs2meta

        # save VOI dict reference for use by self._scale()
        self._voi_meta = voi_meta

    def __str__(self):
        """
        Get string representation of the case.

        Returns
        -------
        str
            String representation of the case.
        """
        return ' '.join([self.source, self.iteration_coordinate, str(self.outputs)])

    def get_design_vars(self, scaled=True, use_indices=True):
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
        vals = self._get_variables_of_type('desvar')
        if scaled:
            return self._apply_voi_meta(vals, scaled, use_indices)
        else:
            return vals

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
        vals = self._get_variables_of_type('objective')
        if scaled:
            return self._apply_voi_meta(vals, scaled, use_indices)
        else:
            return vals

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
        vals = self._get_variables_of_type('constraint')
        if scaled:
            return self._apply_voi_meta(vals, scaled, use_indices)
        else:
            return vals

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
        PromotedToAbsoluteMap
            Map of variables to their values.
        """
        vals = self._get_variables_of_type('response')
        if scaled:
            return self._apply_voi_meta(vals, scaled, use_indices)
        else:
            return vals

    def list_inputs(self,
                    values=True,
                    units=False,
                    hierarchical=True,
                    print_arrays=False,
                    out_stream=_DEFAULT_OUT_STREAM):
        """
        Return and optionally log a list of input names and other optional information.

        Also optionally logs the information to a user defined output stream.

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
        meta = self._abs2meta
        inp_vars = {}
        inputs = []

        if self.inputs is not None:
            for abs_name in self.inputs.absolute_names():
                if abs_name not in inp_vars:
                    inp_vars[abs_name] = {'value': self.inputs[abs_name]}

        if inp_vars is not None and len(inp_vars) > 0:
            for name in inp_vars:
                outs = {}
                if values:
                    outs['value'] = inp_vars[name]['value']
                if units:
                    outs['units'] = meta[name]['units']
                inputs.append((name, outs))

        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if out_stream:
            if len(inp_vars) is 0:
                out_stream.write('WARNING: Inputs not recorded. Make sure your recording ' +
                                 'settings have record_inputs set to True\n')

            self._write_outputs('input', None, inputs, hierarchical, print_arrays, out_stream)

        return inputs

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

        Also optionally logs the information to a user defined output stream.

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
        meta = self._abs2meta
        expl_outputs = []
        impl_outputs = []
        out_vars = {}

        for abs_name in self.outputs.absolute_names():
            out_vars[abs_name] = {'value': self.outputs[abs_name]}
            if self.residuals and abs_name in self.residuals.absolute_names():
                out_vars[abs_name]['residuals'] = self.residuals[abs_name]
            else:
                out_vars[abs_name]['residuals'] = 'Not Recorded'

        if len(out_vars) > 0:
            for name in out_vars:
                if residuals_tol and \
                   out_vars[name]['residuals'] is not 'Not Recorded' and \
                   np.linalg.norm(out_vars[name]['residuals']) < residuals_tol:
                    continue
                outs = {}
                if values:
                    outs['value'] = out_vars[name]['value']
                if prom_name:
                    outs['prom_name'] = self._var_abs2prom['output'][name]
                if residuals:
                    outs['resids'] = out_vars[name]['residuals']
                if units:
                    outs['units'] = meta[name]['units']
                if shape:
                    outs['shape'] = out_vars[name]['value'].shape
                if bounds:
                    outs['lower'] = meta[name]['lower']
                    outs['upper'] = meta[name]['upper']
                if scaling:
                    outs['ref'] = meta[name]['ref']
                    outs['ref0'] = meta[name]['ref0']
                    outs['res_ref'] = meta[name]['res_ref']
                if meta[name]['explicit']:
                    expl_outputs.append((name, outs))
                else:
                    impl_outputs.append((name, outs))

        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if out_stream:
            if len(out_vars) is 0:
                out_stream.write('WARNING: Outputs not recorded. Make sure your recording ' +
                                 'settings have record_outputs set to True\n')

            if explicit:
                self._write_outputs('output', 'Explicit', expl_outputs, hierarchical, print_arrays,
                                    out_stream)

            if implicit:
                self._write_outputs('output', 'Implicit', impl_outputs, hierarchical, print_arrays,
                                    out_stream)

        if explicit and implicit:
            return expl_outputs + impl_outputs
        elif explicit:
            return expl_outputs
        elif implicit:
            return impl_outputs
        else:
            raise RuntimeError('You have excluded both Explicit and Implicit components.')

    def _write_outputs(self, in_out, comp_type, outputs, hierarchical, print_arrays, out_stream):
        """
        Write table of variable names, values, residuals, and metadata to out_stream.

        The output values could actually represent input variables.
        In this context, outputs refers to the data that is being logged to an output stream.

        Parameters
        ----------
        in_out : str, 'input' or 'output'
            indicates whether the values passed in are from inputs or output variables.
        comp_type : str, 'Explicit' or 'Implicit'
            the type of component with the output values.
        outputs : list
            list of (name, dict of vals and metadata) tuples.
        hierarchical : bool
            When True, human readable output shows variables in hierarchical format.
        print_arrays : bool
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format  is affected
            by the values set with numpy.set_printoptions
            Default is False.
        out_stream : file-like object
            Where to send human readable output.
            Set to None to suppress.
        """
        if out_stream is None:
            return

        # Make a dict of outputs. Makes it easier to work with in this method
        dict_of_outputs = OrderedDict()
        for name, vals in outputs:
            dict_of_outputs[name] = vals

        allprocs_abs_names = {
            'input': dict_of_outputs.keys(),
            'output': dict_of_outputs.keys()
        }

        write_outputs(in_out, comp_type, dict_of_outputs, hierarchical, print_arrays, out_stream,
                      'model', allprocs_abs_names)

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
            return PromotedToAbsoluteMap({}, self._prom2abs, self._abs2prom)

        ret_vars = {}
        for var in self.outputs.absolute_names():
            if var_type in self._abs2meta[var]['type']:
                ret_vars[var] = self.outputs[var]

        return PromotedToAbsoluteMap(ret_vars, self._prom2abs, self._abs2prom)

    def _apply_voi_meta(self, vals, scaled=True, use_indices=True):
        """
        Scale the values array and apply indices from _voi_meta per the arguments.

        Parameters
        ----------
        vals : PromotedToAbsoluteMap
            Map of variables to their values.
        scaled : bool
            The unique id of the system/solver/driver/problem that did the recording.
        use_indices : bool
            The full unique identifier for this iteration.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of variables to their scaled values.
        """
        for name in vals.absolute_names():
            if name in self._voi_meta:
                meta = self._voi_meta[name]
                if scaled:
                    # physical to scaled
                    if meta['adder'] is not None:
                        vals[name] += meta['adder']
                    if meta['scaler'] is not None:
                        vals[name] *= meta['scaler']
                if use_indices:
                    if meta['indices'] is not None:
                        vals[name] = vals[name][meta['indices']]

        return vals


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

    def __str__(self):
        """
        Get string representation of the dictionary.

        Returns
        -------
        str
            String representation of the dictionary.
        """
        return super(PromotedToAbsoluteMap, self).__str__()

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
