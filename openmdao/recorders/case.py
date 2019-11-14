"""
A Case class.
"""

import sys
import re
import itertools

from collections import OrderedDict

import numpy as np

from openmdao.recorders.sqlite_recorder import blob_to_array
from openmdao.utils.record_util import deserialize, get_source_system
from openmdao.utils.variable_table import write_var_table
from openmdao.utils.general_utils import warn_deprecation, make_set, \
    var_name_match_includes_excludes
from openmdao.utils.units import get_conversion

_DEFAULT_OUT_STREAM = object()
_AMBIGOUS_PROM_NAME = object()


class Case(object):
    """
    Case wraps the data from a single iteration of a recording to make it more easily accessible.

    Attributes
    ----------
    source : str
        The unique id of the system/solver/driver/problem that did the recording.
    name : str
        The unique identifier for this case.
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
    outputs : PromAbsDict
        Map of outputs to values recorded.
    inputs : PromAbsDict or None
        Map of inputs to values recorded, None if not recorded.
    residuals : PromAbsDict or None
        Map of outputs to residuals recorded, None if not recorded.
    jacobian : PromAbsDict or None
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
    _var_info : dict
        Dictionary with information about variables (scaling, indices, execution order).
    _format_version : int
        A version number specifying the format of array data, if not numpy arrays.
    """

    def __init__(self, source, data, prom2abs, abs2prom, abs2meta, var_info, data_format=None):
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
        var_info : dict
            Dictionary with information about variables (scaling, indices, execution order).
        data_format : int
            A version number specifying the format of array data, if not numpy arrays.
        """
        self.source = source
        self._format_version = data_format

        if 'iteration_coordinate' in data.keys():
            self.name = data['iteration_coordinate']
            parts = self.name.split('|')
            if len(parts) > 2:
                self.parent = '|'.join(parts[:-2])
            else:
                self.parent = None
        elif 'case_name' in data.keys():
            self.name = data['case_name']  # problem cases
            self.parent = None
        else:
            self.name = None
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
                inputs = deserialize(data['inputs'], abs2meta)
            elif data_format in (1, 2):
                inputs = blob_to_array(data['inputs'])
                if type(inputs) is np.ndarray and not inputs.shape:
                    inputs = None
            else:
                inputs = data['inputs']
            if inputs is not None:
                self.inputs = PromAbsDict(inputs, prom2abs['input'], abs2prom['input'])

        if 'outputs' in data.keys():
            if data_format >= 3:
                outputs = deserialize(data['outputs'], abs2meta)
            elif self._format_version in (1, 2):
                outputs = blob_to_array(data['outputs'])
                if type(outputs) is np.ndarray and not outputs.shape:
                    outputs = None
            else:
                outputs = data['outputs']
            if outputs is not None:
                self.outputs = PromAbsDict(outputs, prom2abs['output'], abs2prom['output'])

        if 'residuals' in data.keys():
            if data_format >= 3:
                residuals = deserialize(data['residuals'], abs2meta)
            elif data_format in (1, 2):
                residuals = blob_to_array(data['residuals'])
                if type(residuals) is np.ndarray and not residuals.shape:
                    residuals = None
            else:
                residuals = data['residuals']
            if residuals is not None:
                self.residuals = PromAbsDict(residuals, prom2abs['output'], abs2prom['output'])

        if 'jacobian' in data.keys():
            if data_format >= 2:
                jacobian = blob_to_array(data['jacobian'])
                if type(jacobian) is np.ndarray and not jacobian.shape:
                    jacobian = None
            else:
                jacobian = data['jacobian']
            if jacobian is not None:
                self.jacobian = PromAbsDict(jacobian, prom2abs['output'], abs2prom['output'])

        # save var name & meta dict references for use by self._get_variables_of_type()
        self._prom2abs = prom2abs
        self._abs2prom = abs2prom
        self._abs2meta = abs2meta

        # save VOI dict reference for use by self._scale()
        self._var_info = var_info

    @property
    def iteration_coordinate(self):
        """
        Deprecate the 'iteration_coordinate' attribute.

        Returns
        -------
        str
            The unique identifier for this case.
        """
        warn_deprecation("'iteration_coordinate' has been deprecated. Use 'name' instead.")
        return self.name

    def __str__(self):
        """
        Get string representation of the case.

        Returns
        -------
        str
            String representation of the case.
        """
        return ' '.join([self.source, self.name, str(self.outputs)])

    def __getitem__(self, name):
        """
        Get an output/input variable.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.

        Returns
        -------
        float or ndarray or any python object
            the requested output/input variable.
        """
        if self.outputs is not None:
            try:
                return self.outputs[name]
            except KeyError:
                if self.inputs is not None:
                    return self.inputs[name]
        elif self.inputs is not None:
            return self.inputs[name]

        raise KeyError('Variable name "%s" not found.' % name)

    def get_val(self, name, units=None, indices=None):
        """
        Get an output/input variable.

        Function is used if you want to specify display units.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.
        units : str, optional
            Units to convert to before upon return.
        indices : int or list of ints or tuple of ints or int ndarray or Iterable or None, optional
            Indices or slice to return.

        Returns
        -------
        float or ndarray
            The requested output/input variable.
        """
        val = self[name]

        if indices is not None:
            val = val[indices]

        if units is not None:
            base_units = self._get_units(name)

            if base_units is None:
                msg = "Can't express variable '{}' with units of 'None' in units of '{}'."
                raise TypeError(msg.format(name, units))

            try:
                scale, offset = get_conversion(base_units, units)
            except TypeError:
                msg = "Can't express variable '{}' with units of '{}' in units of '{}'."
                raise TypeError(msg.format(name, base_units, units))

            val = (val + offset) * scale

        return val

    def _get_units(self, name):
        """
        Get the units for a variable name.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.

        Returns
        -------
        str
            Unit string.
        """
        meta = self._abs2meta

        if name in meta:
            return meta[name]['units']

        proms = self._prom2abs

        if name in proms['output']:
            abs_name = proms['output'][name][0]
            return meta[abs_name]['units']

        elif name in proms['input']:
            if len(proms['input'][name]) > 1:
                # The promoted name maps to multiple absolute names, require absolute name.
                msg = "Can't get units for the promoted name '%s' because it refers to " + \
                      "multiple inputs: %s. Access the units using an absolute path name."
                raise RuntimeError(msg % (name, str(proms['input'][name])))

            abs_name = proms['input'][name][0]
            return meta[abs_name]['units']

        raise KeyError('Variable name "{}" not found.'.format(name))

    def get_design_vars(self, scaled=True, use_indices=True):
        """
        Get the values of the design variables, as seen by the driver, for this case.

        Parameters
        ----------
        scaled : bool
            If True, then return scaled values.
        use_indices : bool
            If True, apply indices.

        Returns
        -------
        PromAbsDict
            Map of variables to their values.
        """
        return self._get_variables_of_type('desvar', scaled, use_indices)

    def get_objectives(self, scaled=True, use_indices=True):
        """
        Get the values of the objectives, as seen by the driver, for this case.

        Parameters
        ----------
        scaled : bool
            If True, then return scaled values.
        use_indices : bool
            If True, apply indices.

        Returns
        -------
        PromAbsDict
            Map of variables to their values.
        """
        return self._get_variables_of_type('objective', scaled, use_indices)

    def get_constraints(self, scaled=True, use_indices=True):
        """
        Get the values of the constraints, as seen by the driver, for this case.

        Parameters
        ----------
        scaled : bool
            If True, then return scaled values.
        use_indices : bool
            If True, apply indices.

        Returns
        -------
        PromAbsDict
            Map of variables to their values.
        """
        return self._get_variables_of_type('constraint', scaled, use_indices)

    def get_responses(self, scaled=True, use_indices=True):
        """
        Get the values of the responses, as seen by the driver, for this case.

        Parameters
        ----------
        scaled : bool
            If True, then return scaled values.
        use_indices : bool
            If True, apply indices.

        Returns
        -------
        PromAbsDict
            Map of variables to their values.
        """
        return self._get_variables_of_type('response', scaled, use_indices)

    def list_inputs(self,
                    values=True,
                    prom_name=False,
                    units=False,
                    shape=False,
                    hierarchical=True,
                    print_arrays=False,
                    tags=None,
                    includes=None,
                    excludes=None,
                    out_stream=_DEFAULT_OUT_STREAM):
        """
        Return and optionally log a list of input names and other optional information.

        Also optionally logs the information to a user defined output stream.

        Parameters
        ----------
        values : bool, optional
            When True, display/return input values. Default is True.
        prom_name : bool, optional
            When True, display/return the promoted name of the variable.
            Default is False.
        units : bool, optional
            When True, display/return units. Default is False.
        shape : bool, optional
            When True, display/return the shape of the value. Default is False.
        hierarchical : bool, optional
            When True, human readable output shows variables in hierarchical format.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format is affected
            by the values set with numpy.set_printoptions
            Default is False.
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed. Only inputs with the
            given tags will be listed.
            Default is None, which means there will be no filtering based on tags.
        includes : None or list_like
            List of glob patterns for pathnames to include in the check. Default is None, which
            includes all components in the model.
        excludes : None or list_like
            List of glob patterns for pathnames to exclude from the check. Default is None, which
            excludes nothing.
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list
            list of input names and other optional information about those inputs
        """
        meta = self._abs2meta
        inputs = []

        if self.inputs is not None:
            for var_name in self.inputs.absolute_names():
                # Filter based on tags
                if tags and not (make_set(tags) & make_set(meta[var_name]['tags'])):
                    continue

                if not var_name_match_includes_excludes(var_name,
                                                        self._abs2prom['input'][var_name],
                                                        includes, excludes):
                    continue

                var_meta = {}
                if values:
                    var_meta['value'] = self.inputs[var_name]
                if prom_name:
                    var_meta['prom_name'] = self._abs2prom['input'][var_name]
                if units:
                    var_meta['units'] = meta[var_name]['units']
                if shape:
                    var_meta['shape'] = self.inputs[var_name].shape
                inputs.append((var_name, var_meta))

        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if out_stream:
            if self.inputs is None or len(self.inputs) is 0:
                out_stream.write('WARNING: Inputs not recorded. Make sure your recording ' +
                                 'settings have record_inputs set to True\n')

            self._write_table('input', inputs, hierarchical, print_arrays, out_stream)

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
                     tags=None,
                     includes=None,
                     excludes=None,
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
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed. Only inputs with the
            given tags will be listed.
            Default is None, which means there will be no filtering based on tags.
        includes : None or list_like
            List of glob patterns for pathnames to include in the check. Default is None, which
            includes all components in the model.
        excludes : None or list_like
            List of glob patterns for pathnames to exclude from the check. Default is None, which
            excludes nothing.
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

        for var_name in self.outputs.absolute_names():
            # Filter based on tags
            if tags and not (make_set(tags) & make_set(meta[var_name]['tags'])):
                continue

            if not var_name_match_includes_excludes(var_name, self._abs2prom['output'][var_name],
                                                    includes, excludes):
                continue

            # check if residuals were recorded, skip if within specifed tolerance
            if self.residuals and var_name in self.residuals.absolute_names():
                resids = self.residuals[var_name]
                if residuals_tol and np.linalg.norm(resids) < residuals_tol:
                    continue
            else:
                resids = 'Not Recorded'

            var_meta = {}
            if values:
                var_meta['value'] = self.outputs[var_name]
            if prom_name:
                var_meta['prom_name'] = self._abs2prom['output'][var_name]
            if residuals:
                var_meta['resids'] = resids
            if units:
                var_meta['units'] = meta[var_name]['units']
            if shape:
                var_meta['shape'] = self.outputs[var_name].shape
            if bounds:
                var_meta['lower'] = meta[var_name]['lower']
                var_meta['upper'] = meta[var_name]['upper']
            if scaling:
                var_meta['ref'] = meta[var_name]['ref']
                var_meta['ref0'] = meta[var_name]['ref0']
                var_meta['res_ref'] = meta[var_name]['res_ref']
            if meta[var_name]['explicit']:
                expl_outputs.append((var_name, var_meta))
            else:
                impl_outputs.append((var_name, var_meta))

        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if out_stream:
            if self.outputs is None or len(self.outputs) is 0:
                out_stream.write('WARNING: Outputs not recorded. Make sure your recording ' +
                                 'settings have record_outputs set to True\n')

            if explicit:
                self._write_table('explicit', expl_outputs, hierarchical, print_arrays, out_stream)

            if implicit:
                self._write_table('implicit', impl_outputs, hierarchical, print_arrays, out_stream)

        if explicit and implicit:
            return expl_outputs + impl_outputs
        elif explicit:
            return expl_outputs
        elif implicit:
            return impl_outputs
        else:
            raise RuntimeError('You have excluded both Explicit and Implicit components.')

    def _write_table(self, var_type, var_data, hierarchical, print_arrays, out_stream):
        """
        Write table of variable names, values, residuals, and metadata to out_stream.

        The output values could actually represent input variables.
        In this context, outputs refers to the data that is being logged to an output stream.

        Parameters
        ----------
        var_type : 'input', 'explicit' or 'implicit'
            Indicates type of variables, input or explicit/implicit output.
        var_data : list
            List of (name, dict of vals and metadata) tuples.
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
        var_dict = OrderedDict()
        for name, vals in var_data:
            var_dict[name] = vals

        # determine pathname of the system
        if self.source in ('root', 'driver', 'problem', 'root.nonlinear_solver'):
            pathname = ''
        elif '|' in self.source:
            pathname = get_source_system(self.source)
        else:
            pathname = self.source.replace('root.', '')
            if pathname.endswith('.nonlinear_solver'):
                pathname = pathname[:-17]  # len('.nonlinear_solver') == 17

        # vars should be in execution order
        if 'execution_order' in self._var_info:
            var_order = self._var_info['execution_order']
            var_list = [var_name for var_name in var_order if var_name in var_dict]
        else:
            # don't have execution order, just sort for determinism
            var_list = sorted(var_dict.keys())

        write_var_table(pathname, var_list, var_type, var_dict,
                        hierarchical, print_arrays, out_stream)

    def _get_variables_of_type(self, var_type, scaled=False, use_indices=False):
        """
        Get the variables of a given type and their values.

        Parameters
        ----------
        var_type : str
            String indicating which value for 'type' should be accepted for a variable
            to be included in the returned map.
        scaled : bool
            If True, then return scaled values.
        use_indices : bool
            If True, apply indices.

        Returns
        -------
        PromAbsDict
            Map of variables to their values.
        """
        if self.outputs is None:
            return PromAbsDict({}, self._prom2abs, self._abs2prom)

        ret_vars = {}
        update_vals = scaled or use_indices
        for name in self.outputs.absolute_names():
            if var_type in self._abs2meta[name]['type']:
                ret_vars[name] = val = self.outputs[name]
                if update_vals and name in self._var_info:
                    meta = self._var_info[name]
                    if use_indices and meta['indices'] is not None:
                        val = val[meta['indices']]
                    if scaled:
                        if meta['adder'] is not None:
                            val += meta['adder']
                        if meta['scaler'] is not None:
                            val *= meta['scaler']
                    ret_vars[name] = val

        return PromAbsDict(ret_vars, self._prom2abs['output'], self._abs2prom['output'])


class PromAbsDict(dict):
    """
    A dictionary that enables accessing values via absolute or promoted variable names.

    Attributes
    ----------
    _values : array or dict
        Array or dict of values accessible via absolute variable name.
    _keys : array
        Absolute variable names that map to the values in the _values array.
    _prom2abs : dict
        Dictionary mapping promoted names to absolute names.
    _abs2prom : dict
        Dictionary mapping absolute names to promoted names.
    """

    def __init__(self, values, prom2abs, abs2prom):
        """
        Initialize.

        Parameters
        ----------
        values : array or dict
            Numpy structured array or dictionary of values.
        prom2abs : dict
            Dictionary mapping promoted names to absolute names.
        abs2prom : dict
            Dictionary mapping absolute names to promoted names.
        """
        super(PromAbsDict, self).__init__()

        self._prom2abs = prom2abs
        self._abs2prom = abs2prom

        if isinstance(values, dict):
            # dict of values, keyed on either absolute or promoted names
            self._values = {}
            for key in values.keys():
                if key in abs2prom:
                    # key is absolute name
                    self._values[key] = values[key]
                    prom_key = abs2prom[key]
                    super(PromAbsDict, self).__setitem__(prom_key, values[key])
                elif key in prom2abs:
                    # key is promoted name
                    for abs_key in prom2abs[key]:
                        self._values[abs_key] = values[key]
                    super(PromAbsDict, self).__setitem__(key, values[key])
                elif isinstance(key, tuple) or ',' in key:
                    # derivative keys can be either (of, wrt) or 'of,wrt'
                    abs_keys, prom_key = self._deriv_keys(key)
                    for abs_key in abs_keys:
                        self._values[abs_key] = values[key]
                    super(PromAbsDict, self).__setitem__(prom_key, values[key])
            self._keys = self._values.keys()
        else:
            # numpy structured array, which will always use absolute names
            self._values = values[0]
            self._keys = values.dtype.names
            for key in self._keys:
                if key in abs2prom:
                    prom_key = abs2prom[key]
                    if prom_key in self:
                        # We already set a value for this promoted name, which means
                        # it is an input that maps to multiple absolute names. Set the
                        # value to AMBIGOUS and require access via absolute name.
                        super(PromAbsDict, self).__setitem__(prom_key, _AMBIGOUS_PROM_NAME)
                    else:
                        super(PromAbsDict, self).__setitem__(prom_key, self._values[key])
                elif ',' in key:
                    # derivative keys will be a string in the form of 'of,wrt'
                    abs_keys, prom_key = self._deriv_keys(key)
                    super(PromAbsDict, self).__setitem__(prom_key, self._values[key])

    def __str__(self):
        """
        Get string representation of the dictionary.

        Returns
        -------
        str
            String representation of the dictionary.
        """
        return super(PromAbsDict, self).__str__()

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
        prom2abs = self._prom2abs
        abs2prom = self._abs2prom

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
            val = super(PromAbsDict, self).__getitem__(key)
            if val is _AMBIGOUS_PROM_NAME:
                msg = "The promoted name '%s' is invalid because it refers to multiple " + \
                      "inputs: %s. Access the value using an absolute path name or the " + \
                      "connected output variable instead."
                raise RuntimeError(msg % (key, str(self._prom2abs[key])))
            else:
                return val

        elif isinstance(key, tuple) or ',' in key:
            # derivative keys can be either (of, wrt) or 'of,wrt'
            abs_keys, prom_key = self._deriv_keys(key)
            return super(PromAbsDict, self).__getitem__(prom_key)

        raise KeyError('Variable name "%s" not found.' % key)

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
        if isinstance(key, tuple) or ',' in key:
            # derivative keys can be either (of, wrt) or 'of,wrt'
            abs_keys, prom_key = self._deriv_keys(key)

            for abs_key in abs_keys:
                self._values[abs_key] = value

            super(PromAbsDict, self).__setitem__(prom_key, value)

        elif key in self._keys:
            # absolute name
            self._values[key] = value
            super(PromAbsDict, self).__setitem__(self._abs2prom[key], value)
        else:
            # promoted name, propagate to all connected absolute names
            for abs_key in self._prom2abs[key]:
                if abs_key in self._keys:
                    self._values[abs_key] = value
            super(PromAbsDict, self).__setitem__(key, value)

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
