"""
A Case class.
"""

import sys
import itertools

from collections import OrderedDict

from fnmatch import fnmatchcase

import numpy as np

from openmdao.core.constants import _DEFAULT_OUT_STREAM
from openmdao.core.system import allowed_meta_names
from openmdao.recorders.sqlite_recorder import blob_to_array
from openmdao.utils.record_util import deserialize, get_source_system
from openmdao.utils.variable_table import write_var_table, NA
from openmdao.utils.general_utils import match_prom_or_abs
from openmdao.utils.units import unit_conversion, simplify_unit
from openmdao.recorders.sqlite_recorder import format_version as current_version

_AMBIGOUS_PROM_NAME = object()


class Case(object):
    """
    Case wraps the data from a single iteration of a recording to make it more easily accessible.

    Parameters
    ----------
    source : str
        The unique id of the system/solver/driver/problem that did the recording.
    data : dict-like
        Dictionary of data for a case.
    prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names of all variables to absolute names.
    abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names of all variables to promoted names.
    abs2meta : dict
        Dictionary mapping absolute names of all variables to variable metadata.
    conns : dict
        Dictionary of all model connections.
    auto_ivc_map : dict
        Dictionary that maps all auto_ivc sources to either an absolute input name for single
        connections or a promoted input name for multiple connections. This is for output
        display.
    var_info : dict
        Dictionary with information about variables (scaling, indices, execution order).
    data_format : int
        A version number specifying the format of array data, if not numpy arrays.

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
    derivatives : PromAbsDict or None
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
    _conns : dict
        Dictionary of all model connections.
    _auto_ivc_map : dict
        Dictionary that maps all auto_ivc sources to either an absolute input name for single
        connections or a promoted input name for multiple connections. This is for output display.
    _var_info : dict
        Dictionary with information about variables (scaling, indices, execution order).
    _format_version : int
        A version number specifying the format of array data, if not numpy arrays.
    """

    def __init__(self, source, data, prom2abs, abs2prom, abs2meta, conns, auto_ivc_map, var_info,
                 data_format=-1):
        """
        Initialize.
        """
        self.source = source
        self._format_version = data_format

        # save VOI dict reference for use by self._scale()
        self._var_info = var_info

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

        # for a solver or problem case
        self.abs_err = data['abs_err'] if 'abs_err' in data.keys() else None
        self.rel_err = data['rel_err'] if 'rel_err' in data.keys() else None

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
        self.derivatives = None

        if 'inputs' in data.keys():
            if data_format >= 3:
                inputs = deserialize(data['inputs'], abs2meta, prom2abs, conns)
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
                outputs = deserialize(data['outputs'], abs2meta, prom2abs, conns)
            elif self._format_version in (1, 2):
                outputs = blob_to_array(data['outputs'])
                if type(outputs) is np.ndarray and not outputs.shape:
                    outputs = None
            else:
                outputs = data['outputs']
            if outputs is not None:
                self.outputs = PromAbsDict(outputs, prom2abs['output'], abs2prom['output'],
                                           in_prom2abs=prom2abs['input'],
                                           auto_ivc_map=auto_ivc_map)

        if 'residuals' in data.keys():
            if data_format >= 3:
                residuals = deserialize(data['residuals'], abs2meta, prom2abs, conns)
            elif data_format in (1, 2):
                residuals = blob_to_array(data['residuals'])
                if type(residuals) is np.ndarray and not residuals.shape:
                    residuals = None
            else:
                residuals = data['residuals']
            if residuals is not None:
                self.residuals = PromAbsDict(residuals, prom2abs['output'], abs2prom['output'],
                                             in_prom2abs=prom2abs['input'],
                                             auto_ivc_map=auto_ivc_map)

        if 'jacobian' in data.keys():
            if data_format >= 2:
                jacobian = blob_to_array(data['jacobian'])
                if type(jacobian) is np.ndarray and not jacobian.shape:
                    jacobian = None
            else:
                jacobian = data['jacobian']
            if jacobian is not None:
                self.derivatives = PromAbsDict(jacobian, prom2abs['output'], abs2prom['output'],
                                               in_prom2abs=prom2abs['input'],
                                               auto_ivc_map=auto_ivc_map,
                                               var_info=var_info)

        # save var name & meta dict references for use by self._get_variables_of_type()
        self._prom2abs = prom2abs
        self._abs2prom = abs2prom
        self._abs2meta = abs2meta
        self._conns = conns
        self._auto_ivc_map = auto_ivc_map

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
                if name in self._auto_ivc_map:
                    return self.inputs[self._auto_ivc_map[name]]
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
            simp_units = simplify_unit(units)

            if base_units is None:
                msg = "Can't express variable '{}' with units of 'None' in units of '{}'."
                raise TypeError(msg.format(name, simp_units))

            try:
                scale, offset = unit_conversion(base_units, simp_units)
            except TypeError:
                msg = "Can't express variable '{}' with units of '{}' in units of '{}'."
                raise TypeError(msg.format(name, base_units, simp_units))

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

    def get_io_metadata(self, iotypes=('input', 'output'), metadata_keys=None,
                        includes=None, excludes=None, is_indep_var=None, is_design_var=None,
                        tags=None):
        """
        Retrieve metadata for a filtered list of variables.

        Parameters
        ----------
        iotypes : str or iter of str
            Will contain either 'input', 'output', or both.  Defaults to both.
        metadata_keys : iter of str or None
            Names of metadata entries to be retrieved or None, meaning retrieve all
            available 'allprocs' metadata.  If 'val' or 'src_indices' are required,
            their keys must be provided explicitly since they are not found in the 'allprocs'
            metadata and must be retrieved from local metadata located in each process.
        includes : str, iter of str or None
            Collection of glob patterns for pathnames of variables to include. Default is None,
            which includes all variables.
        excludes : str, iter of str or None
            Collection of glob patterns for pathnames of variables to exclude. Default is None.
        is_indep_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only inputs connected to an output tagged `openmdao:indep_var`.
            If False, list only inputs _not_ connected to outputs tagged `openmdao:indep_var`.
        is_design_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only outputs and connected inputs that are driver design variables.
            If False, list only outputs and connected inputs that are _not_ driver design variables.
        tags : str or iter of strs
            User defined tags that can be used to filter what gets listed. Only inputs with the
            given tags will be listed.
            Default is None, which means there will be no filtering based on tags.

        Returns
        -------
        dict
            A dict of metadata keyed on name, where the metadata is a dict containing
            entries based on the value of the metadata_keys arg.  Every metadata dict will
            always contain two entries, 'prom_name' and 'discrete', to indicate a given
            variable's promoted name and whether or not it is discrete.
        """
        if isinstance(iotypes, str):
            iotypes = (iotypes,)
        if isinstance(includes, str):
            includes = (includes,)
        if isinstance(excludes, str):
            excludes = (excludes,)
        if isinstance(tags, str):
            tags = {tags}

        if metadata_keys is not None:
            keyset = set(metadata_keys)
            diff = keyset - allowed_meta_names
            if diff:
                raise RuntimeError(f"Case: {sorted(diff)} are not valid metadata entry names.")

        abs2meta = self._abs2meta
        abs2prom = self._abs2prom

        result = {}

        if is_design_var is not None:
            des_vars = self.get_design_vars()
            auto_ivc_map = self._auto_ivc_map

        for iotype in iotypes:
            data = getattr(self, f'{iotype}s')

            if data is None:
                # data not recorded for this i/o type
                continue

            for abs_name in data.absolute_names():
                prom = abs2prom[iotype][abs_name]

                if not match_prom_or_abs(abs_name, prom, includes, excludes):
                    continue

                meta = abs2meta[abs_name] if abs_name in abs2meta else None

                if meta is None:
                    continue

                if metadata_keys is None:
                    ret_meta = dict(meta)
                else:
                    ret_meta = {}
                    for key in keyset:
                        try:
                            ret_meta[key] = tuple(meta[key]) if 'shape' in key else meta[key]
                        except KeyError:
                            ret_meta[key] = NA

                # handle is_indep_var
                if is_indep_var is not None:
                    if iotype == 'output':
                        out_meta = meta
                    else:
                        src_name = self._conns[abs_name]
                        out_meta = abs2meta[src_name]

                    src_tags = out_meta['tags'] if 'tags' in out_meta else {}
                    if is_indep_var:
                        if 'openmdao:indep_var' not in src_tags:
                            continue
                    elif 'openmdao:indep_var' in src_tags:
                        continue

                # handle is_design_var
                if is_design_var is not None:
                    if iotype == 'output':
                        out_name = abs_name
                    else:
                        # input, get connected output
                        src_name = self._conns[abs_name]
                        out_name = abs2prom['output'][src_name]

                    if out_name.startswith('_auto_ivc.'):
                        out_name = auto_ivc_map[out_name]

                    if is_design_var:
                        if out_name not in des_vars:
                            continue
                    elif out_name in des_vars:
                        continue

                # handle tags
                if tags:
                    meta_tags = ret_meta.get('tags', {})
                    match_tag = False
                    for tag in tags:
                        for meta_tag in meta_tags:
                            if fnmatchcase(meta_tag, tag):
                                match_tag = True
                                break
                    if not match_tag:
                        continue

                ret_meta['io'] = iotype

                ret_meta['discrete'] = 'discrete' in abs2meta[abs_name]
                ret_meta['prom_name'] = prom

                if iotype == 'output':
                    ret_meta['explicit'] = True if meta.get('explicit') else False

                result[abs_name] = ret_meta

        return result

    def list_vars(self,
                  val=True,
                  prom_name=True,
                  residuals=False,
                  residuals_tol=None,
                  units=False,
                  shape=False,
                  bounds=False,
                  scaling=False,
                  desc=False,
                  print_arrays=False,
                  tags=None,
                  print_tags=False,
                  includes=None,
                  excludes=None,
                  is_indep_var=None,
                  is_design_var=None,
                  list_autoivcs=False,
                  out_stream=_DEFAULT_OUT_STREAM,
                  print_min=False,
                  print_max=False,
                  return_format='list'):
        """
        Write a list of inputs and outputs sorted by component in execution order.

        Parameters
        ----------
        val : bool, optional
            When True, display output values. Default is True.
        prom_name : bool, optional
            When True, display the promoted name of the variable.
            Default is True.
        residuals : bool, optional
            When True, display residual values. Default is False.
        residuals_tol : float, optional
            If set, limits the output of list_outputs to only variables where
            the norm of the resids array is greater than the given 'residuals_tol'.
            Default is None.
        units : bool, optional
            When True, display units. Default is False.
        shape : bool, optional
            When True, display/return the shape of the value. Default is False.
        bounds : bool, optional
            When True, display/return bounds (lower and upper). Default is False.
        scaling : bool, optional
            When True, display/return scaling (ref, ref0, and res_ref). Default is False.
        desc : bool, optional
            When True, display/return description. Default is False.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format  is affected
            by the values set with numpy.set_printoptions
            Default is False.
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed. Only outputs with the
            given tags will be listed.
            Default is None, which means there will be no filtering based on tags.
        print_tags : bool
            When true, display tags in the columnar display.
        includes : None, str, or iter of str
            Collection of glob patterns for pathnames of variables to include. Default is None,
            which includes all output variables.
        excludes : None, str, or iter of str
            Collection of glob patterns for pathnames of variables to exclude. Default is None.
        is_indep_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only outputs tagged `openmdao:indep_var`.
            If False, list only outputs that are _not_ tagged `openmdao:indep_var`.
        is_design_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only inputs connected to outputs that are driver design variables.
            If False, list only inputs _not_ connected to outputs that are driver design variables.
        list_autoivcs : bool
            If True, include auto_ivc outputs in the listing.  Defaults to False.
        out_stream : file-like
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.
        print_min : bool
            When true, if the output value is an array, print its smallest value.
        print_max : bool
            When true, if the output value is an array, print its largest value.
        return_format : str
            Indicates the desired format of the return value. Can have value of 'list' or 'dict'.
            If 'list', the return value is a list of (name, metadata) tuples.
            if 'dict', the return value is a dictionary mapping {name: metadata}.

        Returns
        -------
        list of (name, metadata) or dict of {name: metadata}
            List or dict of output names and other optional information about those outputs.
        """
        if return_format not in ('list', 'dict'):
            badarg = f"'{return_format}'" if isinstance(return_format, str) else f"{return_format}"
            raise ValueError(f"Invalid value ({badarg}) for return_format, "
                             "must be a string value of 'list' or 'dict'")

        keynames = ['val', 'units', 'shape', 'desc', 'tags']
        keyflags = [val, units, shape, desc, tags or print_tags]
        keys = [name for i, name in enumerate(keynames) if keyflags[i]]

        if bounds:
            keys.extend(('lower', 'upper'))
        if scaling:
            keys.extend(('ref', 'ref0', 'res_ref'))

        outputs = self.get_io_metadata('output', keys, includes, excludes,
                                       is_indep_var, is_design_var, tags)

        # filter auto_ivcs if requested
        if outputs and not list_autoivcs:
            outputs = {n: m for n, m in outputs.items() if not n.startswith('_auto_ivc.')}

        # get output values & resids
        if outputs and (val or residuals or residuals_tol):
            to_remove = []
            print_options = np.get_printoptions()
            np_precision = print_options['precision']

            for name, meta in outputs.items():
                if val:
                    # we want value from the case, not from the metadata
                    meta['val'] = self.outputs[name]

                    if isinstance(meta['val'], np.ndarray):
                        if print_min:
                            meta['min'] = np.round(np.min(meta['val']), np_precision)

                        if print_max:
                            meta['max'] = np.round(np.max(meta['val']), np_precision)

                if residuals or residuals_tol:
                    resids = self.residuals[name]
                    if residuals_tol and np.linalg.norm(resids) < residuals_tol:
                        to_remove.append(name)
                    elif residuals:
                        meta['resids'] = resids

            # remove any outputs that don't pass the residuals_tol filter
            for name in to_remove:
                del outputs[name]

        if self.inputs is not None:
            inputs = self.get_io_metadata('input', keys, includes, excludes,
                                          is_indep_var, is_design_var, tags)
        else:
            inputs = {}

        # get input values
        if inputs and val is not None:
            print_options = np.get_printoptions()
            np_precision = print_options['precision']

            for name, meta in inputs.items():
                # we want value from the case, not from the metadata
                meta['val'] = self.inputs[name]

                if isinstance(meta['val'], np.ndarray):
                    if print_min:
                        meta['min'] = np.round(np.min(meta['val']), np_precision)

                    if print_max:
                        meta['max'] = np.round(np.max(meta['val']), np_precision)

        # combine inputs and outputs create return value
        var_dict = inputs
        var_dict.update(outputs)

        # remove metadata we don't want to show/return
        to_remove = ['discrete', 'explicit']
        if not prom_name:
            to_remove.append('prom_name')
        if not print_tags:
            to_remove.append('tags')
        for meta in var_dict.values():
            for key in to_remove:
                try:
                    del meta[key]
                except KeyError:
                    pass

        if var_dict:
            if 'execution_order' in self._var_info:
                var_list = []
                for var_name in self._var_info['execution_order']:
                    if var_name in var_dict:
                        var_list.append(var_name)
            else:
                # don't have execution order, just sort for determinism
                var_list = sorted(var_dict.keys())

            write_var_table('', var_list, 'all', var_dict,
                            True, print_arrays, out_stream)

        return var_dict if return_format == 'dict' else list(var_dict.items())

    def list_inputs(self,
                    val=True,
                    prom_name=True,
                    units=False,
                    shape=False,
                    global_shape=False,
                    desc=False,
                    hierarchical=True,
                    print_arrays=False,
                    tags=None,
                    print_tags=False,
                    includes=None,
                    excludes=None,
                    is_indep_var=None,
                    is_design_var=None,
                    out_stream=_DEFAULT_OUT_STREAM,
                    print_min=False,
                    print_max=False,
                    return_format='list'):
        """
        Return and optionally log a list of input names and other optional information.

        Parameters
        ----------
        val : bool, optional
            When True, display/return input values. Default is True.
        prom_name : bool, optional
            When True, display/return the promoted name of the variable.
            Default is True.
        units : bool, optional
            When True, display/return units. Default is False.
        shape : bool, optional
            When True, display/return the shape of the value. Default is False.
        global_shape : bool, optional
            When True, display/return the global shape of the value. Default is False.
        desc : bool, optional
            When True, display/return description. Default is False.
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
        print_tags : bool
            When true, display tags in the columnar display.
        includes : str, iter of str, or None
            Glob patterns for pathnames to include in the check. Default is None, which
            includes all.
        excludes : str, iter of str, or None
            Glob patterns for pathnames to exclude from the check. Default is None, which
            excludes nothing.
        is_indep_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only inputs connected to an output tagged `openmdao:indep_var`.
            If False, list only inputs _not_ connected to outputs tagged `openmdao:indep_var`.
        is_design_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only inputs connected to outputs that are driver design variables.
            If False, list only inputs _not_ connected to outputs that are driver design variables.
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.
        print_min : bool, optional
            When true, if the input value is an array, print its smallest value.
        print_max : bool, optional
            When true, if the input value is an array, print its largest value.
        return_format : str
            Indicates the desired format of the return value. Can have value of 'list' or 'dict'.
            If 'list', the return value is a list of (name, metadata) tuples.
            if 'dict', the return value is a dictionary mapping {name: metadata}.

        Returns
        -------
        list of (name, metadata) or dict of {name: metadata}
            List or dict of input names and other optional information about those inputs.
        """
        if return_format not in ('list', 'dict'):
            badarg = f"'{return_format}'" if isinstance(return_format, str) else f"{return_format}"
            raise ValueError(f"Invalid value ({badarg}) for return_format, "
                             "must be a string value of 'list' or 'dict'")

        if not self.inputs:
            return {} if return_format == 'dict' else []

        keynames = ['val', 'units', 'shape', 'global_shape', 'desc', 'tags']
        keyvals = [val, units, shape, global_shape, desc, tags or print_tags]
        keys = [n for i, n in enumerate(keynames) if keyvals[i]]

        inputs = self.get_io_metadata('input', keys, includes, excludes,
                                      is_indep_var, is_design_var, tags)

        if inputs and val:
            print_options = np.get_printoptions()
            np_precision = print_options['precision']

            for name, meta in inputs.items():
                meta['val'] = var_val = self.inputs[name]
                if isinstance(var_val, np.ndarray):
                    if print_min:
                        meta['min'] = np.round(np.min(var_val), np_precision)
                    if print_max:
                        meta['max'] = np.round(np.max(var_val), np_precision)

        if out_stream:
            if self.inputs:
                self._write_table('input', inputs, hierarchical, print_arrays, out_stream)
            else:
                ostream = sys.stdout if out_stream is _DEFAULT_OUT_STREAM else out_stream
                ostream.write('WARNING: Inputs not recorded. Make sure your recording ' +
                              'settings have record_inputs set to True\n')

        # remove metadata we don't want to show/return
        to_remove = ['discrete']
        if not prom_name:
            to_remove.append('prom_name')
        if not print_tags:
            to_remove.append('tags')
        for meta in inputs.values():
            for key in to_remove:
                try:
                    del meta[key]
                except KeyError:
                    pass

        return inputs if return_format == 'dict' else list(inputs.items())

    def list_outputs(self,
                     explicit=True, implicit=True,
                     val=True,
                     prom_name=True,
                     residuals=False,
                     residuals_tol=None,
                     units=False,
                     shape=False,
                     global_shape=False,
                     bounds=False,
                     scaling=False,
                     desc=False,
                     hierarchical=True,
                     print_arrays=False,
                     tags=None,
                     print_tags=False,
                     includes=None,
                     excludes=None,
                     is_indep_var=None,
                     is_design_var=None,
                     list_autoivcs=False,
                     out_stream=_DEFAULT_OUT_STREAM,
                     print_min=False,
                     print_max=False,
                     return_format='list'):
        """
        Return and optionally log a list of output names and other optional information.

        Parameters
        ----------
        explicit : bool, optional
            Include outputs from explicit components. Default is True.
        implicit : bool, optional
            Include outputs from implicit components. Default is True.
        val : bool, optional
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
        global_shape : bool, optional
            When True, display/return the global shape of the value. Default is False.
        bounds : bool, optional
            When True, display/return bounds (lower and upper). Default is False.
        scaling : bool, optional
            When True, display/return scaling (ref, ref0, and res_ref). Default is False.
        desc : bool, optional
            When True, display/return description. Default is False.
        hierarchical : bool, optional
            When True, human readable output shows variables in hierarchical format.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format  is affected
            by the values set with numpy.set_printoptions
            Default is False.
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed. Only outputs with the
            given tags will be listed.
            Default is None, which means there will be no filtering based on tags.
        print_tags : bool
            When true, display tags in the columnar display.
        includes : str, iter of str, or None
            Glob patterns for pathnames to include in the check. Default is None, which
            includes all.
        excludes : str, iter of str, or None
            Glob patterns for pathnames to exclude from the check. Default is None, which
            excludes nothing.
        is_indep_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only inputs connected to an output tagged `openmdao:indep_var`.
            If False, list only inputs _not_ connected to outputs tagged `openmdao:indep_var`.
        is_design_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only inputs connected to outputs that are driver design variables.
            If False, list only inputs _not_ connected to outputs that are driver design variables.
        list_autoivcs : bool
            If True, include auto_ivc outputs in the listing.  Defaults to False.
        out_stream : file-like
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.
        print_min : bool, optional
            When true, if the output value is an array, print its smallest value.
        print_max : bool, optional
            When true, if the output value is an array, print its largest value.
        return_format : str
            Indicates the desired format of the return value. Can have value of 'list' or 'dict'.
            If 'list', the return value is a list of (name, metadata) tuples.
            if 'dict', the return value is a dictionary mapping {name: metadata}.

        Returns
        -------
        list of (name, metadata) or dict of {name: metadata}
            List or dict of output names and other optional information about those outputs.
        """
        if return_format not in ('list', 'dict'):
            badarg = f"'{return_format}'" if isinstance(return_format, str) else f"{return_format}"
            raise ValueError(f"Invalid value ({badarg}) for return_format, "
                             "must be a string value of 'list' or 'dict'")

        if not self.outputs:
            return {} if return_format == 'dict' else []

        keynames = ['val', 'units', 'shape', 'global_shape', 'desc', 'tags']
        keyvals = [val, units, shape, global_shape, desc, tags or print_tags]
        keys = [n for i, n in enumerate(keynames) if keyvals[i]]

        if bounds:
            keys.extend(('lower', 'upper'))
        if scaling:
            keys.extend(('ref', 'ref0', 'res_ref'))

        outputs = self.get_io_metadata('output', keys, includes, excludes,
                                       is_indep_var, is_design_var, tags)

        # filter auto_ivcs if requested
        if outputs and not list_autoivcs:
            outputs = {n: m for n, m in outputs.items() if not n.startswith('_auto_ivc.')}

        # get output values & resids
        if outputs and (val or residuals or residuals_tol):
            to_remove = []
            print_options = np.get_printoptions()
            np_precision = print_options['precision']

            for name, meta in outputs.items():
                if val:
                    # we want value from the case, not from the metadata
                    meta['val'] = self.outputs[name]

                    if isinstance(meta['val'], np.ndarray):
                        if print_min:
                            meta['min'] = np.round(np.min(meta['val']), np_precision)

                        if print_max:
                            meta['max'] = np.round(np.max(meta['val']), np_precision)

                if residuals or residuals_tol:
                    try:
                        resids = self.residuals[name]
                        if residuals_tol and np.linalg.norm(resids) < residuals_tol:
                            to_remove.append(name)
                        elif residuals:
                            meta['resids'] = resids
                    except KeyError:
                        if residuals:
                            meta['resids'] = 'Not Recorded'

            # remove any outputs that don't pass the residuals_tol filter
            for name in to_remove:
                del outputs[name]

        expl_outputs = {n: m for n, m in outputs.items() if m['explicit']}
        impl_outputs = {n: m for n, m in outputs.items() if not m['explicit']}

        # remove metadata we don't want to show/return
        to_remove = ['discrete', 'explicit']
        if not prom_name:
            to_remove.append('prom_name')
        if not print_tags:
            to_remove.append('tags')
        for meta in itertools.chain(outputs.values(), expl_outputs.values(), impl_outputs.values()):
            for key in to_remove:
                try:
                    del meta[key]
                except KeyError:
                    pass

        if out_stream:
            if not self.outputs:
                ostream = sys.stdout if out_stream is _DEFAULT_OUT_STREAM else out_stream
                ostream.write('WARNING: Outputs not recorded. Make sure your recording ' +
                              'settings have record_outputs set to True\n')
            if explicit:
                self._write_table('explicit', expl_outputs, hierarchical, print_arrays, out_stream)
            if implicit:
                self._write_table('implicit', impl_outputs, hierarchical, print_arrays, out_stream)

        if explicit and implicit:
            pass
        elif explicit:
            outputs = expl_outputs
        elif implicit:
            outputs = impl_outputs
        else:
            raise RuntimeError('You have excluded both Explicit and Implicit components.')

        return outputs if return_format == 'dict' else list(outputs.items())

    def _write_table(self, var_type, var_data, hierarchical, print_arrays, out_stream):
        """
        Write table of variable names, values, residuals, and metadata to out_stream.

        Parameters
        ----------
        var_type : 'input', 'explicit' or 'implicit'
            Indicates type of variables, input or explicit/implicit output.
        var_data : dict or list
            Dict of {name: metadata} or list of (name, metadata).
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

        # Make a dict of variables. Makes it easier to work with in this method
        if isinstance(var_data, dict):
            var_dict = var_data
        else:
            var_dict = OrderedDict()
            for name, meta in var_data:
                var_dict[name] = meta

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
                        hierarchical=hierarchical, print_arrays=print_arrays,
                        out_stream=out_stream)

    def _get_variables_of_type(self, var_type, scaled=False, use_indices=False):
        """
        Get the variables of a given type and their values.

        Parameters
        ----------
        var_type : str
            String indicating which value for 'type' should be accepted for a variable
            to be included in the returned map.  Allowed values are: ['desvar', 'objective',
            'constraint', 'response'].
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

        abs2meta = self._abs2meta
        prom2abs_in = self._prom2abs['input']
        auto_ivc_map = self._auto_ivc_map

        ret_vars = {}

        for name, meta in self._var_info.items():
            # FIXME: _var_info contains dvs, responses, and 'execution_order'. It
            # should be reorganized to prevent dvs and responses from being at the
            # same level as execution_order.  While unlikely, it is possible that
            # a dv/response could have the name 'execution_order'.  Mainly though,
            # separating them will prevent needing the following kludge.
            if name == 'execution_order':
                continue

            src = meta['source']

            if var_type in abs2meta[src]['type']:
                try:
                    val = self.outputs[src].copy()
                except KeyError:
                    # not recorded
                    continue
                if use_indices and meta['indices'] is not None:
                    val = val[meta['indices']]
                if scaled:
                    if meta['total_adder'] is not None:
                        val += meta['total_adder']
                    if meta['total_scaler'] is not None:
                        val *= meta['total_scaler']
                ret_vars[name] = val

        return PromAbsDict(ret_vars, self._prom2abs['output'], self._abs2prom['output'],
                           in_prom2abs=prom2abs_in, auto_ivc_map=auto_ivc_map,
                           var_info=self._var_info)


class PromAbsDict(dict):
    """
    A dictionary that enables accessing values via absolute or promoted variable names.

    Parameters
    ----------
    values : array or dict
        Numpy structured array or dictionary of values.
    prom2abs : dict
        Dictionary mapping promoted names to absolute names.
    abs2prom : dict
        Dictionary mapping absolute names in the output vector to promoted names.
    data_format : int
        A version number specifying the OpenMDAO SQL case database version.
    in_prom2abs : dict
        Dictionary mapping promoted names in the input vector to absolute names.
    auto_ivc_map : dict
        Dictionary that maps all auto_ivc sources to either an absolute input name for single
        connections or a promoted input name for multiple connections. This is for output
        display.
    var_info : dict
        Dictionary of variable metadata. Needed when there are constraint aliases.

    Attributes
    ----------
    _values : array or dict
        Array or dict of values accessible via absolute variable name.
    _keys : array
        Absolute variable names that map to the values in the _values array.
    _prom2abs : dict
        Dictionary mapping promoted names in the output vector to absolute names.
    _abs2prom : dict
        Dictionary mapping absolute names to promoted names.
    _auto_ivc_map : dict
        Dictionary that maps all auto_ivc sources to either an absolute input name for single
        connections or a promoted input name for multiple connections. This is for output display.
    _var_info : dict
        Dictionary of variable metadata. Needed when there are constraint aliases.
    _DERIV_KEY_SEP : str
        Separator character for derivative keys.
    """

    def __init__(self, values, prom2abs, abs2prom, data_format=current_version,
                 in_prom2abs=None, auto_ivc_map=None, var_info=None):
        """
        Initialize.
        """
        super().__init__()

        self._prom2abs = prom2abs
        self._abs2prom = abs2prom
        auto_ivc_map = auto_ivc_map if auto_ivc_map is not None else {}
        self._var_info = var_info
        self._auto_ivc_map = auto_ivc_map

        if data_format <= 8:
            DERIV_KEY_SEP = self._DERIV_KEY_SEP = ','
        else:
            DERIV_KEY_SEP = self._DERIV_KEY_SEP = '!'

        if isinstance(values, dict):
            # dict of values, keyed on either absolute or promoted names
            self._values = {}
            for key in values.keys():
                if key in auto_ivc_map:
                    # key is auto_ivc, so translate to a readable input name.
                    self._values[key] = values[key]
                    in_key = auto_ivc_map[key]
                    super().__setitem__(in_key, values[key])
                elif key in abs2prom:
                    # key is absolute name
                    self._values[key] = values[key]
                    prom_key = abs2prom[key]
                    super().__setitem__(prom_key, values[key])
                elif key in prom2abs:
                    # key is promoted name
                    for abs_key in prom2abs[key]:
                        self._values[abs_key] = values[key]
                    super().__setitem__(key, values[key])
                elif isinstance(key, tuple) or DERIV_KEY_SEP in key:
                    # derivative keys can be either (of, wrt) or 'of!wrt'
                    abs_keys, prom_key = self._deriv_keys(key)
                    for abs_key in abs_keys:
                        self._values[abs_key] = values[key]
                    super().__setitem__(prom_key, values[key])
                elif in_prom2abs is not None and key in in_prom2abs:
                    # Auto-ivc outputs, use abs source (which is prom source.)
                    self._values[key] = values[key]
                    super().__setitem__(key, values[key])
                else:
                    # Constraint alias support.
                    self._values[key] = values[key]
                    super().__setitem__(key, values[key])

            self._keys = self._values.keys()
        else:
            # numpy structured array, which will always use absolute names
            self._values = values[0]
            self._keys = values.dtype.names
            for key in self._keys:
                if key in auto_ivc_map:
                    # key is auto_ivc, so translate to a readable input name.
                    in_key = auto_ivc_map[key]
                    super().__setitem__(in_key, self._values[key])
                elif key in abs2prom:
                    prom_key = abs2prom[key]
                    if prom_key in self:
                        # We already set a value for this promoted name, which means
                        # it is an input that maps to multiple absolute names. Set the
                        # value to AMBIGOUS and require access via absolute name.
                        super().__setitem__(prom_key, _AMBIGOUS_PROM_NAME)
                    else:
                        super().__setitem__(prom_key, self._values[key])
                elif DERIV_KEY_SEP in key:
                    # derivative keys will be a string in the form of 'of!wrt'
                    abs_keys, prom_key = self._deriv_keys(key)
                    super().__setitem__(prom_key, self._values[key])
                elif in_prom2abs is not None and key in in_prom2abs:
                    # Auto-ivc outputs, use abs source (which is prom source.)
                    # TODO - maybe get rid of this by always saving the source name
                    super().__setitem__(key, self._values[key])

    def _deriv_keys(self, key):
        """
        Get the absolute and promoted name versions of the provided derivative key.

        Parameters
        ----------
        key : tuple or string
            derivative key as either (of, wrt) or 'of!wrt'.

        Returns
        -------
        list of tuples:
            list of (of, wrt) mapping the provided key to absolute names.
        tuple :
            (of, wrt) mapping the provided key to promoted names.
        """
        prom2abs = self._prom2abs
        abs2prom = self._abs2prom

        DERIV_KEY_SEP = self._DERIV_KEY_SEP

        # derivative could be tuple or string, using absolute or promoted names

        of, wrt = key if isinstance(key, tuple) else key.split(DERIV_KEY_SEP)

        if of in abs2prom:
            # if promoted, will map to all connected absolute names
            abs_of = [of]
            prom_of = abs2prom[of]
        elif of in prom2abs:
            abs_of = prom2abs[of]
            prom_of = of
        else:
            # Support for constraint aliases.
            abs_of = [self._var_info[of]['source']]

            # The "of" part of the key name should be the alias.
            prom_of = of

        if wrt in prom2abs:
            abs_wrt = [prom2abs[wrt]][0]
        else:
            abs_wrt = [wrt]

        abs_keys = [f'{o}{DERIV_KEY_SEP}{w}' for o, w in itertools.product(abs_of, abs_wrt)]

        if wrt in abs2prom:
            prom_wrt = abs2prom[wrt]
        else:
            prom_wrt = wrt

        prom_key = (prom_of, prom_wrt)

        return abs_keys, prom_key

    def __getitem__(self, key):
        """
        Use the variable name to get the corresponding value.

        Parameters
        ----------
        key : str
            Absolute or promoted variable name.

        Returns
        -------
        array :
            An array entry value that corresponds to the given variable name.
        """
        if key in self._keys:
            # absolute name
            return self._values[key]

        elif key in self._auto_ivc_map:
            # We allow the user to query with auto_ivc varname.
            src_key = self._auto_ivc_map[key]
            if src_key in self._keys:
                return self._values[self._auto_ivc_map[key]]

        elif key in self:
            # promoted name
            val = super().__getitem__(key)
            if val is _AMBIGOUS_PROM_NAME:
                msg = "The promoted name '%s' is invalid because it refers to multiple " + \
                      "inputs: %s. Access the value using an absolute path name or the " + \
                      "connected output variable instead."
                raise RuntimeError(msg % (key, str(self._prom2abs[key])))
            else:
                return val

        elif isinstance(key, tuple) or self._DERIV_KEY_SEP in key:
            # derivative keys can be either (of, wrt) or 'of!wrt'
            _, prom_key = self._deriv_keys(key)
            return super().__getitem__(prom_key)

        raise KeyError('Variable name "%s" not found.' % key)

    def __setitem__(self, key, value):
        """
        Set the value for the given key, which may use absolute or promoted names.

        Parameters
        ----------
        key : str
            Absolute or promoted variable name.
        value : any
            value for variable
        """
        auto_ivc_map = self._auto_ivc_map
        abs2prom = self._abs2prom
        prom2abs = self._prom2abs

        if isinstance(key, tuple):
            _, prom_key = self._deriv_keys(key)
            self._values[f"{prom_key[0]}!{prom_key[1]}"] = value
            super().__setitem__(prom_key, value)
        elif self._DERIV_KEY_SEP in key:
            # derivative keys can be either (of, wrt) or 'of!wrt'
            _, prom_key = self._deriv_keys(key)

            self._values[f"{prom_key[0]}!{prom_key[1]}"] = value

            super().__setitem__(prom_key, value)

        elif key in abs2prom:
            if key in auto_ivc_map:
                # key is auto_ivc, so translate to a readable input name.
                self._values[key] = value
                in_key = auto_ivc_map[key]
                super().__setitem__(in_key, self._values[key])
            else:
                # absolute name
                self._values[key] = value
                super().__setitem__(self._abs2prom[key], value)
        elif key in prom2abs:
            # promoted name, propagate to all connected absolute names
            for abs_key in self._prom2abs[key]:
                if abs_key in self._keys:
                    self._values[abs_key] = value
            super().__setitem__(key, value)
        else:
            # Design variable by promoted input name.
            self._values[key] = value
            super().__setitem__(key, value)

    def absolute_names(self):
        """
        Yield absolute names for variables contained in this dictionary.

        Similar to keys() but with absolute variable names instead of promoted names.

        Yields
        ------
        str
            absolute names for variables contained in this dictionary.
        """
        DERIV_KEY_SEP = self._DERIV_KEY_SEP

        for key in self._keys:
            if DERIV_KEY_SEP in key:
                # return derivative keys as tuples instead of strings
                of, wrt = key.split(DERIV_KEY_SEP)
                if of in self._prom2abs:
                    of = self._prom2abs[of][0]
                if wrt in self._prom2abs:
                    abswrts = self._prom2abs[wrt]
                    if len(abswrts) == 1:
                        wrt = abswrts[0]
                        # for now, if wrt is ambiguous, leave as promoted
                yield (of, wrt)
            else:
                yield key
