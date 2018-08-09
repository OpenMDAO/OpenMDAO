"""Define the Component class."""

from __future__ import division

from collections import OrderedDict, Iterable
from itertools import product
from six import string_types, iteritems, itervalues

import numpy as np
from numpy import ndarray, isscalar, atleast_1d, atleast_2d, promote_types
from scipy.sparse import issparse

from openmdao.approximation_schemes.complex_step import ComplexStep, DEFAULT_CS_OPTIONS
from openmdao.approximation_schemes.finite_difference import FiniteDifference, DEFAULT_FD_OPTIONS
from openmdao.core.system import System
from openmdao.jacobians.assembled_jacobian import SUBJAC_META_DEFAULTS
from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian
from openmdao.utils.units import valid_units
from openmdao.utils.general_utils import format_as_float_or_array, ensure_compatible, \
    warn_deprecation, find_matches
from openmdao.vectors.vector import INT_DTYPE
from openmdao.utils.name_maps import rel_key2abs_key, abs_key2rel_key


# Suppored methods for derivatives
_supported_methods = {'fd': (FiniteDifference, DEFAULT_FD_OPTIONS),
                      'cs': (ComplexStep, DEFAULT_CS_OPTIONS),
                      'exact': (None, {})}


# the following metadata will be accessible for vars on all procs
global_meta_names = {
    'input': ('units', 'shape', 'size'),
    'output': ('units', 'shape', 'size',
               'ref', 'ref0', 'res_ref', 'distributed', 'lower', 'upper'),
}


def _valid_var_name(name):
    """
    Determine if the proposed name is a valid variable name.

    Parameters
    ----------
    name : str
        Proposed name.

    Returns
    -------
    bool
        True if the proposed name is a valid variable name, else False.
    """
    forbidden_chars = ['.', '*', '?', '!', '[', ']']

    return not any([True for character in forbidden_chars if character in name])


class Component(System):
    """
    Base Component class; not to be directly instantiated.

    Attributes
    ----------
    distributed : bool
        This is True if the component has variables that are distributed across multiple
        processes.
    _approx_schemes : OrderedDict
        A mapping of approximation types to the associated ApproximationScheme.
    _var_rel2data_io : dict
        Dictionary mapping relative names to dicts with keys (prom, rel, my_idx, type_, metadata).
        This is only needed while adding inputs and outputs. During setup, these are used to
        build the dictionaries of metadata.
    _static_var_rel2data_io : dict
        Static version of above - stores data for variables added outside of setup.
    _var_rel_names : {'input': [str, ...], 'output': [str, ...]}
        List of relative names of owned variables existing on current proc.
        This is only needed while adding inputs and outputs. During setup, these are used to
        determine the list of absolute names.
    _static_var_rel_names : dict
        Static version of above - stores names of variables added outside of setup.
    _declared_partials : list
        Cached storage of user-declared partials.
    _approximated_partials : list
        Cached storage of user-declared approximations.
    _declared_partial_checks : list
        Cached storage of user-declared check partial options.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            available here and in all descendants of this system.
        """
        # put these here to prevent them from possibly overriding values set
        # by the user in initialize().
        self.distributed = False

        super(Component, self).__init__(**kwargs)

        self._approx_schemes = OrderedDict()

        self._var_rel_names = {'input': [], 'output': []}
        self._var_rel2data_io = {}

        self._static_var_rel_names = {'input': [], 'output': []}
        self._static_var_rel2data_io = {}

        self._declared_partials = []
        self._approximated_partials = []
        self._declared_partial_checks = []

    def setup(self):
        """
        Declare inputs and outputs.

        Available attributes:
            name
            pathname
            comm
            options
        """
        pass

    def _setup_procs(self, pathname, comm, mode):
        """
        Execute first phase of the setup process.

        Distribute processors, assign pathnames, and call setup on the component.

        Parameters
        ----------
        pathname : str
            Global name of the system, including the path.
        comm : MPI.Comm or <FakeComm>
            MPI communicator object.
        mode : string
            Derivatives calculation mode, 'fwd' for forward, and 'rev' for
            reverse (adjoint). Default is 'rev'.
        """
        self.pathname = pathname
        self.comm = comm
        self._mode = mode
        self._subsystems_proc_range = []

        # Clear out old variable information so that we can call setup on the component.
        self._var_rel_names = {'input': [], 'output': []}
        self._var_rel2data_io = {}
        self._design_vars = OrderedDict()
        self._responses = OrderedDict()

        self._static_mode = False
        self._var_rel2data_io.update(self._static_var_rel2data_io)
        for type_ in ['input', 'output']:
            self._var_rel_names[type_].extend(self._static_var_rel_names[type_])
        self._design_vars.update(self._static_design_vars)
        self._responses.update(self._static_responses)
        self.setup()
        self._static_mode = True

        if self.distributed:
            self._vector_class = self._distributed_vector_class
        else:
            self._vector_class = self._local_vector_class

    def _setup_vars(self, recurse=True):
        """
        Count total variables.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(Component, self)._setup_vars()
        num_var = self._num_var

        for vec_name in self._lin_rel_vec_name_list:
            num_var[vec_name] = {}
            # Compute num_var
            for type_ in ['input', 'output']:
                relnames = self._var_allprocs_relevant_names[vec_name][type_]
                num_var[vec_name][type_] = len(relnames)

        self._num_var['nonlinear'] = self._num_var['linear']

    def _setup_var_data(self, recurse=True):
        """
        Compute the list of abs var names, abs/prom name maps, and metadata dictionaries.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        global global_meta_names
        super(Component, self)._setup_var_data()
        allprocs_abs_names = self._var_allprocs_abs_names
        abs_names = self._var_abs_names
        allprocs_prom2abs_list = self._var_allprocs_prom2abs_list
        abs2prom = self._var_abs2prom
        allprocs_abs2meta = self._var_allprocs_abs2meta
        abs2meta = self._var_abs2meta

        # Compute the prefix for turning rel/prom names into abs names
        if self.pathname:
            prefix = self.pathname + '.'
        else:
            prefix = ''

        for type_ in ['input', 'output']:
            for prom_name in self._var_rel_names[type_]:
                abs_name = prefix + prom_name
                metadata = self._var_rel2data_io[prom_name]['metadata']

                # Compute allprocs_abs_names, abs_names
                allprocs_abs_names[type_].append(abs_name)
                abs_names[type_].append(abs_name)

                # Compute allprocs_prom2abs_list, abs2prom
                allprocs_prom2abs_list[type_][prom_name] = [abs_name]
                abs2prom[type_][abs_name] = prom_name

                # Compute allprocs_abs2meta
                allprocs_abs2meta[abs_name] = {
                    meta_name: metadata[meta_name]
                    for meta_name in global_meta_names[type_]
                }

                # Compute abs2meta
                abs2meta[abs_name] = metadata

    def _setup_var_sizes(self, recurse=True):
        """
        Compute the arrays of local variable sizes for all variables/procs on this system.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(Component, self)._setup_var_sizes()

        iproc = self.comm.rank
        nproc = self.comm.size
        vec_names = self._lin_rel_vec_name_list

        sizes = self._var_sizes

        # Initialize empty arrays
        for vec_name in vec_names:
            sizes[vec_name] = {}

            for type_ in ('input', 'output'):
                sizes[vec_name][type_] = np.zeros((nproc, self._num_var[vec_name][type_]), int)

            # Compute _var_sizes
            abs2meta = self._var_abs2meta
            for type_ in ('input', 'output'):
                sz = sizes[vec_name][type_]
                for idx, abs_name in enumerate(self._var_allprocs_relevant_names[vec_name][type_]):
                    sz[iproc, idx] = abs2meta[abs_name]['size']

        if self.comm.size > 1:
            for vec_name in vec_names:
                sizes = self._var_sizes[vec_name]
                for type_ in ['input', 'output']:
                    self.comm.Allgather(sizes[type_][iproc, :], sizes[type_])

        self._var_sizes['nonlinear'] = self._var_sizes['linear']

        self._setup_global_shapes()

    def _setup_partials(self, recurse=True):
        """
        Process all partials and approximations that the user declared.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        self._subjacs_info = {}
        self._jacobian = DictionaryJacobian(system=self)

        for of, wrt, dependent, rows, cols, val in self._declared_partials:
            self._declare_partials(of, wrt, dependent=dependent, rows=rows, cols=cols, val=val)

        for of, wrt, method, kwargs in self._approximated_partials:
            self._approx_partials(of, wrt, method=method, **kwargs)

    def add_input(self, name, val=1.0, shape=None, src_indices=None, flat_src_indices=None,
                  units=None, desc=''):
        """
        Add an input variable to the component.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray or Iterable
            The initial value of the variable being added in user-defined units.
            Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if src_indices not provided and
            val is not an array. Default is None.
        src_indices : int or list of ints or tuple of ints or int ndarray or Iterable or None
            The global indices of the source variable to transfer data from.
            A value of None implies this input depends on all entries of source.
            Default is None. The shapes of the target and src_indices must match,
            and form of the entries within is determined by the value of 'flat_src_indices'.
        flat_src_indices : bool
            If True, each entry of src_indices is assumed to be an index into the
            flattened source.  Otherwise each entry must be a tuple or list of size equal
            to the number of dimensions of the source.
        units : str or None
            Units in which this input variable will be provided to the component
            during execution. Default is None, which means it is unitless.
        desc : str
            description of the variable

        Returns
        -------
        dict
            metadata for added variable
        """
        if units == 'unitless':
            warn_deprecation("Input '%s' has units='unitless' but 'unitless' "
                             "has been deprecated. Use "
                             "units=None instead.  Note that connecting a "
                             "unitless variable to one with units is no longer "
                             "an error, but will issue a warning instead." %
                             name)
            units = None

        # First, type check all arguments
        if not isinstance(name, str):
            raise TypeError('The name argument should be a string')
        if not _valid_var_name(name):
            raise NameError("'%s' is not a valid input name." % name)
        if not isscalar(val) and not isinstance(val, (list, tuple, ndarray, Iterable)):
            raise TypeError('The val argument should be a float, list, tuple, ndarray or Iterable')
        if shape is not None and not isinstance(shape, (int, tuple, list, np.integer)):
            raise TypeError("The shape argument should be an int, tuple, or list but "
                            "a '%s' was given" % type(shape))
        if src_indices is not None and not isinstance(src_indices, (int, list, tuple,
                                                                    ndarray, Iterable)):
            raise TypeError('The src_indices argument should be an int, list, '
                            'tuple, ndarray or Iterable')
        if units is not None and not isinstance(units, str):
            raise TypeError('The units argument should be a str or None')

        # Check that units are valid
        if units is not None and not valid_units(units):
            raise ValueError("The units '%s' are invalid" % units)

        metadata = {}

        # value, shape: based on args, making sure they are compatible
        metadata['value'], metadata['shape'], src_indices = ensure_compatible(name, val, shape,
                                                                              src_indices)
        metadata['size'] = np.prod(metadata['shape'])

        # src_indices: None or ndarray
        if src_indices is None:
            metadata['src_indices'] = None
        else:
            metadata['src_indices'] = np.asarray(src_indices, dtype=INT_DTYPE)
        metadata['flat_src_indices'] = flat_src_indices

        metadata['units'] = units
        metadata['desc'] = desc

        # We may not know the pathname yet, so we have to use name for now, instead of abs_name.
        if self._static_mode:
            var_rel2data_io = self._static_var_rel2data_io
            var_rel_names = self._static_var_rel_names
        else:
            var_rel2data_io = self._var_rel2data_io
            var_rel_names = self._var_rel_names

        # Disallow dupes
        if name in var_rel2data_io:
            msg = "Variable name '{}' already exists.".format(name)
            raise ValueError(msg)

        var_rel2data_io[name] = {
            'prom': name, 'rel': name,
            'my_idx': len(self._var_rel_names['input']),
            'type': 'input',
            'metadata': metadata
        }
        var_rel_names['input'].append(name)

        return metadata

    def add_output(self, name, val=1.0, shape=None, units=None, res_units=None, desc='',
                   lower=None, upper=None, ref=1.0, ref0=0.0, res_ref=1.0):
        """
        Add an output variable to the component.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if val is not an array.
            Default is None.
        units : str or None
            Units in which the output variables will be provided to the component during execution.
            Default is None, which means it has no units.
        res_units : str or None
            Units in which the residuals of this output will be given to the user when requested.
            Default is None, which means it has no units.
        desc : str
            description of the variable.
        lower : float or list or tuple or ndarray or Iterable or None
            lower bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no lower bound.
            Default is None.
        upper : float or list or tuple or ndarray or or Iterable None
            upper bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no upper bound.
            Default is None.
        ref : float or ndarray
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 1. Default is 1.
        ref0 : float or ndarray
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 0. Default is 0.
        res_ref : float or ndarray
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 1. Default is 1.

        Returns
        -------
        dict
            metadata for added variable
        """
        if units == 'unitless':
            warn_deprecation("Output '%s' has units='unitless' but 'unitless' "
                             "has been deprecated. Use "
                             "units=None instead.  Note that connecting a "
                             "unitless variable to one with units is no longer "
                             "an error, but will issue a warning instead." %
                             name)
            units = None

        # First, type check all arguments
        if not isinstance(name, str):
            raise TypeError('The name argument should be a string')
        if not _valid_var_name(name):
            raise NameError("'%s' is not a valid output name." % name)
        if not isscalar(val) and not isinstance(val, (list, tuple, ndarray, Iterable)):
            msg = 'The val argument should be a float, list, tuple, ndarray or Iterable'
            raise TypeError(msg)
        if not isscalar(ref) and not isinstance(val, (list, tuple, ndarray, Iterable)):
            msg = 'The ref argument should be a float, list, tuple, ndarray or Iterable'
            raise TypeError(msg)
        if not isscalar(ref0) and not isinstance(val, (list, tuple, ndarray, Iterable)):
            msg = 'The ref0 argument should be a float, list, tuple, ndarray or Iterable'
            raise TypeError(msg)
        if not isscalar(res_ref) and not isinstance(val, (list, tuple, ndarray, Iterable)):
            msg = 'The res_ref argument should be a float, list, tuple, ndarray or Iterable'
            raise TypeError(msg)
        if shape is not None and not isinstance(shape, (int, tuple, list, np.integer)):
            raise TypeError("The shape argument should be an int, tuple, or list but "
                            "a '%s' was given" % type(shape))
        if units is not None and not isinstance(units, str):
            raise TypeError('The units argument should be a str or None')
        if res_units is not None and not isinstance(res_units, str):
            raise TypeError('The res_units argument should be a str or None')

        # Check that units are valid
        if units is not None and not valid_units(units):
            raise ValueError("The units '%s' are invalid" % units)

        metadata = {}

        # value, shape: based on args, making sure they are compatible
        metadata['value'], metadata['shape'], _ = ensure_compatible(name, val, shape)
        metadata['size'] = np.prod(metadata['shape'])

        # units, res_units: taken as is
        metadata['units'] = units
        metadata['res_units'] = res_units

        # desc: taken as is
        metadata['desc'] = desc

        if lower is not None:
            lower = ensure_compatible(name, lower, metadata['shape'])[0]
        if upper is not None:
            upper = ensure_compatible(name, upper, metadata['shape'])[0]

        metadata['lower'] = lower
        metadata['upper'] = upper

        # All refs: check the shape if necessary
        for item, item_name in zip([ref, ref0, res_ref], ['ref', 'ref0', 'res_ref']):
            if not isscalar(item):
                it = atleast_1d(item)
                if it.shape != metadata['shape']:
                    raise ValueError("'{}': When adding output '{}', expected shape {} but got "
                                     "shape {} for argument '{}'.".format(self.name, name,
                                                                          metadata['shape'],
                                                                          it.shape, item_name))

        if isscalar(ref):
            self._has_output_scaling |= ref != 1.0
        else:
            self._has_output_scaling |= np.any(ref != 1.0)

        if isscalar(ref0):
            self._has_output_scaling |= ref0 != 0.0
        else:
            self._has_output_scaling |= np.any(ref0)

        if isscalar(res_ref):
            self._has_resid_scaling |= res_ref != 1.0
        else:
            self._has_resid_scaling |= np.any(res_ref != 1.0)

        ref = format_as_float_or_array('ref', ref, flatten=True)
        ref0 = format_as_float_or_array('ref0', ref0, flatten=True)
        res_ref = format_as_float_or_array('res_ref', res_ref, flatten=True)

        metadata['ref'] = ref
        metadata['ref0'] = ref0
        metadata['res_ref'] = res_ref

        metadata['distributed'] = self.distributed

        # We may not know the pathname yet, so we have to use name for now, instead of abs_name.
        if self._static_mode:
            var_rel2data_io = self._static_var_rel2data_io
            var_rel_names = self._static_var_rel_names
        else:
            var_rel2data_io = self._var_rel2data_io
            var_rel_names = self._var_rel_names

        # Disallow dupes
        if name in var_rel2data_io:
            msg = "Variable name '{}' already exists.".format(name)
            raise ValueError(msg)

        var_rel2data_io[name] = {
            'prom': name, 'rel': name,
            'my_idx': len(self._var_rel_names['output']),
            'type': 'output',
            'metadata': metadata
        }
        var_rel_names['output'].append(name)

        return metadata

    def _approx_partials(self, of, wrt, method='fd', **kwargs):
        """
        Inform the framework that the specified derivatives are to be approximated.

        Parameters
        ----------
        of : str or list of str
            The name of the residual(s) that derivatives are being computed for.
            May also contain a glob pattern.
        wrt : str or list of str
            The name of the variables that derivatives are taken with respect to.
            This can contain the name of any input or output variable.
            May also contain a glob pattern.
        method : str
            The type of approximation that should be used. Valid options include:
                - 'fd': Finite Difference
        **kwargs : dict
            Keyword arguments for controlling the behavior of the approximation.
        """
        pattern_matches = self._find_partial_matches(of, wrt)

        for of_bundle, wrt_bundle in product(*pattern_matches):
            of_pattern, of_matches = of_bundle
            wrt_pattern, wrt_matches = wrt_bundle
            if not of_matches:
                raise ValueError('No matches were found for of="{}"'.format(of_pattern))
            if not wrt_matches:
                raise ValueError('No matches were found for wrt="{}"'.format(wrt_pattern))

            info = self._subjacs_info
            for rel_key in product(of_matches, wrt_matches):
                abs_key = rel_key2abs_key(self, rel_key)
                if abs_key in info:
                    meta = info[abs_key]
                else:
                    meta = SUBJAC_META_DEFAULTS.copy()
                meta['method'] = method
                meta.update(kwargs)
                info[abs_key] = meta

    def declare_partials(self, of, wrt, dependent=True, rows=None, cols=None, val=None,
                         method='exact', step=None, form=None, step_calc=None):
        """
        Declare information about this component's subjacobians.

        Parameters
        ----------
        of : str or list of str
            The name of the residual(s) that derivatives are being computed for.
            May also contain a glob pattern.
        wrt : str or list of str
            The name of the variables that derivatives are taken with respect to.
            This can contain the name of any input or output variable.
            May also contain a glob pattern.
        dependent : bool(True)
            If False, specifies no dependence between the output(s) and the
            input(s). This is only necessary in the case of a sparse global
            jacobian, because if 'dependent=False' is not specified and
            declare_partials is not called for a given pair, then a dense
            matrix of zeros will be allocated in the sparse global jacobian
            for that pair.  In the case of a dense global jacobian it doesn't
            matter because the space for a dense subjac will always be
            allocated for every pair.
        rows : ndarray of int or None
            Row indices for each nonzero entry.  For sparse subjacobians only.
        cols : ndarray of int or None
            Column indices for each nonzero entry.  For sparse subjacobians only.
        val : float or ndarray of float or scipy.sparse
            Value of subjacobian.  If rows and cols are not None, this will
            contain the values found at each (row, col) location in the subjac.
        method : str
            The type of approximation that should be used. Valid options include:
            'fd': Finite Difference, 'cs': Complex Step, 'exact': use the component
            defined analytic derivatives. Default is 'exact'.
        step : float
            Step size for approximation. Defaults to None, in which case the approximation
            method provides its default value.
        form : string
            Form for finite difference, can be 'forward', 'backward', or 'central'. Defaults
            to None, in which case the approximation method provides its default value.
        step_calc : string
            Step type for finite difference, can be 'abs' for absolute', or 'rel' for
            relative. Defaults to None, in which case the approximation method provides
            its default value.
        """
        try:
            method_func, default_opts = _supported_methods[method]
        except KeyError:
            msg = 'Method "{}" is not supported, method must be one of {}'
            raise ValueError(msg.format(method, _supported_methods.keys()))

        # Analytic Derivative for this Jacobian pair
        if method_func is None:  # exact

            # If only one of rows/cols is specified
            if (rows is None) ^ (cols is None):
                raise ValueError('If one of rows/cols is specified, then both must be specified')

            self._declared_partials.append((of, wrt, dependent, rows, cols, val))

        # Approximation of the derivative, former API call approx_partials.
        else:

            if method not in self._approx_schemes:
                self._approx_schemes[method] = method_func()

            # If rows/cols is specified
            if rows is not None or cols is not None:
                raise ValueError('Sparse FD specification not supported yet.')

            # Need to declare the Jacobian element too.
            self._declared_partials.append((of, wrt, True, rows, cols, val))

            kwargs = {}
            if step:
                if 'step' in default_opts:
                    kwargs['step'] = step
                else:
                    raise RuntimeError("'step' is not a valid option for '%s'" % method)
            if form:
                if 'form' in default_opts:
                    kwargs['form'] = form
                else:
                    raise RuntimeError("'form' is not a valid option for '%s'" % method)
            if step_calc:
                if 'step_calc' in default_opts:
                    kwargs['step_calc'] = step_calc
                else:
                    raise RuntimeError("'step_calc' is not a valid option for '%s'" % method)

            self._approximated_partials.append((of, wrt, method, kwargs))

    def set_check_partial_options(self, wrt, method='fd', form=None, step=None, step_calc=None):
        """
        Set options that will be used for checking partial derivatives.

        Parameters
        ----------
        wrt : str or list of str
            The name or names of the variables that derivatives are taken with respect to.
            This can contain the name of any input or output variable.
            May also contain a glob pattern.
        method : str
            Method for check: "fd" for finite difference, "cs" for complex step.
        form : str
            Finite difference form for check, can be "forward", "central", or "backward". Leave
            undeclared to keep unchanged from previous or default value.
        step : float
            Step size for finite difference check. Leave undeclared to keep unchanged from previous
            or default value.
        step_calc : str
            Type of step calculation for check, can be "abs" for absolute (default) or "rel" for
            relative.  Leave undeclared to keep unchanged from previous or default value.
        """
        supported_methods = ('fd', 'cs')
        if method not in supported_methods:
            msg = "Method '{}' is not supported, method must be one of {}"
            raise ValueError(msg.format(method, supported_methods))

        if step and not isinstance(step, (int, float)):
            msg = "The value of 'step' must be numeric, but '{}' was specified."
            raise ValueError(msg.format(step))

        supported_step_calc = ('abs', 'rel')
        if step_calc and step_calc not in supported_step_calc:
            msg = "The value of 'step_calc' must be one of {}, but '{}' was specified."
            raise ValueError(msg.format(supported_step_calc, step_calc))

        if not isinstance(wrt, (string_types, list, tuple)):
            msg = "The value of 'wrt' must be a string or list of strings, but a type " \
                  "of '{}' was provided."
            raise ValueError(msg.format(type(wrt).__name__))

        wrt_list = [wrt] if isinstance(wrt, string_types) else wrt
        self._declared_partial_checks.append((wrt_list, method, form, step, step_calc))

    def _get_check_partial_options(self):
        """
        Return dictionary of partial options with pattern matches processed.

        This is called by check_partials.

        Returns
        -------
        dict(wrt : (options))
            Dictionary keyed by name with tuples of options (method, form, step, step_calc)
        """
        opts = {}
        outs = list(self._var_allprocs_prom2abs_list['output'].keys())
        ins = list(self._var_allprocs_prom2abs_list['input'].keys())

        invalid_wrt = []

        for wrt_list, method, form, step, step_calc in self._declared_partial_checks:
            for pattern in wrt_list:
                matches = find_matches(pattern, outs + ins)

                # if a non-wildcard var name was specified and not found, save for later Exception
                if len(matches) == 0 and _valid_var_name(pattern):
                    invalid_wrt.append(pattern)

                for match in matches:
                    if match in opts:
                        opt = opts[match]

                        # New assignments take precedence
                        for name, value in zip(['method', 'form', 'step', 'step_calc'],
                                               [method, form, step, step_calc]):
                            if value is not None:
                                opt[name] = value

                    else:
                        opts[match] = {'method': method,
                                       'form': form,
                                       'step': step,
                                       'step_calc': step_calc}

        if invalid_wrt:
            if len(invalid_wrt) == 1:
                msg = "Invalid 'wrt' variable specified for check_partial options on Component " \
                      "'{}': '{}'.".format(self.pathname, invalid_wrt[0])
            else:
                msg = "Invalid 'wrt' variables specified for check_partial options on Component " \
                      "'{}': {}.".format(self.pathname, invalid_wrt)
            raise ValueError(msg)

        return opts

    def _declare_partials(self, of, wrt, dependent=True, rows=None, cols=None, val=None):
        """
        Store subjacobian metadata for later use.

        Parameters
        ----------
        of : str or list of str
            The name of the residual(s) that derivatives are being computed for.
            May also contain a glob pattern.
        wrt : str or list of str
            The name of the variables that derivatives are taken with respect to.
            This can contain the name of any input or output variable.
            May also contain a glob pattern.
        dependent : bool(True)
            If False, specifies no dependence between the output(s) and the
            input(s). This is only necessary in the case of a sparse global
            jacobian, because if 'dependent=False' is not specified and
            declare_partials is not called for a given pair, then a dense
            matrix of zeros will be allocated in the sparse global jacobian
            for that pair.  In the case of a dense global jacobian it doesn't
            matter because the space for a dense subjac will always be
            allocated for every pair.
        rows : ndarray of int or None
            Row indices for each nonzero entry.  For sparse subjacobians only.
        cols : ndarray of int or None
            Column indices for each nonzero entry.  For sparse subjacobians only.
        val : float or ndarray of float or scipy.sparse
            Value of subjacobian.  If rows and cols are not None, this will
            contain the values found at each (row, col) location in the subjac.
        """
        is_scalar = isscalar(val)

        if dependent:
            if rows is None:
                if val is not None and not is_scalar and not issparse(val):
                    val = atleast_2d(val)
                    val = val.astype(promote_types(val.dtype, float), copy=False)
                rows_max = cols_max = 0
            else:  # sparse list format
                rows = np.array(rows, dtype=INT_DTYPE, copy=False)
                cols = np.array(cols, dtype=INT_DTYPE, copy=False)

                if rows.shape != cols.shape:
                    raise ValueError('rows and cols must have the same shape,'
                                     ' rows: {}, cols: {}'.format(rows.shape, cols.shape))

                if is_scalar:
                    val = np.full(rows.size, val, dtype=float)
                    is_scalar = False
                elif val is not None:
                    # np.promote_types will choose the smallest dtype that can contain
                    # both arguments
                    val = atleast_1d(val)
                    safe_dtype = promote_types(val.dtype, float)
                    val = val.astype(safe_dtype, copy=False)

                    if rows.shape != val.shape:
                        raise ValueError('If rows and cols are specified, val must be a scalar or '
                                         'have the same shape, val: {}, '
                                         'rows/cols: {}'.format(val.shape, rows.shape))
                else:
                    val = np.zeros_like(rows, dtype=float)

                if rows.size > 0:
                    if rows.min() < 0:
                        # of, wrt = abs_key2rel_key(self, abs_key)
                        msg = '{}: d({})/d({}): row indices must be non-negative'
                        raise ValueError(msg.format(self.pathname, of, wrt))
                    if cols.min() < 0:
                        # of, wrt = abs_key2rel_key(self, abs_key)
                        msg = '{}: d({})/d({}): col indices must be non-negative'
                        raise ValueError(msg.format(self.pathname, of, wrt))
                    rows_max = rows.max()
                    cols_max = cols.max()
                else:
                    rows_max = cols_max = 0

        pattern_matches = self._find_partial_matches(of, wrt)
        abs2meta = self._var_abs2meta

        is_array = isinstance(val, ndarray)

        for of_bundle, wrt_bundle in product(*pattern_matches):
            of_pattern, of_matches = of_bundle
            wrt_pattern, wrt_matches = wrt_bundle
            if not of_matches:
                raise ValueError('No matches were found for of="{}"'.format(of_pattern))
            if not wrt_matches:
                raise ValueError('No matches were found for wrt="{}"'.format(wrt_pattern))

            for rel_key in product(of_matches, wrt_matches):
                abs_key = rel_key2abs_key(self, rel_key)
                if not dependent:
                    if abs_key in self._subjacs_info:
                        del self._subjacs_info[abs_key]
                    continue

                if abs_key in self._subjacs_info:
                    meta = self._subjacs_info[abs_key]
                else:
                    meta = SUBJAC_META_DEFAULTS.copy()

                meta['rows'] = rows
                meta['cols'] = cols
                meta['dependent'] = dependent
                meta['shape'] = shape = (abs2meta[abs_key[0]]['size'], abs2meta[abs_key[1]]['size'])

                if val is None:
                    # we can only get here if rows is None  (we're not sparse list format)
                    meta['value'] = np.zeros(shape)
                elif is_array:
                    if rows is None and val.shape != shape and val.size == shape[0] * shape[1]:
                        meta['value'] = val = val.copy().reshape(shape)
                    else:
                        meta['value'] = val.copy()
                elif is_scalar:
                    meta['value'] = np.full(shape, val, dtype=float)
                else:
                    meta['value'] = val

                if rows_max >= shape[0] or cols_max >= shape[1]:
                    of, wrt = abs_key2rel_key(self, abs_key)
                    msg = '{}: d({})/d({}): Expected {}x{} but declared at least {}x{}'
                    raise ValueError(msg.format(self.pathname, of, wrt, shape[0], shape[1],
                                                rows_max + 1, cols_max + 1))

                self._check_partials_meta(abs_key, meta['value'],
                                          shape if rows is None else (rows.shape[0], 1))
                self._subjacs_info[abs_key] = meta

    def _find_partial_matches(self, of, wrt):
        """
        Find all partial derivative matches from of and wrt.

        Parameters
        ----------
        of : str or list of str
            The relative name of the residual(s) that derivatives are being computed for.
            May also contain a glob pattern.
        wrt : str or list of str
            The relative name of the variables that derivatives are taken with respect to.
            This can contain the name of any input or output variable.
            May also contain a glob pattern.

        Returns
        -------
        tuple(list, list)
            Pair of lists containing pattern matches (if any). Returns (of_matches, wrt_matches)
            where of_matches is a list of tuples (pattern, matches) and wrt_matches is a list of
            tuples (pattern, output_matches, input_matches).
        """
        of_list = [of] if isinstance(of, string_types) else of
        wrt_list = [wrt] if isinstance(wrt, string_types) else wrt
        outs = list(self._var_allprocs_prom2abs_list['output'])
        ins = list(self._var_allprocs_prom2abs_list['input'])

        of_pattern_matches = [(pattern, find_matches(pattern, outs)) for pattern in of_list]
        wrt_pattern_matches = [(pattern, find_matches(pattern, outs + ins)) for pattern in wrt_list]
        return of_pattern_matches, wrt_pattern_matches

    def _check_partials_meta(self, abs_key, val, shape):
        """
        Check a given partial derivative and metadata for the correct shapes.

        Parameters
        ----------
        abs_key : tuple(str, str)
            The of/wrt pair (given absolute names) defining the partial derivative.
        val : ndarray
            Subjac value.
        shape : tuple
            Expected shape of val.
        """
        out_size, in_size = shape

        if in_size == 0 and self.comm.rank != 0:  # 'inactive' component
            return

        if val is not None:
            val_shape = val.shape
            if len(val_shape) == 1:
                val_out, val_in = val_shape[0], 1
            else:
                val_out, val_in = val.shape
            if val_out > out_size or val_in > in_size:
                of, wrt = abs_key2rel_key(self, abs_key)
                msg = '{}: d({})/d({}): Expected {}x{} but val is {}x{}'
                raise ValueError(msg.format(self.pathname, of, wrt, out_size, in_size,
                                            val_out, val_in))

    def _set_partials_meta(self):
        """
        Set subjacobian info into our jacobian.
        """
        for key, meta in iteritems(self._subjacs_info):

            if 'method' in meta:
                method = meta['method']
                if method is not None and method in self._approx_schemes:
                    self._approx_schemes[method].add_approximation(key, meta)

        for approx in itervalues(self._approx_schemes):
            approx._init_approximations()

    def _guess_nonlinear(self):
        """
        Provide initial guess for states.

        Does nothing on any non-implicit component.
        """
        pass

    def _clear_iprint(self):
        """
        Clear out the iprint stack from the solvers.

        Components don't have nested solvers, so do nothing to prevent errors.
        """
        pass
