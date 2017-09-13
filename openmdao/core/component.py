"""Define the Component class."""

from __future__ import division

from fnmatch import fnmatchcase
import numpy as np
from itertools import product
from six import string_types, iteritems, itervalues
from scipy.sparse import issparse
from copy import deepcopy
from collections import OrderedDict, Iterable

from openmdao.approximation_schemes.complex_step import ComplexStep
from openmdao.approximation_schemes.finite_difference import FiniteDifference
from openmdao.core.system import System
from openmdao.jacobians.assembled_jacobian import SUBJAC_META_DEFAULTS
from openmdao.utils.units import valid_units
from openmdao.utils.general_utils import format_as_float_or_array, ensure_compatible, \
    warn_deprecation, ContainsAll
from openmdao.utils.name_maps import rel_key2abs_key, abs_key2rel_key


class Component(System):
    """
    Base Component class; not to be directly instantiated.

    Attributes
    ----------
    matrix_free : Bool
        This is set to True if the component overrides the appropriate function with a user-defined
        matrix vector product with the Jacobian.
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
        self.matrix_free = False
        self.distributed = False

        super(Component, self).__init__(**kwargs)

        self._approx_schemes = OrderedDict()

        self._var_rel_names = {'input': [], 'output': []}
        self._var_rel2data_io = {}

        self._static_var_rel_names = {'input': [], 'output': []}
        self._static_var_rel2data_io = {}

        self._declared_partials = []
        self._approximated_partials = []

        # Defaults for the finite difference check used by check_partial_derivs.
        meta = self.metadata
        meta.declare('check_method', default='fd', values=['fd', 'cs'],
                     desc='Method for check: "fd" for finite difference, "cs" for complex step.')
        meta.declare('check_form', default='forward', values=['forward', 'central', 'backward'],
                     desc='Finite difference form for check, can be "forward", "central", or '
                     '"backward".')
        meta.declare('check_step', default=1.0e-6, desc='Step size for finite difference check.')
        meta.declare('check_step_calc', default='abs', values=['abs', 'rel'],
                     desc='Type of step calculation for check, can be "abs" for absolute '
                     '(default) or "rel" for relative.')

    def setup(self):
        """
        Declare inputs and outputs.

        Available attributes:
            name
            pathname
            comm
            metadata
        """
        pass

    def _setup_vars(self, recurse=True):
        """
        Call setup in components and count variables, total and by var_set.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(Component, self)._setup_vars()
        num_var = self._num_var
        num_var_byset = self._num_var_byset
        data = self._var_rel2data_io

        for vec_name in self._lin_rel_vec_name_list:
            num_var[vec_name] = {}
            num_var_byset[vec_name] = {}
            # Compute num_var
            for type_ in ['input', 'output']:
                relnames = self._var_allprocs_relevant_names[vec_name][type_]
                num_var[vec_name][type_] = len(relnames)

                num_var_byset[vec_name][type_] = vbyset = {}
                # Compute num_var_byset
                for name in relnames:
                    set_name = data[name.rsplit('.', 1)[-1]]['metadata']['var_set']
                    if set_name not in vbyset:
                        vbyset[set_name] = 0
                    vbyset[set_name] += 1

        self._num_var['nonlinear'] = self._num_var['linear']
        self._num_var_byset['nonlinear'] = self._num_var_byset['linear']

    def _setup_var_data(self, recurse=True):
        """
        Compute the list of abs var names, abs/prom name maps, and metadata dictionaries.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
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

        meta_names = {
            'input': ('units', 'shape', 'var_set'),
            'output': ('units', 'shape', 'var_set', 'ref', 'ref0', 'distributed'),
        }

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
                allprocs_abs2meta[type_][abs_name] = {
                    meta_name: metadata[meta_name]
                    for meta_name in meta_names[type_]
                }

                # Compute abs2meta
                abs2meta[type_][abs_name] = metadata

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
        relevant = self._relevant
        vec_names = self._lin_rel_vec_name_list

        sizes = self._var_sizes
        sizes_byset = self._var_sizes_byset

        # Initialize empty arrays
        for vec_name in vec_names:
            sizes[vec_name] = {}
            sizes_byset[vec_name] = {}

            for type_ in ('input', 'output'):
                sizes[vec_name][type_] = np.zeros((nproc, self._num_var[vec_name][type_]), int)

                sizes_byset[vec_name][type_] = {}
                for set_name, nvars in iteritems(self._num_var_byset[vec_name][type_]):
                    sizes_byset[vec_name][type_][set_name] = np.zeros((nproc, nvars), int)

            # Compute _var_sizes and _var_sizes_byset
            for type_ in ('input', 'output'):
                sz = sizes[vec_name][type_]
                sz_byset = sizes_byset[vec_name][type_]
                abs2meta_t = self._var_abs2meta[type_]
                allprocs_abs2idx_byset_t = self._var_allprocs_abs2idx_byset[vec_name][type_]
                for idx, abs_name in enumerate(self._var_allprocs_relevant_names[vec_name][type_]):
                    meta = abs2meta_t[abs_name]
                    set_name = meta['var_set']
                    size = np.prod(meta['shape'])
                    idx_byset = allprocs_abs2idx_byset_t[abs_name]

                    sz[iproc, idx] = size
                    sz_byset[set_name][iproc, idx_byset] = size

        if self.comm.size > 1:
            for vec_name in vec_names:
                sizes = self._var_sizes[vec_name]
                sizes_byset = self._var_sizes_byset[vec_name]
                for type_ in ['input', 'output']:
                    self.comm.Allgather(sizes[type_][iproc, :], sizes[type_])
                    for set_name, sbyset in iteritems(sizes_byset[type_]):
                        self.comm.Allgather(sbyset[iproc, :], sbyset)

        self._var_sizes['nonlinear'] = self._var_sizes['linear']
        self._var_sizes_byset['nonlinear'] = self._var_sizes_byset['linear']

        self._setup_global_shapes()

    def _setup_partials(self, recurse=True):
        """
        Process all partials and approximations that the user declared.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(Component, self)._setup_partials()

        for of, wrt, dependent, rows, cols, val in self._declared_partials:
            self._declare_partials(of, wrt, dependent=dependent, rows=rows, cols=cols, val=val)

        for of, wrt, method, kwargs in self._approximated_partials:
            self._approx_partials(of, wrt, method=method, **kwargs)

    def add_input(self, name, val=1.0, shape=None, src_indices=None, flat_src_indices=None,
                  units=None, desc='', var_set=0):
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
        var_set : hashable object
            For advanced users only. ID or color for this variable, relevant for
            reconfigurability. Default is 0.

        Returns
        -------
        dict
            metadata for added variable
        """
        if self._static_mode:
            warn_deprecation("In the future, the 'add_input' method must be "
                             "called from 'setup' rather than "
                             "in the '__init__' function.")

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
        if not np.isscalar(val) and not isinstance(val, (list, tuple, np.ndarray, Iterable)):
            raise TypeError('The val argument should be a float, list, tuple, ndarray or Iterable')
        if shape is not None and not isinstance(shape, (int, tuple, list, np.integer)):
            raise TypeError("The shape argument should be an int, tuple, or list but "
                            "a '%s' was given" % type(shape))
        if src_indices is not None and not isinstance(src_indices, (int, list, tuple,
                                                                    np.ndarray, Iterable)):
            raise TypeError('The src_indices argument should be an int, list, '
                            'tuple, ndarray or Iterable')
        if units is not None and not isinstance(units, str):
            raise TypeError('The units argument should be a str or None')

        # Check that units are valid
        if units is not None and not valid_units(units):
            raise ValueError("The units '%s' are invalid" % units)

        metadata = {}

        # value, shape: based on args, making sure they are compatible
        metadata['value'], metadata['shape'] = ensure_compatible(name, val,
                                                                 shape,
                                                                 src_indices)

        # src_indices: None or ndarray
        if src_indices is None:
            metadata['src_indices'] = None
        else:
            metadata['src_indices'] = np.atleast_1d(src_indices)
        metadata['flat_src_indices'] = flat_src_indices

        # units: taken as is
        metadata['units'] = units

        # desc: taken as is
        metadata['desc'] = desc

        # var_set: taken as is
        metadata['var_set'] = var_set

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
            'type': 'input', 'metadata': metadata}
        var_rel_names['input'].append(name)

        return metadata

    def add_output(self, name, val=1.0, shape=None, units=None, res_units=None, desc='',
                   lower=None, upper=None, ref=1.0, ref0=0.0, res_ref=1.0, var_set=0):
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
        ref : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 1. Default is 1.
        ref0 : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 0. Default is 0.
        res_ref : float
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 1. Default is 1.
        var_set : hashable object
            For advanced users only. ID or color for this variable, relevant for reconfigurability.
            Default is 0.

        Returns
        -------
        dict
            metadata for added variable
        """
        if self._static_mode:
            warn_deprecation("In the future, the 'add_output' method must be "
                             "called from 'setup' rather than "
                             "in the '__init__' function.")

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
        if not np.isscalar(val) and not isinstance(val, (list, tuple, np.ndarray, Iterable)):
            raise TypeError('The val argument should be a float, list, tuple, or ndarray')
        if not np.isscalar(ref) and not isinstance(val, (list, tuple, np.ndarray, Iterable)):
            raise TypeError('The ref argument should be a float, list, tuple, or ndarray')
        if not np.isscalar(ref0) and not isinstance(val, (list, tuple, np.ndarray, Iterable)):
            raise TypeError('The ref0 argument should be a float, list, tuple, or ndarray')
        if not np.isscalar(res_ref) and not isinstance(val, (list, tuple, np.ndarray, Iterable)):
            raise TypeError('The res_ref argument should be a float, list, tuple, or ndarray')
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
        metadata['value'], metadata['shape'] = ensure_compatible(name, val, shape)

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
        for item, msg in zip([ref, ref0, res_ref],
                             ['ref', 'ref0', 'res_ref']):
            if not np.isscalar(item) and \
               np.atleast_1d(item).shape != metadata['shape']:
                raise ValueError('The %s argument has the wrong shape' % msg)

        ref = format_as_float_or_array('ref', ref, flatten=True)
        ref0 = format_as_float_or_array('ref0', ref0, flatten=True)
        res_ref = format_as_float_or_array('res_ref', res_ref, flatten=True)

        # ref, ref0, res_ref: taken as is
        metadata['ref'] = ref
        metadata['ref0'] = ref0
        metadata['res_ref'] = res_ref

        # var_set: taken as is
        metadata['var_set'] = var_set

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
            'type': 'output', 'metadata': metadata}
        var_rel_names['output'].append(name)

        return metadata

    def approx_partials(self, of, wrt, method='fd', **kwargs):
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
                - 'fd': Finite Difference, 'cs': Complex Step
        **kwargs : dict
            Keyword arguments for controlling the behavior of the approximation.
        """
        supported_methods = {'fd': FiniteDifference,
                             'cs': ComplexStep}

        if method not in supported_methods:
            msg = 'Method "{}" is not supported, method must be one of {}'
            raise ValueError(msg.format(method, supported_methods.keys()))

        if method not in self._approx_schemes:
            self._approx_schemes[method] = supported_methods[method]()

        self._approximated_partials.append((of, wrt, method, kwargs))

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

            for rel_key in product(of_matches, wrt_matches):
                meta_changes = {
                    'method': method,
                }
                abs_key = rel_key2abs_key(self, rel_key)
                meta = self._subjacs_info.get(abs_key, SUBJAC_META_DEFAULTS.copy())
                meta.update(meta_changes)
                meta.update(kwargs)
                self._subjacs_info[abs_key] = meta

    def declare_partials(self, of, wrt, dependent=True,
                         rows=None, cols=None, val=None):
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
            set_subjac_info is not called for a given pair, then a dense
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
        # If only one of rows/cols is specified
        if (rows is None) ^ (cols is None):
            raise ValueError('If one of rows/cols is specified, then both must be specified')

        self._declared_partials.append((of, wrt, dependent, rows, cols, val))

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
            set_subjac_info is not called for a given pair, then a dense
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
        if val is not None and not issparse(val):
            val = np.atleast_1d(val)
            # np.promote_types  will choose the smallest dtype that can contain both arguments
            safe_dtype = np.promote_types(val.dtype, float)
            val = val.astype(safe_dtype, copy=False)

        if rows is not None:
            rows = np.array(rows, dtype=int, copy=False)
            cols = np.array(cols, dtype=int, copy=False)

            if rows.shape != cols.shape:
                raise ValueError('rows and cols must have the same shape,'
                                 ' rows: {}, cols: {}'.format(rows.shape, cols.shape))

            if val is not None and val.shape != (1,) and rows.shape != val.shape:
                raise ValueError('If rows and cols are specified, val must be a scalar or have the '
                                 'same shape, val: {}, rows/cols: {}'.format(val.shape, rows.shape))

            if val is None:
                val = np.zeros_like(rows, dtype=float)

        pattern_matches = self._find_partial_matches(of, wrt)

        multiple_items = False

        for of_bundle, wrt_bundle in product(*pattern_matches):
            of_pattern, of_matches = of_bundle
            wrt_pattern, wrt_matches = wrt_bundle
            if not of_matches:
                raise ValueError('No matches were found for of="{}"'.format(of_pattern))
            if not wrt_matches:
                raise ValueError('No matches were found for wrt="{}"'.format(wrt_pattern))

            make_copies = (multiple_items
                           or len(of_matches) > 1
                           or len(wrt_matches) > 1)
            # Setting this to true means that future loop iterations (i.e. if there are multiple
            # items in either of or wrt) will make copies.
            multiple_items = True

            for rel_key in product(of_matches, wrt_matches):
                meta_changes = {
                    'rows': rows,
                    'cols': cols,
                    'value': deepcopy(val) if make_copies else val,
                    'dependent': dependent
                }
                abs_key = rel_key2abs_key(self, rel_key)
                meta = self._subjacs_info.get(abs_key, SUBJAC_META_DEFAULTS.copy())
                meta.update(meta_changes)
                self._check_partials_meta(abs_key, meta)
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
        glob_patterns = {'*', '?', '['}
        outs = list(self._var_allprocs_prom2abs_list['output'].keys())
        ins = list(self._var_allprocs_prom2abs_list['input'].keys())

        def find_matches(pattern, var_list):
            if glob_patterns.intersection(pattern):
                return [name for name in var_list if fnmatchcase(name, pattern)]
            elif pattern in var_list:
                return [pattern]
            return []

        of_pattern_matches = [(pattern, find_matches(pattern, outs)) for pattern in of_list]
        wrt_pattern_matches = [(pattern, find_matches(pattern, outs + ins)) for pattern in wrt_list]
        return of_pattern_matches, wrt_pattern_matches

    def _check_partials_meta(self, abs_key, meta):
        """
        Check a given partial derivative and metadata for the correct shapes.

        Parameters
        ----------
        abs_key : tuple(str,str)
            The of/wrt pair (given absolute names) defining the partial derivative.
        meta : dict
            Metadata dictionary from declare_partials.
        """
        of, wrt = abs_key2rel_key(self, abs_key)
        if meta['dependent']:
            out_size = np.prod(self._var_abs2meta['output'][abs_key[0]]['shape'])
            if abs_key[1] in self._var_abs2meta['input']:
                in_size = np.prod(self._var_abs2meta['input'][abs_key[1]]['shape'])
            else:  # assume output (or get a KeyError)
                in_size = np.prod(self._var_abs2meta['output'][abs_key[1]]['shape'])

            if in_size == 0 and self.comm.rank != 0:  # 'inactive' component
                return

            rows = meta['rows']
            cols = meta['cols']
            if not (rows is None or rows.size == 0):
                if rows.min() < 0:
                    msg = '{}: d({})/d({}): row indices must be non-negative'
                    raise ValueError(msg.format(self.pathname, of, wrt))
                if cols.min() < 0:
                    msg = '{}: d({})/d({}): col indices must be non-negative'
                    raise ValueError(msg.format(self.pathname, of, wrt))
                if rows.max() >= out_size or cols.max() >= in_size:
                    msg = '{}: d({})/d({}): Expected {}x{} but declared at least {}x{}'
                    raise ValueError(msg.format(
                        self.pathname, of, wrt,
                        out_size, in_size,
                        rows.max() + 1, cols.max() + 1))
            elif meta['value'] is not None:
                val = meta['value']
                val_shape = val.shape
                if len(val_shape) == 1:
                    val_out, val_in = val_shape[0], 1
                else:
                    val_out, val_in = val.shape
                if val_out > out_size or val_in > in_size:
                    msg = '{}: d({})/d({}): Expected {}x{} but val is {}x{}'
                    raise ValueError(msg.format(
                        self.pathname, of, wrt,
                        out_size, in_size,
                        val_out, val_in))

    def _set_partials_meta(self):
        """
        Set subjacobian info into our jacobian.
        """
        with self.jacobian_context() as J:
            for key, meta in iteritems(self._subjacs_info):
                self._check_partials_meta(key, meta)
                J._set_partials_meta(key, meta)

                method = meta.get('method', False)
                if method and meta['dependent']:
                    self._approx_schemes[method].add_approximation(key, meta)

        for approx in itervalues(self._approx_schemes):
            approx._init_approximations()

    def _guess_nonlinear(self):
        """
        Provide initial guess for states.

        Does nothing on any non-implicit component.
        """
        pass
