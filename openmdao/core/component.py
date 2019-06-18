"""Define the Component class."""

from __future__ import division

from collections import OrderedDict, Iterable, Counter, defaultdict
from itertools import product
from six import string_types, iteritems, itervalues
import copy

import numpy as np
from numpy import ndarray, isscalar, atleast_1d, atleast_2d, promote_types
from scipy.sparse import issparse

from openmdao.core.system import System, _supported_methods, _DEFAULT_COLORING_META
from openmdao.approximation_schemes.complex_step import ComplexStep
from openmdao.approximation_schemes.finite_difference import FiniteDifference
from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian
from openmdao.vectors.vector import INT_DTYPE
from openmdao.utils.units import valid_units
from openmdao.utils.name_maps import rel_key2abs_key, abs_key2rel_key, rel_name2abs_name
from openmdao.utils.mpi import MPI
from openmdao.utils.general_utils import format_as_float_or_array, ensure_compatible, \
    warn_deprecation, find_matches, simple_warning
from openmdao.utils.coloring import _DYN_COLORING


# the following metadata will be accessible for vars on all procs
global_meta_names = {
    'input': ('units', 'shape', 'size', 'distributed'),
    'output': ('units', 'shape', 'size',
               'ref', 'ref0', 'res_ref', 'distributed', 'lower', 'upper'),
}

_full_slice = slice(None)


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
    _approx_schemes : OrderedDict
        A mapping of approximation types to the associated ApproximationScheme.
    _var_rel2meta : dict
        Dictionary mapping relative names to metadata.
        This is only needed while adding inputs and outputs. During setup, these are used to
        build the dictionaries of metadata.
    _static_var_rel2meta : dict
        Static version of above - stores data for variables added outside of setup.
    _var_rel_names : {'input': [str, ...], 'output': [str, ...]}
        List of relative names of owned variables existing on current proc.
        This is only needed while adding inputs and outputs. During setup, these are used to
        determine the list of absolute names.
    _static_var_rel_names : dict
        Static version of above - stores names of variables added outside of setup.
    _declared_partials : dict
        Cached storage of user-declared partials.
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
        super(Component, self).__init__(**kwargs)

        self._var_rel_names = {'input': [], 'output': []}
        self._var_rel2meta = {}

        self._static_var_rel_names = {'input': [], 'output': []}
        self._static_var_rel2meta = {}

        self._declared_partials = defaultdict(dict)
        self._declared_partial_checks = []

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super(Component, self)._declare_options()

        self.options.declare('distributed', types=bool, default=False,
                             desc='True if the component has variables that are distributed '
                                  'across multiple processes.')

    @property
    def distributed(self):
        """
        Provide 'distributed' property for backwards compatibility.

        Returns
        -------
        bool
            reference to the 'distributed' option.
        """
        warn_deprecation("The 'distributed' property provides backwards compatibility "
                         "with OpenMDAO <= 2.4.0 ; use the 'distributed' option instead.")
        return self.options['distributed']

    @distributed.setter
    def distributed(self, val):
        """
        Provide for setting of the 'distributed' property for backwards compatibility.

        Parameters
        ----------
        val : bool
            True if the component has variables that are distributed across multiple processes.
        """
        warn_deprecation("The 'distributed' property provides backwards compatibility "
                         "with OpenMDAO <= 2.4.0 ; use the 'distributed' option instead.")
        self.options['distributed'] = val

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

    def _setup_procs(self, pathname, comm, mode, prob_options):
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
        prob_options : OptionsDictionary
            Problem level options.
        """
        self.pathname = pathname
        self._problem_options = prob_options

        orig_comm = comm
        if self._num_par_fd > 1:
            if comm.size > 1:
                comm = self._setup_par_fd_procs(comm)
            elif not MPI:
                msg = ("'%s': MPI is not active but num_par_fd = %d. No parallel finite difference "
                       "will be performed." % (self.pathname, self._num_par_fd))
                simple_warning(msg)

        self.comm = comm
        self._mode = mode
        self._subsystems_proc_range = []
        self._first_call_to_linearize = True

        # Clear out old variable information so that we can call setup on the component.
        self._var_rel_names = {'input': [], 'output': []}
        self._var_rel2meta = {}
        self._design_vars = OrderedDict()
        self._responses = OrderedDict()

        self._static_mode = False
        self._var_rel2meta.update(self._static_var_rel2meta)
        for type_ in ['input', 'output']:
            self._var_rel_names[type_].extend(self._static_var_rel_names[type_])
        self._design_vars.update(self._static_design_vars)
        self._responses.update(self._static_responses)
        self.setup()

        # check to make sure that if num_par_fd > 1 that this system is actually doing FD.
        # Unfortunately we have to do this check after system setup has been called because that's
        # when declare_partials generally happens, so we raise an exception here instead of just
        # resetting the value of num_par_fd (because the comm has already been split and possibly
        # used by the system setup).
        if self._num_par_fd > 1 and orig_comm.size > 1 and not (self._owns_approx_jac or
                                                                self._approx_schemes):
            raise RuntimeError("'%s': num_par_fd is > 1 but no FD is active." % self.pathname)

        # check here if declare_coloring was called during setup but declare_partials
        # wasn't.  If declare partials wasn't called, call it with of='*' and wrt='*' so we'll
        # have something to color.
        if self._coloring_info['coloring'] is not None:
            for key, meta in iteritems(self._declared_partials):
                if 'method' in meta and meta['method'] is not None:
                    break
            else:
                method = self._coloring_info['method']
                simple_warning("%s: declare_coloring or use_fixed_coloring was called but no approx"
                               " partials were declared.  Declaring all partials as approximated "
                               "using default metadata and method='%s'." % (self.pathname, method))
                self.declare_partials('*', '*', method=method)

        self._static_mode = True

        if self.options['distributed']:
            if self._distributed_vector_class is not None:
                self._vector_class = self._distributed_vector_class
            else:
                simple_warning("The 'distributed' option is set to True for Component %s, "
                               "but there is no distributed vector implementation (MPI/PETSc) "
                               "available. The default non-distributed vectors will be used."
                               % pathname)
                self._vector_class = self._local_vector_class
        else:
            self._vector_class = self._local_vector_class

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
        allprocs_prom2abs_list = self._var_allprocs_prom2abs_list
        abs2prom = self._var_abs2prom
        allprocs_abs2meta = self._var_allprocs_abs2meta
        abs2meta = self._var_abs2meta

        # Compute the prefix for turning rel/prom names into abs names
        prefix = self.pathname + '.' if self.pathname else ''

        for type_ in ['input', 'output']:
            for prom_name in self._var_rel_names[type_]:
                abs_name = prefix + prom_name
                metadata = self._var_rel2meta[prom_name]

                # Compute allprocs_abs_names, abs_names
                allprocs_abs_names[type_].append(abs_name)

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

            for prom_name, val in iteritems(self._var_discrete[type_]):
                abs_name = prefix + prom_name
                allprocs_prom2abs_list[type_][prom_name] = [abs_name]
                self._var_allprocs_discrete[type_][abs_name] = val

        self._var_allprocs_abs2prom = self._var_abs2prom

        self._var_abs_names = self._var_allprocs_abs_names
        if self._var_discrete['input'] or self._var_discrete['output']:
            self._discrete_inputs = _DictValues(self._var_discrete['input'])
            self._discrete_outputs = _DictValues(self._var_discrete['output'])
        else:
            self._discrete_inputs = self._discrete_outputs = ()

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

        sizes = self._var_sizes
        abs2meta = self._var_abs2meta

        if self._use_derivatives:
            vec_names = self._lin_rel_vec_name_list
        else:
            vec_names = self._vec_names

        # Initialize empty arrays
        for vec_name in vec_names:
            # at component level, _var_allprocs_* is the same as var_* since all vars exist in all
            # procs for a given component, so we don't have to mess with figuring out what vars are
            # local.
            if self._use_derivatives:
                relnames = self._var_allprocs_relevant_names[vec_name]
            else:
                relnames = self._var_allprocs_abs_names

            sizes[vec_name] = {}
            for type_ in ('input', 'output'):
                sizes[vec_name][type_] = sz = np.zeros((nproc, len(relnames[type_])), int)

                # Compute _var_sizes
                for idx, abs_name in enumerate(relnames[type_]):
                    sz[iproc, idx] = abs2meta[abs_name]['size']

        if nproc > 1:
            for vec_name in vec_names:
                sizes = self._var_sizes[vec_name]
                if self.options['distributed']:
                    for type_ in ['input', 'output']:
                        sizes_in = copy.deepcopy(sizes[type_][iproc, :])
                        self.comm.Allgather(sizes_in, sizes[type_])
                else:
                    # if component isn't distributed, we don't need to allgather sizes since
                    # they'll all be the same.
                    for type_ in ['input', 'output']:
                        sizes[type_] = np.tile(sizes[type_][iproc], (nproc, 1))

        if self._use_derivatives:
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

        for key, dct in iteritems(self._declared_partials):
            of, wrt = key
            self._declare_partials(of, wrt, dct)

    def _update_wrt_matches(self):
        """
        Determine the list of wrt variables that match the wildcard(s) given in declare_coloring.
        """
        info = self._coloring_info
        ofs, allwrt = self._get_partials_varlists()
        wrt_patterns = info['wrt_patterns']
        matches_prom = set()
        for w in wrt_patterns:
            matches_prom.update(find_matches(w, allwrt))

        # error if nothing matched
        if not matches_prom:
            raise ValueError("Invalid 'wrt' variable(s) specified for colored approx partial "
                             "options on Component '{}': {}.".format(self.pathname, wrt_patterns))

        info['wrt_matches_prom'] = matches_prom
        info['wrt_matches'] = [rel_name2abs_name(self, n) for n in matches_prom]

    def _update_subjac_sparsity(self, sparsity):
        """
        Update subjac sparsity info based on the given coloring.

        The sparsity of the partial derivatives in this component will be used when computing
        the sparsity of the total jacobian for the entire model.  Without this, all of this
        component's partials would be treated as dense, resulting in an overly conservative
        coloring of the total jacobian.

        Parameters
        ----------
        sparsity : dict
            A nested dict of the form dct[of][wrt] = (rows, cols, shape)
        """
        # sparsity uses relative names, so we need to convert to absolute
        pathname = self.pathname
        for of, sub in iteritems(sparsity):
            of_abs = '.'.join((pathname, of)) if pathname else of
            for wrt, tup in iteritems(sub):
                wrt_abs = '.'.join((pathname, wrt)) if pathname else wrt
                abs_key = (of_abs, wrt_abs)
                if abs_key in self._subjacs_info:
                    # add sparsity info to existing partial info
                    self._subjacs_info[abs_key]['sparsity'] = tup

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
            raise TypeError('The name argument should be a string.')
        if not name:
            raise NameError('The name argument should be a non-empty string.')
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
        metadata['distributed'] = self.options['distributed']

        # We may not know the pathname yet, so we have to use name for now, instead of abs_name.
        if self._static_mode:
            var_rel2meta = self._static_var_rel2meta
            var_rel_names = self._static_var_rel_names
        else:
            var_rel2meta = self._var_rel2meta
            var_rel_names = self._var_rel_names

        # Disallow dupes
        if name in var_rel2meta:
            msg = "Variable name '{}' already exists.".format(name)
            raise ValueError(msg)

        var_rel2meta[name] = metadata
        var_rel_names['input'].append(name)

        return metadata

    def add_discrete_input(self, name, val, desc=''):
        """
        Add a discrete input variable to the component.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : a picklable object
            The initial value of the variable being added.
        desc : str
            description of the variable

        Returns
        -------
        dict
            metadata for added variable
        """
        # First, type check all arguments
        if not isinstance(name, str):
            raise TypeError('The name argument should be a string.')
        if not name:
            raise NameError('The name argument should be a non-empty string.')
        if not _valid_var_name(name):
            raise NameError("'%s' is not a valid input name." % name)

        metadata = {
            'value': val,
            'type': type(val),
            'desc': desc,
        }

        if self._static_mode:
            var_rel2meta = self._static_var_rel2meta
        else:
            var_rel2meta = self._var_rel2meta

        # Disallow dupes
        if name in var_rel2meta:
            msg = "Variable name '{}' already exists.".format(name)
            raise ValueError(msg)

        var_rel2meta[name] = self._var_discrete['input'][name] = metadata

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

        if not isinstance(name, str):
            raise TypeError('The name argument should be a string.')
        if not name:
            raise NameError('The name argument should be a non-empty string.')
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

        metadata['distributed'] = self.options['distributed']

        # We may not know the pathname yet, so we have to use name for now, instead of abs_name.
        if self._static_mode:
            var_rel2meta = self._static_var_rel2meta
            var_rel_names = self._static_var_rel_names
        else:
            var_rel2meta = self._var_rel2meta
            var_rel_names = self._var_rel_names

        # Disallow dupes
        if name in var_rel2meta:
            msg = "Variable name '{}' already exists.".format(name)
            raise ValueError(msg)

        var_rel2meta[name] = metadata
        var_rel_names['output'].append(name)

        return metadata

    def add_discrete_output(self, name, val, desc=''):
        """
        Add an output variable to the component.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : a picklable object
            The initial value of the variable being added.
        desc : str
            description of the variable.

        Returns
        -------
        dict
            metadata for added variable
        """
        if not isinstance(name, str):
            raise TypeError('The name argument should be a string.')
        if not name:
            raise NameError('The name argument should be a non-empty string.')
        if not _valid_var_name(name):
            raise NameError("'%s' is not a valid output name." % name)

        metadata = {
            'value': val,
            'type': type(val),
            'desc': desc
        }

        if self._static_mode:
            var_rel2meta = self._static_var_rel2meta
        else:
            var_rel2meta = self._var_rel2meta

        # Disallow dupes
        if name in var_rel2meta:
            msg = "Variable name '{}' already exists.".format(name)
            raise ValueError(msg)

        var_rel2meta[name] = self._var_discrete['output'][name] = metadata

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
        self._has_approx = True

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
                meta = info[abs_key]
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

        Returns
        -------
        dict
            Metadata dict for the specified partial(s).
        """
        try:
            method_func = _supported_methods[method]
        except KeyError:
            msg = 'Method "{}" is not supported, method must be one of {}'
            raise ValueError(msg.format(method, list(_supported_methods)))

        if isinstance(of, list):
            of = tuple(of)
        if isinstance(wrt, list):
            wrt = tuple(wrt)

        meta = self._declared_partials[of, wrt]
        meta['dependent'] = dependent

        # If only one of rows/cols is specified
        if (rows is None) ^ (cols is None):
            raise ValueError('If one of rows/cols is specified, then both must be specified')

        if dependent:
            meta['value'] = val
            if rows is not None:
                meta['rows'] = rows
                meta['cols'] = cols

                # First, check the length of rows and cols to catch this easy mistake and give a
                # clear message.
                if len(cols) != len(rows):
                    msg = "{0}: declare_partials has been called with rows and cols, which" + \
                          " should be arrays of equal length, but rows is length {1} while " + \
                          "cols is length {2}."
                    raise RuntimeError(msg.format(self.pathname, len(rows), len(cols)))

                # Check for repeated rows/cols indices.
                idxset = set(zip(rows, cols))
                if len(rows) - len(idxset) > 0:
                    dups = [n for n, val in iteritems(Counter(zip(rows, cols))) if val > 1]
                    raise RuntimeError("%s: declare_partials has been called with rows and cols "
                                       "that specify the following duplicate subjacobian entries: "
                                       "%s." % (self.pathname, sorted(dups)))

        if method_func is not None:
            # we're doing approximations
            self._has_approx = True
            meta['method'] = method
            self._get_approx_scheme(method)

            default_opts = method_func.DEFAULT_OPTIONS

            # If rows/cols is specified
            if rows is not None or cols is not None:
                raise ValueError('Sparse FD specification not supported yet.')
        else:
            default_opts = ()

        if step:
            if 'step' in default_opts:
                meta['step'] = step
            else:
                raise RuntimeError("'step' is not a valid option for '%s'" % method)
        if form:
            if 'form' in default_opts:
                meta['form'] = form
            else:
                raise RuntimeError("'form' is not a valid option for '%s'" % method)
        if step_calc:
            if 'step_calc' in default_opts:
                meta['step_calc'] = step_calc
            else:
                raise RuntimeError("'step_calc' is not a valid option for '%s'" % method)

        return meta

    def declare_coloring(self,
                         wrt=_DEFAULT_COLORING_META['wrt_patterns'],
                         method=_DEFAULT_COLORING_META['method'],
                         form=None,
                         step=None,
                         per_instance=_DEFAULT_COLORING_META['per_instance'],
                         num_full_jacs=_DEFAULT_COLORING_META['num_full_jacs'],
                         tol=_DEFAULT_COLORING_META['tol'],
                         orders=_DEFAULT_COLORING_META['orders'],
                         perturb_size=_DEFAULT_COLORING_META['perturb_size'],
                         show_summary=_DEFAULT_COLORING_META['show_summary'],
                         show_sparsity=_DEFAULT_COLORING_META['show_sparsity']):
        """
        Set options for deriv coloring of a set of wrt vars matching the given pattern(s).

        Parameters
        ----------
        wrt : str or list of str
            The name or names of the variables that derivatives are taken with respect to.
            This can contain input names, output names, or glob patterns.
        method : str
            Method used to compute derivative: "fd" for finite difference, "cs" for complex step.
        form : str
            Finite difference form, can be "forward", "central", or "backward". Leave
            undeclared to keep unchanged from previous or default value.
        step : float
            Step size for finite difference. Leave undeclared to keep unchanged from previous
            or default value.
        per_instance : bool
            If True, a separate coloring will be generated for each instance of a given class.
            Otherwise, only one coloring for a given class will be generated and all instances
            of that class will use it.
        num_full_jacs : int
            Number of times to repeat partial jacobian computation when computing sparsity.
        tol : float
            Tolerance used to determine if an array entry is nonzero during sparsity determination.
        orders : int
            Number of orders above and below the tolerance to check during the tolerance sweep.
        perturb_size : float
            Size of input/output perturbation during generation of sparsity.
        show_summary : bool
            If True, display summary information after generating coloring.
        show_sparsity : bool
            If True, display sparsity with coloring info after generating coloring.
        """
        super(Component, self).declare_coloring(wrt, method, form, step, per_instance,
                                                num_full_jacs,
                                                tol, orders, perturb_size,
                                                show_summary, show_sparsity)
        # create approx partials for all matches
        meta = self.declare_partials('*', wrt, method=method, step=step, form=form)
        meta['coloring'] = True

    def set_check_partial_options(self, wrt, method='fd', form=None, step=None, step_calc=None,
                                  directional=False):
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
        directional : bool
            Set to True to perform a single directional derivative for each vector variable in the
            pattern named in wrt.
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

        if not isinstance(directional, bool):
            msg = "The value of 'directional' must be True or False, but a type " \
                  "of '{}' was provided."
            raise ValueError(msg.format(type(directional).__name__))

        wrt_list = [wrt] if isinstance(wrt, string_types) else wrt
        self._declared_partial_checks.append((wrt_list, method, form, step, step_calc,
                                              directional))

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
        of, wrt = self._get_potential_partials_lists()
        invalid_wrt = []

        for wrt_list, method, form, step, step_calc, directional in self._declared_partial_checks:
            for pattern in wrt_list:
                matches = find_matches(pattern, wrt)

                # if a non-wildcard var name was specified and not found, save for later Exception
                if len(matches) == 0 and _valid_var_name(pattern):
                    invalid_wrt.append(pattern)

                for match in matches:
                    if match in opts:
                        opt = opts[match]

                        # New assignments take precedence
                        keynames = ['method', 'form', 'step', 'step_calc', 'directional']
                        for name, value in zip(keynames,
                                               [method, form, step, step_calc, directional]):
                            if value is not None:
                                opt[name] = value

                    else:
                        opts[match] = {'method': method,
                                       'form': form,
                                       'step': step,
                                       'step_calc': step_calc,
                                       'directional': directional}

        if invalid_wrt:
            if len(invalid_wrt) == 1:
                msg = "Invalid 'wrt' variable specified for check_partial options on Component " \
                      "'{}': '{}'.".format(self.pathname, invalid_wrt[0])
            else:
                msg = "Invalid 'wrt' variables specified for check_partial options on Component " \
                      "'{}': {}.".format(self.pathname, invalid_wrt)
            raise ValueError(msg)

        return opts

    def _declare_partials(self, of, wrt, dct):
        """
        Store subjacobian metadata for later use.

        Parameters
        ----------
        of : tuple of str
            The names of the residuals that derivatives are being computed for.
            May also contain glob patterns.
        wrt : tuple of str
            The names of the variables that derivatives are taken with respect to.
            This can contain the name of any input or output variable.
            May also contain glob patterns.
        dct : dict
            Metadata dict specifying shape, and/or approx properties.
        """
        val = dct['value'] if 'value' in dct else None
        is_scalar = isscalar(val)
        dependent = dct['dependent']

        if dependent:
            if 'rows' in dct and dct['rows'] is not None:  # sparse list format
                rows = dct['rows']
                cols = dct['cols']

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
                        msg = '{}: d({})/d({}): row indices must be non-negative'
                        raise ValueError(msg.format(self.pathname, of, wrt))
                    if cols.min() < 0:
                        msg = '{}: d({})/d({}): col indices must be non-negative'
                        raise ValueError(msg.format(self.pathname, of, wrt))
                    rows_max = rows.max()
                    cols_max = cols.max()
                else:
                    rows_max = cols_max = 0
            else:
                if val is not None and not is_scalar and not issparse(val):
                    val = atleast_2d(val)
                    val = val.astype(promote_types(val.dtype, float), copy=False)
                rows_max = cols_max = 0
                rows = None
                cols = None

        pattern_matches = self._find_partial_matches(of, wrt)
        abs2meta = self._var_abs2meta

        is_array = isinstance(val, ndarray)
        patmeta = dict(dct)

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
                    meta = patmeta.copy()

                meta['rows'] = rows
                meta['cols'] = cols
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
                    of, wrt = rel_key
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
        of, wrt = self._get_potential_partials_lists()

        of_pattern_matches = [(pattern, find_matches(pattern, of)) for pattern in of_list]
        wrt_pattern_matches = [(pattern, find_matches(pattern, wrt)) for pattern in wrt_list]
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

    def _set_approx_partials_meta(self):
        """
        Add approximations for those partials registered with method=fd or method=cs.
        """
        self._get_static_wrt_matches()
        subjacs = self._subjacs_info
        for key in self._approx_subjac_keys_iter():
            meta = subjacs[key]
            self._approx_schemes[meta['method']].add_approximation(key, meta)

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

    def _check_first_linearize(self):
        if self._first_call_to_linearize:
            self._first_call_to_linearize = False  # only do this once
            is_dynamic = self._coloring_info['coloring'] is _DYN_COLORING
            coloring = self._get_coloring()
            if coloring is not None:
                if not is_dynamic:
                    coloring._check_config_partial(self)
                self._update_subjac_sparsity(coloring.get_subjac_sparsity())


class _DictValues(object):
    """
    A dict-like wrapper for a dict of metadata, where getitem returns 'value' from metadata.
    """

    def __init__(self, dct):
        self._dict = dct

    def __getitem__(self, key):
        return self._dict[key]['value']

    def __setitem__(self, key, value):
        self._dict[key]['value'] = value

    def __contains__(self, key):
        return key in self._dict

    def __len__(self):
        return len(self._dict)

    def items(self):
        return [(key, self._dict[key]['value']) for key in self._dict]

    def iteritems(self):
        for key, val in self._dict.iteritems():
            yield key, val['value']
