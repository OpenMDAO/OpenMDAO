"""Define the Component class."""

import sys
from collections import defaultdict
from collections.abc import Iterable
from itertools import product

from numbers import Integral
import numpy as np
from numpy import ndarray, isscalar, ndim, atleast_1d, atleast_2d, promote_types
from scipy.sparse import issparse, coo_matrix

from openmdao.core.system import System, _supported_methods, _DEFAULT_COLORING_META, \
    global_meta_names, _MetadataDict, collect_errors
from openmdao.core.constants import INT_DTYPE
from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian
from openmdao.utils.array_utils import shape_to_len
from openmdao.utils.units import simplify_unit
from openmdao.utils.name_maps import abs_key_iter, abs_key2rel_key, rel_name2abs_name
from openmdao.utils.mpi import MPI
from openmdao.utils.general_utils import format_as_float_or_array, ensure_compatible, \
    find_matches, make_set, convert_src_inds
from openmdao.utils.indexer import Indexer, indexer
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.om_warnings import issue_warning, MPIWarning, DistributedComponentWarning, \
    DerivativesWarning, SetupWarning, warn_deprecation

_forbidden_chars = {'.', '*', '?', '!', '[', ']'}
_whitespace = {' ', '\t', '\r', '\n'}
_allowed_types = (list, tuple, ndarray, Iterable)


def _valid_var_name(name):
    """
    Determine if the proposed name is a valid variable name.

    Leading and trailing whitespace is illegal, and a specific list of characters
    are illegal anywhere in the string.

    Parameters
    ----------
    name : str
        Proposed name.

    Returns
    -------
    bool
        True if the proposed name is a valid variable name, else False.
    """
    global _forbidden_chars, _whitespace
    if not name:
        return False
    if _forbidden_chars.intersection(name):
        return False
    return name is name.strip()


class Component(System):
    """
    Base Component class; not to be directly instantiated.

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Available here and in all descendants of this system.

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
    _no_check_partials : bool
        If True, the check_partials function will ignore this component.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        self._var_rel_names = {'input': [], 'output': []}
        self._var_rel2meta = {}

        self._static_var_rel_names = {'input': [], 'output': []}
        self._static_var_rel2meta = {}

        self._declared_partials = defaultdict(dict)
        self._declared_partial_checks = []
        self._no_check_partials = False

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('distributed', types=bool, default=False,
                             desc='If True, set all variables in this component as distributed '
                                  'across multiple processes')
        self.options.declare('run_root_only', types=bool, default=False,
                             desc='If True, call compute, compute_partials, linearize, '
                                  'apply_linear, apply_nonlinear, and compute_jacvec_product '
                                  'only on rank 0 and broadcast the results to the other ranks.')

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

    def _setup_procs(self, pathname, comm, mode, prob_meta):
        """
        Execute first phase of the setup process.

        Distribute processors, assign pathnames, and call setup on the component.

        Parameters
        ----------
        pathname : str
            Global name of the system, including the path.
        comm : MPI.Comm or <FakeComm>
            MPI communicator object.
        mode : str
            Derivatives calculation mode, 'fwd' for forward, and 'rev' for
            reverse (adjoint). Default is 'rev'.
        prob_meta : dict
            Problem level metadata.
        """
        super()._setup_procs(pathname, comm, mode, prob_meta)

        orig_comm = comm
        if self._num_par_fd > 1:
            if comm.size > 1:
                comm = self._setup_par_fd_procs(comm)
            elif not MPI:
                issue_warning(f"MPI is not active but num_par_fd = {self._num_par_fd}. No parallel "
                              "finite difference will be performed.",
                              prefix=self.msginfo, category=MPIWarning)
                self._num_par_fd = 1

        self.comm = comm
        nprocs = comm.size

        # Clear out old variable information so that we can call setup on the component.
        self._var_rel_names = {'input': [], 'output': []}
        self._var_rel2meta = {}
        if comm.size == 1:
            self._has_distrib_vars = False

        for meta in self._static_var_rel2meta.values():
            # variable isn't distributed if we're only running on 1 proc
            if nprocs == 1 and 'distributed' in meta and meta['distributed']:
                meta['distributed'] = False

            # reset shape if any dynamic shape parameters are set in case this is a resetup
            # NOTE: this is necessary because we allow variables to be added in __init__.
            if 'shape_by_conn' in meta and (meta['shape_by_conn'] or
                                            meta['copy_shape'] is not None):
                meta['shape'] = None
                if not isscalar(meta['val']):
                    if meta['val'].size > 0:
                        meta['val'] = meta['val'].flatten()[0]
                    else:
                        meta['val'] = 1.0

        self._var_rel2meta.update(self._static_var_rel2meta)
        for io in ['input', 'output']:
            self._var_rel_names[io].extend(self._static_var_rel_names[io])

        self.setup()
        self._setup_check()

        self._set_vector_class()

    def _set_vector_class(self):
        if self._has_distrib_vars:
            dist_vec_class = self._problem_meta['distributed_vector_class']
            if dist_vec_class is not None:
                self._vector_class = dist_vec_class
            else:
                issue_warning("Component contains distributed variables, "
                              "but there is no distributed vector implementation (MPI/PETSc) "
                              "available. The default non-distributed vectors will be used.",
                              prefix=self.msginfo, category=DistributedComponentWarning)

                self._vector_class = self._problem_meta['local_vector_class']
        else:
            self._vector_class = self._problem_meta['local_vector_class']

    def _configure_check(self):
        """
        Do any error checking on i/o configuration.
        """
        # Check here if declare_coloring was called during setup but declare_partials wasn't.
        # If declare partials wasn't called, call it with of='*' and wrt='*' so we'll have
        # something to color.
        if self._coloring_info['coloring'] is not None:
            for key, meta in self._declared_partials.items():
                if 'method' in meta and meta['method'] is not None:
                    break
            else:
                method = self._coloring_info['method']
                issue_warning("declare_coloring or use_fixed_coloring was called but no approx"
                              " partials were declared.  Declaring all partials as approximated "
                              f"using default metadata and method='{method}'.", prefix=self.msginfo,
                              category=DerivativesWarning)
                self.declare_partials('*', '*', method=method)

        super()._configure_check()

    def _setup_var_data(self):
        """
        Compute the list of abs var names, abs/prom name maps, and metadata dictionaries.
        """
        global global_meta_names
        super()._setup_var_data()

        allprocs_prom2abs_list = self._var_allprocs_prom2abs_list
        abs2prom = self._var_allprocs_abs2prom = self._var_abs2prom
        abs_in2prom_info = self._problem_meta['abs_in2prom_info']

        # Compute the prefix for turning rel/prom names into abs names
        prefix = self.pathname + '.' if self.pathname else ''

        for io in ['input', 'output']:
            abs2meta = self._var_abs2meta[io]
            allprocs_abs2meta = self._var_allprocs_abs2meta[io]

            is_input = io == 'input'
            for prom_name in self._var_rel_names[io]:
                abs_name = prefix + prom_name
                abs2meta[abs_name] = metadata = self._var_rel2meta[prom_name]

                # Compute allprocs_prom2abs_list, abs2prom
                allprocs_prom2abs_list[io][prom_name] = [abs_name]
                abs2prom[io][abs_name] = prom_name

                allprocs_abs2meta[abs_name] = {
                    meta_name: metadata[meta_name]
                    for meta_name in global_meta_names[io]
                }
                if is_input and 'src_indices' in metadata:
                    allprocs_abs2meta[abs_name]['has_src_indices'] = \
                        metadata['src_indices'] is not None
                    if metadata['add_input_src_indices'] and abs_name not in abs_in2prom_info:
                        # need a level for each system including '', so we don't
                        # subtract 1 from abs_in.split('.') which includes the var name
                        abs_in2prom_info[abs_name] = [None for s in abs_name.split('.')]

            for prom_name, val in self._var_discrete[io].items():
                abs_name = prefix + prom_name

                # Compute allprocs_prom2abs_list, abs2prom
                allprocs_prom2abs_list[io][prom_name] = [abs_name]
                abs2prom[io][abs_name] = prom_name

                # Compute allprocs_discrete (metadata for discrete vars)
                self._var_allprocs_discrete[io][abs_name] = v = val.copy()
                del v['val']

        if self._var_discrete['input'] or self._var_discrete['output']:
            self._discrete_inputs = _DictValues(self._var_discrete['input'])
            self._discrete_outputs = _DictValues(self._var_discrete['output'])
        else:
            self._discrete_inputs = self._discrete_outputs = ()

    @collect_errors
    def _setup_var_sizes(self):
        """
        Compute the arrays of variable sizes for all variables/procs on this system.
        """
        iproc = self.comm.rank
        abs2idx = self._var_allprocs_abs2idx = {}

        for io in ('input', 'output'):
            sizes = self._var_sizes[io] = np.zeros((self.comm.size, len(self._var_rel_names[io])),
                                                   dtype=INT_DTYPE)

            for i, (name, metadata) in enumerate(self._var_allprocs_abs2meta[io].items()):
                sz = metadata['size']
                sizes[iproc, i] = 0 if sz is None else sz
                abs2idx[name] = i

            if self.comm.size > 1:
                my_sizes = sizes[iproc, :].copy()
                self.comm.Allgather(my_sizes, sizes)

        self._owned_sizes = self._var_sizes['output']

    def _setup_partials(self):
        """
        Process all partials and approximations that the user declared.
        """
        self._subjacs_info = {}
        self._jacobian = DictionaryJacobian(system=self)

        self.setup_partials()  # hook for component writers to specify sparsity patterns

        # check to make sure that if num_par_fd > 1 that this system is actually doing FD.
        # Unfortunately we have to do this check after system setup has been called because that's
        # when declare_partials generally happens, so we raise an exception here instead of just
        # resetting the value of num_par_fd (because the comm has already been split and possibly
        # used by the system setup).
        orig_comm = self._full_comm if self._full_comm is not None else self.comm
        if self._num_par_fd > 1 and orig_comm.size > 1 and not (self._owns_approx_jac or
                                                                self._approx_schemes):
            raise RuntimeError("%s: num_par_fd is > 1 but no FD is active." % self.msginfo)

        for key, dct in self._declared_partials.items():
            of, wrt = key
            self._declare_partials(of, wrt, dct)

        if self.matrix_free and self._subjacs_info:
            issue_warning("matrix free component has declared the following "
                          f"partials: {sorted(self._subjacs_info)}, which will allocate "
                          "(possibly unnecessary) memory for each of those sub-jacobians.",
                          prefix=self.msginfo, category=DerivativesWarning)

    def setup_partials(self):
        """
        Declare partials.

        This is meant to be overridden by component classes.  All partials should be
        declared here since this is called after all size/shape information is known for
        all variables.
        """
        pass

    def _run_root_only(self):
        """
        Return the value of the run_root_only option and check for possible errors.

        Returns
        -------
        bool
            True if run_root_only is active.
        """
        if self.options['run_root_only']:
            if self.comm.size > 1 or (self._full_comm is not None and self._full_comm.size > 1):
                if self._has_distrib_vars:
                    raise RuntimeError(f"{self.msginfo}: Can't set 'run_root_only' option when "
                                       "a component has distributed variables.")
                if self._num_par_fd > 1:
                    raise RuntimeError(f"{self.msginfo}: Can't set 'run_root_only' option when "
                                       "using parallel FD.")
                if self._problem_meta['using_par_deriv_color']:
                    raise RuntimeError(f"{self.msginfo}: Can't set 'run_root_only' option when "
                                       "using parallel_deriv_color.")
                return True
        return False

    def _update_wrt_matches(self, info):
        """
        Determine the list of wrt variables that match the wildcard(s) given in declare_coloring.

        Parameters
        ----------
        info : dict
            Coloring metadata dict.
        """
        _, allwrt = self._get_partials_varlists()
        wrt_patterns = info['wrt_patterns']
        if wrt_patterns is None or '*' in wrt_patterns:
            info['wrt_matches_rel'] = None
            info['wrt_matches'] = None
            return

        matches_rel = set()
        for w in wrt_patterns:
            matches_rel.update(find_matches(w, allwrt))

        # error if nothing matched
        if not matches_rel:
            raise ValueError("{}: Invalid 'wrt' variable(s) specified for colored approx partial "
                             "options: {}.".format(self.msginfo, wrt_patterns))

        info['wrt_matches_rel'] = matches_rel
        info['wrt_matches'] = [rel_name2abs_name(self, n) for n in matches_rel]

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
        prefix = self.pathname + '.' if self.pathname else None
        for of, sub in sparsity.items():
            if prefix:
                of = prefix + of
            for wrt, tup in sub.items():
                if prefix:
                    wrt = prefix + wrt
                abs_key = (of, wrt)
                if abs_key in self._subjacs_info:
                    # add sparsity info to existing partial info
                    self._subjacs_info[abs_key]['sparsity'] = tup

    def add_input(self, name, val=1.0, shape=None, src_indices=None, flat_src_indices=None,
                  units=None, desc='', tags=None, shape_by_conn=False, copy_shape=None,
                  distributed=None):
        """
        Add an input variable to the component.

        Parameters
        ----------
        name : str
            Name of the variable in this component's namespace.
        val : float or list or tuple or ndarray or Iterable
            The initial value of the variable being added in user-defined units.
            Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if src_indices not provided and
            val is not an array. Default is None.
        src_indices : int or list or tuple or int ndarray or Iterable or None
            The global indices of the source variable to transfer data from.
            A value of None implies this input depends on all entries of the source array.
            Default is None. The shapes of the target and src_indices must match,
            and the form of the entries within is determined by the value of 'flat_src_indices'.
        flat_src_indices : bool
            If True and the source is non-flat, each entry of src_indices is assumed to be an index
            into the flattened source.  Ignored if the source is flat.
        units : str or None
            Units in which this input variable will be provided to the component
            during execution. Default is None, which means it is unitless.
        desc : str
            Description of the variable.
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed when calling
            list_inputs and list_outputs.
        shape_by_conn : bool
            If True, shape this input to match its connected output.
        copy_shape : str or None
            If a str, that str is the name of a variable. Shape this input to match that of
            the named variable.
        distributed : bool
            If True, this variable is a distributed variable, so it can have different sizes/values
            across MPI processes.

        Returns
        -------
        dict
            Metadata for added variable.
        """
        # First, type check all arguments
        if not isinstance(name, str):
            raise TypeError('%s: The name argument should be a string.' % self.msginfo)
        if not _valid_var_name(name):
            raise NameError("%s: '%s' is not a valid input name." % (self.msginfo, name))

        if not isscalar(val) and not isinstance(val, _allowed_types):
            raise TypeError('%s: The val argument should be a float, list, tuple, ndarray or '
                            'Iterable' % self.msginfo)
        if shape is not None and not isinstance(shape, (Integral, tuple, list)):
            raise TypeError("%s: The shape argument should be an int, tuple, or list but "
                            "a '%s' was given" % (self.msginfo, type(shape)))
        if src_indices is not None:
            try:
                src_indices = indexer(src_indices, flat_src=flat_src_indices)
            except Exception as err:
                raise TypeError(f"{self.msginfo}: When specifying src_indices for input "
                                f"'{name}': {err}")
        if units is not None:
            if not isinstance(units, str):
                raise TypeError('%s: The units argument should be a str or None.' % self.msginfo)
            units = simplify_unit(units, msginfo=self.msginfo)

        if tags is not None and not isinstance(tags, (str, list)):
            raise TypeError('The tags argument should be a str or list')

        if (shape_by_conn or copy_shape):
            if shape is not None or ndim(val) > 0:
                raise ValueError("%s: If shape is to be set dynamically using 'shape_by_conn' or "
                                 "'copy_shape', 'shape' and 'val' should be a scalar, "
                                 "but shape of '%s' and val of '%s' was given for variable '%s'."
                                 % (self.msginfo, shape, val, name))
            if src_indices is not None:
                raise ValueError("%s: Setting of 'src_indices' along with 'shape_by_conn' or "
                                 "'copy_shape' for variable '%s' is currently unsupported." %
                                 (self.msginfo, name))
        else:
            # value, shape: based on args, making sure they are compatible
            val, shape = ensure_compatible(name, val, shape, src_indices)

        if src_indices is not None:
            warn_deprecation(f"{self.msginfo}: Passing `src_indices` as an arg to "
                             f"`add_input('{name}', ...` is deprecated and will become an error "
                             "in a future release.  Add `src_indices` to a `promotes` or `connect` "
                             "call instead.")

        # until we get rid of component level distributed option, handle the case where
        # component distributed has been set to True but variable distributed has been set
        # to False by the caller.
        if distributed is not False:
            if distributed is None:
                distributed = False
            # using ._dict below to avoid tons of deprecation warnings
            distributed = distributed or self.options._dict['distributed']['val']

        metadata = _MetadataDict()

        metadata.update({
            'val': val,
            'shape': shape,
            'size': shape_to_len(shape),
            'src_indices': src_indices,
            'flat_src_indices': flat_src_indices,
            'add_input_src_indices': src_indices is not None,
            'units': units,
            'desc': desc,
            'distributed': distributed,
            'tags': make_set(tags),
            'shape_by_conn': shape_by_conn,
            'copy_shape': copy_shape,
        })

        # this will get reset later if comm size is 1
        self._has_distrib_vars |= metadata['distributed']

        if self._static_mode:
            var_rel2meta = self._static_var_rel2meta
            var_rel_names = self._static_var_rel_names
        else:
            var_rel2meta = self._var_rel2meta
            var_rel_names = self._var_rel_names

        # Disallow dupes
        if name in var_rel2meta:
            raise ValueError("{}: Variable name '{}' already exists.".format(self.msginfo, name))

        var_rel2meta[name] = metadata
        var_rel_names['input'].append(name)

        self._var_added(name)

        return metadata

    def add_discrete_input(self, name, val, desc='', tags=None):
        """
        Add a discrete input variable to the component.

        Parameters
        ----------
        name : str
            Name of the variable in this component's namespace.
        val : a picklable object
            The initial value of the variable being added.
        desc : str
            Description of the variable.
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed when calling
            list_inputs and list_outputs.

        Returns
        -------
        dict
            Metadata for added variable.
        """
        # First, type check all arguments
        if not isinstance(name, str):
            raise TypeError('%s: The name argument should be a string.' % self.msginfo)
        if not _valid_var_name(name):
            raise NameError("%s: '%s' is not a valid input name." % (self.msginfo, name))
        if tags is not None and not isinstance(tags, (str, list)):
            raise TypeError('%s: The tags argument should be a str or list' % self.msginfo)

        metadata = _MetadataDict()

        metadata.update({
            'val': val,
            'type': type(val),
            'desc': desc,
            'tags': make_set(tags),
        })

        if metadata['type'] == np.ndarray:
            metadata.update({'shape': val.shape})

        if self._static_mode:
            var_rel2meta = self._static_var_rel2meta
        else:
            var_rel2meta = self._var_rel2meta

        # Disallow dupes
        if name in var_rel2meta:
            raise ValueError("{}: Variable name '{}' already exists.".format(self.msginfo, name))

        var_rel2meta[name] = self._var_discrete['input'][name] = metadata

        self._var_added(name)

        return metadata

    def add_output(self, name, val=1.0, shape=None, units=None, res_units=None, desc='',
                   lower=None, upper=None, ref=1.0, ref0=0.0, res_ref=None, tags=None,
                   shape_by_conn=False, copy_shape=None, distributed=None):
        """
        Add an output variable to the component.

        Parameters
        ----------
        name : str
            Name of the variable in this component's namespace.
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
            Description of the variable.
        lower : float or list or tuple or ndarray or Iterable or None
            Lower bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no lower bound.
            Default is None.
        upper : float or list or tuple or ndarray or or Iterable None
            Upper bound(s) in user-defined units. It can be (1) a float, (2) an array_like
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
        tags : str or list of strs or set of strs
            User defined tags that can be used to filter what gets listed when calling
            list_inputs and list_outputs.
        shape_by_conn : bool
            If True, shape this output to match its connected input(s).
        copy_shape : str or None
            If a str, that str is the name of a variable. Shape this output to match that of
            the named variable.
        distributed : bool
            If True, this variable is a distributed variable, so it can have different sizes/values
            across MPI processes.

        Returns
        -------
        dict
            Metadata for added variable.
        """
        global _allowed_types

        # First, type check all arguments
        if (shape_by_conn or copy_shape) and (shape is not None or ndim(val) > 0):
            raise ValueError("%s: If shape is to be set dynamically using 'shape_by_conn' or "
                             "'copy_shape', 'shape' and 'val' should be scalar, "
                             "but shape of '%s' and val of '%s' was given for variable '%s'."
                             % (self.msginfo, shape, val, name))

        if not isinstance(name, str):
            raise TypeError('%s: The name argument should be a string.' % self.msginfo)
        if not _valid_var_name(name):
            raise NameError("%s: '%s' is not a valid output name." % (self.msginfo, name))

        if shape is not None and not isinstance(shape, (int, tuple, list, np.integer)):
            raise TypeError("%s: The shape argument should be an int, tuple, or list but "
                            "a '%s' was given" % (self.msginfo, type(shape)))
        if res_units is not None:
            if not isinstance(res_units, str):
                msg = '%s: The res_units argument should be a str or None' % self.msginfo
                raise TypeError(msg)
            res_units = simplify_unit(res_units, msginfo=self.msginfo)

        if units is not None:
            if not isinstance(units, str):
                raise TypeError('%s: The units argument should be a str or None' % self.msginfo)
            units = simplify_unit(units, msginfo=self.msginfo)

        if tags is not None and not isinstance(tags, (str, set, list)):
            raise TypeError('The tags argument should be a str, set, or list')

        if not (copy_shape or shape_by_conn):
            if not isscalar(val) and not isinstance(val, _allowed_types):
                msg = '%s: The val argument should be a float, list, tuple, ndarray or Iterable'
                raise TypeError(msg % self.msginfo)

            # value, shape: based on args, making sure they are compatible
            val, shape = ensure_compatible(name, val, shape)

            if lower is not None:
                lower = ensure_compatible(name, lower, shape)[0]
                self._has_bounds = True
            if upper is not None:
                upper = ensure_compatible(name, upper, shape)[0]
                self._has_bounds = True

            # All refs: check the shape if necessary
            for item, item_name in zip([ref, ref0, res_ref], ['ref', 'ref0', 'res_ref']):
                if item is not None and not isscalar(item):
                    if not isinstance(item, _allowed_types):
                        raise TypeError(f'{self.msginfo}: The {item_name} argument should be a '
                                        'float, list, tuple, ndarray or Iterable')

                    it = atleast_1d(item)
                    if it.shape != shape:
                        raise ValueError(f"{self.msginfo}: When adding output '{name}', expected "
                                         f"shape {shape} but got shape {it.shape} for argument "
                                         f"'{item_name}'.")

        if isscalar(ref):
            self._has_output_scaling |= ref != 1.0
        else:
            self._has_output_scaling |= np.any(ref != 1.0)

        if isscalar(ref0):
            self._has_output_scaling |= ref0 != 0.0
            self._has_output_adder |= ref0 != 0.0
        else:
            self._has_output_scaling |= np.any(ref0)
            self._has_output_adder |= np.any(ref0)

        if isscalar(res_ref):
            self._has_resid_scaling |= res_ref != 1.0
        else:
            self._has_resid_scaling |= np.any(res_ref != 1.0)

        # until we get rid of component level distributed option, handle the case where
        # component distributed has been set to True but variable distributed has been set
        # to False by the caller.
        if distributed is not False:
            if distributed is None:
                distributed = False
            # using ._dict below to avoid tons of deprecation warnings
            distributed = distributed or self.options._dict['distributed']['val']

        metadata = _MetadataDict()

        metadata.update({
            'val': val,
            'shape': shape,
            'size': shape_to_len(shape),
            'units': units,
            'res_units': res_units,
            'desc': desc,
            'distributed': distributed,
            'tags': make_set(tags),
            'ref': format_as_float_or_array('ref', ref, flatten=True),
            'ref0': format_as_float_or_array('ref0', ref0, flatten=True),
            'res_ref': format_as_float_or_array('res_ref', res_ref, flatten=True, val_if_none=None),
            'lower': lower,
            'upper': upper,
            'shape_by_conn': shape_by_conn,
            'copy_shape': copy_shape
        })

        # this will get reset later if comm size is 1
        self._has_distrib_vars |= metadata['distributed']

        # We may not know the pathname yet, so we have to use name for now, instead of abs_name.
        if self._static_mode:
            var_rel2meta = self._static_var_rel2meta
            var_rel_names = self._static_var_rel_names
        else:
            var_rel2meta = self._var_rel2meta
            var_rel_names = self._var_rel_names

        # Disallow dupes
        if name in var_rel2meta:
            raise ValueError("{}: Variable name '{}' already exists.".format(self.msginfo, name))

        var_rel2meta[name] = metadata
        var_rel_names['output'].append(name)

        self._var_added(name)

        return metadata

    def add_discrete_output(self, name, val, desc='', tags=None):
        """
        Add an output variable to the component.

        Parameters
        ----------
        name : str
            Name of the variable in this component's namespace.
        val : a picklable object
            The initial value of the variable being added.
        desc : str
            Description of the variable.
        tags : str or list of strs or set of strs
            User defined tags that can be used to filter what gets listed when calling
            list_inputs and list_outputs.

        Returns
        -------
        dict
            Metadata for added variable.
        """
        if not isinstance(name, str):
            raise TypeError('%s: The name argument should be a string.' % self.msginfo)
        if not _valid_var_name(name):
            raise NameError("%s: '%s' is not a valid output name." % (self.msginfo, name))
        if tags is not None and not isinstance(tags, (str, set, list)):
            raise TypeError('%s: The tags argument should be a str, set, or list' % self.msginfo)

        metadata = _MetadataDict()

        metadata.update({
            'val': val,
            'type': type(val),
            'desc': desc,
            'tags': make_set(tags)
        })

        if metadata['type'] == np.ndarray:
            metadata.update({'shape': val.shape})

        if self._static_mode:
            var_rel2meta = self._static_var_rel2meta
        else:
            var_rel2meta = self._var_rel2meta

        # Disallow dupes
        if name in var_rel2meta:
            raise ValueError("{}: Variable name '{}' already exists.".format(self.msginfo, name))

        var_rel2meta[name] = self._var_discrete['output'][name] = metadata

        self._var_added(name)

        return metadata

    def _var_added(self, name):
        """
        Notify config that a variable has been added to this Component.

        Parameters
        ----------
        name : str
            Name of the added variable.
        """
        if self._problem_meta is not None and self._problem_meta['config_info'] is not None:
            self._problem_meta['config_info']._var_added(self.pathname, name)

    def _update_dist_src_indices(self, abs_in2out, all_abs2meta, all_abs2idx, all_sizes):
        """
        Set default src_indices for any distributed inputs where they aren't set.

        Parameters
        ----------
        abs_in2out : dict
            Mapping of connected inputs to their source.  Names are absolute.
        all_abs2meta : dict
            Mapping of absolute names to metadata for all variables in the model.
        all_abs2idx : dict
            Dictionary mapping an absolute name to its allprocs variable index.
        all_sizes : dict
            Mapping of types to sizes of each variable in all procs.

        Returns
        -------
        list
            Names of inputs where src_indices were added.
        """
        iproc = self.comm.rank
        abs2meta_in = self._var_abs2meta['input']
        all_abs2meta_in = all_abs2meta['input']
        all_abs2meta_out = all_abs2meta['output']
        abs_in2prom_info = self._problem_meta['abs_in2prom_info']

        sizes_in = self._var_sizes['input']
        sizes_out = all_sizes['output']
        added_src_inds = []
        # loop over continuous inputs
        for i, (iname, meta_in) in enumerate(abs2meta_in.items()):
            if meta_in['src_indices'] is None and iname not in abs_in2prom_info:
                src = abs_in2out[iname]
                dist_in = meta_in['distributed']
                dist_out = all_abs2meta_out[src]['distributed']
                if dist_in or dist_out:
                    gsize_out = all_abs2meta_out[src]['global_size']
                    gsize_in = all_abs2meta_in[iname]['global_size']
                    vout_sizes = sizes_out[:, all_abs2idx[src]]

                    offset = None
                    if gsize_out == gsize_in or (not dist_out and np.sum(vout_sizes)
                                                 == gsize_in):
                        # This assumes one of:
                        # 1) a distributed output with total size matching the total size of a
                        #    distributed input
                        # 2) a non-distributed output with local size matching the total size of a
                        #    distributed input
                        # 3) a non-distributed output with total size matching the total size of a
                        #    distributed input
                        if dist_in:
                            offset = np.sum(sizes_in[:iproc, i])
                            end = offset + sizes_in[iproc, i]
                        else:
                            if src.startswith('_auto_ivc.'):
                                nzs = np.nonzero(vout_sizes)[0]
                                if nzs.size == 1:
                                    # special case where we have a 'distributed' auto_ivc output
                                    # that has a nonzero value in only one proc, so we can treat
                                    # it like a non-distributed output. This happens in cases
                                    # where an auto_ivc output connects to a variable that is
                                    # remote on at least one proc.
                                    offset = 0
                                    end = vout_sizes[nzs[0]]

                    # total sizes differ and output is distributed, so can't determine mapping
                    if offset is None:
                        self._collect_error(f"{self.msginfo}: Can't determine src_indices "
                                            f"automatically for input '{iname}'. They must be "
                                            "supplied manually.", ident=(self.pathname, iname))
                        continue

                    if dist_in and not dist_out:
                        src_shape = self._get_full_dist_shape(src, all_abs2meta_out[src]['shape'])
                    else:
                        src_shape = all_abs2meta_out[src]['global_shape']

                    if offset == end:
                        idx = np.zeros(0, dtype=INT_DTYPE)
                    else:
                        idx = slice(offset, end)
                    meta_in['src_indices'] = indexer(idx, flat_src=True, src_shape=src_shape)
                    meta_in['flat_src_indices'] = True
                    added_src_inds.append(iname)

        return added_src_inds

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
                raise ValueError('{}: No matches were found for of="{}"'.format(self.msginfo,
                                                                                of_pattern))
            if not wrt_matches:
                raise ValueError('{}: No matches were found for wrt="{}"'.format(self.msginfo,
                                                                                 wrt_pattern))

            info = self._subjacs_info
            for abs_key in abs_key_iter(self, of_matches, wrt_matches):
                meta = info[abs_key]
                meta['method'] = method
                meta.update(kwargs)
                info[abs_key] = meta

    def declare_partials(self, of, wrt, dependent=True, rows=None, cols=None, val=None,
                         method='exact', step=None, form=None, step_calc=None, minimum_step=None):
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
        form : str
            Form for finite difference, can be 'forward', 'backward', or 'central'. Defaults
            to None, in which case the approximation method provides its default value.
        step_calc : str
            Step type for computing the size of the finite difference step. It can be 'abs' for
            absolute, 'rel_avg' for a size relative to the absolute value of the vector input, or
            'rel_element' for a size relative to each value in the vector input. In addition, it
            can be 'rel_legacy' for a size relative to the norm of the vector.  For backwards
            compatibilty, it can be 'rel', which currently defaults to 'rel_legacy', but in the
            future will default to 'rel_avg'. Defaults to None, in which case the approximation
            method provides its default value.
        minimum_step : float
            Minimum step size allowed when using one of the relative step_calc options.

        Returns
        -------
        dict
            Metadata dict for the specified partial(s).
        """
        try:
            method_func = _supported_methods[method]
        except KeyError:
            msg = '{}: d({})/d({}): method "{}" is not supported, method must be one of {}'
            raise ValueError(msg.format(self.msginfo, of, wrt, method, sorted(_supported_methods)))

        # lists aren't hashable so convert to tuples
        if isinstance(of, list):
            of = tuple(of)
        if isinstance(wrt, list):
            wrt = tuple(wrt)

        meta = self._declared_partials[of, wrt]
        meta['dependent'] = dependent

        # If only one of rows/cols is specified
        if (rows is None) ^ (cols is None):
            raise ValueError('{}: d({})/d({}): If one of rows/cols is specified, then '
                             'both must be specified.'.format(self.msginfo, of, wrt))

        if dependent:
            meta['val'] = val
            if rows is not None:
                rows = np.array(rows, dtype=INT_DTYPE, copy=False)
                cols = np.array(cols, dtype=INT_DTYPE, copy=False)

                # Check the length of rows and cols to catch this easy mistake and give a
                # clear message.
                if len(cols) != len(rows):
                    raise RuntimeError("{}: d({})/d({}): declare_partials has been called "
                                       "with rows and cols, which should be arrays of equal length,"
                                       " but rows is length {} while cols is length "
                                       "{}.".format(self.msginfo, of, wrt, len(rows), len(cols)))

                if rows.size > 0 and rows.min() < 0:
                    msg = '{}: d({})/d({}): row indices must be non-negative'
                    raise ValueError(msg.format(self.msginfo, of, wrt))
                if cols.size > 0 and cols.min() < 0:
                    msg = '{}: d({})/d({}): col indices must be non-negative'
                    raise ValueError(msg.format(self.msginfo, of, wrt))

                meta['rows'] = rows
                meta['cols'] = cols

                # Check for repeated rows/cols indices.
                size = len(rows)
                if size > 0:
                    coo = coo_matrix((np.ones(size, dtype=np.short), (rows, cols)))
                    dsize = coo.data.size
                    csc = coo.tocsc()
                    # csc adds values at duplicate indices together, so result will be that data
                    # size is less if there are duplicates
                    if csc.data.size < dsize:
                        coo = csc.tocoo()
                        del csc
                        inds = np.where(coo.data > 1.)
                        dups = list(zip(coo.row[inds], coo.col[inds]))
                        raise RuntimeError("{}: d({})/d({}): declare_partials has been called "
                                           "with rows and cols that specify the following duplicate"
                                           " subjacobian entries: {}.".format(self.msginfo, of, wrt,
                                                                              sorted(dups)))

        if method_func is not None:
            # we're doing approximations
            self._has_approx = True
            meta['method'] = method
            self._get_approx_scheme(method)

            default_opts = method_func.DEFAULT_OPTIONS
        else:
            default_opts = ()

        if step:
            if 'step' in default_opts:
                meta['step'] = step
            else:
                raise RuntimeError("{}: d({})/d({}): 'step' is not a valid option for "
                                   "'{}'".format(self.msginfo, of, wrt, method))
        if minimum_step is not None:
            if 'minimum_step' in default_opts:
                meta['minimum_step'] = minimum_step
            else:
                raise RuntimeError("{}: d({})/d({}): 'minimum_step' is not a valid option for "
                                   "'{}'".format(self.msginfo, of, wrt, method))
        if form:
            if 'form' in default_opts:
                meta['form'] = form
            else:
                raise RuntimeError("{}: d({})/d({}): 'form' is not a valid option for "
                                   "'{}'".format(self.msginfo, of, wrt, method))
        if step_calc:
            if 'step_calc' in default_opts:
                meta['step_calc'] = step_calc
            else:
                raise RuntimeError("{}: d({})/d({}): 'step_calc' is not a valid option "
                                   "for '{}'".format(self.msginfo, of, wrt, method))

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
                         min_improve_pct=_DEFAULT_COLORING_META['min_improve_pct'],
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
        min_improve_pct : float
            If coloring does not improve (decrease) the number of solves more than the given
            percentage, coloring will not be used.
        show_summary : bool
            If True, display summary information after generating coloring.
        show_sparsity : bool
            If True, display sparsity with coloring info after generating coloring.
        """
        super().declare_coloring(wrt, method, form, step, per_instance,
                                 num_full_jacs,
                                 tol, orders, perturb_size, min_improve_pct,
                                 show_summary, show_sparsity)

        # create approx partials for all matches
        meta = self.declare_partials('*', wrt, method=method, step=step, form=form)
        meta['coloring'] = True

    def set_check_partial_options(self, wrt, method='fd', form=None, step=None, step_calc=None,
                                  minimum_step=None, directional=False):
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
            Step type for computing the size of the finite difference step. It can be 'abs' for
            absolute, 'rel_avg' for a size relative to the absolute value of the vector input, or
            'rel_element' for a size relative to each value in the vector input. In addition, it
            can be 'rel_legacy' for a size relative to the norm of the vector.  For backwards
            compatibilty, it can be 'rel', which currently defaults to 'rel_legacy', but in the
            future will default to 'rel_avg'. Defaults to None, in which case the approximation
            method provides its default value.
        minimum_step : float
            Minimum step size allowed when using one of the relative step_calc options.
        directional : bool
            Set to True to perform a single directional derivative for each vector variable in the
            pattern named in wrt.
        """
        supported_methods = ('fd', 'cs')
        if method not in supported_methods:
            msg = "{}: Method '{}' is not supported, method must be one of {}"
            raise ValueError(msg.format(self.msginfo, method, supported_methods))

        if step and not isinstance(step, (int, float)):
            msg = "{}: The value of 'step' must be numeric, but '{}' was specified."
            raise ValueError(msg.format(self.msginfo, step))

        supported_step_calc = ('abs', 'rel', 'rel_legacy', 'rel_avg', 'rel_element')
        if step_calc and step_calc not in supported_step_calc:
            msg = "{}: The value of 'step_calc' must be one of {}, but '{}' was specified."
            raise ValueError(msg.format(self.msginfo, supported_step_calc, step_calc))

        if not isinstance(wrt, (str, list, tuple)):
            msg = "{}: The value of 'wrt' must be a string or list of strings, but a type " \
                  "of '{}' was provided."
            raise ValueError(msg.format(self.msginfo, type(wrt).__name__))

        if not isinstance(directional, bool):
            msg = "{}: The value of 'directional' must be True or False, but a type " \
                  "of '{}' was provided."
            raise ValueError(msg.format(self.msginfo, type(directional).__name__))

        wrt_list = [wrt] if isinstance(wrt, str) else wrt
        self._declared_partial_checks.append((wrt_list, method, form, step, step_calc,
                                              minimum_step, directional))

    def _get_check_partial_options(self):
        """
        Return dictionary of partial options with pattern matches processed.

        This is called by check_partials.

        Returns
        -------
        dict(wrt: (options))
            Dictionary keyed by name with tuples of options (method, form, step, step_calc,
            minimum_step, directional)
        """
        if not self._declared_partial_checks:
            return {}
        opts = {}
        _, wrt = self._get_partials_varlists()
        invalid_wrt = []
        matrix_free = self.matrix_free

        if matrix_free:
            n_directional = 0

        for data_tup in self._declared_partial_checks:
            wrt_list, method, form, step, step_calc, minimum_step, directional = data_tup

            for pattern in wrt_list:
                matches = find_matches(pattern, wrt)

                # if a non-wildcard var name was specified and not found, save for later Exception
                if len(matches) == 0 and _valid_var_name(pattern):
                    invalid_wrt.append(pattern)

                for match in matches:
                    if match in opts:
                        opt = opts[match]

                        # New assignments take precedence
                        keynames = ['method', 'form', 'step', 'step_calc', 'minimum_step',
                                    'directional']
                        for name, value in zip(keynames,
                                               [method, form, step, step_calc, minimum_step,
                                                directional]):
                            if value is not None:
                                opt[name] = value

                    else:
                        opts[match] = {'method': method,
                                       'form': form,
                                       'step': step,
                                       'step_calc': step_calc,
                                       'minimum_step': minimum_step,
                                       'directional': directional}

                    if matrix_free and directional:
                        n_directional += 1

        if invalid_wrt:
            msg = "{}: Invalid 'wrt' variables specified for check_partial options: {}."
            raise ValueError(msg.format(self.msginfo, invalid_wrt))

        if matrix_free:
            if n_directional > 0 and n_directional < len(wrt):
                msg = "{}: For matrix free components, directional should be set to True for " + \
                      "all inputs."
                raise ValueError(msg.format(self.msginfo))

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
        val = dct['val'] if 'val' in dct else None
        is_scalar = isscalar(val)
        dependent = dct['dependent']

        if dependent:
            if 'rows' in dct and dct['rows'] is not None:  # sparse list format
                rows = dct['rows']
                cols = dct['cols']

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
                        raise ValueError('{}: d({})/d({}): If rows and cols are specified, val '
                                         'must be a scalar or have the same shape, val: {}, '
                                         'rows/cols: {}'.format(self.msginfo, of, wrt,
                                                                val.shape, rows.shape))
                else:
                    val = np.zeros_like(rows, dtype=float)

                if rows.size > 0:
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

        pattern_matches = self._find_partial_matches(of, '*' if wrt is None else wrt)
        abs2meta_in = self._var_abs2meta['input']
        abs2meta_out = self._var_abs2meta['output']

        is_array = isinstance(val, ndarray)
        patmeta = dict(dct)
        patmeta_not_none = {k: v for k, v in dct.items() if v is not None}

        for of_bundle, wrt_bundle in product(*pattern_matches):
            of_pattern, of_matches = of_bundle
            wrt_pattern, wrt_matches = wrt_bundle
            if not of_matches:
                raise ValueError('{}: No matches were found for of="{}"'.format(self.msginfo,
                                                                                of_pattern))
            if not wrt_matches:
                raise ValueError('{}: No matches were found for wrt="{}"'.format(self.msginfo,
                                                                                 wrt_pattern))

            for abs_key in abs_key_iter(self, of_matches, wrt_matches):
                if not dependent:
                    if abs_key in self._subjacs_info:
                        del self._subjacs_info[abs_key]
                    continue

                if abs_key in self._subjacs_info:
                    meta = self._subjacs_info[abs_key]
                    meta.update(patmeta_not_none)
                else:
                    meta = patmeta.copy()

                of, wrt = abs_key
                meta['rows'] = rows
                meta['cols'] = cols
                csz = abs2meta_in[wrt]['size'] if wrt in abs2meta_in else abs2meta_out[wrt]['size']
                meta['shape'] = shape = (abs2meta_out[of]['size'], csz)
                dist_out = abs2meta_out[of]['distributed']
                if wrt in abs2meta_in:
                    dist_in = abs2meta_in[wrt]['distributed']
                else:
                    dist_in = abs2meta_out[wrt]['distributed']

                if dist_in and not dist_out and not self.matrix_free:
                    rel_key = abs_key2rel_key(self, abs_key)
                    raise RuntimeError(f"{self.msginfo}: component has defined partial {rel_key} "
                                       "which is a non-distributed output wrt a distributed input."
                                       " This is only supported using the matrix free API.")

                if shape[0] == 0 or shape[1] == 0:
                    msg = "{}: '{}' is an array of size 0"
                    if shape[0] == 0:
                        if dist_out:
                            # distributed vars are allowed to have zero size inputs on some procs
                            rows_max = -1
                        else:
                            # non-distributed vars are not allowed to have zero size inputs
                            raise ValueError(msg.format(self.msginfo, of))
                    if shape[1] == 0:
                        if not dist_in:
                            # non-distributed vars are not allowed to have zero size outputs
                            raise ValueError(msg.format(self.msginfo, wrt))
                        else:
                            # distributed vars are allowed to have zero size outputs on some procs
                            cols_max = -1

                if val is None:
                    # we can only get here if rows is None  (we're not sparse list format)
                    meta['val'] = np.zeros(shape)
                elif is_array:
                    if rows is None and val.shape != shape and val.size == shape[0] * shape[1]:
                        meta['val'] = val = val.copy().reshape(shape)
                    else:
                        meta['val'] = val.copy()
                elif is_scalar:
                    meta['val'] = np.full(shape, val, dtype=float)
                else:
                    meta['val'] = val

                if rows_max >= shape[0] or cols_max >= shape[1]:
                    of, wrt = abs_key2rel_key(self, abs_key)
                    raise ValueError(f"{self.msginfo}: d({of})/d({wrt}): Expected {shape[0]}x"
                                     f"{shape[1]} but declared at least {rows_max + 1}x"
                                     f"{cols_max + 1}")

                self._check_partials_meta(abs_key, meta['val'],
                                          shape if rows is None else (rows.shape[0], 1))

                self._subjacs_info[abs_key] = meta

    def _find_partial_matches(self, of_pattern, wrt_pattern, use_resname=False):
        """
        Find all partial derivative matches from of and wrt.

        Parameters
        ----------
        of_pattern : str or list of str
            The relative name of the residual(s) that derivatives are being computed for.
            May also contain a glob pattern.
        wrt_pattern : str or list of str
            The relative name of the variables that derivatives are taken with respect to.
            This can contain the name of any input or output variable.
            May also contain a glob pattern.
        use_resname : bool
            If True, use residual names for 'of' patterns.

        Returns
        -------
        tuple(list, list)
            Pair of lists containing pattern matches (if any). Returns (of_matches, wrt_matches)
            where of_matches is a list of tuples (pattern, matches) and wrt_matches is a list of
            tuples (pattern, output_matches, input_matches).
        """
        of_list = [of_pattern] if isinstance(of_pattern, str) else of_pattern
        wrt_list = [wrt_pattern] if isinstance(wrt_pattern, str) else wrt_pattern
        ofs, wrts = self._get_partials_varlists(use_resname=use_resname)

        of_pattern_matches = [(pattern, find_matches(pattern, ofs)) for pattern in of_list]
        wrt_pattern_matches = [(pattern, find_matches(pattern, wrts)) for pattern in wrt_list]
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
                raise ValueError(msg.format(self.msginfo, of, wrt, out_size, in_size,
                                            val_out, val_in))

    def _set_approx_partials_meta(self):
        """
        Add approximations for those partials registered with method=fd or method=cs.
        """
        self._get_static_wrt_matches()
        subjacs = self._subjacs_info
        wrtset = set()
        subjac_keys = self._get_approx_subjac_keys()
        # go through subjac keys in reverse and only add approx for the last of each wrt
        # (this prevents warnings that could confuse users)
        for i in range(len(subjac_keys) - 1, -1, -1):
            key = subjac_keys[i]
            if key[1] not in wrtset:
                wrtset.add(key[1])
                meta = subjacs[key]
                self._approx_schemes[meta['method']].add_approximation(key, self, meta)

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
            if coloring_mod._use_partial_sparsity:
                coloring = self._get_coloring()
                if coloring is not None:
                    if not self._coloring_info['dynamic']:
                        coloring._check_config_partial(self)
                    self._update_subjac_sparsity(coloring.get_subjac_sparsity())
                self._jacobian._restore_approx_sparsity()

    def _resolve_src_inds(self):
        abs2prom = self._var_abs2prom['input']
        abs_in2prom_info = self._problem_meta['abs_in2prom_info']
        all_abs2meta_in = self._var_allprocs_abs2meta['input']
        abs2meta_in = self._var_abs2meta['input']
        conns = self._problem_meta['model_ref']()._conn_global_abs_in2out
        all_abs2meta_out = self._problem_meta['model_ref']()._var_allprocs_abs2meta['output']

        for tgt, meta in abs2meta_in.items():
            if tgt in abs_in2prom_info:
                pinfo = abs_in2prom_info[tgt][-1]  # component always last in the plist
                if pinfo is not None:
                    inds, flat, shape = pinfo
                    if inds is None:
                        if meta['add_input_src_indices']:
                            if shape is None:
                                shape = all_abs2meta_out[conns[tgt]]['global_shape']
                            meta['src_shape'] = shape
                            inds = meta['src_indices']
                    else:
                        all_abs2meta_in[tgt]['has_src_indices'] = True
                        meta['src_shape'] = shape = all_abs2meta_out[conns[tgt]]['global_shape']
                        if meta['add_input_src_indices']:
                            inds = convert_src_inds(inds, shape, meta['src_indices'], shape)
                        elif inds._flat_src:
                            meta['flat_src_indices'] = True
                        elif meta['flat_src_indices'] is None:
                            meta['flat_src_indices'] = flat

                    if inds is not None:
                        try:
                            if not isinstance(inds, Indexer):
                                meta['src_indices'] = inds = indexer(inds, flat_src=flat,
                                                                     src_shape=shape)
                            else:
                                meta['src_indices'] = inds = inds.copy()
                                inds.set_src_shape(shape)
                                self._var_prom2inds[abs2prom[tgt]] = [shape, inds, flat]
                        except Exception:
                            type_exc, exc, tb = sys.exc_info()
                            self._collect_error(f"When accessing '{conns[tgt]}' with src_shape "
                                                f"{shape} from '{pinfo.prom_path()}' using "
                                                f"src_indices {inds}: {exc}", exc_type=type_exc,
                                                tback=tb, ident=(conns[tgt], tgt))

            elif meta['add_input_src_indices']:
                self._var_prom2inds[abs2prom[tgt]] = [meta['shape'], meta['src_indices'],
                                                      meta['src_indices']._flat_src]

    def _has_fast_rel_lookup(self):
        """
        Return True if this System should have fast relative variable name lookup in vectors.

        Returns
        -------
        bool
            True if this System should have fast relative variable name lookup in vectors.
        """
        return True


class _DictValues(object):
    """
    A dict-like wrapper for a dict of metadata, where getitem returns 'val' from metadata.
    """

    def __init__(self, dct):
        self._dict = dct

    def __getitem__(self, key):
        return self._dict[key]['val']

    def __setitem__(self, key, value):
        self._dict[key]['val'] = value

    def __contains__(self, key):
        return key in self._dict

    def __len__(self):
        return len(self._dict)

    def items(self):
        return [(key, self._dict[key]['val']) for key in self._dict]
