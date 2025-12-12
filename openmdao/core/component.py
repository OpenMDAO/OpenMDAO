"""Define the Component class."""

import sys
import inspect
from collections import defaultdict
from collections.abc import Iterable
from itertools import product, chain
from io import StringIO

from numbers import Integral
import numpy as np
from numpy import ndarray, isscalar, ndim, atleast_1d
from scipy.sparse import issparse, coo_matrix, csr_matrix

from openmdao.core.system import System, _supported_methods, _DEFAULT_COLORING_META, \
    global_meta_names, collect_errors, _iter_derivs
from openmdao.core.constants import INT_DTYPE, _DEFAULT_OUT_STREAM, _SetupStatus
from openmdao.jacobians.subjac import Subjac
from openmdao.jacobians.dictionary_jacobian import _CheckingJacobian
from openmdao.utils.units import simplify_unit
from openmdao.utils.name_maps import abs_key_iter, abs_key2rel_key, rel_key2abs_key
from openmdao.utils.mpi import MPI
from openmdao.utils.array_utils import shape_to_len, submat_sparsity_iter, sparsity_diff_viz
from openmdao.utils.deriv_display import _deriv_display, _deriv_display_compact
from openmdao.utils.general_utils import format_as_float_or_array, ensure_compatible, \
    find_matches, make_set, inconsistent_across_procs, LocalRangeIterable
from openmdao.utils.indexer import Indexer, indexer
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.om_warnings import issue_warning, MPIWarning, DistributedComponentWarning, \
    DerivativesWarning, warn_deprecation, OMInvalidCheckDerivativesOptionsWarning
from openmdao.utils.code_utils import is_lambda, LambdaPickleWrapper, get_function_deps, \
    get_return_names
from openmdao.approximation_schemes.complex_step import ComplexStep
from openmdao.approximation_schemes.finite_difference import FiniteDifference


_forbidden_chars = {'.', '*', '?', '!', '[', ']'}
_whitespace = {' ', '\t', '\r', '\n'}
_allowed_types = (list, tuple, ndarray, Iterable)

_no_matvec_scope = (None, frozenset())


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
    _var_rel2meta : dict
        Dictionary mapping relative names to metadata.
        This is only needed while adding inputs and outputs. During setup, these are used to
        build the dictionaries of metadata.  This contains both continuous and discrete variables.
    _static_var_rel2meta : dict
        Static version of above - stores data for variables added outside of setup.
    _var_rel_names : {'input': [str, ...], 'output': [str, ...]}
        List of relative names of owned variables existing on current proc.
        This is only needed while adding inputs and outputs. During setup, these are used to
        determine the list of absolute names. This includes only continuous variables.
    _static_var_rel_names : dict
        Static version of above - stores names of variables added outside of setup.
    _declared_partials_patterns : dict
        Dictionary of declared partials patterns.  Each key is a tuple of the form
        (of, wrt) where of and wrt may be glob patterns.
    _declared_partial_checks : list
        Cached storage of user-declared check partial options.
    _no_check_partials : bool
        If True, the check_partials function will ignore this component.
    _has_distrib_outputs : bool
        If True, this component has at least one distributed output.
    _compute_primals_out_shape : tuple or None
        Cached (shape, istuple) of the output from compute_primal function.  If istuple is True,
        then shape is a tuple of shapes, otherwise it is a single shape.
    _valid_name_map : dict
        Mapping of declared input/output names to valid Python names.
    _orig_compute_primal : function
        The original compute_primal method.
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

        self._declared_partials_patterns = {}
        self._declared_partial_checks = []
        self._no_check_partials = False
        self._has_distrib_outputs = False
        self._compute_primals_out_shape = None
        self._valid_name_map = {}
        self._orig_compute_primal = getattr(self, 'compute_primal')

    def _tree_flatten(self):
        """
        Return a flattened pytree representation of this component.

        We treat this component, when passed as 'self' into a function that is used by jax, as a
        pytree with no continuous data.

        Returns
        -------
        Tuple
            A tuple containing continuous and static data.
        """
        return ((), {'_self_': self, '_statics_': self.get_self_statics()})

    @staticmethod
    def _tree_unflatten(aux_data, children):
        """
        Return the same instance of this component that was returned by the _tree_flatten method.
        """
        return aux_data['_self_']

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
                                  'apply_linear, apply_nonlinear, solve_linear, solve_nonlinear, '
                                  'and compute_jacvec_product only on rank 0 and broadcast the '
                                  'results to the other ranks.')
        self.options.declare('always_opt', types=bool, default=False,
                             desc='If True, force nonlinear operations on this component to be '
                                  'included in the optimization loop even if this component is not '
                                  'relevant to the design variables and responses.')
        self.options.declare('use_jit', types=bool, default=True,
                             desc='If True, attempt to use jit on compute_primal, assuming jax or '
                             'some other AD package capable of jitting is active.')
        self.options.declare('default_shape', types=tuple, default=(1,),
                             desc='Default shape for variables that do not set val to a non-scalar '
                             'value or set shape, shape_by_conn, copy_shape, or compute_shape.'
                             ' Default is (1,).')

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

    def _setup_procs(self, pathname, comm, prob_meta):
        """
        Execute first phase of the setup process.

        Distribute processors, assign pathnames, and call setup on the component.

        Parameters
        ----------
        pathname : str
            Global name of the system, including the path.
        comm : MPI.Comm or <FakeComm>
            MPI communicator object.
        prob_meta : dict
            Problem level metadata.
        """
        super()._setup_procs(pathname, comm, prob_meta)

        if self._num_par_fd > 1:
            if comm.size > 1:
                comm = self._setup_par_fd_procs(comm)
            elif not MPI:
                issue_warning(f"MPI is not active but num_par_fd = {self._num_par_fd}. No parallel "
                              "finite difference will be performed.",
                              prefix=self.msginfo, category=MPIWarning)
                self._num_par_fd = 1

        self.comm = comm

        # Clear out old variable information so that we can call setup on the component.
        self._var_rel_names = {'input': [], 'output': []}
        self._var_rel2meta = {}
        if comm.size == 1:
            self._has_distrib_vars = self._has_distrib_outputs = False

        for meta in self._static_var_rel2meta.values():
            # reset shape if any dynamic shape parameters are set in case this is a resetup
            # NOTE: this is necessary because we allow variables to be added in __init__.
            if 'shape_by_conn' in meta and (meta['shape_by_conn'] or
                                            meta['compute_shape'] is not None):
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
        if self._coloring_info.coloring is not None:
            for meta in self._declared_partials_patterns.values():
                if 'method' in meta and meta['method'] is not None:
                    break
            else:
                method = self._coloring_info.method
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
        super()._setup_var_data()

        # Compute the prefix for turning rel/prom names into abs names
        prefix = self.pathname + '.'

        for io in ['input', 'output']:
            abs2meta = self._var_abs2meta[io]
            allprocs_abs2meta = self._var_allprocs_abs2meta[io]

            is_input = io == 'input'
            for prom_name in self._var_rel_names[io]:
                abs_name = prefix + prom_name
                abs2meta[abs_name] = metadata = self._var_rel2meta[prom_name]
                self._resolver.add_mapping(abs_name, prom_name, io,
                                           local=True, distributed=metadata['distributed'])

                allprocs_abs2meta[abs_name] = {
                    meta_name: metadata[meta_name]
                    for meta_name in global_meta_names[io]
                }
                if is_input and 'src_indices' in metadata:
                    allprocs_abs2meta[abs_name]['has_src_indices'] = \
                        metadata['src_indices'] is not None

            for prom_name, val in self._var_discrete[io].items():
                abs_name = prefix + prom_name
                self._resolver.add_mapping(abs_name, prom_name, io,
                                           local=True, continuous=False)

                # Compute allprocs_discrete (metadata for discrete vars)
                self._var_allprocs_discrete[io][abs_name] = v = val.copy()
                del v['val']

        if self._var_discrete['input'] or self._var_discrete['output']:
            self._discrete_inputs = _DictValues(self._var_discrete['input'])
            self._discrete_outputs = _DictValues(self._var_discrete['output'])
        else:
            self._discrete_inputs = self._discrete_outputs = {}

        if self.comm.size > 1:
            # check that same variables are declared on all procs
            vnames = (list(self._var_rel_names['output']), list(self._var_rel_names['input']))
            allnames = self.comm.gather(vnames, root=0)
            if self.comm.rank == 0:
                outset, inset = vnames
                msg = ''
                for oset, iset in allnames:
                    if iset != inset or oset != outset:
                        msg = self._missing_vars_error(allnames)
                        break
                self.comm.bcast(msg, root=0)
            else:
                msg = self.comm.bcast(None, root=0)

            if msg:
                raise RuntimeError(msg)

        if self.compute_primal is not None:
            self._check_compute_primal_args()
            self._check_compute_primal_returns()

        self._serial_idxs = None
        self._inconsistent_keys = set()

    def _missing_vars_error(self, allnames):
        msg = ''
        outset, inset = allnames[0]
        for rank, (olist, ilist) in enumerate(allnames):
            if rank != 0 and (olist != outset or ilist != inset):
                idiff = set(inset).symmetric_difference(ilist)
                odiff = set(outset).symmetric_difference(olist)
                if idiff or odiff:
                    varnames = sorted(idiff | odiff)
                    if len(varnames) == 1:
                        varmsg = f"Variable '{varnames[0]}' exists on some ranks and not others."
                    else:
                        varmsg = f"Variables {varnames} exist on some ranks and not others."
                else:
                    varmsg = "Variables have not been declared in the same order on all ranks."

                msg = (f"{self.msginfo}: {varmsg} A component must declare all variables in "
                       "the same order on all ranks, even if the size of the variable is 0 on "
                       "some ranks.")
                break
        return msg

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

        self._owned_output_sizes = self._var_sizes['output']

    def _setup_partials(self):
        """
        Process all partials and approximations that the user declared.
        """
        self._subjacs_info = {}

        self.setup_partials()  # hook for component writers to specify sparsity patterns

        if self.options['derivs_method'] in ('cs', 'fd'):
            if self.matrix_free:
                raise RuntimeError(f"{self.msginfo}: derivs_method of 'cs' or 'fd' is not "
                                   "allowed for a matrix free component.")
            self._has_approx = True
            method = self.options['derivs_method']
            if method in ('cs', 'fd'):
                self._get_approx_scheme(method)
            if not self._declared_partials_patterns:
                if self.compute_primal is None:
                    raise RuntimeError(f"{self.msginfo}: compute_primal must be defined if using "
                                       "a derivs_method option of 'cs' or 'fd'")
                # declare all partials as 'cs' or 'fd'
                for of, wrt in get_function_deps(self._orig_compute_primal,
                                                 self._var_rel_names['output']):
                    if of in self._discrete_outputs or wrt in self._discrete_inputs:
                        continue
                    self.declare_partials(of, wrt, method=method)
            else:
                # update only those partials that have been declared
                for meta in self._declared_partials_patterns.values():
                    meta['method'] = method

        # check to make sure that if num_par_fd > 1 that this system is actually doing FD.
        # Unfortunately we have to do this check after system setup has been called because that's
        # when declare_partials generally happens, so we raise an exception here instead of just
        # resetting the value of num_par_fd (because the comm has already been split and possibly
        # used by the system setup).
        orig_comm = self._full_comm if self._full_comm is not None else self.comm
        if self._num_par_fd > 1 and orig_comm.size > 1 and not (self._owns_approx_jac or
                                                                self._approx_schemes):
            raise RuntimeError("%s: num_par_fd is > 1 but no FD is active." % self.msginfo)

        for key, pattern_meta in self._declared_partials_patterns.items():
            of, wrt = key
            self._resolve_partials_patterns(of, wrt, pattern_meta)

    def setup_partials(self):
        """
        Declare partials.

        This is meant to be overridden by component classes.  All partials should be
        declared here since this is called after all size/shape information is known for
        all variables.
        """
        pass

    def _setup_residuals(self):
        """
        Process hook to call user-defined setup_residuals method if provided.
        """
        pass

    def _declared_partials_iter(self):
        """
        Iterate over all declared partials.

        Yields
        ------
        key : tuple (of, wrt)
            Subjacobian key.  Names are absolute.
        """
        yield from self._subjacs_info.keys()

    _subjac_keys_iter = _declared_partials_iter

    def _get_missing_partials(self, missing):
        """
        Provide (of, wrt) tuples for which derivatives have not been declared in the component.

        Parameters
        ----------
        missing : dict
            Dictionary containing set of missing derivatives keyed by system pathname.
        """
        if ('*', '*') in self._declared_partials_patterns or \
                (('*',), ('*',)) in self._declared_partials_patterns:
            return

        # keep old default behavior where matrix free components are assumed to have
        # 'dense' whole variable to whole variable partials if no partials are declared.
        if self.matrix_free and not self._declared_partials_patterns:
            return

        keyset = self._subjacs_info
        mset = set()
        for of in self._var_allprocs_abs2meta['output']:
            for wrt in self._var_allprocs_abs2meta['input']:
                if (of, wrt) not in keyset:
                    mset.add((of, wrt))

        if mset:
            missing[self.pathname] = mset

    @property
    def checking(self):
        """
        Return True if check_partials or check_totals is executing.

        Returns
        -------
        bool
            True if we're running within check_partials or check_totals.
        """
        return self._problem_meta is not None and self._problem_meta['checking']

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
                if self._problem_meta['has_par_deriv_color']:
                    raise RuntimeError(f"{self.msginfo}: Can't set 'run_root_only' option when "
                                       "using parallel_deriv_color.")
                return True
        return False

    def _promoted_wrt_iter(self):
        yield from self._get_partials_wrts()

    def add_input(self, name, val=1.0, shape=None, units=None, desc='', tags=None,
                  shape_by_conn=False, copy_shape=None, compute_shape=None,
                  units_by_conn=False, copy_units=None, compute_units=None,
                  require_connection=False, distributed=None, primal_name=None):
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
            Shape of this variable, only required if val is not an array. Default is None.
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
        compute_shape : function
            A function taking a dict arg containing names and shapes of this component's outputs
            and returning the shape of this input.
        units_by_conn : bool
            If True, set units of this input to match its connected output.
        copy_units : str or None
            If a str, that str is the name of a variable. Set the units of this input to match those
            of the named variable.
        compute_units : function
            A function taking a dict arg containing names and PhysicalUnits of this component's
            outputs and returning the PhysicalUnits of this input.
        require_connection : bool
            If True and this input is not a design variable, it must be connected to an output.
        distributed : bool
            If True, this variable is a distributed variable, so it can have different sizes/values
            across MPI processes.
        primal_name : str or None
            Valid python name to represent the variable in compute_primal if 'name' is not a valid
            python name.

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
        if units is not None:
            if not isinstance(units, str):
                raise TypeError('%s: The units argument should be a str or None.' % self.msginfo)
            units = simplify_unit(units, msginfo=self.msginfo)

        if tags is not None and not isinstance(tags, (str, list, set)):
            raise TypeError('The tags argument should be a str, set, or list')

        if copy_shape and compute_shape:
            raise ValueError(f"{self.msginfo}: Only one of 'copy_shape' or 'compute_shape' can "
                             "be specified.")
        if copy_units and compute_units:
            raise ValueError(f"{self.msginfo}: Only one of 'copy_units' or 'compute_units' can "
                             "be specified.")

        if copy_shape and not isinstance(copy_shape, str):
            raise TypeError(f"{self.msginfo}: The copy_shape argument should be a str or None but "
                            f"a '{type(copy_shape).__name__}' was given.")
        if copy_units and not isinstance(copy_units, str):
            raise TypeError(f"{self.msginfo}: The copy_units argument should be a str or None but "
                            f"a '{type(copy_units).__name__}' was given.")

        if compute_shape and not callable(compute_shape):
            raise TypeError(f"{self.msginfo}: The compute_shape argument should be callable but "
                            f"a '{type(compute_shape).__name__}' was given.")
        if compute_units and not callable(compute_units):
            raise TypeError(f"{self.msginfo}: The compute_units argument should be callable but "
                            f"a '{type(compute_units).__name__}' was given.")

        if shape_by_conn or copy_shape or compute_shape:
            if shape or ndim(val) > 0:
                raise ValueError("%s: If shape is to be set dynamically, 'shape' and 'val' should "
                                 "be a scalar, but shape of '%s' and val of '%s' was given for "
                                 "variable '%s'." % (self.msginfo, shape, val, name))
        else:
            # value, shape: based on args, making sure they are compatible
            val, shape = ensure_compatible(name, val, shape,
                                           default_shape=self.options['default_shape'])

        if (units_by_conn or copy_units or compute_units) and units is not None:
            raise ValueError("%s: If units is to be set dynamically using 'units_by_conn', "
                             "'copy_units', or 'compute_units', 'units' should be None, but "
                             "units of '%s' was given for variable '%s'."
                             % (self.msginfo, units, name))

        # until we get rid of component level distributed option, handle the case where
        # component distributed has been set to True but variable distributed has been set
        # to False by the caller.
        if distributed is not False:
            if distributed is None:
                distributed = False
            # using ._dict below to avoid tons of deprecation warnings
            distributed = distributed or ('distributed' in self.options and
                                          self.options._dict['distributed']['val'])

        if compute_shape is not None and is_lambda(compute_shape):
            compute_shape = LambdaPickleWrapper(compute_shape)
        if compute_units is not None and is_lambda(compute_units):
            compute_units = LambdaPickleWrapper(compute_units)

        if primal_name is not None:
            self._valid_name_map[name] = primal_name

        metadata = {
            'val': val,
            'shape': shape,
            'size': shape_to_len(shape),
            'src_indices': None,
            'flat_src_indices': None,
            'units': units,
            'desc': desc,
            'tags': make_set(tags),
            'shape_by_conn': shape_by_conn,
            'compute_shape': compute_shape,
            'copy_shape': copy_shape,
            'units_by_conn': units_by_conn,
            'compute_units': compute_units,
            'copy_units': copy_units,
            'require_connection': require_connection,
            'distributed': distributed,
        }

        # this will get reset later if comm size is 1
        self._has_distrib_vars |= distributed

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

    def add_discrete_input(self, name, val, desc='', tags=None, primal_name=None):
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
        primal_name : str or None
            Valid python name to represent the variable in compute_primal if 'name' is not a valid
            python name.

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

        if primal_name is not None:
            self._valid_name_map[name] = primal_name

        metadata = {}

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
                   shape_by_conn=False, copy_shape=None, compute_shape=None,
                   units_by_conn=False, copy_units=None, compute_units=None,
                   distributed=None, primal_name=None):
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
        compute_shape : function
            A function taking a dict arg containing names and shapes of this component's inputs
            and returning the shape of this output.
        units_by_conn : bool
            If True, set the units of this output to match its connected input(s).
        copy_units : str or None
            If a str, that str is the name of a variable. Set the units of this output to match
            those of the named variable.
        compute_units : function
            A function taking a dict arg containing names and PhysicalUnits of this component's
            inputs and returning the PhysicalUnits of this output.
        distributed : bool
            If True, this variable is a distributed variable, so it can have different sizes/values
            across MPI processes.
        primal_name : str or None
            Valid python name to represent the variable in compute_primal if 'name' is not a valid
            python name.

        Returns
        -------
        dict
            Metadata for added variable.
        """
        # First, type check all arguments
        if (shape_by_conn or copy_shape or compute_shape) and (shape is not None or ndim(val) > 0):
            raise ValueError("%s: If shape is to be set dynamically using 'shape_by_conn', "
                             "'copy_shape', or 'compute_shape', 'shape' and 'val' should be scalar,"
                             " but shape of '%s' and val of '%s' was given for variable '%s'."
                             % (self.msginfo, shape, val, name))

        if (units_by_conn or copy_units or compute_units) and units is not None:
            raise ValueError("%s: If units is to be set dynamically using 'units_by_conn', "
                             "'copy_units', or 'compute_units', 'units' should be None, but "
                             "units of '%s' was given for variable '%s'."
                             % (self.msginfo, units, name))

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

        if not (copy_shape or shape_by_conn or compute_shape):
            if not isscalar(val) and not isinstance(val, _allowed_types):
                msg = '%s: The val argument should be a float, list, tuple, ndarray or Iterable'
                raise TypeError(msg % self.msginfo)

            default_shape = self.options['default_shape']
            # value, shape: based on args, making sure they are compatible
            val, shape = ensure_compatible(name, val, shape, default_shape=default_shape)

            if lower is not None:
                lower = ensure_compatible(name, lower, shape, default_shape=default_shape)[0]
                self._has_bounds = True
            if upper is not None:
                upper = ensure_compatible(name, upper, shape, default_shape=default_shape)[0]
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

        if res_ref is not None:
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
            distributed = distributed or ('distributed' in self.options and
                                          self.options._dict['distributed']['val'])

        if copy_shape and compute_shape:
            raise ValueError(f"{self.msginfo}: Only one of 'copy_shape' or 'compute_shape' can "
                             "be specified.")
        if copy_units and compute_units:
            raise ValueError(f"{self.msginfo}: Only one of 'copy_units' or 'compute_units' can "
                             "be specified.")

        if copy_shape and not isinstance(copy_shape, str):
            raise TypeError(f"{self.msginfo}: The copy_shape argument should be a str or None but "
                            f"a '{type(copy_shape).__name__}' was given.")
        if copy_units and not isinstance(copy_units, str):
            raise TypeError(f"{self.msginfo}: The copy_units argument should be a str or None but "
                            f"a '{type(copy_units).__name__}' was given.")

        if compute_shape and not callable(compute_shape):
            raise TypeError(f"{self.msginfo}: The compute_shape argument should be callable but "
                            f"a '{type(compute_shape).__name__}' was given.")
        if compute_units and not callable(compute_units):
            raise TypeError(f"{self.msginfo}: The compute_units argument should be callable but "
                            f"a '{type(compute_units).__name__}' was given.")

        if compute_shape is not None and is_lambda(compute_shape):
            compute_shape = LambdaPickleWrapper(compute_shape)
        if compute_units is not None and is_lambda(compute_units):
            compute_units = LambdaPickleWrapper(compute_units)

        if primal_name is not None:
            self._valid_name_map[name] = primal_name

        metadata = {
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
            'compute_shape': compute_shape,
            'copy_shape': copy_shape,
            'units_by_conn': units_by_conn,
            'compute_units': compute_units,
            'copy_units': copy_units,
        }

        # this will get reset later if comm size is 1
        self._has_distrib_vars |= distributed
        self._has_distrib_outputs |= distributed

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

    def add_discrete_output(self, name, val, desc='', tags=None, primal_name=None):
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
        primal_name : str or None
            Valid python name to represent the variable in compute_primal if 'name' is not a valid
            python name.

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

        if primal_name is not None:
            self._valid_name_map[name] = primal_name

        metadata = {}

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
            Dictionary mapping an absolute name to its allprocs variable index for the
            whole model.
        all_sizes : dict
            Mapping of types to sizes of each variable in all procs for the whole model.

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
        # src is outside of this component, so we need to use the all_sizes dict to get the sizes
        # of the output variables
        sizes_out = all_sizes['output']
        added_src_inds = []
        # loop over continuous local inputs
        for iname, meta_in in abs2meta_in.items():
            if meta_in['src_indices'] is None and iname not in abs_in2prom_info:
                src = abs_in2out[iname]
                dist_in = meta_in['distributed']
                dist_out = all_abs2meta_out[src]['distributed']
                if dist_in or dist_out:
                    i = self._var_allprocs_abs2idx[iname]
                    gsize_out = all_abs2meta_out[src]['global_size']
                    gsize_in = all_abs2meta_in[iname]['global_size']
                    vout_sizes = sizes_out[:, all_abs2idx[src]]

                    offset = None
                    if gsize_out == gsize_in or (not dist_out and np.sum(vout_sizes) == gsize_in):
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

                    # total sizes differ and output is distributed, so can't determine mapping
                    if offset is None:
                        self._collect_error(f"{self.msginfo}: Can't determine src_indices "
                                            f"automatically for input '{iname}'. They must be "
                                            "supplied manually.", ident=(self.pathname, iname))
                        continue

                    if dist_in and not dist_out:
                        src_shape = self._get_full_dist_shape(src, all_abs2meta_out[src]['shape'],
                                                              'output')
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
        self._has_approx = True
        info = self._subjacs_info

        for abs_key in self._matching_key_iter(of, wrt):
            meta = info[abs_key]
            meta['method'] = method
            meta.update(kwargs)

    def declare_partials(self, of, wrt, dependent=True, rows=None, cols=None, val=None,
                         method='exact', step=None, form=None, step_calc=None, minimum_step=None,
                         diagonal=None):
        """
        Declare information about this component's subjacobians.

        Parameters
        ----------
        of : str or iter of str
            The name of the residual(s) that derivatives are being computed for.
            May also contain a glob pattern.
        wrt : str or iter of str
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
            compatibilty, it can be 'rel', which is now equivalent to 'rel_avg'. Defaults to None,
            in which case the approximation method provides its default value.
        minimum_step : float
            Minimum step size allowed when using one of the relative step_calc options.
        diagonal : bool
            If True, the subjacobian is a diagonal matrix.

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

        if not isinstance(of, (str, Iterable)):
            raise ValueError(f"{self.msginfo}: in declare_partials, the 'of' arg must be a string "
                             f"or an iter of strings, but got {of}.")
        if not isinstance(wrt, (str, Iterable)):
            raise ValueError(f"{self.msginfo}: in declare_partials, the 'wrt' arg must be a "
                             f"string or an iter of strings, but got {wrt}.")

        of = of if isinstance(of, str) else tuple(of)
        wrt = wrt if isinstance(wrt, str) else tuple(wrt)

        key = (of, wrt)
        if key not in self._declared_partials_patterns:
            self._declared_partials_patterns[key] = {}   # SUBJAC_META_DEFAULTS.copy()
        meta = self._declared_partials_patterns[key]
        meta['dependent'] = dependent

        # If only one of rows/cols is specified
        if (rows is None) ^ (cols is None):
            raise ValueError('{}: d({})/d({}): If one of rows/cols is specified, then '
                             'both must be specified.'.format(self.msginfo, of, wrt))

        if dependent:
            meta['val'] = val
            meta['diagonal'] = diagonal

            if val is not None:
                _val = val.data if issparse(val) else val
                if np.all(_val == 0):
                    warn_deprecation(f'{self.msginfo}: d({of})/d({wrt}): Partial was declared to be'
                                     ' exactly zero. This is inefficient and the declaration '
                                     'should be removed. In a future version of OpenMDAO this '
                                     'behavior will raise an error.')

            if rows is not None:
                rows = np.asarray(rows, dtype=INT_DTYPE)
                cols = np.asarray(cols, dtype=INT_DTYPE)

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
            compatibilty, it can be 'rel', which is now equivalent to 'rel_avg'. Defaults to None,
            in which case the approximation method provides its default value..
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
        wrt = self._get_partials_wrts()
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

    def _get_approx_partial_options(self, key, method='fd', checkopts=None):
        approx = self._get_approx_scheme(method)
        options = approx.DEFAULT_OPTIONS.copy()
        if checkopts is None:
            options.update(approx._wrt_meta)
        else:
            options.update(checkopts)
            options.update(approx._wrt_meta)

        if not approx._wrt_meta:
            del self._approx_schemes[method]

        abs_key = rel_key2abs_key(self, key)
        if abs_key in self._subjacs_info:
            meta = self._subjacs_info[abs_key]
            options.update({k: v for k, v in meta.items() if v is not None and k in options})

        return options

    def _resolve_partials_patterns(self, of, wrt, pattern_meta):
        """
        Store subjacobian metadata for specific of, wrt pairs after resolving glob patterns.

        Parameters
        ----------
        of : tuple of str
            The names of the residuals that derivatives are being computed for.
            May also contain glob patterns.
        wrt : tuple of str
            The names of the variables that derivatives are taken with respect to.
            This can contain the name of any input or output variable.
            May also contain glob patterns.
        pattern_meta : dict
            Metadata dict specifying shape, and/or approx properties, keyed by (of, wrt) as
            described above.
        """
        val = pattern_meta['val'] if 'val' in pattern_meta else None
        dependent = pattern_meta['dependent']
        matfree = self.matrix_free
        if matfree:
            if 'val' in pattern_meta:
                if pattern_meta['val'] is not None:
                    issue_warning(f"{self.msginfo}: 'val' was passed to declare_partials for {of} "
                                  f"wrt {wrt} but matrix_free is True, so 'val' will be ignored.")
                del pattern_meta['val']
            val = None
        elif isinstance(val, list):
            val = pattern_meta['val'] = np.asarray(val, dtype=float)

        rows_max = cols_max = 0
        rows = cols = None

        if dependent:

            if 'rows' in pattern_meta and pattern_meta['rows'] is not None:  # sparse list format
                rows = pattern_meta['rows']
                cols = pattern_meta['cols']
                if rows.size > 0:
                    rows_max = rows.max()
                    cols_max = cols.max()

                if not matfree:
                    if not isscalar(val) and val is not None:
                        if rows.shape != val.shape:
                            raise ValueError('{}: d({})/d({}): If rows and cols are specified, val '
                                             'must be a scalar or have the same shape, val: {}, '
                                             'rows/cols: {}'.format(self.msginfo, of, wrt,
                                                                    val.shape, rows.shape))

        abs2meta_in = self._var_abs2meta['input']
        abs2meta_out = self._var_abs2meta['output']

        pattern_meta = dict(pattern_meta)

        for abs_key in self._matching_key_iter(of, '*' if wrt is None else wrt):
            if not dependent:
                if abs_key in self._subjacs_info:
                    del self._subjacs_info[abs_key]
                continue

            _of, _wrt = abs_key
            wrtmeta = abs2meta_in[_wrt] if _wrt in abs2meta_in else abs2meta_out[_wrt]
            shape = (abs2meta_out[_of]['size'], wrtmeta['size'])

            dist_out = abs2meta_out[_of]['distributed']
            dist_in = wrtmeta['distributed']

            if dist_in and not dist_out and not matfree:
                rel_key = abs_key2rel_key(self, abs_key)
                raise RuntimeError(f"{self.msginfo}: component has defined partial {rel_key} "
                                   "which is a non-distributed output wrt a distributed input."
                                   " This is only supported using the matrix free API.")

            if shape[0] == 0 or shape[1] == 0:
                if shape[0] == 0:
                    if dist_out:
                        # distributed vars are allowed to have zero size inputs on some procs
                        rows_max = -1
                    else:
                        # non-distributed vars are not allowed to have zero size inputs
                        raise ValueError(f"{self.msginfo}: '{_of}' is an array of size 0.")
                if shape[1] == 0:
                    if not dist_in:
                        # non-distributed vars are not allowed to have zero size outputs
                        raise ValueError(f"{self.msginfo}: '{_wrt}' is an array of size 0.")
                    else:
                        # distributed vars are allowed to have zero size outputs on some procs
                        cols_max = -1

            if rows_max >= shape[0] or cols_max >= shape[1]:
                relof, relwrt = abs_key2rel_key(self, abs_key)
                raise ValueError(f"{self.msginfo}: d({relof})/d({relwrt}): Expected {shape[0]}x"
                                 f"{shape[1]} but declared at least {rows_max + 1}x"
                                 f"{cols_max + 1}.")

            if abs_key in self._subjacs_info:
                prev_meta = self._subjacs_info[abs_key]
            else:
                prev_meta = None

            self._subjacs_info[abs_key] = Subjac.get_instance_metadata(pattern_meta, prev_meta,
                                                                       shape, self, abs_key)

    def _column_iotypes(self):
        """
        Return a tuple of the iotypes that make up columns of the jacobian.

        Returns
        -------
        tuple of the form ('input',)
            The iotypes that make up columns of the jacobian.
        """
        return ('input',)

    def _get_partials_wrts(self):
        """
        Get list of 'wrt' variables that form the partial jacobian.

        Returns
        -------
        list
            List of 'wrt' relative variable names.
        """
        namelists = [self._var_rel_names[io] for io in self._column_iotypes()]
        return [n for n in chain(*namelists)]

    def _get_partials_ofs(self, use_resname=False):
        """
        Get lists of 'of' variables that form the partial jacobian.

        Parameters
        ----------
        use_resname : bool
            Ignored.

        Returns
        -------
        list
            List of 'of' relative variable names.
        """
        return list(self._var_rel_names['output'])

    def _matching_key_iter(self, of_patterns, wrt_patterns, use_resname=False):
        """
        Iterate over all combinations of matching keys for the given patterns.

        Parameters
        ----------
        of_patterns : list of str
            List of variable names and/or glob patterns for the 'of' variables.
        wrt_patterns : list of str
            List of variable names and/or glob patterns for the 'wrt' variables.
        use_resname : bool, optional
            If True, match of_patterns against residuals instead of outputs.

        Yields
        ------
        tuple
            A tuple of matching keys, where the first element is the 'of' key and the second
            element is the 'wrt' key.  Both are absolute names.
        """
        of_bundles = self._find_of_matches(of_patterns, use_resname=use_resname)
        wrt_bundles = self._find_wrt_matches(wrt_patterns)

        for of_bundle, wrt_bundle in product(of_bundles, wrt_bundles):
            of_pattern, of_matches = of_bundle
            wrt_pattern, wrt_matches = wrt_bundle
            if not of_matches:
                raise ValueError(f'{self.msginfo}: No matches were found for of="{of_pattern}"')
            if not wrt_matches:
                raise ValueError(f'{self.msginfo}: No matches were found for wrt="{wrt_pattern}"')
            yield from abs_key_iter(self, of_matches, wrt_matches)

    def _find_of_matches(self, pattern, use_resname=False):
        """
        Find all matches for the given 'of' pattern.

        Parameters
        ----------
        pattern : str
            Glob pattern or relative variable name.
        use_resname : bool
            If True, match residual names instead of output names.

        Returns
        -------
        list
            List of tuples of the form (abs_name, meta) where abs_name is the absolute name of the
            matching variable and meta is the metadata for that variable.
        """
        of_list = [pattern] if isinstance(pattern, str) else pattern
        return [(pattern, find_matches(pattern, self._get_partials_ofs(use_resname=use_resname)))
                for pattern in of_list]

    def _find_wrt_matches(self, pattern):
        """
        Find all matches for the given 'wrt' pattern.

        Parameters
        ----------
        pattern : str
            Glob pattern or relative variable name.

        Returns
        -------
        list
            List of tuples of the form (abs_name, meta) where abs_name is the absolute name of the
            matching variable and meta is the metadata for that variable.
        """
        patterns = [pattern] if isinstance(pattern, str) else pattern
        return [(pattern, find_matches(pattern, self._get_partials_wrts())) for pattern in patterns]

    def _add_approximations(self, use_relevance=True):
        """
        Add approximations for those partials registered with method=fd or method=cs.

        Parameters
        ----------
        use_relevance : bool
            If True, use relevance to determine which partials to approximate.
        """
        subjacs_info = self._subjacs_info
        wrtset = set()
        subjac_keys = self._get_approx_subjac_keys(use_relevance=use_relevance, initialize=True)
        methods = list(self._approx_schemes)
        self._approx_schemes = {}
        for method in methods:
            self._get_approx_scheme(method)

        # go through subjac keys in reverse and only add approx for the last of each wrt
        # (this prevents warnings that could confuse users)
        for i in range(len(subjac_keys) - 1, -1, -1):
            key = subjac_keys[i]
            wrt = key[1]
            if wrt not in wrtset:
                wrtset.add(wrt)
                meta = subjacs_info[key]
                self._approx_schemes[meta['method']].add_approximation(wrt, self, meta)

        # get rid of any empty approx schemes
        to_remove = [name for name, scheme in self._approx_schemes.items() if not scheme._wrt_meta]
        for name in to_remove:
            del self._approx_schemes[name]

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
                self._get_coloring()

    def _resolve_src_inds(self):
        abs2prom = self._resolver.abs2prom
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
                    if inds is not None:
                        all_abs2meta_in[tgt]['has_src_indices'] = True
                        meta['src_shape'] = shape = all_abs2meta_out[conns[tgt]]['global_shape']
                        if inds._flat_src:
                            meta['flat_src_indices'] = True
                        elif meta['flat_src_indices'] is None:
                            meta['flat_src_indices'] = flat

                        try:
                            if not isinstance(inds, Indexer):
                                meta['src_indices'] = inds = indexer(inds, flat_src=flat,
                                                                     src_shape=shape)
                            else:
                                meta['src_indices'] = inds = inds.copy()
                                inds.set_src_shape(shape)
                                self._var_prom2inds[abs2prom(tgt, iotype='input')] = \
                                    [shape, inds, flat]
                        except Exception:
                            type_exc, exc, tb = sys.exc_info()
                            self._collect_error(f"When accessing '{conns[tgt]}' with src_shape "
                                                f"{shape} from '{pinfo.prom_path()}' using "
                                                f"src_indices {inds}: {exc}", exc_type=type_exc,
                                                tback=tb, ident=(conns[tgt], tgt))

    def _check_consistent_serial_dinputs(self, nz_dist_outputs):
        """
        Check consistency across ranks for serial d_inputs variables.

        This is used primarily to test that `compute_jacvec_product` and `apply_linear` methods
        follow the OpenMDAO convention that in reverse mode, the component should perform
        'allreduce' sorts of operations only for derivatives of distributed outputs with-respect-to
        serial inputs.  This should result in serial input derivatives being consistent across all
        ranks in the Component's communicator.

        Parameters
        ----------
        nz_dist_outputs : set or list
            Set of distributed outputs with nonzero values for the most recent _apply_linear call.
        """
        if not self.checking or not self._has_distrib_outputs or self.comm.size < 2:
            return

        if self._serial_idxs is None:
            ranges = defaultdict(list)
            output_len = 0 if self.is_explicit(is_comp=True) else len(self._outputs)
            for _, offset, end, vec, slc, dist_sizes in self._get_jac_wrts():
                if dist_sizes is None:  # not distributed
                    if offset != end:
                        if vec is self._outputs:
                            ranges[vec].append(range(offset, end))
                        else:
                            ranges[vec].append(range(offset - output_len, end - output_len))

            self._serial_idxs = []
            for vec, rlist in ranges.items():
                if rlist:
                    self._serial_idxs.append((vec, np.hstack(rlist)))

        for vec, inds in self._serial_idxs:
            # _jac_wrt_iter gives us _input and possibly _output vecs (for implicit comps), but we
            # want to check _dinputs and _doutputs
            v = self._dinputs if vec is self._inputs else self._doutputs
            result = inconsistent_across_procs(self.comm, v.asarray()[inds])
            if self.comm.rank == 0 and np.any(result):
                bad_inds = np.arange(len(v), dtype=INT_DTYPE)[inds][result]
                bad_mask = np.zeros(len(v), dtype=bool)
                bad_mask[bad_inds] = True
                for inname, start, stop in v.ranges():
                    if np.any(bad_mask[start:stop]):
                        for outname in nz_dist_outputs:
                            key = (outname, inname)
                            self._inconsistent_keys.add(key)

    def _get_dist_nz_dresids(self):
        """
        Get names of distributed resids that are non-zero prior to computing derivatives.

        This should only be called when 'rev' mode is active.

        Returns
        -------
        list of str
            List of names of distributed resids that have nonzero entries.
        """
        nzresids = []
        dresids = self._dresiduals.asarray()
        for of, start, end, _, dist_sizes in self._get_jac_ofs():
            if dist_sizes is not None:
                if np.any(dresids[start:end]):
                    nzresids.append(of)

        full_nzresids = set()
        if self.comm.rank == 0:
            for nzoutlist in self.comm.gather(nzresids, root=0):
                full_nzresids.update(nzoutlist)
            return full_nzresids

        self.comm.gather(nzresids, root=0)
        return nzresids

    def _get_graph_node_meta(self):
        """
        Return metadata to add to this system's graph node.

        Returns
        -------
        dict
            Metadata for this system's graph node.
        """
        meta = super()._get_graph_node_meta()
        meta['base'] = 'ExplicitComponent' if self.is_explicit() else 'ImplicitComponent'
        return meta

    def compute_fd_jac(self, jac, method='fd'):
        """
        Force the use of finite difference to compute a jacobian.

        This can be used to compute sparsity for a component that computes derivatives analytically
        in order to check the accuracy of the declared sparsity.

        Parameters
        ----------
        jac : Jacobian
            The Jacobian object that will contain the computed jacobian.
        method : str
            The type of finite difference to perform. Valid options are 'fd' for forward difference,
            or 'cs' for complex step.
        """
        if method == 'jax':
            method = 'fd'
        fd_methods = {'fd': _supported_methods['fd'], 'cs': _supported_methods['cs']}
        try:
            approximation = fd_methods[method]()
        except KeyError:
            raise ValueError(f"Method '{method}' is not a recognized finite difference method.")

        local_opts = self._get_check_partial_options()
        added_wrts = set()

        for rel_key in product(self._get_partials_ofs(), self._get_partials_wrts()):
            fd_options = self._get_approx_partial_options(rel_key, method=method,
                                                          checkopts=local_opts)
            _, rel_wrt = rel_key
            wrt = self._resolver.rel2abs(rel_wrt)

            # prevent adding multiple approxs with same wrt (and confusing users with warnings)
            if wrt not in added_wrts:
                approximation.add_approximation(wrt, self, fd_options)
                added_wrts.add(wrt)

        # Perform the FD here.
        with self._unscaled_context(outputs=[self._outputs], residuals=[self._residuals]):
            approximation.compute_approximations(self, jac=jac)

    def check_sparsity(self, method='fd', max_nz=90., out_stream=_DEFAULT_OUT_STREAM):
        """
        Check the sparsity of the computed jacobian against the declared sparsity.

        Check is skipped if one of the dimensions of the jacobian is 1 or if the percentage of
        nonzeros in the computed jacobian is greater than max_nz%.

        Parameters
        ----------
        method : str
            The type of finite difference to perform. Valid options are 'fd' for forward difference,
            or 'cs' for complex step.
        max_nz : float
            If the percentage of nonzeros in a sub-jacobian exceeds this, no warning is issued if
            the computed sparsity does not match the declared sparsity.
        out_stream : file-like object
            Where to send the output.  If None, output will be suppressed.

        Returns
        -------
        list
            A list of tuples, one for each subjacobian that has a mismatch between the computed
            sparsity and the declared sparsity.  Each tuple has the form (of, wrt, computed_rows,
            computed_cols, declared_rows, declared_cols, shape, pct_nonzero).
        """
        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        def rowsizeiter():
            for of, start, end, _, _ in self._get_jac_ofs():
                yield of, end - start

        def colsizeiter():
            for wrt, start, end, _, _, _ in self._get_jac_wrts():
                yield wrt, end - start

        sparsity, _ = self.compute_fd_sparsity(method=method)

        prefix = self.pathname + '.'
        plen = len(prefix)
        ret = []
        for of, wrt, nzrows, nzcols, shape in submat_sparsity_iter(rowsizeiter(), colsizeiter(),
                                                                   sparsity.row, sparsity.col,
                                                                   sparsity.shape):
            if 1 in shape:
                continue
            key = (of, wrt)
            if key in self._subjacs_info:
                meta = self._subjacs_info[key]
                computed = sorted(zip(nzrows, nzcols))
                if meta['rows'] is None:
                    if meta['diagonal']:
                        rows = np.arange(shape[0])
                        cols = np.arange(shape[1])
                        declared = sorted(zip(rows, cols))
                    else:
                        rows = []
                        cols = []
                        declared = []
                else:
                    rows = meta['rows']
                    cols = meta['cols']
                    declared = sorted(zip(rows, cols))

                if declared != computed:
                    pct_nonzero = 100. * len(nzrows) / (shape[0] * shape[1])
                    if pct_nonzero > max_nz:
                        continue
                    if shape[0] > 200 or shape[1] > 200:
                        mstr = "Sparsity matrix too large to show."
                    else:
                        stream = StringIO()
                        val_map = {0: '.', 1: 'C', 3: 'D', 4: 'x'}
                        sparsity_diff_viz(csr_matrix((np.ones(len(nzrows)), (nzrows, nzcols)),
                                                     shape=shape, dtype=bool),
                                          csr_matrix((np.ones(len(rows)), (rows, cols)),
                                                     shape=shape, dtype=bool),
                                          val_map=val_map,
                                          stream=stream)
                        mstr = stream.getvalue()
                    wrn = (f"{self.msginfo}:\n(D)eclared sparsity pattern != (C)omputed sparsity "
                           f"pattern for sub-jacobian ({of[plen:]}, {wrt[plen:]}) with shape "
                           f"{shape} and {pct_nonzero:.2f}% nonzeros:\n{mstr}\n")
                    ret.append((of, wrt, nzrows, nzcols, rows, cols, shape, pct_nonzero, wrn))
                    if out_stream is not None:
                        print(wrn, file=out_stream)

        return ret

    def _check_fds_differ(self, method, step, form, step_calc, minimum_step):
        # Check to make sure the method and settings used for checking
        #   is different from the method used to calc the derivatives
        # Could do this later in this method but at that point some computations could have been
        #   made and it would just waste time before the user is told there is an error and the
        #   program errs out
        requested_method = method
        alloc_complex = self._outputs._alloc_complex

        local_opts = self._get_check_partial_options()

        nocs = False
        if not alloc_complex:
            if method == 'cs':
                nocs = True

            for meta in local_opts.values():
                if 'method' in meta and meta['method'] == 'cs':
                    nocs = True
                    break

        for keypats, meta in self._declared_partials_patterns.items():

            # Get the complete set of options, including defaults
            #    for the computing of the derivs for this component
            if 'method' not in meta:
                meta_with_defaults = {}
                meta_with_defaults['method'] = 'exact'
            elif meta['method'] == 'cs':
                meta_with_defaults = ComplexStep.DEFAULT_OPTIONS.copy()
            else:
                meta_with_defaults = FiniteDifference.DEFAULT_OPTIONS.copy()
            meta_with_defaults.update(meta)

            _, wrtpats = keypats
            # For each of the partials, check to see if the
            #   check partials options are different than the options used to compute
            #   the partials
            for _, wrtvars in self._find_wrt_matches(wrtpats):
                for var in wrtvars:
                    # we now have individual vars like 'x'
                    # get the options for checking partials
                    fd_options = _get_fd_options(var, requested_method, local_opts, step,
                                                 form, step_calc, alloc_complex, minimum_step)

                    if not alloc_complex:
                        if meta_with_defaults['method'] == 'cs' or fd_options['method'] == 'cs':
                            nocs = True

                    # compare the compute options to the check options
                    if fd_options['method'] != meta_with_defaults['method']:
                        all_same = False
                    else:
                        all_same = True
                        if fd_options['method'] == 'fd':
                            option_names = ['form', 'step', 'step_calc', 'minimum_step',
                                            'directional']
                        else:
                            option_names = ['step', 'directional']
                        for name in option_names:
                            if fd_options[name] != meta_with_defaults[name]:
                                all_same = False
                                break
                    if all_same:
                        msg = (f"Checking partials with respect to variable '{var}' in component "
                               f"'{self.pathname}' using the same method and options as are used "
                               "to compute the component's derivatives will not provide any "
                               "relevant information on the accuracy.\n"
                               "To correct this, change the options to do the "
                               "check_partials using either:\n"
                               "     - arguments to Problem.check_partials.\n"
                               "     - arguments to Component.set_check_partial_options")

                        issue_warning(msg, prefix=self.msginfo,
                                      category=OMInvalidCheckDerivativesOptionsWarning)

        if nocs:
            self._nocs_warning()

    def check_partials(self, out_stream=_DEFAULT_OUT_STREAM,
                       compact_print=False, abs_err_tol=0.0, rel_err_tol=1e-6,
                       method='fd', step=None, form='forward', step_calc='abs',
                       minimum_step=1e-12, force_dense=True, show_only_incorrect=False,
                       show_worst=True, rich_print=True):
        """
        Check partial derivatives comprehensively for this component.

        Parameters
        ----------
        out_stream : file-like object
            Where to send human readable output. By default it goes to stdout.
            Set to None to suppress.
        compact_print : bool
            Set to True to just print the essentials, one line per input-output pair.
        abs_err_tol : float
            Threshold value for absolute error.  Errors about this value will have a '*' displayed
            next to them in output, making them easy to search for. Default is 1.0E-6.
        rel_err_tol : float
            Threshold value for relative error.  Errors about this value will have a '*' displayed
            next to them in output, making them easy to search for. Note at times there may be a
            significant relative error due to a minor absolute error.  Default is 1.0E-6.
        method : str
            Method, 'fd' for finite difference or 'cs' for complex step. Default is 'fd'.
        step : None, float, or list/tuple of float
            Step size(s) for approximation. Default is None, which means 1e-6 for 'fd' and 1e-40 for
            'cs'.
        form : str
            Form for finite difference, can be 'forward', 'backward', or 'central'. Default
            'forward'.
        step_calc : str
            Step type for computing the size of the finite difference step. It can be 'abs' for
            absolute, 'rel_avg' for a size relative to the absolute value of the vector input, or
            'rel_element' for a size relative to each value in the vector input. In addition, it
            can be 'rel_legacy' for a size relative to the norm of the vector.  For backwards
            compatibilty, it can be 'rel', which is now equivalent to 'rel_avg'. Defaults to None,
            in which case the approximation method provides its default value.
        minimum_step : float
            Minimum step size allowed when using one of the relative step_calc options.
        force_dense : bool
            If True, analytic derivatives will be coerced into arrays. Default is True.
        show_only_incorrect : bool, optional
            Set to True if output should print only the subjacs found to be incorrect.
        show_worst : bool, optional
            Set to False to suppress the display of the worst subjac.
        rich_print : bool, optional
            If True, print using rich if available.

        Returns
        -------
        tuple of the form (derivs_dict, worst)
            Where derivs_dict is a dict, where the top key is the component pathname.
            Under the top key, the subkeys are the (of, wrt) keys of the subjacs.
            Within the (of, wrt) entries are the following keys:
            'tol violation', 'magnitude', 'J_fd', 'J_fwd', 'J_rev', 'vals_at_max_error',
            and 'rank_inconsistent'.
            For 'J_fd', 'J_fwd', 'J_rev' the value is a numpy array representing the computed
            Jacobian for the three different methods of computation.
            For 'tol violation' and 'vals_at_max_error' the value is a
            tuple containing values for forward - fd, reverse - fd, forward - reverse. For
            'magnitude' the value is a tuple indicating the maximum magnitude of values found in
            Jfwd, Jrev, and Jfd.
            The boolean 'rank_inconsistent' indicates if the derivative wrt a serial variable is
            inconsistent across MPI ranks.

            worst is either None or a tuple of the form (error, table_row, header)
            where error is the max error found, table_row is the formatted table row
            containing the max error, and header is the formatted table header.  'worst'
            is not None only if compact_print is True.
        """
        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if self._problem_meta is not None:
            if self._problem_meta['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
                raise RuntimeError(f"{self.msginfo}: Can't check_partials before final_setup. Also,"
                                   " make sure to set valid input values before calling "
                                   "check_partials either manually or by running the model.")

        self._check_fds_differ(method, step, form, step_calc, minimum_step)

        # Make sure we're in a valid state
        self.run_apply_nonlinear()

        input_cache = self._inputs.asarray(copy=True)
        output_cache = self._outputs.asarray(copy=True)

        local_opts = self._get_check_partial_options()

        if self.matrix_free:
            directions = ('fwd', 'rev')
        else:
            # TODO: replace 'fwd' with self.best_partial_deriv_direction(). Currently fails
            # when it equals 'rev' for directional derivatives.
            directions = ('fwd',)  # rev same as fwd for analytic jacobians
            self.run_linearize(sub_do_ln=False)

        nondep_derivs = set()
        of_list = self._get_partials_ofs()
        wrt_list = self._get_partials_wrts()
        axis = {'fwd': 1, 'rev': 0}
        mfree_directions = {}
        abs2meta_in = self._var_allprocs_abs2meta['input']
        abs2meta_out = self._var_allprocs_abs2meta['output']
        partials_data = defaultdict(dict)
        requested_method = method
        probmeta = self._problem_meta
        prefix = self.pathname + '.'

        # ensure we don't miss any pertials due to relevance
        with self._relevance.active(False):

            for mode in directions:
                jac_key = 'J_' + mode

                with self._unscaled_context(outputs=[self._outputs], residuals=[self._residuals]):

                    # Matrix-free components need to calculate Jacobian by matrix-vector product.
                    if self.matrix_free:
                        dstate = self._doutputs
                        if mode == 'fwd':
                            dinputs = self._dinputs
                            doutputs = self._dresiduals
                            in_list = wrt_list
                            out_list = of_list
                        else:
                            dinputs = self._dresiduals
                            doutputs = self._dinputs
                            in_list = of_list
                            out_list = wrt_list

                        for inp in in_list:
                            inp_abs = prefix + inp
                            if mode == 'fwd':
                                directional = inp in local_opts and local_opts[inp]['directional']
                            else:
                                directional = len(mfree_directions) > 0

                            try:
                                flat_view = dinputs._abs_get_val(inp_abs)
                            except KeyError:
                                # Implicit state
                                flat_view = dstate._abs_get_val(inp_abs)

                            if directional:
                                n_in = 1
                                idxs = range(1)
                                if inp in mfree_directions:
                                    perturb = mfree_directions[inp]
                                else:
                                    perturb = 2.0 * np.random.random(len(flat_view)) - 1.0
                                    mfree_directions[inp] = perturb

                            else:
                                n_in = len(flat_view)
                                idxs = LocalRangeIterable(self, inp_abs, use_vec_offset=False)
                                perturb = 1.0

                            for idx in idxs:

                                dinputs.set_val(0.0)
                                dstate.set_val(0.0)

                                if directional:
                                    flat_view[:] = perturb
                                elif idx is not None:
                                    flat_view[idx] = perturb

                                # Matrix Vector Product
                                probmeta['checking'] = True
                                try:
                                    self.run_apply_linear(mode)
                                finally:
                                    probmeta['checking'] = False

                                for out in out_list:
                                    out_abs = prefix + out

                                    try:
                                        derivs = doutputs._abs_get_val(out_abs)
                                    except KeyError:
                                        # Implicit state
                                        derivs = dstate._abs_get_val(out_abs)

                                    if mode == 'fwd':
                                        key = out, inp
                                        deriv = partials_data[key]

                                        # Allocate first time
                                        if jac_key not in deriv:
                                            shape = (len(derivs), n_in)
                                            deriv[jac_key] = np.zeros(shape)

                                        if idx is not None:
                                            deriv[jac_key][:, idx] = derivs

                                    else:  # rev
                                        key = inp, out
                                        deriv = partials_data[key]

                                        if directional:
                                            # Dot product test for adjoint validity.
                                            m = mfree_directions[out]
                                            d = mfree_directions[inp]
                                            mhat = derivs
                                            dhat = deriv['J_fwd'][:, idx]
                                            deriv['directional_fwd_rev'] = \
                                                (dhat.dot(d), mhat.dot(m))
                                        else:
                                            meta = abs2meta_in[out_abs] if out_abs in abs2meta_in \
                                                else abs2meta_out[out_abs]
                                            if not meta['distributed']:  # serial input or state
                                                if inconsistent_across_procs(self.comm, derivs,
                                                                             return_array=False):
                                                    deriv['rank_inconsistent'] = True

                                        # Allocate first time
                                        if jac_key not in deriv:
                                            shape = (n_in, len(derivs))
                                            deriv[jac_key] = np.zeros(shape)

                                        if idx is not None:
                                            deriv[jac_key][idx, :] = derivs

                    # This component already has a Jacobian with calculated derivatives.
                    else:

                        subjacs = self._get_jacobian()._get_subjacs(self)

                        for rel_key in product(of_list, wrt_list):
                            abs_key = rel_key2abs_key(self, rel_key)
                            of, wrt = abs_key

                            if mode == 'fwd':
                                inp = rel_key[1]
                                directional = inp in local_opts and local_opts[inp]['directional']
                            else:
                                directional = len(mfree_directions) > 0

                            if wrt in self._var_abs2meta['input']:
                                wrt_meta = self._var_abs2meta['input'][wrt]
                            else:
                                wrt_meta = self._var_abs2meta['output'][wrt]

                            copy = True

                            # No need to calculate partials; they are already stored
                            try:
                                subjac = subjacs[abs_key]
                                if force_dense and not directional:
                                    deriv_value = subjac.todense()
                                else:
                                    deriv_value = subjac.get_val()
                            except KeyError:
                                rows = None
                                # Missing derivatives are assumed 0.
                                in_size = 1 if directional else wrt_meta['size']
                                out_size = self._var_abs2meta['output'][of]['size']
                                deriv_value = None
                                copy = False
                                nondep_derivs.add(rel_key)
                            else:
                                if subjac.info['diagonal']:
                                    rows = cols = np.arange(subjac.nrows)
                                else:
                                    rows = subjac.info['rows']
                                    cols = subjac.info['cols']
                                # Testing for pairs that are not dependent so that we suppress
                                # printing them unless the fd is nonzero.
                                # Note: subjacs_info is empty for undeclared partials, which is the
                                # default behavior now.
                                try:
                                    if not subjac.info['dependent']:
                                        nondep_derivs.add(rel_key)
                                except KeyError:
                                    nondep_derivs.add(rel_key)

                            if force_dense:
                                if directional:
                                    if rows is not None:
                                        in_size = wrt_meta['size']
                                        out_size = self._var_abs2meta['output'][of]['size']
                                        # if a scalar value is provided (in declare_partials),
                                        # expand to the correct size array value for zipping
                                        if deriv_value.size == 1:
                                            deriv_value *= np.ones(rows.size)
                                        deriv_value = coo_matrix((deriv_value, (rows, cols)),
                                                                 shape=(out_size, in_size))
                                        deriv_value = \
                                            np.atleast_2d(deriv_value.sum(axis=axis[mode]))
                                        if deriv_value.shape[0] < deriv_value.shape[1]:
                                            deriv_value = deriv_value.T
                                        copy = False

                                    elif issparse(deriv_value):
                                        if directional:
                                            deriv_value = \
                                                np.atleast_2d(deriv_value.sum(axis=axis[mode]))
                                            if deriv_value.shape[0] < deriv_value.shape[1]:
                                                deriv_value = deriv_value.T
                                        copy = False
                                    else:
                                        deriv_value = \
                                            np.atleast_2d(np.sum(deriv_value, axis=axis[mode])).T
                                else:
                                    if rows is not None or issparse(deriv_value):
                                        copy = False

                            if copy:
                                deriv_value = deriv_value.copy()

                            if deriv_value is not None:
                                partials_data[rel_key][jac_key] = deriv_value

                self._inputs.set_val(input_cache)
                self._outputs.set_val(output_cache)

            all_fd_options = {}

            of = self._get_partials_ofs()
            wrt = self._get_partials_wrts()

            if step is None or isinstance(step, (float, int)):
                steps = [step]
            else:
                steps = step

            actual_steps = defaultdict(list)
            alloc_complex = self._outputs._alloc_complex
            rel2abs = self._resolver.rel2abs

            for step in steps:
                self.run_apply_nonlinear()
                approximations = {'fd': FiniteDifference(), 'cs': ComplexStep()}

                added_wrts = set()

                # Load up approximation objects with the requested settings.

                for rel_key in product(of, wrt):
                    _, rel_wrt = rel_key

                    fd_options = _get_fd_options(rel_wrt, requested_method,
                                                 local_opts, step, form, step_calc,
                                                 alloc_complex, minimum_step)

                    actual_steps[rel_key].append(fd_options['step'])

                    # Determine if fd or cs.
                    method = requested_method

                    all_fd_options[rel_wrt] = fd_options
                    if rel_wrt in mfree_directions:
                        vector = mfree_directions.get(rel_wrt)
                    else:
                        vector = None

                    # prevent adding multiple approxs with same wrt (and confusing users with
                    # warnings)
                    abs_wrt = rel2abs(rel_wrt)
                    if abs_wrt not in added_wrts:
                        approximations[fd_options['method']].add_approximation(abs_wrt, self,
                                                                               fd_options,
                                                                               vector=vector)
                        added_wrts.add(abs_wrt)

                approx_jac = _CheckingJacobian(self)
                for approximation in approximations.values():
                    # Perform the FD here.
                    approximation.compute_approximations(self, jac=approx_jac)

                for abs_key, fd_partial in approx_jac.items():
                    rel_key = abs_key2rel_key(self, abs_key)
                    deriv = partials_data[rel_key]
                    subjacs_info = approx_jac._subjacs[abs_key].info
                    _of, _wrt = rel_key

                    if 'J_fd' not in deriv:
                        deriv['J_fd'] = []
                        deriv['steps'] = []
                    deriv['J_fd'].append(fd_partial)
                    deriv['steps'] = actual_steps[rel_key]
                    deriv['rows'] = subjacs_info['rows']
                    deriv['cols'] = subjacs_info['cols']

                    if 'uncovered_nz' in subjacs_info:
                        deriv['uncovered_nz'] = subjacs_info['uncovered_nz']
                        deriv['uncovered_threshold'] = subjacs_info['uncovered_threshold']

                    if _wrt in local_opts and local_opts[_wrt]['directional']:
                        if self.matrix_free:
                            # Dot product test for adjoint validity.
                            m = mfree_directions[_of].flatten()
                            d = mfree_directions[_wrt].flatten()
                            mhat = fd_partial.flatten()
                            dhat = deriv['J_rev'].flatten()

                            if 'directional_fd_rev' not in deriv:
                                deriv['directional_fd_rev'] = []
                            deriv['directional_fd_rev'].append((dhat.dot(d), mhat.dot(m)))

        # convert to regular dict from defaultdict
        partials_data = {key: dict(d) for key, d in partials_data.items()}

        incon_keys = self._get_inconsistent_keys()

        # if compact_print is True, we'll show all derivatives, even non-dependent ones
        nondeps = set() if compact_print else nondep_derivs
        # force iterator to run so that error info will be added to partials_data
        err_iter = list(_iter_derivs(partials_data, show_only_incorrect, all_fd_options, False,
                                     nondeps, self.matrix_free, abs_err_tol, rel_err_tol,
                                     incon_keys))
        worst = None
        if out_stream is not None:
            if compact_print:
                worst = _deriv_display_compact(self, err_iter, partials_data, out_stream,
                                               totals=False,
                                               show_only_incorrect=show_only_incorrect,
                                               show_worst=show_worst, rich_print=rich_print)
            else:
                _deriv_display(self, err_iter, partials_data, rel_err_tol, abs_err_tol, out_stream,
                               all_fd_options, False, show_only_incorrect, rich_print=rich_print)

        # check for zero subjacs that are declared as dependent
        zero_keys = set()
        for key, meta in partials_data.items():
            if key in nondep_derivs:
                continue
            abs_key = rel_key2abs_key(self, key)
            if abs_key in self._subjacs_info:
                maxmag = max([mag.max() for mag in meta['magnitude']])
                if maxmag == 0.0:
                    zero_keys.add(key)

        if zero_keys:
            issue_warning(f"\nComponent '{self.pathname}' has zero derivatives for the "
                          "following variable pairs that were declared as 'dependent': "
                          f"{sorted(zero_keys)}.\n",
                          category=DerivativesWarning)

        # add pathname to the partials dict to make it compatible with the return value
        # from Problem.check_partials and passable to assert_check_partials.
        return {self.pathname: partials_data}, worst

    def _nocs_warning(self):
        issue_warning(f"Component '{self.pathname}' requested complex step, but "
                      "force_alloc_complex has not been set to True, so finite difference was "
                      "used.\nTo enable complex step, specify 'force_alloc_complex=True' when "
                      "calling setup on the problem, e.g. "
                      "'problem.setup(force_alloc_complex=True)'.", category=DerivativesWarning)

    def _check_compute_primal_args(self):
        """
        Check that the compute_primal method args are in the correct order.
        """
        args = list(inspect.signature(self._orig_compute_primal).parameters)
        if args and args[0] == 'self':
            args = args[1:]
        compargs = self._get_compute_primal_argnames()
        if args != compargs:
            raise RuntimeError(f"{self.msginfo}: compute_primal method args {args} don't match "
                               f"the args {compargs} mapped from this component's inputs. To "
                               "map inputs to the compute_primal method, set the name used in "
                               "compute_primal to the 'primal_name' arg when calling "
                               "add_input/add_discrete_input. This is only necessary if the "
                               "declared component input name is not a valid Python name.")

    def _check_compute_primal_returns(self):
        """
        Check that the compute_primal method returns a tuple.
        """
        retnames = get_return_names(self.compute_primal)
        if self._valid_name_map:
            expected_names = [self._valid_name_map.get(n, n) for n in self._var_rel_names['output']]
            expected_names.extend([self._valid_name_map.get(n, n) for n in self._discrete_outputs])
        else:
            expected_names = list(self._var_rel_names['output'])
            expected_names.extend(self._discrete_outputs)

        if len(retnames) != len(expected_names):
            raise RuntimeError(f"{self.msginfo}: compute_primal method returns {len(retnames)} "
                               f"values but expected {len(expected_names)}.")

        for i, (expname, rname) in enumerate(zip(expected_names, retnames)):
            if rname is not None and expname != rname:
                raise RuntimeError(f"{self.msginfo}: compute_primal method returns {rname} "
                                   f"for return value {i} but the name of the output that was "
                                   f"mapped for this component is {expname}. To map outputs to "
                                   "the compute_primal method, set the name used in compute_primal "
                                   "to the 'primal_name' arg when calling "
                                   "add_output/add_discrete_output. This is only necessary if "
                                   "the declared component output name is not a valid Python name.")

    def get_declare_partials_calls(self, sparsity=None):
        """
        Return a string containing declare_partials() calls based on the subjac sparsity.

        Parameters
        ----------
        sparsity : coo_matrix or None
            Sparsity matrix to use. If None, compute_sparsity will be called to compute it.

        Returns
        -------
        str
            A string containing a declare_partials() call for each nonzero subjac. This
            string may be cut and pasted into a component's setup() method.
        """
        lines = []
        for of, wrt, nzrows, nzcols, _ in self.subjac_sparsity_iter(sparsity=sparsity):
            lines.append(f"    self.declare_partials(of='{of}', wrt='{wrt}', "
                         f"rows={list(nzrows)}, cols={list(nzcols)})")
        return '\n'.join(lines)

    def _get_matvec_scope(self):
        """
        Find the input and output variables that are needed for a particular matvec product.

        Returns
        -------
        (set, set)
            Sets of output and input variables.
        """
        return _no_matvec_scope


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

    def __iter__(self):
        return iter(self._dict)

    def __bool__(self):
        return bool(self._dict)

    def keys(self):
        return self._dict.keys()

    def items(self):
        return [(key, meta['val']) for key, meta in self._dict.items()]

    def values(self):
        return [meta['val'] for meta in self._dict.values()]

    def set_vals(self, vals):
        for key, val in zip(self._dict, vals):
            self[key] = val


def _get_fd_options(var, global_method, local_opts, global_step, global_form, global_step_calc,
                    alloc_complex, global_minimum_step):
    local_wrt = var

    # Determine if fd or cs.
    method = global_method
    if local_wrt in local_opts:
        local_method = local_opts[local_wrt]['method']
        if local_method:
            method = local_method

    # We can't use CS if we haven't allocated a complex vector, so we fall back on fd.
    if method == 'cs' and not alloc_complex:
        method = 'fd'

    fd_options = {'order': None,
                  'method': method}

    if method == 'cs':
        fd_options = ComplexStep.DEFAULT_OPTIONS.copy()
        fd_options['method'] = 'cs'

        fd_options['form'] = None
        fd_options['step_calc'] = None
        fd_options['minimum_step'] = None

    elif method == 'fd':
        fd_options = FiniteDifference.DEFAULT_OPTIONS.copy()
        fd_options['method'] = 'fd'

        fd_options['form'] = global_form
        fd_options['step_calc'] = global_step_calc
        fd_options['minimum_step'] = global_minimum_step

    if global_step and global_method == method:
        fd_options['step'] = global_step

    fd_options['directional'] = False

    # Precedence: component options > global options > defaults
    if local_wrt in local_opts:
        for name in ['form', 'step', 'step_calc', 'minimum_step', 'directional']:
            value = local_opts[local_wrt][name]
            if value is not None:
                fd_options[name] = value

    return fd_options
