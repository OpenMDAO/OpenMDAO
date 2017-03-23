"""Define the Problem class and a FakeComm class for non-MPI users."""

from __future__ import division
from collections import OrderedDict, defaultdict, namedtuple
import sys

from six import string_types, iteritems, iterkeys
from six.moves import range
from itertools import product, chain

import numpy as np
import scipy.sparse as sparse

from openmdao.approximation_schemes.finite_difference import FiniteDifference, DEFAULT_FD_OPTIONS
from openmdao.assemblers.default_assembler import DefaultAssembler
from openmdao.components.deprecated_component import Component as DepComponent
from openmdao.core.component import Component
from openmdao.core.driver import Driver
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.error_checking.check_config import check_config
from openmdao.vectors.default_vector import DefaultVector

from openmdao.utils.class_util import overrides_method
from openmdao.utils.general_utils import warn_deprecation, ensure_compatible
from openmdao.utils.name_maps import rel_key2abs_key, abs_key2rel_key, rel_name2abs_name

ErrorTuple = namedtuple('ErrorTuple', ['forward', 'reverse', 'forward_reverse'])
MagnitudeTuple = namedtuple('MagnitudeTuple', ['forward', 'reverse', 'fd'])


class FakeComm(object):
    """
    Fake MPI communicator class used if mpi4py is not installed.

    Attributes
    ----------
    rank : int
        index of current proc; value is 0 because there is only 1 proc.
    size : int
        number of procs in the comm; value is 1 since MPI is not available.
    """

    def __init__(self):
        """
        Initialize attributes.
        """
        self.rank = 0
        self.size = 1


class Problem(object):
    """
    Top-level container for the systems and drivers.

    Attributes
    ----------
    model : <System>
        pointer to the top-level <System> object (root node in the tree).
    comm : MPI.Comm or <FakeComm>
        the global communicator; the same as that of assembler and model.
    driver : <Driver>
        Slot for the driver. The default driver is `Driver`, which just runs
        the model once.
    _assembler : <Assembler>
        pointer to the global <Assembler> object.
    _use_ref_vector : bool
        if True, allocate vectors to store ref. values.
    """

    def __init__(self, model=None, comm=None, assembler_class=None,
                 use_ref_vector=True, root=None):
        """
        Initialize attributes.

        Parameters
        ----------
        model : <System> or None
            pointer to the top-level <System> object (root node in the tree).
        comm : MPI.Comm or <FakeComm> or None
            the global communicator; the same as that of assembler and model.
        assembler_class : <Assembler> or None
            pointer to the global <Assembler> object.
        use_ref_vector : bool
            if True, allocate vectors to store ref. values.
        root : <System> or None
            Deprecated kwarg for `model`.
        """
        if comm is None:
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            except ImportError:
                comm = FakeComm()
        if assembler_class is None:
            assembler_class = DefaultAssembler

        if root is not None:
            if model is not None:
                raise ValueError("cannot specify both `root` and `model`. `root` has been "
                                 "deprecated, please use model")

            warn_deprecation("The 'root' argument provides backwards compatibility "
                             "with OpenMDAO <= 1.x ; use 'model' instead.")

            model = root

        if model is None:
            model = Group()

        self.model = model
        self.comm = comm
        self.driver = Driver()

        self._assembler = assembler_class(comm)
        self._use_ref_vector = use_ref_vector

    def __getitem__(self, name):
        """
        Get an output/input variable.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.

        Returns
        -------
        float or ndarray
            the requested output/input variable.
        """
        if name in self.model._outputs:
            return self.model._outputs[name]
        elif name in self.model._inputs:
            return self.model._inputs[name]
        else:
            msg = 'Variable name "{}" not found.'
            raise KeyError(msg.format(name))

    def __setitem__(self, name, value):
        """
        Set an output/input variable.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.
        value : float or ndarray or list
            value to set this variable to.
        """
        if name in self.model._outputs:
            self.model._outputs[name] = value
        elif name in self.model._inputs:
            self.model._inputs[name] = value
        else:
            msg = 'Variable name "{}" not found.'
            raise KeyError(msg.format(name))

    @property
    def root(self):
        """
        Provide 'root' property for backwards compatibility.

        Returns
        -------
        <Group>
            reference to the 'model' property.
        """
        warn_deprecation("The 'root' property provides backwards compatibility "
                         "with OpenMDAO <= 1.x ; use 'model' instead.")
        return self.model

    @root.setter
    def root(self, model):
        """
        Provide for setting the 'root' property for backwards compatibility.

        Parameters
        -------
        model : <Group>
            reference to a <Group> to be assigned to the 'model' property.
        """
        warn_deprecation("The 'root' property provides backwards compatibility "
                         "with OpenMDAO <= 1.x ; use 'model' instead.")
        self.model = model

    def run_model(self):
        """
        Run the model by calling the root system's solve_nonlinear.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        return self.model.run_solve_nonlinear()

    def run_driver(self):
        """
        Run the driver on the model.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        with self.model._scaled_context():
            return self.driver.run()

    def run_once(self):
        """
        Backward compatible call for run_model.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        warn_deprecation('This method provides backwards compatibility with '
                         'OpenMDAO <= 1.x ; use run_driver instead.')

        return self.run_model()

    def run(self):
        """
        Backward compatible call for run_driver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        warn_deprecation('This method provides backwards compatibility with '
                         'OpenMDAO <= 1.x ; use run_driver instead.')

        return self.run_driver()

    def setup(self, vector_class=DefaultVector, check=True, logger=None,
              mode='auto'):
        """
        Set up everything (model, assembler, vector, solvers, drivers).

        Parameters
        ----------
        vector_class : type
            reference to an actual <Vector> class; not an instance.
        check : boolean
            whether to run error check after setup is complete.
        logger : object
            Object for logging config checks if check is True.
        mode : string
            Derivatives calculation mode, 'fwd' for forward, and 'rev' for
            reverse (adjoint). Default is 'auto', which lets OpenMDAO choose
            the best mode for your problem.

        Returns
        -------
        self : <Problem>
            this enables the user to instantiate and setup in one line.
        """
        model = self.model
        comm = self.comm
        assembler = self._assembler

        if mode not in ['fwd', 'rev', 'auto']:
            msg = "Unsupported mode: '%s'" % mode
            raise ValueError(msg)

        # TODO: Support automatic determination of mode
        if mode == 'auto':
            mode = 'rev'
        self._mode = mode

        # Recursive system setup
        model._setup_processors('', comm, {}, assembler, [0, comm.size])
        allprocs_abs_names = model._setup_variables()
        model._setup_variable_indices({'input': 0, 'output': 0})
        model._setup_partials()
        model._setup_connections()

        # Assembler setup: variable metadata and indices
        assembler._setup_variables(allprocs_abs_names,
                                   model._var_abs2data_io,
                                   model._var_abs_names)

        # Assembler setup: variable connections
        assembler._setup_connections(model._manual_connections_abs,
                                     model._var_allprocs_prom2abs_list,
                                     model._var_abs2data_io)

        # Assembler setup: global transfer indices vector
        assembler._setup_src_indices(model._var_abs2data_io,
                                     model._var_abs_names)

        # Assembler setup: compute data required for units/scaling
        assembler._setup_src_data(model._var_abs_names['output'],
                                  model._var_abs2data_io)

        # Set up scaling vectors
        model._setup_scaling()

        # Set up lower and upper bounds vectors
        lower_bounds = vector_class('lower', 'output', self.model)
        upper_bounds = vector_class('upper', 'output', self.model)
        model._setup_bounds_vectors(lower_bounds, upper_bounds, True)

        # Vector setup for the basic execution vector
        self.setup_vector('nonlinear', vector_class, self._use_ref_vector)

        # Vector setup for the linear vector
        self.setup_vector('linear', vector_class, self._use_ref_vector)

        model._setup_jacobians()

        for system in model.system_iter(include_self=True, recurse=True):
            # set up all the solvers.
            if system._nl_solver is not None:
                system._nl_solver._setup_solvers(system, 0)
            if system._ln_solver is not None:
                system._ln_solver._setup_solvers(system, 0)

        # Finally, prepare the driver for execution.
        self.driver._setup_driver(self)

        if check:
            check_config(self, logger)

        return self

    def setup_vector(self, vec_name, vector_class, use_ref_vector):
        """
        Set up the 'vec_name' <Vector>.

        Parameters
        ----------
        vec_name : str
            name of the vector.
        vector_class : type
            reference to the actual <Vector> class.
        use_ref_vector : bool
            if True, allocate vectors to store ref. values.
        """
        model = self.model
        assembler = self._assembler

        vectors = {}
        for key in ['input', 'output', 'residual']:
            if key is 'residual':
                typ = 'output'
            else:
                typ = key

            vectors[key] = vector_class(vec_name, typ, self.model)

        # TODO: implement this properly
        ind1, ind2 = self.model._var_allprocs_idx_range['output']
        vector_var_ids = np.arange(ind1, ind2)

        self.model._setup_vector(vectors, vector_var_ids, use_ref_vector)

    def check_partial_derivatives(self, out_stream=sys.stdout, comps=None,
                                  compact_print=False, abs_err_tol=1e-6,
                                  rel_err_tol=1e-6, global_options=None,
                                  force_dense=True):
        """
        Check partial derivatives comprehensively for all components in your model.

        Parameters
        ----------
        out_stream : file_like
            Where to send human readable output. Default is sys.stdout. Set to None to suppress.
        comps : None or list_like
            List of component names to check the partials of (all others will be skipped). Set to
             None (default) to run all components.
        compact_print : bool
            Set to True to just print the essentials, one line per unknown-param pair.
        abs_err_tol : float
            Threshold value for absolute error.  Errors about this value will have a '*' displayed
            next to them in output, making them easy to search for. Default is 1.0E-6.
        rel_err_tol : float
            Threshold value for relative error.  Errors about this value will have a '*' displayed
            next to them in output, making them easy to search for. Note at times there may be a
            significant relative error due to a minor absolute error.  Default is 1.0E-6.
        global_options : dict
            Dictionary of options that override options specified in ALL components. Only
            'form', 'step', 'step_calc', and 'method' can be specified in this way.
        force_dense : bool
            If True, analytic derivatives will be coerced into arrays.

        Returns
        -------
        dict of dicts of dicts
            First key:
                is the component name;
            Second key:
                is the (output, input) tuple of strings;
            Third key:
                is one of ['rel error', 'abs error', 'magnitude', 'J_fd', 'J_fwd', 'J_rev'];

            For 'rel error', 'abs error', 'magnitude' the value is: A tuple containing norms for
                forward - fd, adjoint - fd, forward - adjoint.
            For 'J_fd', 'J_fwd', 'J_rev' the value is: A numpy array representing the computed
                Jacobian for the three different methods of computation.
        """
        if not global_options:
            global_options = DEFAULT_FD_OPTIONS.copy()
            global_options['method'] = 'fd'

        if global_options['method'] == 'fd':
            scheme = FiniteDifference
        else:
            raise ValueError('Unrecognized method: "{}"'.format(global_options['method']))

        model = self.model

        # TODO: Once we're tracking iteration counts, run the model if it has not been run before.

        all_comps = model.system_iter(typ=Component)
        if comps is None:
            comps = [comp for comp in all_comps]
        else:
            all_comp_names = {c.pathname for c in all_comps}
            requested = set(comps)
            extra = requested.difference(all_comp_names)
            if extra:
                msg = 'The following are not valid comp names: {}'.format(sorted(list(extra)))
                raise ValueError(msg)
            comps = [model.get_subsystem(c_name) for c_name in comps]

        current_mode = self._mode
        current_suppresion = model.suppress_solver_output

        # This is a defaultdict of (defaultdict of dicts).
        partials_data = defaultdict(lambda: defaultdict(dict))

        # Caching current point to restore after setups.
        input_cache = model._inputs._clone()
        output_cache = model._outputs._clone()

        # Analytic Jacobians
        for mode in ('fwd', 'rev'):
            self.setup(mode=mode, check=False)
            model.suppress_solver_output = True
            model._inputs.set_vec(input_cache)
            model._outputs.set_vec(output_cache)
            # Make sure we're in a valid state
            model.run_apply_nonlinear()
            model.run_linearize()

            jac_key = 'J_' + mode

            for comp in comps:

                # Skip IndepVarComps
                if isinstance(comp, IndepVarComp):
                    continue

                explicit = isinstance(comp, ExplicitComponent)
                deprecated = isinstance(comp, DepComponent)
                matrix_free = comp._matrix_free
                c_name = comp.pathname

                # TODO: Check deprecated deriv_options.

                with comp._units_scaling_context(scale_jac=True):
                    subjacs = comp._jacobian._subjacs
                    if explicit:
                        comp._negate_jac()

                    of_list = list(comp._var_allprocs_prom2abs_list['output'].keys())
                    wrt_list = list(comp._var_allprocs_prom2abs_list['input'].keys())

                    # The only outputs in wrt should be implicit states.
                    if deprecated:
                        wrt_list.extend(comp._state_names)
                    elif not explicit:
                        wrt_list.extend(of_list)

                    # Matrix-free components need to calculate their Jacobian by matrix-vector
                    # product.
                    if matrix_free:

                        dstate = comp._vectors['output']['linear']
                        if mode == 'fwd':
                            dinputs = comp._vectors['input']['linear']
                            doutputs = comp._vectors['residual']['linear']
                            in_list = wrt_list
                            out_list = of_list
                        else:
                            dinputs = comp._vectors['residual']['linear']
                            doutputs = comp._vectors['input']['linear']
                            in_list = of_list
                            out_list = wrt_list

                        for inp in in_list:
                            inp_abs = rel_name2abs_name(comp, inp)

                            try:
                                flat_view = dinputs._views_flat[inp_abs]
                            except KeyError:
                                # Implicit state
                                flat_view = dstate._views_flat[inp_abs]

                            n_in = len(flat_view)
                            for idx in range(n_in):

                                dinputs.set_const(0.0)
                                dstate.set_const(0.0)

                                # TODO - Sort out the minus sign difference.
                                perturb = 1.0 if (deprecated or not explicit) else -1.0

                                # Dictionary access returns a scaler for 1d input, and we
                                # need a vector for clean code, so use _views_flat.
                                flat_view[idx] = perturb

                                # Matrix Vector Product
                                comp._apply_linear(['linear'], mode)

                                for out in out_list:
                                    out_abs = rel_name2abs_name(comp, out)

                                    try:
                                        derivs = doutputs._views_flat[out_abs]
                                    except KeyError:
                                        # Implicit state
                                        derivs = dstate._views_flat[out_abs]

                                    if mode == 'fwd':
                                        key = out, inp
                                        deriv = partials_data[c_name][key]

                                        # Allocate first time
                                        if jac_key not in deriv:
                                            shape = (len(derivs), n_in)
                                            deriv[jac_key] = np.zeros(shape)

                                        deriv[jac_key][:, idx] = derivs

                                    else:
                                        key = inp, out
                                        deriv = partials_data[c_name][key]

                                        # Allocate first time
                                        if jac_key not in deriv:
                                            shape = (n_in, len(derivs))
                                            deriv[jac_key] = np.zeros(shape)

                                        deriv[jac_key][idx, :] = derivs

                    # These components already have a Jacobian with calculated derivatives.
                    else:

                        for rel_key in product(of_list, wrt_list):
                            abs_key = rel_key2abs_key(comp, rel_key)
                            of, wrt = abs_key

                            # No need to calculate partials; they are already stored
                            deriv_value = subjacs.get(abs_key)

                            if deriv_value is None:
                                # Missing derivatives are assumed 0.
                                in_size = np.prod(comp._var_abs2data_io[wrt]['metadata']['shape'])
                                out_size = np.prod(comp._var_abs2data_io[of]['metadata']['shape'])
                                deriv_value = np.zeros((out_size, in_size))

                            if force_dense:
                                if isinstance(deriv_value, list):
                                    in_size = np.prod(
                                        comp._var_abs2data_io[wrt]['metadata']['shape'])
                                    out_size = np.prod(
                                        comp._var_abs2data_io[of]['metadata']['shape'])
                                    tmp_value = np.zeros((out_size, in_size))
                                    jac_val, jac_i, jac_j = deriv_value
                                    # if a scalar value is provided (in declare_partials),
                                    # expand to the correct size array value for zipping
                                    if jac_val.size == 1:
                                        jac_val = jac_val * np.ones(jac_i.size)
                                    for i, j, val in zip(jac_i, jac_j, jac_val):
                                        tmp_value[i, j] += val
                                    deriv_value = tmp_value

                                elif sparse.issparse(deriv_value):
                                    deriv_value = deriv_value.todense()

                            partials_data[c_name][rel_key][jac_key] = deriv_value

                    if explicit:
                        comp._negate_jac()

        self.setup(mode=current_mode)
        model._inputs.set_vec(input_cache)
        model._outputs.set_vec(output_cache)
        model.run_apply_nonlinear()

        # Finite Difference (or TODO: Complex Step) to calculate Jacobian
        jac_key = 'J_fd'
        for comp in comps:

            c_name = comp.pathname

            # Skip IndepVarComps
            if isinstance(comp, IndepVarComp):
                continue

            subjac_info = comp._subjacs_info
            explicit = isinstance(comp, ExplicitComponent)
            deprecated = isinstance(comp, DepComponent)
            approximation = scheme()

            of = list(comp._var_allprocs_prom2abs_list['output'].keys())
            wrt = list(comp._var_allprocs_prom2abs_list['input'].keys())

            # The only outputs in wrt should be implicit states.
            if deprecated:
                wrt.extend(comp._state_names)
            elif not explicit:
                wrt.extend(of)

            for rel_key in product(of, wrt):
                abs_key = rel_key2abs_key(comp, rel_key)
                approximation.add_approximation(abs_key, global_options)

            approx_jac = {}
            approximation._init_approximations()

            # Peform the FD here.
            approximation.compute_approximations(comp, jac=approx_jac)

            for rel_key, partial in iteritems(approx_jac):
                abs_key = rel_key2abs_key(comp, rel_key)
                # Since all partials for outputs for explicit comps are declared, assume anything
                # missing is an input deriv.
                if (explicit and (abs_key not in subjac_info or
                                  subjac_info[abs_key]['type'] == 'input')):
                    partials_data[c_name][rel_key][jac_key] = -partial
                else:
                    partials_data[c_name][rel_key][jac_key] = partial

        # Conversion of defaultdict to dicts
        partials_data = {comp_name: dict(outer) for comp_name, outer in iteritems(partials_data)}
        model.suppress_solver_output = current_suppresion

        _assemble_derivative_data(partials_data, rel_err_tol, abs_err_tol, out_stream,
                                  compact_print, comps, global_options)

        return partials_data

    def compute_total_derivs(self, of=None, wrt=None, return_format='flat_dict'):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        Parameters
        ----------
        of : list of variable name strings or None
            Variables whose derivatives will be computed. Default is None, which
            uses the driver's objectives and constraints.
        wrt : list of variable name strings or None
            Variables with respect to which the derivatives will be computed.
            Default is None, which uses the driver's desvars.
        return_format : string
            Format to return the derivatives. Default is a 'flat_dict', which
            returns them in a dictionary whose keys are tuples of form (of, wrt).

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        with self.model._scaled_context():
            totals = self._compute_total_derivs(of=of, wrt=wrt, return_format=return_format,
                                                global_names=False)

        return totals

    def _compute_total_derivs(self, of=None, wrt=None, return_format='flat_dict',
                              global_names=True):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        Parameters
        ----------
        of : list of variable name strings or None
            Variables whose derivatives will be computed. Default is None, which
            uses the driver's objectives and constraints.
        wrt : list of variable name strings or None
            Variables with respect to which the derivatives will be computed.
            Default is None, which uses the driver's desvars.
        return_format : string
            Format to return the derivatives. Default is a 'flat_dict', which
            returns them in a dictionary whose keys are tuples of form (of, wrt).
        global_names : bool
            Set to True when passing in global names to skip some translation steps.

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        model = self.model
        mode = self._mode
        vec_dinput = model._vectors['input']
        vec_doutput = model._vectors['output']
        vec_dresid = model._vectors['residual']

        # TODO - Pull 'of' and 'wrt' from driver if unspecified.
        if wrt is None:
            raise NotImplementedError("Need to specify 'wrt' for now.")
        if of is None:
            raise NotImplementedError("Need to specify 'of' for now.")

        # A number of features will need to be supported here as development
        # goes forward.
        # -------------------------------------------------------------------
        # TODO: Make sure we can function in parallel when some params or
        # functions are not local.
        # TODO: Support parallel adjoint and parallel forward derivatives
        #       Aside: how are they specified, and do we have to pick up any
        #       that are missed?
        # TODO: Handle driver scaling.
        # TODO: Might be some additional adjustments needed to set the 'one'
        #       into the PETSC vector.
        # TODO: support parmeter/constraint indices
        # TODO: Support for any other return_format we need.
        # TODO: Support constraint sparsity (i.e., skip in/out that are not
        #       relevant for this constraint) (desvars too?)
        # TODO: Don't calculate for inactive constraints
        # TODO: Support full-model FD. Don't know how this'll work, but we
        #       used to need a separate function for that.
        # TODO: poi_indices and qoi_indices requires special support
        # -------------------------------------------------------------------

        # Prepare model for calculation by cleaning out the derivatives
        # vectors.
        for subname in vec_dinput:

            # TODO: Do all three deriv vectors have the same keys?

            # Skip nonlinear because we don't need to mess with it?
            if subname == 'nonlinear':
                continue

            vec_dinput[subname].set_const(0.0)
            vec_doutput[subname].set_const(0.0)
            vec_dresid[subname].set_const(0.0)

        # Linearize Model
        model._linearize()

        of = [(n,) if isinstance(n, string_types) else n for n in of]
        wrt = [(n,) if isinstance(n, string_types) else n for n in wrt]

        # Create data structures (and possibly allocate space) for the total
        # derivatives that we will return.
        if return_format == 'flat_dict':

            totals = OrderedDict()

            for okeys in of:
                for okey in okeys:
                    for ikeys in wrt:
                        for ikey in ikeys:
                            totals[(okey, ikey)] = None

        elif return_format == 'dict':

            totals = OrderedDict()

            for okeys in of:
                for okey in okeys:
                    totals[okey] = OrderedDict()
                    for ikeys in wrt:
                        for ikey in ikeys:
                            totals[okey][ikey] = None

        else:
            msg = "Unsupported return format '%s." % return_format
            raise NotImplementedError(msg)

        # Convert of and wrt names from promoted to unpromoted
        # (which is absolute path since we're at the top)
        if global_names:
            oldwrt, oldof = wrt, of
        else:
            oldof = of
            of = []
            for names in oldof:
                of.append(tuple(model._var_allprocs_prom2abs_list['output'][name][0]
                                for name in names))

            oldwrt = wrt
            wrt = []
            for names in oldwrt:
                wrt.append(tuple(model._var_allprocs_prom2abs_list['output'][name][0]
                                 for name in names))

        if mode == 'fwd':
            input_list, output_list = wrt, of
            old_input_list, old_output_list = oldwrt, oldof
            input_vec, output_vec = vec_dresid, vec_doutput
        else:
            input_list, output_list = of, wrt
            old_input_list, old_output_list = oldof, oldwrt
            input_vec, output_vec = vec_doutput, vec_dresid

        # TODO : Parallel adjoint setup loop goes here.
        # NOTE : Until we support it, we will just limit ourselves to the
        # 'linear' vector.
        vecname = 'linear'
        dinputs = input_vec[vecname]
        doutputs = output_vec[vecname]

        # If Forward mode, solve linear system for each 'wrt'
        # If Adjoint mode, solve linear system for each 'of'
        for icount, input_names in enumerate(input_list):
            for iname_count, input_name in enumerate(input_names):
                flat_view = dinputs._views_flat[input_name]
                n_in = len(flat_view)
                for idx in range(n_in):
                    # Maybe we don't need to clean up so much at the beginning,
                    # since we clean this every time.
                    dinputs.set_const(0.0)

                    # Dictionary access returns a scaler for 1d input, and we
                    # need a vector for clean code, so use _views_flat.
                    flat_view[idx] = 1.0

                    # The root system solves here.
                    model._solve_linear([vecname], mode)

                    # Pull out the answers and pack into our data structure.
                    for ocount, output_names in enumerate(output_list):
                        for oname_count, output_name in enumerate(output_names):
                            deriv_val = doutputs._views_flat[output_name]
                            len_val = len(deriv_val)

                            if return_format == 'flat_dict':
                                if mode == 'fwd':

                                    key = (old_output_list[ocount][oname_count],
                                           old_input_list[icount][iname_count])

                                    if totals[key] is None:
                                        totals[key] = np.empty((len_val, n_in))
                                    totals[key][:, idx] = deriv_val

                                else:

                                    key = (old_input_list[icount][iname_count],
                                           old_output_list[ocount][oname_count])

                                    if totals[key] is None:
                                        totals[key] = np.empty((n_in, len_val))
                                    totals[key][idx, :] = deriv_val

                            elif return_format == 'dict':
                                if mode == 'fwd':

                                    okey = old_output_list[ocount][oname_count]
                                    ikey = old_input_list[icount][iname_count]

                                    if totals[okey][ikey] is None:
                                        totals[okey][ikey] = np.empty((len_val, n_in))
                                    totals[okey][ikey][:, idx] = deriv_val

                                else:

                                    okey = old_input_list[icount][iname_count]
                                    ikey = old_output_list[ocount][oname_count]

                                    if totals[okey][ikey] is None:
                                        totals[okey][ikey] = np.empty((n_in, len_val))
                                    totals[okey][ikey][idx, :] = deriv_val

        return totals


def _assemble_derivative_data(derivative_data, rel_error_tol, abs_error_tol, out_stream,
                              compact_print, system_list, global_options):
    """
    Compute the relative and absolute errors in the given derivatives and print to `out_stream`.

    Parameters
    ----------
    derivative_data : dict
        Dictionary containing derivative information keyed by system name.
    rel_error_tol : float
        Relative error tolerance.
    abs_error_tol : float
        Absolute error tolerance.
    out_stream : File-like
        File-like stream (or None) to which results are written.
    compact_print : bool
        If results should be printed verbosely or in a table.
    system_list : Iterable
        The systems (in the proper order) that were checked.0
    global_options : dict
        Dictionary containing the options for the approximation.
    """
    fd_desc = "{}:{}".format(global_options['method'],
                             global_options['form'])
    if compact_print:
        check_desc = "    (Check Type: {})".format(fd_desc)
        deriv_line = "{0} wrt {1} | {2:.4e} | {3:.4e} | {4:.4e} | {5:.4e} | {6:.4e} | {7:.4e}"\
                     " | {8:.4e} | {9:.4e} | {10:.4e}\n"
    else:
        check_desc = ""

    for system in system_list:
        # No need to see derivatives of IndepVarComps
        if isinstance(system, IndepVarComp):
            continue

        sys_name = system.pathname
        explicit = False

        # Match header to appropriate type.
        if isinstance(system, Component):
            sys_type = 'Component'
            explicit = isinstance(system, ExplicitComponent)
        elif isinstance(system, Group):
            sys_type = 'Group'
        else:
            sys_type = type(system).__name__

        derivatives = derivative_data[sys_name]

        if out_stream:
            out_stream.write('-' * (len(sys_name) + 15) + '\n')
            out_stream.write("{}: '{}'{}\n".format(sys_type, sys_name, check_desc))
            out_stream.write('-' * (len(sys_name) + 15) + '\n')

            if compact_print:
                # Error Header
                header = "{0} wrt {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10}\n"\
                    .format(
                        _pad_name('<output>', 13, True),
                        _pad_name('<variable>', 13, True),
                        _pad_name('fwd mag.'),
                        _pad_name('rev mag.'),
                        _pad_name('check mag.'),
                        _pad_name('a(fwd-chk)'),
                        _pad_name('a(rev-chk)'),
                        _pad_name('a(fwd-rev)'),
                        _pad_name('r(fwd-chk)'),
                        _pad_name('r(rev-chk)'),
                        _pad_name('r(fwd-rev)')
                    )
                out_stream.write(header)
                out_stream.write('-' * len(header) + '\n')

        # Sorted keys ensures deterministic ordering
        sorted_keys = sorted(iterkeys(derivatives))

        for of, wrt in sorted_keys:
            derivative_info = derivatives[of, wrt]
            forward = derivative_info['J_fwd']
            reverse = derivative_info['J_rev']
            fd = derivative_info['J_fd']

            fwd_error = np.linalg.norm(forward - fd)
            rev_error = np.linalg.norm(reverse - fd)
            fwd_rev_error = np.linalg.norm(forward - reverse)

            fwd_norm = np.linalg.norm(forward)
            rev_norm = np.linalg.norm(reverse)
            fd_norm = np.linalg.norm(fd)

            derivative_info['abs error'] = abs_err = ErrorTuple(fwd_error, rev_error, fwd_rev_error)
            derivative_info['magnitude'] = magnitude = MagnitudeTuple(fwd_norm, rev_norm, fd_norm)

            if fd_norm == 0.:
                nan = float('nan')
                derivative_info['rel error'] = rel_err = ErrorTuple(nan, nan, nan)
            else:
                derivative_info['rel error'] = rel_err = ErrorTuple(fwd_error / fd_norm,
                                                                    rev_error / fd_norm,
                                                                    fwd_rev_error / fd_norm)

            if out_stream:
                if compact_print:
                    out_stream.write(deriv_line.format(
                        _pad_name(of, 13, True),
                        _pad_name(wrt, 13, True),
                        magnitude.forward,
                        magnitude.reverse,
                        magnitude.fd,
                        abs_err.forward,
                        abs_err.reverse,
                        abs_err.forward_reverse,
                        rel_err.forward,
                        rel_err.reverse,
                        rel_err.forward_reverse,
                    ))
                else:
                    # Magnitudes
                    out_stream.write("  {}: '{}' wrt '{}'\n\n".format(sys_name, of, wrt))
                    out_stream.write('    Forward Magnitude : {:.6e}\n'.format(magnitude.forward))
                    out_stream.write('    Reverse Magnitude : {:.6e}\n'.format(magnitude.reverse))
                    out_stream.write('         Fd Magnitude : {:.6e} ({})\n\n'.format(magnitude.fd,
                                                                                      fd_desc))
                    # Absolute Errors
                    error_descs = ('(Jfor  - Jfd) ', '(Jrev  - Jfd) ', '(Jfor  - Jrev)')
                    for error, desc in zip(abs_err, error_descs):
                        error_str = _format_error(error, abs_error_tol)
                        out_stream.write('    Absolute Error {}: {}\n'.format(desc, error_str))
                    out_stream.write('\n')

                    # Relative Errors
                    for error, desc in zip(rel_err, error_descs):
                        error_str = _format_error(error, rel_error_tol)
                        out_stream.write('    Relative Error {}: {}\n'.format(desc, error_str))
                    out_stream.write('\n')

                    # Raw Derivatives
                    out_stream.write('    Raw Forward Derivative (Jfor)\n\n')
                    out_stream.write(str(forward))
                    out_stream.write('\n\n')

                    out_stream.write('    Raw Reverse Derivative (Jfor)\n\n')
                    out_stream.write(str(reverse))
                    out_stream.write('\n\n')

                    out_stream.write('    Raw FD Derivative (Jfd)\n\n')
                    out_stream.write(str(fd))
                    out_stream.write('\n\n')

                    out_stream.write(' -' * 30 + '\n')


def _pad_name(name, pad_num=10, quotes=False):
    """
    Pad a string so that they all line up when stacked.

    Parameters
    ----------
    name : str
        The string to pad.
    pad_num : int
        The number of total spaces the string should take up.
    quotes : bool
        If name should be quoted.

    Returns
    -------
    str
        Padded string
    """
    l_name = len(name)
    if l_name < pad_num:
        pad = pad_num - l_name
        if quotes:
            pad_str = "'{name}'{sep:<{pad}}"
        else:
            pad_str = "{name}{sep:<{pad}}"
        pad_name = pad_str.format(name=name, sep='', pad=pad)
        return pad_name
    else:
        return '{0}'.format(name)


def _format_error(error, tol):
    """
    Format the error, flagging if necessary.

    Parameters
    ----------
    error : float
        The absolute or relative error.
    tol : float
        Tolerance above which errors are flagged

    Returns
    -------
    str
        Formatted and possibly flagged error.
    """
    if np.isnan(error) or error < tol:
        return '{:.6e}'.format(error)
    return '{:.6e} *'.format(error)
