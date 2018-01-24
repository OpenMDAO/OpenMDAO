"""Define the Problem class and a FakeComm class for non-MPI users."""

from __future__ import division, print_function

import sys

from collections import OrderedDict, defaultdict, namedtuple
from itertools import product
import logging

from six import iteritems, iterkeys, itervalues
from six.moves import range

import numpy as np
import scipy.sparse as sparse

from openmdao.approximation_schemes.complex_step import ComplexStep, DEFAULT_CS_OPTIONS
from openmdao.approximation_schemes.finite_difference import FiniteDifference, DEFAULT_FD_OPTIONS
from openmdao.core.component import Component
from openmdao.core.driver import Driver
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.error_checking.check_config import check_config
from openmdao.recorders.recording_iteration_stack import recording_iteration
from openmdao.utils.general_utils import warn_deprecation, ContainsAll
from openmdao.utils.logger_utils import get_logger
from openmdao.utils.mpi import MPI, FakeComm
from openmdao.utils.name_maps import prom_name2abs_name
from openmdao.vectors.default_vector import DefaultVector

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.utils.name_maps import rel_key2abs_key, rel_name2abs_name

ErrorTuple = namedtuple('ErrorTuple', ['forward', 'reverse', 'forward_reverse'])
MagnitudeTuple = namedtuple('MagnitudeTuple', ['forward', 'reverse', 'fd'])

_contains_all = ContainsAll()

CITATION = """@inproceedings{2014_openmdao_derivs,
    Author = {Justin S. Gray and Tristan A. Hearn and Kenneth T. Moore
              and John Hwang and Joaquim Martins and Andrew Ning},
    Booktitle = {15th AIAA/ISSMO Multidisciplinary Analysis and Optimization Conference},
    Doi = {doi:10.2514/6.2014-2042},
    Month = {2014/07/08},
    Publisher = {American Institute of Aeronautics and Astronautics},
    Title = {Automatic Evaluation of Multidisciplinary Derivatives Using
             a Graph-Based Problem Formulation in OpenMDAO},
    Year = {2014}
}"""


class Problem(object):
    """
    Top-level container for the systems and drivers.

    Attributes
    ----------
    model : <System>
        Pointer to the top-level <System> object (root node in the tree).
    comm : MPI.Comm or <FakeComm>
        The global communicator.
    driver : <Driver>
        Slot for the driver. The default driver is `Driver`, which just runs
        the model once.
    _mode : 'fwd' or 'rev'
        Derivatives calculation mode, 'fwd' for forward, and 'rev' for
        reverse (adjoint).
    _use_ref_vector : bool
        If True, allocate vectors to store ref. values.
    _solver_print_cache : list
        Allows solver iprints to be set to requested values after setup calls.
    _initial_condition_cache : dict
        Any initial conditions that are set at the problem level via setitem are cached here
        until they can be processed.
    _setup_status : int
        Current status of the setup in _model.
        0 -- Newly initialized problem or newly added model.
        1 -- The `setup` method has been called, but vectors not initialized.
        2 -- The `final_setup` has been run, everything ready to run.
    cite : str
        Listing of relevant citataions that should be referenced when
        publishing work that uses this class.
    """

    _post_setup_func = None

    def __init__(self, model=None, comm=None, use_ref_vector=True, root=None):
        """
        Initialize attributes.

        Parameters
        ----------
        model : <System> or None
            Pointer to the top-level <System> object (root node in the tree).
        comm : MPI.Comm or <FakeComm> or None
            The global communicator.
        use_ref_vector : bool
            If True, allocate vectors to store ref. values.
        root : <System> or None
            Deprecated kwarg for `model`.
        """
        self.cite = CITATION

        if comm is None:
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            except ImportError:
                comm = FakeComm()

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

        self._use_ref_vector = use_ref_vector
        self._solver_print_cache = []

        self._mode = None  # mode is assigned in setup()

        recording_iteration.stack = []

        self._initial_condition_cache = {}

        # Status of the setup of _model.
        # 0 -- Newly initialized problem or newly added model.
        # 1 -- The `setup` method has been called, but vectors not initialized.
        # 2 -- The `final_setup` has been run, everything ready to run.
        self._setup_status = 0

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
        # Caching only needed if vectors aren't allocated yet.
        if self._setup_status == 1:

            # We have set and cached already
            if name in self._initial_condition_cache:
                val = self._initial_condition_cache[name]

            # Vector not setup, so we need to pull values from saved metadata request.
            else:
                proms = self.model._var_allprocs_prom2abs_list
                meta = self.model._var_abs2meta
                if name in meta['input']:
                    if name in self.model._conn_abs_in2out:
                        src_name = self.model._conn_abs_in2out[name]
                        val = meta['input'][src_name]['value']
                    else:
                        val = meta['input'][name]['value']

                elif name in meta['output']:
                    val = meta['output'][name]['value']

                elif name in proms['input']:
                    abs_name = proms['input'][name][0]
                    if abs_name in self.model._conn_abs_in2out:
                        src_name = self.model._conn_abs_in2out[abs_name]
                        # So, if the inputs and outputs are promoted to the same name, then we
                        # allow getitem, but if they aren't, then we raise an error due to non
                        # uniqueness.
                        if name not in proms['output']:
                            # This triggers a check for unconnected non-unique inputs, and
                            # raises the same error as vector access.
                            abs_name = prom_name2abs_name(self.model, name, 'input')
                        val = meta['output'][src_name]['value']
                    else:
                        # This triggers a check for unconnected non-unique inputs, and
                        # raises the same error as vector access.
                        abs_name = prom_name2abs_name(self.model, name, 'input')
                        val = meta['input'][abs_name]['value']

                elif name in proms['output']:
                    abs_name = prom_name2abs_name(self.model, name, 'output')
                    val = meta['output'][abs_name]['value']

                else:
                    msg = 'Variable name "{}" not found.'
                    raise KeyError(msg.format(name))

                self._initial_condition_cache[name] = val

        elif name in self.model._outputs:
            val = self.model._outputs[name]

        elif name in self.model._inputs:
            val = self.model._inputs[name]

        else:
            msg = 'Variable name "{}" not found.'
            raise KeyError(msg.format(name))

        # Need to cache the "get" in case the user calls in-place numpy operations.
        self._initial_condition_cache[name] = val

        return val

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
        # Caching only needed if vectors aren't allocated yet.
        if self._setup_status == 1:
            self._initial_condition_cache[name] = value
        else:
            if name in self.model._outputs:
                self.model._outputs[name] = value
            elif name in self.model._inputs:
                self.model._inputs[name] = value
            else:
                msg = 'Variable name "{}" not found.'
                raise KeyError(msg.format(name))

    def _set_initial_conditions(self):
        """
        Set all initial conditions that have been saved in cache after setup.
        """
        for name, value in iteritems(self._initial_condition_cache):
            self[name] = value

        # Clean up cache
        self._initial_condition_cache = {}

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
        ----------
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
        if self._mode is None:
            raise RuntimeError("The `setup` method must be called before `run_model`.")

        self.final_setup()
        self.model._clear_iprint()
        return self.model.run_solve_nonlinear()

    def run_driver(self):
        """
        Run the driver on the model.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        if self._mode is None:
            raise RuntimeError("The `setup` method must be called before `run_driver`.")

        self.final_setup()
        self.model._clear_iprint()
        with self.model._scaled_context_all():
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
        warn_deprecation("The 'run_once' method provides backwards compatibility with "
                         "OpenMDAO <= 1.x ; use 'run_model' instead.")

        return self.run_model()

    def run(self):
        """
        Backward compatible call for run_driver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        warn_deprecation("The 'run' method provides backwards compatibility with "
                         "OpenMDAO <= 1.x ; use 'run_driver' instead.")

        return self.run_driver()

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        self.driver.cleanup()

    def setup(self, vector_class=DefaultVector, check=False, logger=None, mode='rev',
              force_alloc_complex=False):
        """
        Set up the model hierarchy.

        When `setup` is called, the model hierarchy is assembled, the processors are allocated
        (for MPI), and variables and connections are all assigned. This method traverses down
        the model hierarchy to call `setup` on each subsystem, and then traverses up te model
        hierarchy to call `configure` on each subsystem.

        Parameters
        ----------
        vector_class : type
            reference to an actual <Vector> class; not an instance.
        check : boolean
            whether to run config check after setup is complete.
        logger : object
            Object for logging config checks if check is True.
        mode : string
            Derivatives calculation mode, 'fwd' for forward, and 'rev' for
            reverse (adjoint). Default is 'rev'.
        force_alloc_complex : bool
            Force allocation of imaginary part in nonlinear vectors. OpenMDAO can generally
            detect when you need to do this, but in some cases (e.g., complex step is used
            after a reconfiguration) you may need to set this to True.

        Returns
        -------
        self : <Problem>
            this enables the user to instantiate and setup in one line.
        """
        model = self.model
        comm = self.comm

        # PETScVector is required for MPI
        if PETScVector and comm.size > 1 and vector_class is not PETScVector:
            msg = ("The `vector_class` argument must be `PETScVector` when "
                   "running in parallel under MPI but '%s' was specified."
                   % vector_class.__name__)
            raise ValueError(msg)

        if mode not in ['fwd', 'rev']:
            msg = "Unsupported mode: '%s'. Use either 'fwd' or 'rev'." % mode
            raise ValueError(msg)

        self._mode = mode

        model._setup(comm, 'full', mode)

        # Cache all args for final setup.
        self._vector_class = vector_class
        self._check = check
        self._logger = logger
        self._force_alloc_complex = force_alloc_complex

        self._setup_status = 1

        return self

    def final_setup(self):
        """
        Perform final setup phase on problem in preparation for run.

        This is the second phase of setup, and is done automatically at the start of `run_driver`
        and `run_model`. At the beginning of final_setup, we have a model hierarchy with defined
        variables, solvers, case_recorders, and derivative settings. During this phase, the vectors
        are created and populated, the drivers and solvers are initialized, and the recorders are
        started, and the rest of the framework is prepared for execution.
        """
        vector_class = self._vector_class
        force_alloc_complex = self._force_alloc_complex

        comm = self.comm
        mode = self._mode

        if self._setup_status < 2:
            self.model._final_setup(comm, vector_class, 'full',
                                    force_alloc_complex=force_alloc_complex)

        self.driver._setup_driver(self)

        # Now that setup has been called, we can set the iprints.
        for items in self._solver_print_cache:
            self.set_solver_print(level=items[0], depth=items[1], type_=items[2])

        if self._check and comm.rank == 0:
            check_config(self, self._logger)

        if self._setup_status < 2:
            self._setup_status = 2
            self._set_initial_conditions()

        # check for post-setup hook
        if Problem._post_setup_func is not None:
            Problem._post_setup_func(self)

    def check_partials(self, logger=None, comps=None, compact_print=False,
                       abs_err_tol=1e-6, rel_err_tol=1e-6,
                       method='fd', step=None, form=DEFAULT_FD_OPTIONS['form'],
                       step_calc=DEFAULT_FD_OPTIONS['step_calc'],
                       force_dense=True, suppress_output=False):
        """
        Check partial derivatives comprehensively for all components in your model.

        Parameters
        ----------
        logger : object
            Logging object to capture output. All output is at logging.INFO level.
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
        method : str
            Method, 'fd' for finite difference or 'cs' for complex step. Default is 'fd'.
        step : float
            Step size for approximation. Default is None.
        form : string
            Form for finite difference, can be 'forward', 'backward', or 'central'. The
            default value is the value of DEFAULT_FD_OPTIONS['form']. Default is
            the value of DEFAULT_FD_OPTIONS['form']
        step_calc : string
            Step type for finite difference, can be 'abs' for absolute', or 'rel' for
            relative. The default value is the value of DEFAULT_FD_OPTIONS['step_calc']
        force_dense : bool
            If True, analytic derivatives will be coerced into arrays. Default is True.
        suppress_output : bool
            Set to True to suppress all output. Default is False.

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
        if self._setup_status < 2:
            self.final_setup()

        model = self.model
        logger = logger if logger else get_logger('check_partials')

        # TODO: Once we're tracking iteration counts, run the model if it has not been run before.

        all_comps = model.system_iter(typ=Component, include_self=True)
        if comps is None:
            comps = [comp for comp in all_comps]
        else:
            all_comp_names = {c.pathname for c in all_comps}
            requested = set(comps)
            extra = requested.difference(all_comp_names)
            if extra:
                msg = 'The following are not valid comp names: {}'.format(sorted(list(extra)))
                raise ValueError(msg)
            comps = [model._get_subsystem(c_name) for c_name in comps]

        current_mode = self._mode
        self.set_solver_print(level=0)

        # This is a defaultdict of (defaultdict of dicts).
        partials_data = defaultdict(lambda: defaultdict(dict))

        # Caching current point to restore after setups.
        input_cache = model._inputs._clone()
        output_cache = model._outputs._clone()

        # Keep track of derivative keys that are declared dependent so that we don't print them
        # unless they are in error.
        indep_key = {}

        # Analytic Jacobians
        for mode in ('fwd', 'rev'):
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
                matrix_free = comp.matrix_free
                c_name = comp.pathname
                indep_key[c_name] = set()

                # TODO: Check deprecated deriv_options.

                with comp._unscaled_context():
                    subjacs = comp._jacobian._subjacs
                    if explicit:
                        comp._negate_jac()

                    of_list = list(comp._var_allprocs_prom2abs_list['output'].keys())
                    wrt_list = list(comp._var_allprocs_prom2abs_list['input'].keys())

                    # The only outputs in wrt should be implicit states.
                    if not explicit:
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
                                perturb = 1.0 if not explicit else -1.0

                                # Dictionary access returns a scaler for 1d input, and we
                                # need a vector for clean code, so use _views_flat.
                                flat_view[idx] = perturb

                                # Matrix Vector Product
                                comp._apply_linear(['linear'], _contains_all, mode)

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

                            # Testing for pairs that are not dependent so that we suppress printing
                            # them unless the fd is non zero. Note: subjacs_info is empty for
                            # undeclared partials, which is the default behavior now.
                            try:
                                if comp._jacobian._subjacs_info[abs_key][0]['dependent'] is False:
                                    indep_key[c_name].add(rel_key)
                            except KeyError:
                                indep_key[c_name].add(rel_key)

                            if deriv_value is None:
                                # Missing derivatives are assumed 0.
                                try:
                                    in_size = comp._var_abs2meta['input'][wrt]['size']
                                except KeyError:
                                    in_size = comp._var_abs2meta['output'][wrt]['size']

                                out_size = comp._var_abs2meta['output'][of]['size']
                                deriv_value = np.zeros((out_size, in_size))

                            if force_dense:
                                if isinstance(deriv_value, list):
                                    try:
                                        in_size = comp._var_abs2meta['input'][wrt]['size']
                                    except KeyError:
                                        in_size = comp._var_abs2meta['output'][wrt]['size']
                                    out_size = comp._var_abs2meta['output'][of]['size']
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

                            partials_data[c_name][rel_key][jac_key] = deriv_value.copy()

                    if explicit:
                        comp._negate_jac()

        model._inputs.set_vec(input_cache)
        model._outputs.set_vec(output_cache)
        model.run_apply_nonlinear()

        # Finite Difference to calculate Jacobian
        jac_key = 'J_fd'
        alloc_complex = model._outputs._alloc_complex
        all_fd_options = {}
        for comp in comps:

            # Skip IndepVarComps
            if isinstance(comp, IndepVarComp):
                continue

            c_name = comp.pathname
            all_fd_options[c_name] = {}
            explicit = isinstance(comp, ExplicitComponent)

            approximations = {'fd': FiniteDifference(),
                              'cs': ComplexStep()}

            of = list(comp._var_allprocs_prom2abs_list['output'].keys())
            wrt = list(comp._var_allprocs_prom2abs_list['input'].keys())

            # The only outputs in wrt should be implicit states.
            if not explicit:
                wrt.extend(of)

            # Load up approximation objects with the requested settings.
            local_opts = comp._get_check_partial_options()
            for rel_key in product(of, wrt):
                abs_key = rel_key2abs_key(comp, rel_key)
                local_wrt = rel_key[1]

                # Determine if fd or cs.
                if local_wrt in local_opts:
                    local_method = local_opts[local_wrt]['method']
                    if local_method:
                        method = local_method

                fd_options = {'order': None,
                              'method': method}

                if method == 'cs':
                    if not alloc_complex:
                        msg = 'In order to check partials with complex step, you need to set ' + \
                            '"force_alloc_complex" to True during setup.'
                        raise RuntimeError(msg)

                    defaults = DEFAULT_CS_OPTIONS

                    fd_options['form'] = None
                    fd_options['step_calc'] = None

                elif method == 'fd':
                    defaults = DEFAULT_FD_OPTIONS

                    fd_options['form'] = form
                    fd_options['step_calc'] = step_calc

                if step:
                    fd_options['step'] = step
                else:
                    fd_options['step'] = defaults['step']

                # Precedence: component options > global options > defaults
                if local_wrt in local_opts:
                    for name in ['form', 'step', 'step_calc']:
                        value = local_opts[local_wrt][name]
                        if value is not None:
                            fd_options[name] = value

                all_fd_options[c_name][local_wrt] = fd_options

                approximations[fd_options['method']].add_approximation(abs_key, fd_options)

            approx_jac = {}
            for approximation in itervalues(approximations):
                approximation._init_approximations()

                # Peform the FD here.
                approximation.compute_approximations(comp, jac=approx_jac)

            for rel_key, partial in iteritems(approx_jac):
                abs_key = rel_key2abs_key(comp, rel_key)
                # Since all partials for outputs for explicit comps are declared, assume anything
                # missing is an input deriv.
                if explicit and abs_key[1] in comp._var_abs_names['input']:
                    partials_data[c_name][rel_key][jac_key] = -partial
                else:
                    partials_data[c_name][rel_key][jac_key] = partial

        # Conversion of defaultdict to dicts
        partials_data = {comp_name: dict(outer) for comp_name, outer in iteritems(partials_data)}

        _assemble_derivative_data(partials_data, rel_err_tol, abs_err_tol, logger, compact_print,
                                  comps, all_fd_options, suppress_output=suppress_output,
                                  indep_key=indep_key)

        return partials_data

    def check_totals(self, of=None, wrt=None, logger=None, compact_print=False, abs_err_tol=1e-6,
                     rel_err_tol=1e-6, method='fd', step=1e-6, form='forward', step_calc='abs',
                     suppress_output=False):
        """
        Check total derivatives for the model vs. finite difference.

        Parameters
        ----------
        of : list of variable name strings or None
            Variables whose derivatives will be computed. Default is None, which
            uses the driver's objectives and constraints.
        wrt : list of variable name strings or None
            Variables with respect to which the derivatives will be computed.
            Default is None, which uses the driver's desvars.
        logger : object
            Logging object to capture output. All output is at logging.INFO level.
        compact_print : bool
            Set to True to just print the essentials, one line per unknown-param pair.
        abs_err_tol : float
            Threshold value for absolute error.  Errors about this value will have a '*' displayed
            next to them in output, making them easy to search for. Default is 1.0E-6.
        rel_err_tol : float
            Threshold value for relative error.  Errors about this value will have a '*' displayed
            next to them in output, making them easy to search for. Note at times there may be a
            significant relative error due to a minor absolute error.  Default is 1.0E-6.
        method : str
            Method, 'fd' for finite difference or 'cs' for complex step. Default is 'fd'
        step : float
            Step size for approximation. Default is 1e-6.
        form : string
            Form for finite difference, can be 'forward', 'backward', or 'central'. Default
            'forward'.
        step_calc : string
            Step type for finite difference, can be 'abs' for absolute', or 'rel' for relative.
            Default is 'abs'.
        suppress_output : bool
            Set to True to suppress all output. Default is False.

        Returns
        -------
        Dict of Dicts of Tuples of Floats

            First key:
                is the (output, input) tuple of strings;
            Second key:
                is one of ['rel error', 'abs error', 'magnitude', 'fdstep'];

            For 'rel error', 'abs error', 'magnitude' the value is: A tuple containing norms for
                forward - fd, adjoint - fd, forward - adjoint.
        """
        model = self.model
        global_names = False

        logger = logger if logger else get_logger('check_totals')

        # TODO: Once we're tracking iteration counts, run the model if it has not been run before.

        if wrt is None:
            wrt = list(self.driver._designvars)
            global_names = True
        if of is None:
            of = list(self.driver._objs)
            of.extend(list(self.driver._cons))
            global_names = True

        with self.model._scaled_context_all():

            # Calculate Total Derivatives
            Jcalc = self._compute_totals(of=of, wrt=wrt, global_names=global_names)

            # Approximate FD
            fd_args = {
                'step': step,
                'form': form,
                'step_calc': step_calc,
            }
            model.approx_totals(method=method, **fd_args)
            Jfd = self._compute_totals_approx(of=of, wrt=wrt, global_names=global_names,
                                              initialize=True)

        # Assemble and Return all metrics.
        data = {}
        data[''] = {}
        for key, val in iteritems(Jcalc):
            data[''][key] = {}
            data[''][key]['J_fwd'] = Jcalc[key]
            data[''][key]['J_fd'] = Jfd[key]
        fd_args['method'] = 'fd'

        _assemble_derivative_data(data, rel_err_tol, abs_err_tol, logger, compact_print, [model],
                                  {'': fd_args}, totals=True, suppress_output=suppress_output)
        return data['']

    def compute_totals(self, of=None, wrt=None, return_format='flat_dict'):
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
        if self._setup_status < 2:
            self.final_setup()

        with self.model._scaled_context_all():
            if self.model._owns_approx_jac:
                totals = self._compute_totals_approx(of=of, wrt=wrt,
                                                     return_format=return_format,
                                                     global_names=False,
                                                     initialize=True)
            else:
                totals = self._compute_totals(of=of, wrt=wrt,
                                              return_format=return_format,
                                              global_names=False)
        return totals

    def _compute_totals_approx(self, of=None, wrt=None, return_format='flat_dict',
                               global_names=True, initialize=False):
        """
        Compute derivatives of desired quantities with respect to desired inputs.

        Uses an approximation method, e.g., fd or cs to calculate the derivatives.

        Parameters
        ----------
        of : list of variable name strings or None
            Variables whose derivatives will be computed.
        wrt : list of variable name strings or None
            Variables with respect to which the derivatives will be computed.
        return_format : string
            Format to return the derivatives. Default is a 'flat_dict', which
            returns them in a dictionary whose keys are tuples of form (of, wrt).
        global_names : bool
            Set to True when passing in global names to skip some translation steps.
        initialize : bool
            Set to True to re-initialize the FD in model. This is only needed when manually
            calling compute_totals on the problem.

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        recording_iteration.stack.append(('_compute_totals', 0))
        model = self.model
        mode = self._mode
        vec_dinput = model._vectors['input']
        vec_doutput = model._vectors['output']
        vec_dresid = model._vectors['residual']
        approx = model._owns_approx_jac
        prom2abs = model._var_allprocs_prom2abs_list['output']

        # TODO - Pull 'of' and 'wrt' from driver if unspecified.
        if wrt is None:
            raise NotImplementedError("Need to specify 'wrt' for now.")
        if of is None:
            raise NotImplementedError("Need to specify 'of' for now.")

        # Prepare model for calculation by cleaning out the derivatives
        # vectors.
        for vec_name in model._lin_vec_names:

            # TODO: Do all three deriv vectors have the same keys?

            vec_dinput[vec_name].set_const(0.0)
            vec_doutput[vec_name].set_const(0.0)
            vec_dresid[vec_name].set_const(0.0)

        # Convert of and wrt names from promoted to absolute path
        oldwrt, oldof = wrt, of
        if not global_names:
            of = [prom2abs[name][0] for name in oldof]
            wrt = [prom2abs[name][0] for name in oldwrt]

        input_list, output_list = wrt, of
        old_input_list, old_output_list = oldwrt, oldof

        # Solve for derivs with the approximation_scheme.
        # This cuts out the middleman by grabbing the Jacobian directly after linearization.

        # Re-initialize so that it is clean.
        if initialize:

            if model._approx_schemes:
                method = list(model._approx_schemes.keys())[0]
                kwargs = model._owns_approx_jac_meta
                model.approx_totals(method=method, **kwargs)
            else:
                model.approx_totals(method='fd')

        # Initialization based on driver (or user) -requested "of" and "wrt".
        if not model._owns_approx_jac or model._owns_approx_of != set(of) \
           or model._owns_approx_wrt != set(wrt):
            model._owns_approx_of = set(of)
            model._owns_approx_wrt = set(wrt)

            # Support for indices defined on driver vars.
            dvs = self.driver._designvars
            cons = self.driver._cons
            objs = self.driver._objs

            response_idx = {key: val['indices'] for key, val in iteritems(cons)
                            if val['indices'] is not None}
            for key, val in iteritems(objs):
                if val['indices'] is not None:
                    response_idx[key] = val['indices']

            model._owns_approx_of_idx = response_idx

            model._owns_approx_wrt_idx = {key: val['indices'] for key, val in iteritems(dvs)
                                          if val['indices'] is not None}

        model._setup_jacobians(recurse=False)

        # Need to temporarily disable size checking to support indices in des_vars and quantities.
        model.jacobian._override_checks = True

        # Linearize Model
        model._linearize()

        model.jacobian._override_checks = False
        approx_jac = model._jacobian._subjacs

        # Create data structures (and possibly allocate space) for the total
        # derivatives that we will return.
        totals = OrderedDict()

        if return_format == 'flat_dict':
            for ocount, output_name in enumerate(output_list):
                okey = old_output_list[ocount]
                for icount, input_name in enumerate(input_list):
                    ikey = old_input_list[icount]
                    jac = approx_jac[output_name, input_name]
                    if isinstance(jac, list):
                        # This is a design variable that was declared as an obj/con.
                        totals[okey, ikey] = np.eye(len(jac[0]))
                        odx = model._owns_approx_of_idx.get(okey)
                        idx = model._owns_approx_wrt_idx.get(ikey)
                        if odx is not None:
                            totals[okey, ikey] = totals[okey, ikey][odx, :]
                        if idx is not None:
                            totals[okey, ikey] = totals[okey, ikey][:, idx]
                    else:
                        totals[okey, ikey] = -jac

        elif return_format == 'dict':
            for ocount, output_name in enumerate(output_list):
                okey = old_output_list[ocount]
                totals[okey] = tot = OrderedDict()
                for icount, input_name in enumerate(input_list):
                    ikey = old_input_list[icount]
                    jac = approx_jac[output_name, input_name]
                    if isinstance(jac, list):
                        # This is a design variable that was declared as an obj/con.
                        tot[ikey] = np.eye(len(jac[0]))
                    else:
                        tot[ikey] = -jac
        else:
            msg = "Unsupported return format '%s." % return_format
            raise NotImplementedError(msg)

        recording_iteration.stack.pop()
        return totals

    def _get_voi_info(self, voi_lists, inp2rhs_name, input_vec, output_vec, input_vois):
        voi_info = {}
        model = self.model
        nproc = self.comm.size
        iproc = model.comm.rank

        for vois in itervalues(voi_lists):
            for input_name, old_input_name, _, _, _ in vois:
                vecname = inp2rhs_name[input_name]
                if vecname not in input_vec:
                    continue
                sizes = model._var_sizes[vecname]['output']
                dinputs = input_vec[vecname]
                doutputs = output_vec[vecname]

                in_var_idx = model._var_allprocs_abs2idx[vecname]['output'][input_name]
                in_var_meta = model._var_allprocs_abs2meta['output'][input_name]

                dup = not in_var_meta['distributed']

                start = np.sum(sizes[:iproc, in_var_idx])
                end = np.sum(sizes[:iproc + 1, in_var_idx])

                in_idxs = None
                if input_name in input_vois:
                    in_voi_meta = input_vois[input_name]
                    if 'indices' in in_voi_meta:
                        in_idxs = in_voi_meta['indices']

                if in_idxs is not None:
                    neg = in_idxs[in_idxs < 0]
                    irange = in_idxs
                    if neg:
                        irange[neg] += end
                    max_i = np.max(in_idxs)
                    min_i = np.min(in_idxs)
                    loc_size = len(in_idxs)
                else:
                    irange = np.arange(in_var_meta['global_size'], dtype=int)
                    max_i = in_var_meta['global_size'] - 1
                    min_i = 0
                    loc_size = end - start

                if loc_size == 0:
                    # var is not local. get size of var in owned proc
                    for rank in range(nproc):
                        sz = sizes[rank, in_var_idx]
                        if sz > 0:
                            loc_size = sz
                            break

                # set totals to zeros instead of None in those cases when none
                # of the specified indices are within the range of interest
                # for this proc.
                store = True if ((start <= min_i < end) or (start <= max_i < end)) else dup

                if store:
                    loc_idxs = irange
                    if min_i > 0:
                        loc_idxs = irange - min_i
                else:
                    loc_idxs = []

                voi_info[input_name] = (dinputs, doutputs, irange, loc_idxs, max_i, min_i,
                                        loc_size, start, end, dup, store)

        return voi_info

    def _compute_totals(self, of=None, wrt=None, return_format='flat_dict', global_names=True):
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
        recording_iteration.stack.append(('_compute_totals', 0))
        model = self.model
        mode = self._mode
        vec_dinput = model._vectors['input']
        vec_doutput = model._vectors['output']
        vec_dresid = model._vectors['residual']
        nproc = self.comm.size
        iproc = model.comm.rank
        sizes = model._var_sizes['nonlinear']['output']
        relevant = model._relevant
        fwd = (mode == 'fwd')
        prom2abs = model._var_allprocs_prom2abs_list['output']
        abs2idx_out = model._var_allprocs_abs2idx['linear']['output']

        if wrt is None:
            wrt = list(self.driver._designvars)
        if of is None:
            of = list(self.driver._objs)
            of.extend(list(self.driver._cons))

        # A number of features will need to be supported here as development
        # goes forward.
        # -------------------------------------------------------------------
        # TODO: Support constraint sparsity (i.e., skip in/out that are not
        #       relevant for this constraint) (desvars too?)
        # TODO: Don't calculate for inactive constraints
        # -------------------------------------------------------------------

        # Prepare model for calculation by cleaning out the derivatives
        # vectors.
        matmat = False
        for vec_name in model._lin_vec_names:
            vec_dinput[vec_name].set_const(0.0)
            vec_doutput[vec_name].set_const(0.0)
            vec_dresid[vec_name].set_const(0.0)

        # Linearize Model
        model._linearize()

        # Create data structures (and possibly allocate space) for the total
        # derivatives that we will return.
        totals = OrderedDict()

        if return_format == 'flat_dict':
            for okey in of:
                for ikey in wrt:
                    totals[(okey, ikey)] = None
        elif return_format == 'dict':
            for okey in of:
                totals[okey] = OrderedDict()
                for ikey in wrt:
                    totals[okey][ikey] = None
        else:
            msg = "Unsupported return format '%s." % return_format
            raise NotImplementedError(msg)

        # Convert of and wrt names from promoted to unpromoted
        # (which is absolute path since we're at the top)
        oldwrt, oldof = wrt, of
        if not global_names:
            of = [prom2abs[name][0] for name in oldof]
            wrt = [prom2abs[name][0] for name in oldwrt]

        owning_ranks = self.model._owning_rank['output']

        # we don't do simultaneous derivatives when compute_totals is called for linear constaints
        has_lin_constraints = False

        if fwd:
            input_list, output_list = wrt, of
            old_input_list, old_output_list = oldwrt, oldof
            input_vec, output_vec = vec_dresid, vec_doutput
            input_vois = self.driver._designvars
            output_vois = self.driver._responses
            remote_outputs = self.driver._remote_responses
            for n in of:
                if n in output_vois and 'linear' in output_vois[n] and output_vois[n]['linear']:
                    has_lin_constraints = True
                    break
        else:  # rev
            input_list, output_list = of, wrt
            old_input_list, old_output_list = oldof, oldwrt
            input_vec, output_vec = vec_doutput, vec_dresid
            input_vois = self.driver._responses
            output_vois = self.driver._designvars
            remote_outputs = self.driver._remote_dvs

        # Solve for derivs using linear solver.

        # this maps either a parallel_deriv_color to a list of tuples of (absname, oldname)
        # of variables in that group, or, for variables that aren't in an
        # parallel_deriv_color, it maps the variable name to a one entry list containing
        # the tuple (absname, oldname) for that variable.  oldname will be
        # the promoted name if the 'global_names' arg is False, else it will
        # be the same as absname (the absolute variable name).
        voi_lists = OrderedDict()

        # this maps the names from input_list to the corresponding names
        # of the RHS vectors.  For any variables that are not part of an
        # parallel_deriv_color, they will map to 'linear'.  All parallel_deriv_color'ed variables
        # will just map to their own name.
        inp2rhs_name = {}

        # if not all inputs are VOIs, then this is a test where most likely
        # design vars, constraints, and objectives were not specified, so
        # don't do relevance checking (because we only want to analyze the
        # dependency graph for VOIs rather than for every input/output
        # in the model)
        use_rel_reduction = True

        for i, name in enumerate(input_list):
            if name in input_vois:
                meta = input_vois[name]
                parallel_deriv_color = meta['parallel_deriv_color']
                simul_coloring = meta['simul_deriv_color']
                matmat = meta['vectorize_derivs']
            else:
                parallel_deriv_color = simul_coloring = None
                use_rel_reduction = False
                matmat = False

            if simul_coloring is not None:
                if parallel_deriv_color:
                    raise RuntimeError("Using both simul_coloring and parallel_deriv_color with "
                                       "variable '%s' is not supported." % name)
                if matmat:
                    raise RuntimeError("Using both simul_coloring and vectorize_derivs with "
                                       "variable '%s' is not supported." % name)

            if parallel_deriv_color is None:  # variable is not in a parallel_deriv_color
                if name in voi_lists:
                    raise RuntimeError("Variable name '%s' matches a parallel_deriv_color name." %
                                       name)
                else:
                    # store the absolute name along with the original name, which
                    # can be either promoted or absolute depending on the value
                    # of the 'global_names' flag.
                    voi_lists[name] = [(name, old_input_list[i], parallel_deriv_color, matmat,
                                        simul_coloring)]
                    inp2rhs_name[name] = name if matmat else 'linear'
            else:
                if parallel_deriv_color not in voi_lists:
                    voi_lists[parallel_deriv_color] = []
                voi_lists[parallel_deriv_color].append((name, old_input_list[i],
                                                        parallel_deriv_color, matmat,
                                                        simul_coloring))
                inp2rhs_name[name] = name

        lin_vec_names = sorted(set(inp2rhs_name.values()))

        voi_info = self._get_voi_info(voi_lists, inp2rhs_name, input_vec, output_vec, input_vois)

        for vois in itervalues(voi_lists):
            # If Forward mode, solve linear system for each 'wrt'
            # If Adjoint mode, solve linear system for each 'of'

            matmats = [m for _, _, _, m, _ in vois]
            matmat = matmats[0]
            if any(matmats) and not all(matmats):
                raise RuntimeError("Mixing of vectorized and non-vectorized derivs in the same "
                                   "parallel color group (%s) is not supported." %
                                   [name for _, name, _, _, _ in vois])
            simul_coloring = vois[0][4] if not has_lin_constraints else None

            if use_rel_reduction:
                rel_systems = set()
                for voi, _, _, _, _ in vois:
                    rel_systems.update(relevant[voi]['@all'][1])
            else:
                rel_systems = _contains_all

            if matmat:
                idx_iter = range(1)
            elif simul_coloring is not None:
                loc_idx_dict = defaultdict(lambda: -1)
                # here we're guaranteed that there is only one voi
                input_name = vois[0][0]
                info = voi_info[input_name]
                dinputs = info[0]
                colors = set(simul_coloring)
                if not isinstance(simul_coloring, np.ndarray):
                    simul_coloring = np.array(simul_coloring, dtype=int)

                def idx_iter():
                    for c in colors:
                        # iterate over negative colors individually
                        if c < 0:
                            for i in np.nonzero(simul_coloring == c)[0]:
                                yield (c, i)
                        else:
                            nzs = np.nonzero(simul_coloring == c)[0]
                            if nzs.size == 1:
                                yield (c, nzs[0])
                            else:
                                yield (c, nzs)
                idx_iter = idx_iter()
            else:
                loc_idx_dict = defaultdict(lambda: -1)
                max_len = max(len(voi_info[name][2]) for name, _, _, _, _ in vois)
                idx_iter = range(max_len)

            for i in idx_iter:
                if simul_coloring is not None:
                    color, i = i
                    do_color_iter = isinstance(i, np.ndarray) and i.size > 1
                else:
                    do_color_iter = False

                # this sets dinputs for the current parallel_deriv_color to 0
                # dinputs is dresids in fwd, doutouts in rev
                if fwd:
                    vec_dresid['linear'].set_const(0.0)
                    if use_rel_reduction:
                        vec_doutput['linear'].set_const(0.0)
                else:  # rev
                    vec_doutput['linear'].set_const(0.0)
                    if use_rel_reduction:
                        vec_dinput['linear'].set_const(0.0)

                for input_name, old_input_name, pd_color, matmat, simul in vois:
                    dinputs, doutputs, idxs, _, max_i, min_i, loc_size, start, end, dup, _ = \
                        voi_info[input_name]

                    if matmat:
                        if input_name in dinputs:
                            vec = dinputs._views_flat[input_name]
                            if vec.size == 1:
                                if start <= idxs[0] < end:
                                    vec[idxs[0] - start] = 1.0
                            else:
                                for ii, idx in enumerate(idxs):
                                    if start <= idx < end:
                                        vec[idx - start, ii] = 1.0
                    elif simul is not None:
                        ii = idxs[i]
                        final_idxs = ii[np.logical_and(ii >= start, ii < end)]
                        vec_dinput['linear'].set_const(0.0)
                        dinputs._views_flat[input_name][final_idxs] = 1.0
                    else:
                        if i >= len(idxs):
                            # reuse the last index if loop iter is larger than current var size
                            idx = idxs[-1]
                        else:
                            idx = idxs[i]
                        if start <= idx < end and input_name in dinputs._views_flat:
                            # Dictionary access returns a scaler for 1d input, and we
                            # need a vector for clean code, so use _views_flat.
                            dinputs._views_flat[input_name][idx - start] = 1.0

                model._solve_linear(lin_vec_names, mode, rel_systems)

                for input_name, old_input_name, _, matmat, simul in vois:
                    dinputs, doutputs, idxs, loc_idxs, max_i, min_i, loc_size, start, end, \
                        dup, store = voi_info[input_name]
                    ncol = dinputs._ncol

                    if matmat:
                        loc_idx = loc_idxs
                    elif simul is not None and do_color_iter:
                        loc_idx = loc_idxs[i - start]
                    else:
                        if simul is None:
                            if i >= len(idxs):
                                idx = idxs[-1]  # reuse the last index
                                delta_loc_idx = 0  # don't increment local_idx
                            else:
                                idx = idxs[i]
                                delta_loc_idx = 1
                            # totals to zeros instead of None in those cases when none
                            # of the specified indices are within the range of interest
                            # for this proc.
                            if store:
                                loc_idx_dict[input_name] += delta_loc_idx
                            loc_idx = loc_idx_dict[input_name]
                        else:
                            idx = loc_idx = i

                    # Pull out the answers and pack into our data structure.
                    for ocount, output_name in enumerate(output_list):
                        out_idxs = None
                        if output_name in output_vois:
                            out_voi_meta = output_vois[output_name]
                            out_idxs = out_voi_meta['indices']

                        if use_rel_reduction and output_name not in relevant[input_name]:
                            # irrelevant output, just give zeros
                            if out_idxs is None:
                                out_var_idx = abs2idx_out[output_name]
                                if output_name in remote_outputs:
                                    _, sz = remote_outputs[output_name]
                                else:
                                    sz = sizes[iproc, out_var_idx]
                                if ncol > 1:
                                    deriv_val = np.zeros((sz, ncol))
                                else:
                                    deriv_val = np.zeros(sz)
                            elif ncol > 1:
                                deriv_val = np.zeros((len(out_idxs), ncol))
                            else:
                                deriv_val = np.zeros(len(out_idxs))
                        else:  # relevant output
                            if output_name in doutputs._views_flat:
                                deriv_val = doutputs._views_flat[output_name]
                                size = deriv_val.size
                            else:
                                deriv_val = None

                            if out_idxs is not None:
                                size = out_idxs.size
                                if deriv_val is not None:
                                    deriv_val = deriv_val[out_idxs]

                            if dup and nproc > 1:
                                out_var_idx = abs2idx_out[output_name]
                                root = owning_ranks[output_name]
                                if deriv_val is None:
                                    if out_idxs is not None:
                                        sz = size
                                    else:
                                        sz = sizes[root, out_var_idx]
                                    if ncol > 1:
                                        deriv_val = np.empty((sz, ncol))
                                    else:
                                        deriv_val = np.empty(sz)
                                self.comm.Bcast(deriv_val, root=root)

                        len_val = len(deriv_val)

                        if store and ncol > 1 and len(deriv_val.shape) == 1:
                            deriv_val = np.atleast_2d(deriv_val).T

                        if return_format == 'flat_dict':
                            if fwd:
                                key = (old_output_list[ocount], old_input_name)

                                if totals[key] is None:
                                    totals[key] = np.zeros((len_val, loc_size))
                                if store:
                                    if simul is not None and do_color_iter:
                                        smap = output_vois[output_name]['simul_map']
                                        if (smap is not None and input_name in smap and
                                                color in smap[input_name]):
                                            col_idxs = smap[input_name][color][1]
                                            if col_idxs:
                                                row_idxs = smap[input_name][color][0]
                                                mat = totals[key]
                                                for idx, col in enumerate(col_idxs):
                                                    mat[row_idxs[idx], col] = \
                                                        deriv_val[row_idxs[idx]]
                                    else:
                                        totals[key][:, loc_idx] = deriv_val
                            else:
                                key = (old_input_name, old_output_list[ocount])

                                if totals[key] is None:
                                    totals[key] = np.zeros((loc_size, len_val))
                                if store:
                                    totals[key][loc_idx, :] = deriv_val.T

                        elif return_format == 'dict':
                            if fwd:
                                okey = old_output_list[ocount]

                                if totals[okey][old_input_name] is None:
                                    totals[okey][old_input_name] = np.zeros((len_val, loc_size))
                                if store:
                                    if simul is not None and do_color_iter:
                                        smap = output_vois[output_name]['simul_map']
                                        if (smap is not None and input_name in smap and
                                                color in smap[input_name]):
                                            col_idxs = smap[input_name][color][1]
                                            if col_idxs:
                                                row_idxs = smap[input_name][color][0]
                                                mat = totals[okey][old_input_name]
                                                for idx, col in enumerate(col_idxs):
                                                    mat[row_idxs[idx], col] = \
                                                        deriv_val[row_idxs[idx]]
                                    else:
                                        totals[okey][old_input_name][:, loc_idx] = deriv_val
                            else:
                                ikey = old_output_list[ocount]

                                if totals[old_input_name][ikey] is None:
                                    totals[old_input_name][ikey] = np.zeros((loc_size, len_val))
                                if store:
                                    totals[old_input_name][ikey][loc_idx, :] = deriv_val.T
                        else:
                            raise RuntimeError("unsupported return format")

        recording_iteration.stack.pop()

        return totals

    def set_solver_print(self, level=2, depth=1e99, type_='all'):
        """
        Control printing for solvers and subsolvers in the model.

        Parameters
        ----------
        level : int
            iprint level. Set to 2 to print residuals each iteration; set to 1
            to print just the iteration totals; set to 0 to disable all printing
            except for failures, and set to -1 to disable all printing including failures.
        depth : int
            How deep to recurse. For example, you can set this to 0 if you only want
            to print the top level linear and nonlinear solver messages. Default
            prints everything.
        type_ : str
            Type of solver to set: 'LN' for linear, 'NL' for nonlinear, or 'all' for all.
        """
        if (level, depth, type_) not in self._solver_print_cache:
            self._solver_print_cache.append((level, depth, type_))

        self.model._set_solver_print(level=level, depth=depth, type_=type_)


def _assemble_derivative_data(derivative_data, rel_error_tol, abs_error_tol, logger,
                              compact_print, system_list, global_options, totals=False,
                              suppress_output=False, indep_key=None):
    """
    Compute the relative and absolute errors in the given derivatives and print to the logger.

    Parameters
    ----------
    derivative_data : dict
        Dictionary containing derivative information keyed by system name.
    rel_error_tol : float
        Relative error tolerance.
    abs_error_tol : float
        Absolute error tolerance.
    logger : object
        Logging object.
    compact_print : bool
        If results should be printed verbosely or in a table.
    system_list : Iterable
        The systems (in the proper order) that were checked.0
    global_options : dict
        Dictionary containing the options for the approximation.
    totals : bool
        Set to True if we are doing check_totals to skip a bunch of stuff.
    suppress_output : bool
        Set to True to suppress all output. Just calculate errors and add the keys.
    indep_key : dict of sets, optional
        Keyed by component name, contains the of/wrt keys that are declared not dependent.
    """
    nan = float('nan')

    if compact_print:
        if totals:
            deriv_line = "{0} wrt {1} | {2:.4e} | {3:.4e} | {4:.4e} | {5:.4e}"
        else:
            deriv_line = "{0} wrt {1} | {2:.4e} | {3:.4e} | {4:.4e} | {5:.4e} | {6:.4e} | {7:.4e}"\
                         " | {8:.4e} | {9:.4e} | {10:.4e}"

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

        if totals:
            sys_name = 'Full Model'

        if not suppress_output:
            logger.info('-' * (len(sys_name) + 15))
            logger.info("{}: '{}'".format(sys_type, sys_name))
            logger.info('-' * (len(sys_name) + 15))

            if compact_print:
                # Error Header
                if totals:
                    header = "{0} wrt {1} | {2} | {3} | {4} | {5}"\
                        .format(
                            _pad_name('<output>', 30, True),
                            _pad_name('<variable>', 30, True),
                            _pad_name('calc mag.'),
                            _pad_name('check mag.'),
                            _pad_name('a(cal-chk)'),
                            _pad_name('r(cal-chk)'),
                        )
                else:
                    header = "{0} wrt {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10}"\
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
                logger.info(header)
                logger.info('-' * len(header) + '\n')

        # Sorted keys ensures deterministic ordering
        sorted_keys = sorted(iterkeys(derivatives))

        for of, wrt in sorted_keys:

            derivative_info = derivatives[of, wrt]
            forward = derivative_info['J_fwd']
            if not totals:
                reverse = derivative_info.get('J_rev')
            fd = derivative_info['J_fd']

            fwd_error = np.linalg.norm(forward - fd)
            if totals:
                rev_error = fwd_rev_error = None
            else:
                rev_error = np.linalg.norm(reverse - fd)
                fwd_rev_error = np.linalg.norm(forward - reverse)

            fwd_norm = np.linalg.norm(forward)
            if totals:
                rev_norm = None
            else:
                rev_norm = np.linalg.norm(reverse)
            fd_norm = np.linalg.norm(fd)

            derivative_info['abs error'] = abs_err = ErrorTuple(fwd_error, rev_error, fwd_rev_error)
            derivative_info['magnitude'] = magnitude = MagnitudeTuple(fwd_norm, rev_norm, fd_norm)

            if fd_norm == 0.:
                derivative_info['rel error'] = rel_err = ErrorTuple(nan, nan, nan)
            else:
                if totals:
                    derivative_info['rel error'] = rel_err = ErrorTuple(fwd_error / fd_norm,
                                                                        nan,
                                                                        nan)
                else:
                    derivative_info['rel error'] = rel_err = ErrorTuple(fwd_error / fd_norm,
                                                                        rev_error / fd_norm,
                                                                        fwd_rev_error / fd_norm)

            # Skip printing the dependent keys if the derivatives are fine.
            if indep_key is not None:
                rel_key = (of, wrt)
                if rel_key in indep_key[sys_name] and fd_norm < abs_error_tol:
                    del derivative_data[sys_name][rel_key]
                    continue

            if not suppress_output:
                if compact_print:
                    if totals:
                        logger.info(deriv_line.format(
                            _pad_name(of, 30, True),
                            _pad_name(wrt, 30, True),
                            magnitude.forward,
                            magnitude.fd,
                            abs_err.forward,
                            rel_err.forward,
                        ))
                    else:
                        logger.info(deriv_line.format(
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

                    if totals:
                        fd_desc = "{}:{}".format(global_options['']['method'],
                                                 global_options['']['form'])

                    else:
                        fd_desc = "{}:{}".format(global_options[sys_name][wrt]['method'],
                                                 global_options[sys_name][wrt]['form'])

                    if compact_print:
                        check_desc = "    (Check Type: {})".format(fd_desc)
                    else:
                        check_desc = ""

                    # Magnitudes
                    logger.info("  {}: '{}' wrt '{}'\n".format(sys_name, of, wrt))
                    logger.info('    Forward Magnitude : {:.6e}'.format(magnitude.forward))
                    if not totals:
                        txt = '    Reverse Magnitude : {:.6e}'
                        logger.info(txt.format(magnitude.reverse))
                    logger.info('         Fd Magnitude : {:.6e} ({})\n'.format(magnitude.fd,
                                                                               fd_desc))
                    # Absolute Errors
                    if totals:
                        error_descs = ('(Jfor  - Jfd) ', )
                    else:
                        error_descs = ('(Jfor  - Jfd) ', '(Jrev  - Jfd) ', '(Jfor  - Jrev)')
                    for error, desc in zip(abs_err, error_descs):
                        error_str = _format_error(error, abs_error_tol)
                        logger.info('    Absolute Error {}: {}'.format(desc, error_str))
                    logger.info('')

                    # Relative Errors
                    for error, desc in zip(rel_err, error_descs):
                        error_str = _format_error(error, rel_error_tol)
                        logger.info('    Relative Error {}: {}'.format(desc, error_str))
                    logger.info('')

                    # Raw Derivatives
                    logger.info('    Raw Forward Derivative (Jfor)\n')
                    logger.info(str(forward))
                    logger.info('')

                    if not totals:
                        logger.info('    Raw Reverse Derivative (Jfor)\n')
                        logger.info(str(reverse))
                        logger.info('')

                    logger.info('    Raw FD Derivative (Jfd)\n')
                    logger.info(str(fd))
                    logger.info('')

                    logger.info(' -' * 30)


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
