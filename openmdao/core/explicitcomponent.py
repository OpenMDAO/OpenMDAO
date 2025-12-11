"""Define the ExplicitComponent class."""

from itertools import chain
from types import MethodType

from openmdao.jacobians.dictionary_jacobian import ExplicitDictionaryJacobian
from openmdao.jacobians.jacobian import JacobianUpdateContext
from openmdao.utils.coloring import _ColSparsityJac
from openmdao.core.component import Component
from openmdao.vectors.vector import _full_slice
from openmdao.utils.class_util import overrides_method
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.core.constants import _UNDEFINED
from openmdao.utils.general_utils import is_undefined


_tuplist = (tuple, list)


class ExplicitComponent(Component):
    """
    Class to inherit from when all output variables are explicit.

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Component options.

    Attributes
    ----------
    _has_compute_partials : bool
        If True, the instance overrides compute_partials.
    _vjp_hash : int or None
        Hash value for the last set of inputs to the compute_primal function.
    _vjp_fun : function or None
        The vector-Jacobian product function.
    """

    def __init__(self, **kwargs):
        """
        Store some bound methods so we can detect runtime overrides.
        """
        super().__init__(**kwargs)

        self._has_compute_partials = overrides_method('compute_partials', self, ExplicitComponent)
        self.options.undeclare('assembled_jac_type')
        self._vjp_hash = None
        self._vjp_fun = None

    @property
    def nonlinear_solver(self):
        """
        Get the nonlinear solver for this system.
        """
        return self._nonlinear_solver

    @nonlinear_solver.setter
    def nonlinear_solver(self, solver):
        """
        Raise an exception.
        """
        raise RuntimeError(f"{self.msginfo}: Explicit components don't support nonlinear solvers.")

    @property
    def linear_solver(self):
        """
        Get the linear solver for this system.
        """
        return self._linear_solver

    @linear_solver.setter
    def linear_solver(self, solver):
        """
        Raise an exception.
        """
        raise RuntimeError(f"{self.msginfo}: Explicit components don't support linear solvers.")

    def _configure(self):
        """
        Configure this system to assign children settings and detect if matrix_free.
        """
        if is_undefined(self.matrix_free):
            self.matrix_free = overrides_method('compute_jacvec_product', self, ExplicitComponent)

    def override_method(self, name, method):
        """
        Dynamically add a method to this component instance.

        This allows users to create an `ExplicitComponent` that has a
        `compute_partials` or `compute_jacvec_product` that isn't defined
        statically, but instead is dynamically created during `setup`. The
        motivating use case is the `omjlcomps` library, where the
        `compute_partials` or `compute_jacvec_product` methods are implemented
        in the Julia programming language (see `omjlcomps.JuliaExplicitComp` in
        byuflowlab/OpenMDAO.jl).

        Parameters
        ----------
        name : str
            The name of the method to add. Must be either 'compute_partials' or
            'compute_jacvec_product'.
        method : function
            The function to add as a method. Will be converted to a MethodType if necessary.

        Raises
        ------
        ValueError
            If name is not 'compute_partials' or 'compute_jacvec_product'.
        """
        if name not in ('compute_partials', 'compute_jacvec_product'):
            raise ValueError(f"{self.msginfo}: name must be either 'compute_partials' or "
                             f"'compute_jacvec_product', but got '{name}'.")

        # Convert to MethodType if not already a bound method
        if not isinstance(method, MethodType):
            method = MethodType(method, self)

        # Set the method on the instance
        setattr(self, name, method)

        # Update the appropriate attribute
        if name == 'compute_partials':
            self._has_compute_partials = True
        elif name == 'compute_jacvec_product':
            self.matrix_free = True

    def _jac_wrt_iter(self, wrt_matches=None):
        """
        Iterate over (name, start, end, vec, slice, dist_sizes) for each column var in the jacobian.

        Parameters
        ----------
        wrt_matches : set or None
            Only include row vars that are contained in this set.  This will determine what
            the actual offsets are, i.e. the offsets will be into a reduced jacobian
            containing only the matching columns.

        Yields
        ------
        str
            Absolute name of 'wrt' variable.
        int
            Starting index.
        int
            Ending index.
        Vector
            The _inputs vector.
        slice
            A full slice.  In the total derivative version of this function, which only is called
            on the top level Group, this may not be a full slice, but rather indices specified
            for the corresponding design variable.
        ndarray or None
            Distributed sizes if var is distributed else None
        """
        start = end = 0
        toidx = self._var_allprocs_abs2idx
        sizes = self._var_sizes['input']
        for wrt, meta in self._var_abs2meta['input'].items():
            if wrt_matches is None or wrt in wrt_matches:
                end += meta['size']
                dist_sizes = sizes[:, toidx[wrt]] if meta['distributed'] else None
                yield wrt, start, end, self._inputs, _full_slice, dist_sizes
                start = end

    def _setup_residuals(self):
        """
        Prevent the user from implementing setup_residuals for explicit components.
        """
        if overrides_method('setup_residuals', self, ExplicitComponent):
            raise RuntimeError(f'{self.msginfo}: Class overrides setup_residuals but '
                               'is an ExplicitComponent. setup_residuals may only be '
                               'overridden by ImplicitComponents.')

    def _setup_partials(self):
        """
        Call setup_partials in components.
        """
        super()._setup_partials()

        if self.matrix_free:
            return

        # Note: These declare calls are outside of setup_partials so that users do not have to
        # call the super version of setup_partials. This is still in the final setup.
        for out_abs, meta in self._var_abs2meta['output'].items():

            size = meta['size']
            if size > 0:
                # ExplicitComponent jacobians have -1 on the diagonal.
                self._subjacs_info[out_abs, out_abs] = {
                    'rows': None,
                    'cols': None,
                    'diagonal': True,
                    'shape': (size, size),
                    'val': -1.0,
                    'dependent': True,
                }

    def _get_jacobian(self, use_relevance=True):
        """
        Initialize the jacobian if it is not already initialized.

        Parameters
        ----------
        use_relevance : bool
            If True, use relevance to determine which partials to approximate.

        Returns
        -------
        Jacobian
            The initialized jacobian.
        """
        if self._relevance_changed() and not isinstance(self._jacobian, _ColSparsityJac):
            self._jacobian = None

        if not self.matrix_free and self._jacobian is None:
            self._jacobian = ExplicitDictionaryJacobian(self)
            if self._has_approx:
                self._get_static_wrt_matches()
                self._add_approximations(use_relevance=use_relevance)

        return self._jacobian

    def add_output(self, name, val=1.0, shape=None, units=None, res_units=None, desc='',
                   lower=None, upper=None, ref=1.0, ref0=0.0, res_ref=None, tags=None,
                   shape_by_conn=False, copy_shape=None, compute_shape=None,
                   units_by_conn=False, compute_units=None, copy_units=None,
                   distributed=None, primal_name=None):
        """
        Add an output variable to the component.

        For ExplicitComponent, res_ref defaults to the value in res unless otherwise specified.

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
        lower : float or list or tuple or ndarray or None
            Lower bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no lower bound.
            Default is None.
        upper : float or list or tuple or ndarray or None
            Upper bound(s) in user-defined units. It can be (1) a float, (2) an array_like
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
            when the scaled value is 1. Default is None, which means residual scaling matches
            output scaling.
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed when calling
            list_inputs and list_outputs and also when listing results from case recorders.
        shape_by_conn : bool
            If True, shape this output to match its connected input(s).
        copy_shape : str or None
            If a str, that str is the name of a variable. Shape this output to match that of
            the named variable.
        compute_shape : function or None
            If a function, that function is called to determine the shape of this output.
        units_by_conn : bool
            If True, units are computed by the connected input(s).
        compute_units : function or None
            If a function, that function is called to determine the units of this output.
        copy_units : str or None
            If a str, that str is the name of a variable. Units this output to match that of
            the named variable.
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
        if res_ref is None:
            res_ref = ref

        return super().add_output(name, val=val, shape=shape, units=units,
                                  res_units=res_units, desc=desc,
                                  lower=lower, upper=upper,
                                  ref=ref, ref0=ref0, res_ref=res_ref,
                                  tags=tags, shape_by_conn=shape_by_conn,
                                  copy_shape=copy_shape, compute_shape=compute_shape,
                                  units_by_conn=units_by_conn, compute_units=compute_units,
                                  copy_units=copy_units,
                                  distributed=distributed, primal_name=primal_name)

    def _approx_subjac_keys_iter(self):
        is_input = self._inputs._contains_abs
        for abs_key, meta in self._subjacs_info.items():
            if 'method' in meta and is_input(abs_key[1]):
                method = meta['method']
                if method in self._approx_schemes:
                    yield abs_key

    def _compute_wrapper(self):
        """
        Call compute based on the value of the "run_root_only" option.
        """
        with self._call_user_function('compute'):
            if self._run_root_only():
                if self.comm.rank == 0:
                    if self._discrete_inputs or self._discrete_outputs:
                        self.compute(self._inputs, self._outputs,
                                     self._discrete_inputs, self._discrete_outputs)
                    else:
                        self.compute(self._inputs, self._outputs)
                    self.comm.bcast([self._outputs.asarray(), self._discrete_outputs], root=0)
                else:
                    new_outs, new_disc_outs = self.comm.bcast(None, root=0)
                    self._outputs.set_val(new_outs)
                    if new_disc_outs:
                        for name, val in new_disc_outs.items():
                            self._discrete_outputs[name] = val
            else:
                if self._discrete_inputs or self._discrete_outputs:
                    self.compute(self._inputs, self._outputs,
                                 self._discrete_inputs, self._discrete_outputs)
                else:
                    self.compute(self._inputs, self._outputs)

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        outputs = self._outputs
        residuals = self._residuals
        with self._unscaled_context(outputs=[outputs], residuals=[residuals]):
            residuals.set_vec(outputs)

            # Sign of the residual is minus the sign of the output vector.
            residuals *= -1.0
            self._compute_wrapper()
            residuals += outputs
            outputs -= residuals

        self.iter_count_apply += 1

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.
        """
        with Recording(self.pathname + '._solve_nonlinear', self.iter_count, self):
            self._residuals.set_val(0.0)
            with self._unscaled_context(outputs=[self._outputs]):
                self._compute_wrapper()

            # Iteration counter is incremented in the Recording context manager at exit.

    def _compute_jacvec_product_wrapper(self, inputs, d_inputs, d_resids, mode,
                                        discrete_inputs=None):
        """
        Call compute_jacvec_product based on the value of the "run_root_only" option.

        Parameters
        ----------
        inputs : Vector
            Nonlinear input vector.
        d_inputs : Vector
            Linear input vector.
        d_resids : Vector
            Linear residual vector.
        mode : str
            Indicates direction of derivative computation, either 'fwd' or 'rev'.
        discrete_inputs : dict or None
            Mapping of variable name to discrete value.
        """
        if self._run_root_only():
            if self.comm.rank == 0:
                if discrete_inputs:
                    self.compute_jacvec_product(inputs, d_inputs, d_resids, mode, discrete_inputs)
                else:
                    self.compute_jacvec_product(inputs, d_inputs, d_resids, mode)
                if mode == 'fwd':
                    self.comm.bcast(d_resids.asarray(), root=0)
                else:  # rev
                    self.comm.bcast(d_inputs.asarray(), root=0)
            else:
                new_vals = self.comm.bcast(None, root=0)
                if mode == 'fwd':
                    d_resids.set_val(new_vals)
                else:  # rev
                    d_inputs.set_val(new_vals)
        else:
            dochk = self._problem_meta['checking'] and mode == 'rev' and self.comm.size > 1

            if dochk:
                nzdresids = self._get_dist_nz_dresids()

            if discrete_inputs:
                self.compute_jacvec_product(inputs, d_inputs, d_resids, mode, discrete_inputs)
            else:
                self.compute_jacvec_product(inputs, d_inputs, d_resids, mode)

            if dochk:
                self._check_consistent_serial_dinputs(nzdresids)

    def _apply_linear(self, mode, scope_out=None, scope_in=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        mode : str
            'fwd' or 'rev'.
        scope_out : set or None
            Set of absolute output names in the scope of this mat-vec product.
            If None, all are in the scope.
        scope_in : set or None
            Set of absolute input names in the scope of this mat-vec product.
            If None, all are in the scope.
        """
        with self._matvec_context(scope_out, scope_in, mode) as vecs:
            d_inputs, d_outputs, d_residuals = vecs

            if not self.matrix_free:
                # Jacobian and vectors are all scaled, unitless
                self._get_jacobian()._apply(self, d_inputs, d_outputs, d_residuals, mode)
                return

            # Jacobian and vectors are all unscaled, dimensional
            with self._unscaled_context(outputs=[self._outputs], residuals=[d_residuals]):

                # set appropriate vectors to read_only to help prevent user error
                if mode == 'fwd':
                    d_inputs.read_only = True
                else:  # rev
                    d_residuals.read_only = True

                try:
                    # handle identity subjacs (output_or_resid wrt itself)
                    if d_outputs._names:
                        get_dresid = d_residuals._abs_get_val
                        get_doutput = d_outputs._abs_get_val

                        # 'val' in the code below is a reference to the part of the
                        # output or residual array corresponding to the variable 'v'
                        if mode == 'fwd':
                            for v in d_outputs._names:
                                if (v, v) not in self._subjacs_info:
                                    val = get_dresid(v)
                                    val -= get_doutput(v)
                        else:  # rev
                            for v in d_outputs._names:
                                if (v, v) not in self._subjacs_info:
                                    val = get_doutput(v)
                                    val -= get_dresid(v)

                    # We used to negate the residual here, and then re-negate after the hook
                    with self._call_user_function('compute_jacvec_product'):
                        self._compute_jacvec_product_wrapper(self._inputs, d_inputs, d_residuals,
                                                             mode, self._discrete_inputs)
                finally:
                    d_inputs.read_only = d_residuals.read_only = False

    def _solve_linear(self, mode, scope_out=_UNDEFINED, scope_in=_UNDEFINED):
        """
        Apply inverse jac product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        mode : str
            'fwd' or 'rev'.
        scope_out : set, None, or _UNDEFINED
            Outputs relevant to possible lower level calls to _apply_linear on Components.
        scope_in : set, None, or _UNDEFINED
            Inputs relevant to possible lower level calls to _apply_linear on Components.
        """
        d_outputs = self._doutputs
        d_residuals = self._dresiduals

        if mode == 'fwd':
            if self._has_resid_scaling:
                with self._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
                    d_outputs.set_vec(d_residuals)
            else:
                d_outputs.set_vec(d_residuals)

            # ExplicitComponent jacobian defined with -1 on diagonal.
            d_outputs *= -1.0

        else:  # rev
            if self._has_resid_scaling:
                with self._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
                    d_residuals.set_vec(d_outputs)
            else:
                d_residuals.set_vec(d_outputs)

            # ExplicitComponent jacobian defined with -1 on diagonal.
            d_residuals *= -1.0

    def _compute_partials_wrapper(self, jac):
        """
        Call compute_partials based on the value of the "run_root_only" option.
        """
        with self._call_user_function('compute_partials'):
            if self._run_root_only():
                if self.comm.rank == 0:
                    if self._discrete_inputs:
                        self.compute_partials(self._inputs, jac, self._discrete_inputs)
                    else:
                        self.compute_partials(self._inputs, jac)
                    self.comm.bcast(list(jac.items()), root=0)
                else:
                    for key, val in self.comm.bcast(None, root=0):
                        jac[key] = val
            else:
                if self._discrete_inputs:
                    self.compute_partials(self._inputs, jac, self._discrete_inputs)
                else:
                    self.compute_partials(self._inputs, jac)

    def _linearize(self, sub_do_ln=False):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        sub_do_ln : bool
            Ignored.
        """
        self._check_first_linearize()

        if self.matrix_free or not (self._has_compute_partials or self._has_approx):
            return

        with JacobianUpdateContext(self) as jac:

            with self._unscaled_context(outputs=[self._outputs], residuals=[self._residuals]):
                # Computing the approximation before the call to compute_partials allows users to
                # override FD'd values.
                for approximation in self._approx_schemes.values():
                    approximation.compute_approximations(self, jac=jac)

                if self._has_compute_partials:
                    # We used to negate the jacobian here, and then re-negate after the hook.
                    self._compute_partials_wrapper(jac)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute outputs given inputs. The model is assumed to be in an unscaled state.

        An inherited component may choose to either override this function or to define a
        compute_primal function.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        discrete_inputs : dict-like or None
            If not None, dict-like object containing discrete input values.
        discrete_outputs : dict-like or None
            If not None, dict-like object containing discrete output values.
        """
        if self.compute_primal is None:
            return

        returns = self.compute_primal(*self._get_compute_primal_invals(inputs, discrete_inputs))

        if not isinstance(returns, _tuplist):
            returns = (returns,)

        if discrete_outputs:
            outputs.set_vals(returns[:outputs.nvars()])
            self._discrete_outputs.set_vals(returns[outputs.nvars():])
        else:
            outputs.set_vals(returns)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Sub-jac components written to partials[output_name, input_name]..
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        """
        pass

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):
        r"""
        Compute jac-vector product. The model is assumed to be in an unscaled state.

        If mode is:
            'fwd': d_inputs \|-> d_outputs

            'rev': d_outputs \|-> d_inputs

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        d_inputs : Vector
            See inputs; product must be computed only if var_name in d_inputs.
        d_outputs : Vector
            See outputs; product must be computed only if var_name in d_outputs.
        mode : str
            Either 'fwd' or 'rev'.
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        """
        pass

    def is_explicit(self, is_comp=True):
        """
        Return True if this is an explicit component.

        Parameters
        ----------
        is_comp : bool
            If True, return True if this is an explicit component.
            If False, return True if this is an explicit component or group.

        Returns
        -------
        bool
            True if this is an explicit component.
        """
        return True

    def _get_compute_primal_invals(self, inputs=None, discrete_inputs=None):
        """
        Yield the inputs expected by the compute_primal method.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables Vector.
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.

        Yields
        ------
        any
            Inputs expected by the compute_primal method.
        """
        if inputs is None:
            inputs = self._inputs
        if discrete_inputs is None:
            discrete_inputs = self._discrete_inputs

        yield from inputs.values()
        if discrete_inputs:
            yield from discrete_inputs.values()

    def _get_compute_primal_argnames(self):
        """
        Return the expected argnames for the compute_primal method.

        Returns
        -------
        list
            List of argnames expected by the compute_primal method.
        """
        if self._valid_name_map:
            return [self._valid_name_map.get(n, n) for n in chain(self._var_rel_names['input'],
                                                                  self._discrete_inputs)]
        else:
            return list(chain(self._var_rel_names['input'], self._discrete_inputs))

    def compute_fd_sparsity(self, method='fd', num_full_jacs=2, perturb_size=1e-9):
        """
        Use finite difference to compute a sparsity matrix.

        Parameters
        ----------
        method : str
            The type of finite difference to perform. Valid options are 'fd' for forward difference,
            or 'cs' for complex step.
        num_full_jacs : int
            Number of times to repeat jacobian computation using random perturbations.
        perturb_size : float
            Size of the random perturbation.

        Returns
        -------
        coo_matrix
            The sparsity matrix.
        """
        jac = _ColSparsityJac(self)
        for _ in self._perturbation_iter(num_full_jacs, perturb_size,
                                         (self._inputs,), (self._outputs, self._residuals)):
            self._apply_nonlinear()
            self.compute_fd_jac(jac=jac, method=method)
        return jac.get_sparsity()
