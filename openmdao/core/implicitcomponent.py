"""Define the ImplicitComponent class."""

from openmdao.core.component import Component
from openmdao.core.constants import _UNDEFINED
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.class_util import overrides_method


_inst_functs = ['apply_linear']


class ImplicitComponent(Component):
    """
    Class to inherit from when all output variables are implicit.

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Component options.

    Attributes
    ----------
    _inst_functs : dict
        Dictionary of names mapped to bound methods.
    """

    def __init__(self, **kwargs):
        """
        Store some bound methods so we can detect runtime overrides.
        """
        super().__init__(**kwargs)

        self._inst_functs = {name: getattr(self, name, None) for name in _inst_functs}

    def _configure(self):
        """
        Configure this system to assign children settings.

        Also tag component if it provides a guess_nonlinear.
        """
        self._has_guess = overrides_method('guess_nonlinear', self, ImplicitComponent)

        new_apply_linear = getattr(self, 'apply_linear', None)

        self.matrix_free = (overrides_method('apply_linear', self, ImplicitComponent) or
                            (new_apply_linear is not None and
                             self._inst_functs['apply_linear'] != new_apply_linear))

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        with self._unscaled_context(outputs=[self._outputs], residuals=[self._residuals]):
            with self._call_user_function('apply_nonlinear', protect_outputs=True):
                args = [self._inputs, self._outputs, self._residuals]
                if self._discrete_inputs or self._discrete_outputs:
                    args += [self._discrete_inputs, self._discrete_outputs]

                if self._run_root_only():
                    if self.comm.rank == 0:
                        self.apply_nonlinear(*args)
                        self.comm.bcast([self._residuals.asarray(), self._discrete_outputs], root=0)
                    else:
                        new_res, new_disc_outs = self.comm.bcast(None, root=0)
                        self._residuals.set_val(new_res)
                        if new_disc_outs:
                            for name, val in new_disc_outs.items():
                                self._discrete_outputs[name] = val
                else:
                    self.apply_nonlinear(*args)

        self.iter_count_apply += 1

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.
        """
        if self._nonlinear_solver is not None:
            with Recording(self.pathname + '._solve_nonlinear', self.iter_count, self):
                self._nonlinear_solver.solve()
        else:
            with self._unscaled_context(outputs=[self._outputs]):
                with Recording(self.pathname + '._solve_nonlinear', self.iter_count, self):
                    with self._call_user_function('solve_nonlinear'):
                        args = [self._inputs, self._outputs]
                        if self._discrete_inputs or self._discrete_outputs:
                            args += [self._discrete_inputs, self._discrete_outputs]
                        if self._run_root_only():
                            if self.comm.rank == 0:
                                self.solve_nonlinear(*args)
                                self.comm.bcast([self._outputs.asarray(), self._discrete_outputs],
                                                root=0)
                            else:
                                new_res, new_disc_outs = self.comm.bcast(None, root=0)
                                self._outputs.set_val(new_res)
                                if new_disc_outs:
                                    for name, val in new_disc_outs.items():
                                        self._discrete_outputs[name] = val
                        else:
                            self.solve_nonlinear(*args)

        # Iteration counter is incremented in the Recording context manager at exit.

    def _guess_nonlinear(self):
        """
        Provide initial guess for states.
        """
        if self._has_guess:
            self._apply_nonlinear()
            complex_step = self._inputs._under_complex_step

            try:
                with self._unscaled_context(outputs=[self._outputs], residuals=[self._residuals]):
                    if complex_step:
                        self._inputs.set_complex_step_mode(False)
                        self._outputs.set_complex_step_mode(False)
                        self._residuals.set_complex_step_mode(False)

                    with self._call_user_function('guess_nonlinear', protect_residuals=True):
                        if self._discrete_inputs or self._discrete_outputs:
                            self.guess_nonlinear(self._inputs, self._outputs, self._residuals,
                                                 self._discrete_inputs, self._discrete_outputs)
                        else:
                            self.guess_nonlinear(self._inputs, self._outputs, self._residuals)
            finally:
                if complex_step:
                    self._inputs.set_complex_step_mode(True)
                    self._outputs.set_complex_step_mode(True)
                    self._residuals.set_complex_step_mode(True)

    def _apply_linear_wrapper(self, *args):
        """
        Call apply_linear based on the value of the "run_root_only" option.

        Parameters
        ----------
        *args : list
            List of positional arguments.
        """
        inputs, outputs, d_inputs, d_outputs, d_residuals, mode = args
        if self._run_root_only():
            if self.comm.rank == 0:
                self.apply_linear(inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
                if mode == 'fwd':
                    self.comm.bcast(d_residuals.asarray(), root=0)
                else:  # rev
                    self.comm.bcast((d_inputs.asarray(), d_outputs.asarray()), root=0)
            else:
                if mode == 'fwd':
                    new_res = self.comm.bcast(None, root=0)
                    d_residuals.set_val(new_res)
                else:  # rev
                    new_ins, new_outs = self.comm.bcast(None, root=0)
                    d_inputs.set_val(new_ins)
                    d_outputs.set_val(new_outs)
        else:
            self.apply_linear(inputs, outputs, d_inputs, d_outputs, d_residuals, mode)

    def _apply_linear(self, jac, rel_systems, mode, scope_out=None, scope_in=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use assembled jacobian jac.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.
        mode : str
            Either 'fwd' or 'rev'.
        scope_out : set or None
            Set of absolute output names in the scope of this mat-vec product.
            If None, all are in the scope.
        scope_in : set or None
            Set of absolute input names in the scope of this mat-vec product.
            If None, all are in the scope.
        """
        if jac is None:
            jac = self._assembled_jac if self._assembled_jac is not None else self._jacobian

        with self._matvec_context(scope_out, scope_in, mode) as vecs:
            d_inputs, d_outputs, d_residuals = vecs

            # Jacobian and vectors are all scaled, unitless
            jac._apply(self, d_inputs, d_outputs, d_residuals, mode)

            # if we're not matrix free, we can skip the bottom of
            # this loop because apply_linear does nothing.
            if not self.matrix_free:
                return

            # Jacobian and vectors are all unscaled, dimensional
            with self._unscaled_context(
                    outputs=[self._outputs, d_outputs], residuals=[d_residuals]):

                # set appropriate vectors to read_only to help prevent user error
                if mode == 'fwd':
                    d_inputs.read_only = d_outputs.read_only = True
                elif mode == 'rev':
                    d_residuals.read_only = True

                try:
                    with self._call_user_function('apply_linear', protect_outputs=True):
                        self._apply_linear_wrapper(self._inputs, self._outputs,
                                                   d_inputs, d_outputs, d_residuals, mode)
                finally:
                    d_inputs.read_only = d_outputs.read_only = d_residuals.read_only = False

    def _solve_linear(self, mode, rel_systems, scope_out=_UNDEFINED, scope_in=_UNDEFINED):
        """
        Apply inverse jac product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.
        scope_out : set, None, or _UNDEFINED
            Outputs relevant to possible lower level calls to _apply_linear on Components.
        scope_in : set, None, or _UNDEFINED
            Inputs relevant to possible lower level calls to _apply_linear on Components.
        """
        if self._linear_solver is not None:
            self._linear_solver._set_matvec_scope(scope_out, scope_in)
            self._linear_solver.solve(mode, rel_systems)

        else:
            d_outputs = self._doutputs
            d_residuals = self._dresiduals

            with self._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
                # set appropriate vectors to read_only to help prevent user error
                if mode == 'fwd':
                    d_residuals.read_only = True
                elif mode == 'rev':
                    d_outputs.read_only = True

                try:
                    with self._call_user_function('solve_linear'):
                        self.solve_linear(d_outputs, d_residuals, mode)
                finally:
                    d_outputs.read_only = d_residuals.read_only = False

    def _approx_subjac_keys_iter(self):
        for abs_key, meta in self._subjacs_info.items():
            if 'method' in meta:
                method = meta['method']
                if method is not None and method in self._approx_schemes:
                    yield abs_key

    def _linearize_wrapper(self):
        """
        Call linearize based on the value of the "run_root_only" option.
        """
        with self._call_user_function('linearize', protect_outputs=True):
            args = [self._inputs, self._outputs, self._jacobian]
            if self._discrete_inputs or self._discrete_outputs:
                args += [self._discrete_inputs, self._discrete_outputs]

            if self._run_root_only():
                if self.comm.rank == 0:
                    self.linearize(*args)
                    self.comm.bcast(list(self._jacobian.items()), root=0)
                else:
                    for key, val in self.comm.bcast(None, root=0):
                        self._jacobian[key] = val
            else:
                self.linearize(*args)

    def _linearize(self, jac=None, sub_do_ln=True):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use assembled jacobian jac.
        sub_do_ln : bool
            Flag indicating if the children should call linearize on their linear solvers.
        """
        self._check_first_linearize()

        with self._unscaled_context(outputs=[self._outputs]):
            # Computing the approximation before the call to compute_partials allows users to
            # override FD'd values.
            for approximation in self._approx_schemes.values():
                approximation.compute_approximations(self, jac=self._jacobian)

            self._linearize_wrapper()

        if (jac is None or jac is self._assembled_jac) and self._assembled_jac is not None:
            self._assembled_jac._update(self)

    def add_output(self, name, val=1.0, **kwargs):
        """
        Add an output variable to the component.

        Parameters
        ----------
        name : str
            Name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        **kwargs : dict
            Keyword args to store.  The value corresponding to each key is a dict containing the
            metadata for the input name that matches that key.

        Returns
        -------
        dict
            Metadata for added variable.
        """
        metadata = super().add_output(name, val, **kwargs)

        metadata['tags'].add('openmdao:allow_desvar')

        return metadata

    def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs=None,
                        discrete_outputs=None):
        """
        Compute residuals given inputs and outputs.

        The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        residuals : Vector
            Unscaled, dimensional residuals written to via residuals[key].
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        discrete_outputs : dict or None
            If not None, dict containing discrete output values.
        """
        pass

    def solve_nonlinear(self, inputs, outputs):
        """
        Compute outputs given inputs. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        """
        pass

    def guess_nonlinear(self, inputs, outputs, residuals,
                        discrete_inputs=None, discrete_outputs=None):
        """
        Provide initial guess for states.

        Override this method to set the initial guess for states.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        residuals : Vector
            Unscaled, dimensional residuals written to via residuals[key].
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        discrete_outputs : dict or None
            If not None, dict containing discrete output values.
        """
        pass

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        r"""
        Compute jac-vector product. The model is assumed to be in an unscaled state.

        If mode is:
            'fwd': (d_inputs, d_outputs) \|-> d_residuals

            'rev': d_residuals \|-> (d_inputs, d_outputs)

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        d_inputs : Vector
            See inputs; product must be computed only if var_name in d_inputs.
        d_outputs : Vector
            See outputs; product must be computed only if var_name in d_outputs.
        d_residuals : Vector
            See outputs.
        mode : str
            Either 'fwd' or 'rev'.
        """
        pass

    def solve_linear(self, d_outputs, d_residuals, mode):
        r"""
        Apply inverse jac product. The model is assumed to be in an unscaled state.

        If mode is:
            'fwd': d_residuals \|-> d_outputs

            'rev': d_outputs \|-> d_residuals

        Note : this is not the linear solution for the implicit component. We use identity so
        that simple implicit components can function in a preconditioner under linear gauss-seidel.
        To correctly solve this component, you should slot a solver in linear_solver or override
        this method.

        Parameters
        ----------
        d_outputs : Vector
            Unscaled, dimensional quantities read via d_outputs[key].
        d_residuals : Vector
            Unscaled, dimensional quantities read via d_residuals[key].
        mode : str
            Either 'fwd' or 'rev'.
        """
        if mode == 'fwd':
            d_outputs.set_vec(d_residuals)
        else:  # rev
            d_residuals.set_vec(d_outputs)

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):
        """
        Compute sub-jacobian parts and any applicable matrix factorizations.

        The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        jacobian : Jacobian
            Sub-jac components written to jacobian[output_name, input_name].
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        discrete_outputs : dict or None
            If not None, dict containing discrete output values.
        """
        pass

    def _list_states(self):
        """
        Return list of all states at and below this system.

        If final setup has not been performed yet, return relative names for this system only.

        Returns
        -------
        list
            List of all states.
        """
        prefix = self.pathname + '.' if self.pathname else ''
        return sorted(list(self._var_allprocs_abs2meta['output']) +
                      [prefix + n for n in self._var_discrete['output']])

    def _list_states_allprocs(self):
        """
        Return list of all states for this component.

        Returns
        -------
        list
            List of all states.
        """
        return self._list_states()
