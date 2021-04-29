"""Define the ImplicitComponent class."""

import numpy as np

from openmdao.core.component import Component
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.class_util import overrides_method

_inst_functs = ['apply_linear', 'apply_multi_linear', 'solve_multi_linear']


class ImplicitComponent(Component):
    """
    Class to inherit from when all output variables are implicit.

    Attributes
    ----------
    _inst_functs : dict
        Dictionary of names mapped to bound methods.
    """

    def __init__(self, **kwargs):
        """
        Store some bound methods so we can detect runtime overrides.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
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
        new_apply_multi_linear = getattr(self, 'apply_multi_linear', None)
        new_solve_multi_linear = getattr(self, 'solve_multi_linear', None)

        self.matrix_free = (overrides_method('apply_linear', self, ImplicitComponent) or
                            (new_apply_linear is not None and
                             self._inst_functs['apply_linear'] != new_apply_linear))
        self.has_apply_multi_linear = (overrides_method('apply_multi_linear',
                                                        self, ImplicitComponent) or
                                       (new_apply_multi_linear is not None and
                                        self._inst_functs['apply_multi_linear'] !=
                                        new_apply_multi_linear))
        self.has_solve_multi_linear = (overrides_method('solve_multi_linear',
                                                        self, ImplicitComponent) or
                                       (new_solve_multi_linear is not None and
                                        self._inst_functs['solve_multi_linear'] !=
                                        new_solve_multi_linear))

        self.supports_multivecs = self.has_apply_multi_linear or self.has_solve_multi_linear
        self.matrix_free |= self.has_apply_multi_linear

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

    def _apply_linear(self, jac, vec_names, rel_systems, mode, scope_out=None, scope_in=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use assembled jacobian jac.
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.
        mode : str
            'fwd' or 'rev'.
        scope_out : set or None
            Set of absolute output names in the scope of this mat-vec product.
            If None, all are in the scope.
        scope_in : set or None
            Set of absolute input names in the scope of this mat-vec product.
            If None, all are in the scope.
        """
        if jac is None:
            jac = self._assembled_jac if self._assembled_jac is not None else self._jacobian

        for vec_name in vec_names:
            if vec_name not in self._rel_vec_names:
                continue

            with self._matvec_context(vec_name, scope_out, scope_in, mode) as vecs:
                d_inputs, d_outputs, d_residuals = vecs

                # Jacobian and vectors are all scaled, unitless
                jac._apply(self, d_inputs, d_outputs, d_residuals, mode)

                # if we're not matrix free, we can skip the bottom of
                # this loop because apply_linear does nothing.
                if not self.matrix_free:
                    continue

                # Jacobian and vectors are all unscaled, dimensional
                with self._unscaled_context(
                        outputs=[self._outputs, d_outputs], residuals=[d_residuals]):

                    # set appropriate vectors to read_only to help prevent user error
                    if mode == 'fwd':
                        d_inputs.read_only = d_outputs.read_only = True
                    elif mode == 'rev':
                        d_residuals.read_only = True

                    try:
                        if d_inputs._ncol > 1:
                            if self.has_apply_multi_linear:
                                with self._call_user_function('apply_multi_linear',
                                                              protect_outputs=True):
                                    self.apply_multi_linear(self._inputs, self._outputs,
                                                            d_inputs, d_outputs, d_residuals, mode)
                            else:
                                with self._call_user_function('apply_linear',
                                                              protect_outputs=True):
                                    for i in range(d_inputs._ncol):
                                        # need to make the multivecs look like regular single vecs
                                        # since the component doesn't know about multivecs.
                                        d_inputs._icol = i
                                        d_outputs._icol = i
                                        d_residuals._icol = i
                                        self._apply_linear_wrapper(self._inputs, self._outputs,
                                                                   d_inputs, d_outputs, d_residuals,
                                                                   mode)
                                d_inputs._icol = None
                                d_outputs._icol = None
                                d_residuals._icol = None
                        else:
                            with self._call_user_function('apply_linear', protect_outputs=True):
                                self._apply_linear_wrapper(self._inputs, self._outputs,
                                                           d_inputs, d_outputs, d_residuals, mode)
                    finally:
                        d_inputs.read_only = d_outputs.read_only = d_residuals.read_only = False

    def _solve_linear(self, vec_names, mode, rel_systems):
        """
        Apply inverse jac product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.
        """
        if self._linear_solver is not None:
            self._linear_solver.solve(vec_names, mode, rel_systems)

        else:
            failed = False
            for vec_name in vec_names:
                if vec_name not in self._rel_vec_names:
                    continue
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]

                with self._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
                    # set appropriate vectors to read_only to help prevent user error
                    if mode == 'fwd':
                        d_residuals.read_only = True
                    elif mode == 'rev':
                        d_outputs.read_only = True

                    try:
                        if d_outputs._ncol > 1:
                            if self.has_solve_multi_linear:
                                with self._call_user_function('solve_multi_linear'):
                                    self.solve_multi_linear(d_outputs, d_residuals, mode)
                            else:
                                with self._call_user_function('solve_linear'):
                                    for i in range(d_outputs._ncol):
                                        # need to make the multivecs look like regular single vecs
                                        # since the component doesn't know about multivecs.
                                        d_outputs._icol = i
                                        d_residuals._icol = i
                                        self.solve_linear(d_outputs, d_residuals, mode)

                                    d_outputs._icol = None
                                    d_residuals._icol = None
                        else:
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
        sub_do_ln : boolean
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

    def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs=None,
                        discrete_outputs=None):
        """
        Compute residuals given inputs and outputs.

        The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
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
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
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
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
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
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        d_inputs : Vector
            see inputs; product must be computed only if var_name in d_inputs
        d_outputs : Vector
            see outputs; product must be computed only if var_name in d_outputs
        d_residuals : Vector
            see outputs
        mode : str
            either 'fwd' or 'rev'
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
            unscaled, dimensional quantities read via d_outputs[key]
        d_residuals : Vector
            unscaled, dimensional quantities read via d_residuals[key]
        mode : str
            either 'fwd' or 'rev'
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
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        jacobian : Jacobian
            sub-jac components written to jacobian[output_name, input_name]
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
