"""Define the ImplicitComponent class."""

from __future__ import division

import numpy as np
from six import itervalues
from six.moves import range

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
        super(ImplicitComponent, self).__init__(**kwargs)

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
        self._inputs.read_only = self._outputs.read_only = True

        with self._unscaled_context(outputs=[self._outputs], residuals=[self._residuals]):
            with Recording(self.pathname + '._apply_nonlinear', self.iter_count, self):
                self.apply_nonlinear(self._inputs, self._outputs, self._residuals)

        self._inputs.read_only = self._outputs.read_only = False

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        # Reconfigure if needed.
        super(ImplicitComponent, self)._solve_nonlinear()

        self._inputs.read_only = True

        if self._nonlinear_solver is not None:
            with Recording(self.pathname + '._solve_nonlinear', self.iter_count, self):
                result = self._nonlinear_solver.solve()
        else:
            with self._unscaled_context(outputs=[self._outputs]):
                with Recording(self.pathname + '._solve_nonlinear', self.iter_count, self):
                    result = self.solve_nonlinear(self._inputs, self._outputs)

        self._inputs.read_only = False

        if result is None:
            return False, 0., 0.
        elif type(result) is bool:
            return result, 0., 0.
        else:
            return result

    def _guess_nonlinear(self):
        """
        Provide initial guess for states.
        """
        with self._unscaled_context(outputs=[self._outputs], residuals=[self._residuals]):
            self.guess_nonlinear(self._inputs, self._outputs, self._residuals)

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
                with self.jacobian_context(jac):
                    jac._apply(d_inputs, d_outputs, d_residuals, mode)

                # if we're not matrix free, we can skip the bottom of
                # this loop because apply_linear does nothing.
                if not self.matrix_free:
                    continue

                # Jacobian and vectors are all unscaled, dimensional
                with self._unscaled_context(
                        outputs=[self._outputs, d_outputs], residuals=[d_residuals]):
                    with Recording(self.pathname + '._apply_linear', self.iter_count, self):
                        if d_inputs._ncol > 1:
                            if self.has_apply_multi_linear:
                                self.apply_multi_linear(self._inputs, self._outputs,
                                                        d_inputs, d_outputs, d_residuals, mode)
                            else:
                                for i in range(d_inputs._ncol):
                                    # need to make the multivecs look like regular single vecs
                                    # since the component doesn't know about multivecs.
                                    d_inputs._icol = i
                                    d_outputs._icol = i
                                    d_residuals._icol = i
                                    self.apply_linear(self._inputs, self._outputs,
                                                      d_inputs, d_outputs, d_residuals, mode)
                                d_inputs._icol = None
                                d_outputs._icol = None
                                d_residuals._icol = None
                        else:
                            self.apply_linear(self._inputs, self._outputs,
                                              d_inputs, d_outputs, d_residuals, mode)

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

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        if self._linear_solver is not None:
            with Recording(self.pathname + '._solve_linear', self.iter_count, self):
                result = self._linear_solver.solve(vec_names, mode, rel_systems)

            return result

        else:
            failed = False
            abs_errors = [0.0]
            rel_errors = [0.0]
            for vec_name in vec_names:
                if vec_name not in self._rel_vec_names:
                    continue
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]

                with self._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
                    with Recording(self.pathname + '._solve_linear', self.iter_count, self):
                        if d_outputs._ncol > 1:
                            if self.has_solve_multi_linear:
                                result = self.solve_multi_linear(d_outputs, d_residuals, mode)
                            else:
                                for i in range(d_outputs._ncol):
                                    # need to make the multivecs look like regular single vecs
                                    # since the component doesn't know about multivecs.
                                    d_outputs._icol = i
                                    d_residuals._icol = i
                                    result = self.solve_linear(d_outputs, d_residuals, mode)
                                    if isinstance(result, bool):
                                        failed |= result
                                    elif result is not None:
                                        failed = failed or result[0]
                                        abs_errors.append(result[1])
                                        rel_errors.append(result[2])

                                d_outputs._icol = None
                                d_residuals._icol = None
                        else:
                            result = self.solve_linear(d_outputs, d_residuals, mode)

                if isinstance(result, bool):
                    failed |= result
                elif result is not None:
                    failed = failed or result[0]
                    abs_errors.append(result[1])
                    rel_errors.append(result[2])

            return failed, np.linalg.norm(abs_errors), np.linalg.norm(rel_errors)

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
        with self._unscaled_context(outputs=[self._outputs]):
            # Computing the approximation before the call to compute_partials allows users to
            # override FD'd values.
            for approximation in itervalues(self._approx_schemes):
                approximation.compute_approximations(self, jac=self._jacobian)

            self._inputs.read_only = self._outputs.read_only = True

            self.linearize(self._inputs, self._outputs, self._jacobian)

            self._inputs.read_only = self._outputs.read_only = False

        if (jac is None or jac is self._assembled_jac) and self._assembled_jac is not None:
            self._assembled_jac._update(self)

    def apply_nonlinear(self, inputs, outputs, residuals):
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

        Returns
        -------
        None or bool or (bool, float, float)
            The bool is the failure flag; and the two floats are absolute and relative error.
        """
        pass

    def guess_nonlinear(self, inputs, outputs, residuals):
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
        """
        pass

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
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

        Note: this is not the linear solution for the implicit component. We use identity so
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

        Returns
        -------
        None or bool or (bool, float, float)
            The bool is the failure flag; and the two floats are absolute and relative error.
        """
        if mode == 'fwd':
            d_outputs.set_vec(d_residuals)
        else:  # rev
            d_residuals.set_vec(d_outputs)

        return False, 0., 0.

    def linearize(self, inputs, outputs, jacobian):
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
        """
        pass

    def _list_states(self):
        """
        Return list of all states at and below this system.

        Returns
        -------
        list
            List of all states.
        """
        return [name for name in self._outputs._names]

    def _list_states_allprocs(self):
        """
        Return list of all states for this component.

        Returns
        -------
        list
            List of all states.
        """
        return self._list_states()
