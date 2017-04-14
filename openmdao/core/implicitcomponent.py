"""Define the ImplicitComponent class."""

from __future__ import division

import numpy as np
from six import itervalues

from openmdao.core.component import Component
from openmdao.utils.class_util import overrides_method


class ImplicitComponent(Component):
    """
    Class to inherit from when all output variables are implicit.
    """

    def __init__(self, **kwargs):
        """
        Check if we are matrix-free.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            available here and in all descendants of this system.
        """
        super(ImplicitComponent, self).__init__(**kwargs)

        if overrides_method('apply_linear', self, ImplicitComponent):
            self._matrix_free = True

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        with self._unscaled_context(
                outputs=[self._outputs], residuals=[self._residuals]):
            self.apply_nonlinear(self._inputs, self._outputs, self._residuals)

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
        if self._nl_solver is not None:
            return self._nl_solver.solve()
        else:
            with self._unscaled_context(outputs=[self._outputs]):
                result = self.solve_nonlinear(self._inputs, self._outputs)

            if result is None:
                return False, 0., 0.
            elif type(result) is bool:
                return result, 0., 0.
            else:
                return result

    def _apply_linear(self, vec_names, mode, scope_out=None, scope_in=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        scope_out : set or None
            Set of absolute output names in the scope of this mat-vec product.
            If None, all are in the scope.
        scope_in : set or None
            Set of absolute input names in the scope of this mat-vec product.
            If None, all are in the scope.
        """
        for vec_name in vec_names:
            with self._matvec_context(vec_name, scope_out, scope_in, mode) as vecs:
                d_inputs, d_outputs, d_residuals = vecs

                # Jacobian and vectors are all scaled, unitless
                with self.jacobian_context() as J:
                    J._apply(d_inputs, d_outputs, d_residuals, mode)

                # Jacobian and vectors are all unscaled, dimensional
                with self._unscaled_context(
                        outputs=[self._outputs, d_outputs], residuals=[d_residuals]):
                    self.apply_linear(self._inputs, self._outputs,
                                      d_inputs, d_outputs, d_residuals, mode)

    def _solve_linear(self, vec_names, mode):
        """
        Apply inverse jac product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        if self._ln_solver is not None:
            return self._ln_solver.solve(vec_names, mode)
        else:
            failed = False
            abs_errors = []
            rel_errors = []
            for vec_name in vec_names:
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]

                with self._unscaled_context(
                        outputs=[d_outputs], residuals=[d_residuals]):
                    result = self.solve_linear(d_outputs, d_residuals, mode)

                if result is None:
                    result = False, 0., 0.
                elif type(result) is bool:
                    result = result, 0., 0.

                failed = failed or result[0]
                abs_errors.append(result[1])
                rel_errors.append(result[2])

            return failed, np.linalg.norm(abs_errors), np.linalg.norm(rel_errors)

    def _linearize(self, do_nl=True, do_ln=True):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        do_nl : boolean
            Flag indicating if the nonlinear solver should be linearized.
        do_ln : boolean
            Flag indicating if the linear solver should be linearized.
        """
        with self.jacobian_context() as J:
            with self._unscaled_context(outputs=[self._outputs]):
                # Computing the approximation before the call to compute_partials allows users to
                # override FD'd values.
                for approximation in itervalues(self._approx_schemes):
                    approximation.compute_approximations(self, jac=J)
                self.linearize(self._inputs, self._outputs, J)

            if self._owns_assembled_jac:
                J._update()

        if self._nl_solver is not None and do_nl:
            self._nl_solver._linearize()

        if self._ln_solver is not None and do_ln:
            self._ln_solver._linearize()

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
        To correctly solve this component, you should slot a solver in ln_solver or override this
        method.

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
        elif mode == 'rev':
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
