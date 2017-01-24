"""Define the ImplicitComponent class."""

from __future__ import division

import collections

import numpy
from six import string_types

from openmdao.core.component import Component


class ImplicitComponent(Component):
    """Class to inherit from when all output variables are implicit."""

    def _apply_nonlinear(self):
        """Compute residuals."""
        self._inputs.scale(self._scaling_to_phys['input'])
        self._outputs.scale(self._scaling_to_phys['output'])
        self._residuals.scale(self._scaling_to_phys['residual'])

        self.apply_nonlinear(self._inputs, self._outputs, self._residuals)

        self._inputs.scale(self._scaling_to_norm['input'])
        self._outputs.scale(self._scaling_to_norm['output'])
        self._residuals.scale(self._scaling_to_norm['residual'])

    def _solve_nonlinear(self):
        """Compute outputs.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        if self._nl_solver is not None:
            self._nl_solver.solve()
        else:
            self._inputs.scale(self._scaling_to_phys['input'])
            self._outputs.scale(self._scaling_to_phys['output'])
            self._residuals.scale(self._scaling_to_phys['residual'])

            self.solve_nonlinear(self._inputs, self._outputs)

            self._inputs.scale(self._scaling_to_norm['input'])
            self._outputs.scale(self._scaling_to_norm['output'])
            self._residuals.scale(self._scaling_to_norm['residual'])

    def _apply_linear(self, vec_names, mode, var_inds=None):
        """Compute jac-vec product.

        Args
        ----
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        var_inds : [int, int, int, int] or None
            ranges of variable IDs involved in this matrix-vector product.
            The ordering is [lb1, ub1, lb2, ub2].
        """
        for vec_name in vec_names:
            with self._matvec_context(vec_name, var_inds, mode) as vecs:
                d_inputs, d_outputs, d_residuals = vecs

                self._inputs.scale(self._scaling_to_phys['input'])
                self._outputs.scale(self._scaling_to_phys['output'])
                d_inputs.scale(self._scaling_to_phys['input'])
                d_outputs.scale(self._scaling_to_phys['output'])
                d_residuals.scale(self._scaling_to_phys['residual'])

                self.apply_linear(self._inputs, self._outputs,
                                  d_inputs, d_outputs, d_residuals, mode)

                self._inputs.scale(self._scaling_to_norm['input'])
                self._outputs.scale(self._scaling_to_norm['output'])
                d_inputs.scale(self._scaling_to_norm['input'])
                d_outputs.scale(self._scaling_to_norm['output'])
                d_residuals.scale(self._scaling_to_norm['residual'])

                self._jacobian._system = self
                self._jacobian._apply(d_inputs, d_outputs, d_residuals, mode)

    def _solve_linear(self, vec_names, mode):
        """Apply inverse jac product.

        Args
        ----
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        if self._ln_solver is not None:
            return self._ln_solver(vec_names, mode)
        else:
            success = True
            for vec_name in vec_names:
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]

                d_outputs.scale(self._scaling_to_phys['output'])
                d_residuals.scale(self._scaling_to_phys['residual'])

                tmp = self.solve_linear(d_outputs, d_residuals, mode)

                d_outputs.scale(self._scaling_to_norm['output'])
                d_residuals.scale(self._scaling_to_norm['residual'])

                success = success and tmp
            return success

    def _linearize(self):
        """Compute jacobian / factorization."""
        self._jacobian._system = self

        self._inputs.scale(self._scaling_to_phys['input'])
        self._outputs.scale(self._scaling_to_phys['output'])

        self.linearize(self._inputs, self._outputs, self._jacobian)

        self._inputs.scale(self._scaling_to_norm['input'])
        self._outputs.scale(self._scaling_to_norm['output'])

        if self._jacobian._top_name == self.pathname:
            self._jacobian._update()

    def apply_nonlinear(self, inputs, outputs, residuals):
        """Compute residuals given inputs and outputs.

        Args
        ----
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        pass

    def solve_nonlinear(self, inputs, outputs):
        """Compute outputs given inputs.

        Args
        ----
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        pass

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
        r"""Compute jac-vector product.

        If mode is:
            'fwd': (d_inputs, d_outputs) \|-> d_residuals

            'rev': d_residuals \|-> (d_inputs, d_outputs)

        Args
        ----
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
        r"""Apply inverse jac product.

        If mode is:
            'fwd': d_residuals \|-> d_outputs

            'rev': d_outputs \|-> d_residuals

        Args
        ----
        d_outputs : Vector
            unscaled, dimensional quantities read via d_outputs[key]
        d_residuals : Vector
            unscaled, dimensional quantities read via d_residuals[key]
        mode : str
            either 'fwd' or 'rev'
        """
        pass

    def linearize(self, inputs, outputs, jacobian):
        """Compute sub-jacobian parts / factorization.

        Args
        ----
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        jacobian : Jacobian
            sub-jac components written to jacobian[output_name, input_name]
        """
        pass
