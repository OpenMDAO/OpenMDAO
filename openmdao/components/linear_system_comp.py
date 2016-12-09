""" A component that solves a linear system. """

import numpy as np
from scipy import linalg

from openmdao.core.implicitcomponent import ImplicitComponent


class LinearSystemComp(ImplicitComponent):
    """Solve a linear system, Ax=b.

    Attributes
    ----------
    _lup : object
        lup returned from scipy.linag.lu_factor
    """

    def __init__(self, **kwargs):
        """Define additional attributes."""
        super(LinearSystemComp, self).__init__(**kwargs)

        self._lup = None

    def initialize(self):
        """Define size parameter."""
        self.metadata.declare('size', value=1, typ=int,
                              desc='the size of the linear system')

    def initialize_variables(self):
        """Matrix and RHS are inputs, solution vector is the output."""
        size = self.metadata['size']

        self.add_input("A", val=np.eye(size))
        self.add_input("b", val=np.ones(size))
        self.add_output("x", shape=size)

    def solve_nonlinear(self, inputs, outputs):
        """Use numpy to solve Ax=b for x."""
        # lu factorization for use with solve_linear
        self.lup = linalg.lu_factor(inputs['A'])
        outputs['x'] = linalg.lu_solve(self.lup, inputs['b'])

    def apply_nonlinear(self, inputs, outputs, residuals):
        """R = Ax - b."""
        residuals['x'] = inputs['A'].dot(outputs['x']) - inputs['b']

    # @skip_lint
    def apply_linear(self, inputs, outputs, d_inputs, d_outputs,
                     d_residuals, mode):
        if mode == 'fwd':

            if 'x' in d_outputs:
                d_residuals['x'] += inputs['A'].dot(d_outputs['x'])
            if 'A' in d_inputs:
                d_residuals['x'] += d_inputs['A'].dot(outputs['x'])
            if 'b' in d_inputs:
                d_residuals['x'] -= d_inputs['b']

        elif mode == 'rev':

            if 'x' in d_outputs:
                d_outputs['x'] += inputs['A'].T.dot(d_residuals['x'])
            if 'A' in d_inputs:
                d_inputs['A'] += np.outer(outputs['x'], d_residuals['x']).T
            if 'b' in d_inputs:
                d_inputs['b'] -= d_residuals['x']

    def solve_linear(self, d_outputs, d_residuals, mode):
        """Back-substitution to solve the derivatives of the linear system."""
        if mode == 'fwd':
            sol_vec, rhs_vec = d_outputs, d_residuals
            t = 0
        else:
            sol_vec, rhs_vec = d_residuals, d_outputs
            t = 1

        sol_vec['x'] = linalg.lu_solve(self.lup, rhs_vec['x'], trans=t)
