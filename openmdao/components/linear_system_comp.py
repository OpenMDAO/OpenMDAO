"""Define the LinearSystemComp class."""
from __future__ import division, print_function

from six.moves import range

import numpy as np
from scipy import linalg

from openmdao.core.implicitcomponent import ImplicitComponent


class LinearSystemComp(ImplicitComponent):
    """
    Component that solves a linear system, Ax=b.

    Designed to handle small and dense linear systems that can be
    efficiently solved with lu-decomposition

    Attributes
    ----------
    _lup : object
        matrix factorization returned from scipy.linag.lu_factor
    """

    def __init__(self, **kwargs):
        """
        Intialize the LinearSystemComp component.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            available here and in all descendants of this system.
        """
        super(LinearSystemComp, self).__init__(**kwargs)
        self._lup = None

    def initialize(self):
        """
        Declare metadata.
        """
        self.metadata.declare('size', default=1, types=int, desc='the size of the linear system')
        self.metadata.declare('partial_type', default='dense',
                              values=['dense', 'sparse', 'matrix_free'],
                              desc='the way the derivatives are defined')

    def setup(self):
        """
        Matrix and RHS are inputs, solution vector is the output.
        """
        size = self.metadata['size']

        self._lup = None

        if self.metadata['partial_type'] == "matrix_free":
            self.apply_linear = self._mat_vec_prod

        self.add_input("A", val=np.eye(size))
        self.add_input("b", val=np.ones(size))
        self.add_output("x", shape=size, val=.1)

        # Set up the derivatives according to the user specified mode.

        partial_type = self.metadata['partial_type']

        size = self.metadata['size']
        row_col = np.arange(size, dtype="int")

        if partial_type == 'sparse':
            self.declare_partials('x', 'b', val=-np.ones(size), rows=row_col, cols=row_col)
            # self.declare_partials('x', 'b', val=-1, rows=row_col, cols=row_col)

            rows = []
            cols = []
            for i in range(size):
                for j in range(size):
                    rows.append(i)
                    cols.append(i * size + j)

            self.dx_da_rows = rows
            self.dx_da_cols = cols

            self.declare_partials('x', 'A', val=np.ones(size**2), rows=rows, cols=cols)

        elif partial_type == "dense":
            self.declare_partials(of='x', wrt='b', val=-np.eye(size))
            self.declare_partials(of='x', wrt='A')

        self.declare_partials(of='x', wrt='x')

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        R = Ax - b.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        residuals['x'] = inputs['A'].dot(outputs['x']) - inputs['b']

    def solve_nonlinear(self, inputs, outputs):
        """
        Use numpy to solve Ax=b for x.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        # lu factorization for use with solve_linear
        self._lup = linalg.lu_factor(inputs['A'])
        outputs['x'] = linalg.lu_solve(self._lup, inputs['b'])

    def linearize(self, inputs, outputs, J):
        """
        Compute the non-constant partial derivatives.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        J : Jacobian
            sub-jac components written to jacobian[output_name, input_name]
        """
        partial_type = self.metadata['partial_type']
        if partial_type == "matrix_free":
            return

        x = outputs['x']
        size = self.metadata['size']
        if partial_type == "dense":
            dx_dA = np.zeros((size, size**2))

            for i in range(size):
                dx_dA[i, i * size:(i + 1) * size] = x

            J['x', 'A'] = dx_dA
            J['x', 'x'] = inputs['A']

            # constant, defined int setup
            # J['x', 'b'] = -np.eye()

        else:  # sparse

            J['x', 'A'] = np.tile(x, size)
            J['x', 'x'] = inputs['A']

            # constant, defined int setup
            # J['x', 'b'] = -np.ones(size)

    def _mat_vec_prod(self, inputs, outputs, d_inputs, d_outputs,
                      d_residuals, mode):
        """
        Compute jac-vector product.

        linear operator for the partial derivative jacobian, only used if the 'partial_type'
        metadata is set to 'matrix_free'.

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
        r"""
        Back-substitution to solve the derivatives of the linear system.

        If mode is:
            'fwd': d_residuals \|-> d_outputs

            'rev': d_outputs \|-> d_residuals

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
            d_outputs['x'] = linalg.lu_solve(self._lup, d_residuals['x'], trans=0)
        else:  # rev
            d_residuals['x'] = linalg.lu_solve(self._lup, d_outputs['x'], trans=1)
