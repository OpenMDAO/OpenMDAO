"""Define the LinearSystemComp class."""
from __future__ import division, print_function

from six.moves import range

import numpy as np
from scipy import linalg

from openmdao.core.implicitcomponent import ImplicitComponent


class LinearSystemComp(ImplicitComponent):
    """
    Component that solves a linear system, Ax=b.

    Designed to handle small and dense linear systems that can be efficiently solved with
    lu-decomposition.

    Attributes
    ----------
    _lup : None or list(object)
        matrix factorizations returned from scipy.linag.lu_factor for each A matrix
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
        self.metadata.declare('size', default=1, types=int, desc='The size of the linear system.')
        self.metadata.declare('vec_size', types=int, default=1,
                              desc='Number of linear systems to solve.')
        self.metadata.declare('vectorize_A', default=False, types=bool,
                              desc='Set to True to vectorize the A matrix.')
    def setup(self):
        """
        Matrix and RHS are inputs, solution vector is the output.
        """
        size = self.metadata['size']
        vec_size = self.metadata['vec_size']
        vec_size_A = self.vec_size_A = vec_size if self.metadata['vectorize_A'] else 1

        self._lup = []
        shape = (vec_size, size) if vec_size > 1 else (size, )
        shape_A = (vec_size_A, size, size) if vec_size_A > 1 else (size, size)

        multi_eye = np.eye(size).reshape(shape_A)
        self.add_input("A", val=np.repeat(multi_eye, vec_size_A, axis=0))
        self.add_input("b", val=np.ones(shape))
        self.add_output("x", shape=shape, val=.1)

        # Set up the derivatives.
        size = self.metadata['size']
        row_col = np.arange(size, dtype="int")

        self.declare_partials('x', 'b', val=-np.ones(size), rows=row_col, cols=row_col)

        rows = []
        cols = []
        for i in range(size):
            for j in range(size):
                rows.append(i)
                cols.append(i * size + j)

        self.declare_partials('x', 'A', val=np.ones(size**2), rows=rows, cols=cols)

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
        vec_size = self.metadata['vec_size']
        vec_size_A = self.vec_size_A

        # lu factorization for use with solve_linear
        self._lup = []
        if vec_size > 1:
            for j in range(vec_size_A):
                self._lup.append(linalg.lu_factor(inputs['A'][j]))

            for j in range(vec_size):
                idx = j if vec_size_A > 1 else 0
                outputs['x'][j] = linalg.lu_solve(self._lup[idx], inputs['b'][j])
        else:
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
        x = outputs['x']
        size = self.metadata['size']

        J['x', 'A'] = np.tile(x, size)

        J['x', 'x'] = inputs['A']

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
        vec_size = self.metadata['vec_size']
        vec_size_A = self.vec_size_A

        if mode == 'fwd':
            if vec_size > 1:
                for j in range(vec_size):
                    idx = j if vec_size_A > 1 else 0
                    d_outputs['x'][j] = linalg.lu_solve(self._lup[idx], d_residuals['x'][j], trans=0)
            else:
                d_outputs['x'] = linalg.lu_solve(self._lup, d_residuals['x'], trans=0)

        else:  # rev
            if vec_size > 1:
                for j in range(vec_size):
                    idx = j if vec_size_A > 1 else 0
                    d_residuals['x'][j] = linalg.lu_solve(self._lup[idx], d_outputs['x'][j], trans=1)
            else:
                d_residuals['x'] = linalg.lu_solve(self._lup, d_outputs['x'], trans=1)