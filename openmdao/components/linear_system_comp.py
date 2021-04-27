"""Define the LinearSystemComp class."""

import numpy as np
from scipy import linalg

from openmdao.core.implicitcomponent import ImplicitComponent


class LinearSystemComp(ImplicitComponent):
    """
    Component that solves a linear system, Ax=b.

    Designed to handle small, dense linear systems (Ax=B) that can be efficiently solved with
    lu-decomposition. It can be vectorized to either solve for multiple right hand sides,
    or to solve multiple linear systems.

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
            Keyword arguments that will be mapped into the Component options.
        """
        super().__init__(**kwargs)
        self._lup = None

        self._no_check_partials = True

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('size', default=1, types=int, desc='The size of the linear system.')
        self.options.declare('vec_size', types=int, default=1,
                             desc='Number of linear systems to solve.')
        self.options.declare('vectorize_A', default=False, types=bool,
                             desc='Set to True to vectorize the A matrix.')

    def setup(self):
        """
        Matrix and RHS are inputs, solution vector is the output.
        """
        vec_size = self.options['vec_size']
        vec_size_A = self.vec_size_A = vec_size if self.options['vectorize_A'] else 1
        size = self.options['size']

        self._lup = []
        shape = (vec_size, size) if vec_size > 1 else (size, )
        shape_A = (vec_size_A, size, size) if vec_size_A > 1 else (size, size)

        init_A = np.eye(size)
        if vec_size_A > 1:
            init_A = np.repeat(init_A.reshape(1, size, size), vec_size_A, axis=0)

        self.add_input("A", val=init_A)
        self.add_input("b", val=np.ones(shape))
        self.add_output("x", shape=shape, val=.1)

    def setup_partials(self):
        """
        Set up the derivatives.
        """
        vec_size = self.options['vec_size']
        vec_size_A = self.vec_size_A = vec_size if self.options['vectorize_A'] else 1
        size = self.options['size']
        mat_size = size * size
        full_size = size * vec_size

        row_col = np.arange(full_size, dtype="int")

        self.declare_partials('x', 'b', val=np.full(full_size, -1.0), rows=row_col, cols=row_col)

        rows = np.repeat(np.arange(full_size), size)

        if vec_size_A > 1:
            cols = np.arange(mat_size * vec_size)
        else:
            cols = np.tile(np.arange(mat_size), vec_size)

        self.declare_partials('x', 'A', rows=rows, cols=cols)

        cols = np.tile(np.arange(size), size)
        cols = np.tile(cols, vec_size) + np.repeat(np.arange(vec_size), mat_size) * size

        self.declare_partials(of='x', wrt='x', rows=rows, cols=cols)

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
        if self.options['vec_size'] > 1:
            if self.vec_size_A > 1:
                residuals['x'] = np.einsum('ijk,ik->ij', inputs['A'], outputs['x']) - inputs['b']
            else:
                residuals['x'] = np.einsum('jk,ik->ij', inputs['A'], outputs['x']) - inputs['b']

        else:
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
        vec_size = self.options['vec_size']
        vec_size_A = self.vec_size_A

        # lu factorization for use with solve_linear
        self._lup = []
        if vec_size > 1:
            for j in range(vec_size_A):
                lhs = inputs['A'][j] if vec_size_A > 1 else inputs['A']
                self._lup.append(linalg.lu_factor(lhs))

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
        size = self.options['size']
        vec_size = self.options['vec_size']

        J['x', 'A'] = np.tile(x, size).flat
        if self.vec_size_A > 1:
            J['x', 'x'] = inputs['A'].flat
        else:
            J['x', 'x'] = np.tile(inputs['A'].flat, vec_size)

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
        vec_size = self.options['vec_size']
        vec_size_A = self.vec_size_A

        if mode == 'fwd':
            if vec_size > 1:
                for j in range(vec_size):
                    idx = j if vec_size_A > 1 else 0
                    d_outputs['x'][j] = linalg.lu_solve(self._lup[idx], d_residuals['x'][j],
                                                        trans=0)
            else:
                d_outputs['x'] = linalg.lu_solve(self._lup, d_residuals['x'], trans=0)

        else:  # rev
            if vec_size > 1:
                for j in range(vec_size):
                    idx = j if vec_size_A > 1 else 0
                    d_residuals['x'][j] = linalg.lu_solve(self._lup[idx], d_outputs['x'][j],
                                                          trans=1)
            else:
                d_residuals['x'] = linalg.lu_solve(self._lup, d_outputs['x'], trans=1)
