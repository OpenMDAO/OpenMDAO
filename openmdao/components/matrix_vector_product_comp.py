"""Definition of the Matrix Vector Product Component."""


from six import string_types

import numpy as np
import scipy.linalg as spla

from openmdao.core.explicitcomponent import ExplicitComponent


class MatrixVectorProductComp(ExplicitComponent):
    """
    Computes a vectorized matrix-vector product.

    math::
        b = np.dot(A, x)

    where A is of shape (vec_size, n, m)
          x is of shape (vec_size, m)
          b is of shape (vec_size, m)

    if vec_size > 1 and

    where A is of shape (n, m)
          x is of shape (m,)
          b is of shape (m,)

    otherwise.

    The size of vectors x and b is determined by the number of rows in m at each point.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('vec_size', types=int, default=1,
                             desc='The number of points at which the matrix'
                                  '-vector product is to be computed')
        self.options.declare('A_name', types=string_types, default='A',
                             desc='The variable name for the matrix.')
        self.options.declare('A_shape', types=tuple, default=(3, 3),
                             desc='The shape of the input matrix at a single point.')
        self.options.declare('A_units', types=string_types, allow_none=True, default=None,
                             desc='The units of the input matrix.')
        self.options.declare('x_name', types=string_types, default='x',
                             desc='The name of the input vector.')
        self.options.declare('x_units', types=string_types, default=None, allow_none=True,
                             desc='The units of the input vector.')
        self.options.declare('b_name', types=string_types, default='b',
                             desc='The variable name of the output vector.')
        self.options.declare('b_units', types=string_types, allow_none=True, default=None,
                             desc='The units of the output vector.')

    def setup(self):
        """
        Declare inputs, outputs, and derivatives for the matrix vector product component.
        """
        opts = self.options
        vec_size = opts['vec_size']

        n_rows, n_cols = opts['A_shape']

        self.add_input(name=opts['A_name'],
                       shape=(vec_size,) + opts['A_shape'],
                       units=opts['A_units'])

        self.add_input(name=opts['x_name'],
                       shape=(vec_size,) + (n_cols,),
                       units=opts['x_units'])

        b_shape = (vec_size,) + (n_rows,) if vec_size > 1 else (n_rows,)

        self.add_output(name=opts['b_name'],
                        val=np.ones(shape=b_shape),
                        units=opts['b_units'])

        # Make a dummy version of A so we can figure out the nonzero indices
        A = np.ones(shape=(vec_size,) + opts['A_shape'])
        x = np.ones(shape=(vec_size,) + (n_cols,))
        bd_A = spla.block_diag(*A)
        x_repeat = np.repeat(x, A.shape[1], axis=0)
        bd_x_repeat = spla.block_diag(*x_repeat)
        db_dx_rows, db_dx_cols = np.nonzero(bd_A)
        db_dA_rows, db_dA_cols = np.nonzero(bd_x_repeat)

        self.declare_partials(of=opts['b_name'], wrt=opts['A_name'],
                              rows=db_dA_rows, cols=db_dA_cols)
        self.declare_partials(of=opts['b_name'], wrt=opts['x_name'],
                              rows=db_dx_rows, cols=db_dx_cols)

    def compute(self, inputs, outputs):
        """
        Compute the matrix vector product of inputs `A` and `x` using np.einsum.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        opts = self.options
        A = inputs[opts['A_name']]
        x = inputs[opts['x_name']]

        # ... here allows b to be shaped either (n, i) or (i,)
        outputs[opts['b_name']][...] = np.einsum('nij,nj->ni', A, x)

    def compute_partials(self, inputs, partials):
        """
        Compute the sparse partials for the matrix vector product w.r.t. the inputs.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        opts = self.options
        A_name = opts['A_name']
        x_name = opts['x_name']
        b_name = opts['b_name']
        A = inputs[A_name]
        x = inputs[x_name]

        # Use the following for sparse partials
        partials[b_name, A_name] = np.repeat(x, A.shape[1], axis=0).ravel()
        partials[b_name, x_name] = A.ravel()
