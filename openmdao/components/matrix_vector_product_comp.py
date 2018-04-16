from six import string_types

import numpy as np
import scipy.linalg as spla

from openmdao.api import ExplicitComponent


class MatrixVectorProductComp(ExplicitComponent):
    """
    Computes a vectorized matrix-vector product

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

    The size of vectors x and b is determined by the number of for rows in m at any point.
    """

    def initialize(self):
        self.metadata.declare('vec_size', types=int, default=1,
                              desc='The number of points at which the matrix'
                                   '-vector product is to be computed')
        self.metadata.declare('A_name', types=string_types, default='A',
                              desc='The variable name for the matrix.')
        self.metadata.declare('A_shape', types=tuple, default=(3, 3),
                              desc='The shape of the input matrix at a single point.')
        self.metadata.declare('A_units', types=string_types, allow_none=True, default=None,
                              desc='The units of the input matrix.')
        self.metadata.declare('x_name', types=string_types, default='x',
                              desc='The name of the input vector.')
        self.metadata.declare('x_units', types=string_types, default=None, allow_none=True,
                              desc='The units of the input vector.')
        self.metadata.declare('b_name', types=string_types, default='b',
                              desc='The variable name of the output vector.')
        self.metadata.declare('b_units', types=string_types, allow_none=True, default=None,
                              desc='The units of the output vector.')

    def setup(self):
        meta = self.metadata
        vec_size = meta['vec_size']

        n_rows, n_cols = meta['A_shape']

        self.add_input(name=meta['A_name'],
                       shape=(vec_size,) + meta['A_shape'],
                       units=meta['A_units'])

        self.add_input(name=meta['x_name'],
                       shape=(vec_size,) + (n_cols,),
                       units=meta['x_units'])

        b_shape = (vec_size,) + (n_rows,) if vec_size > 1 else (n_rows,)

        self.add_output(name=meta['b_name'],
                        val=np.ones(shape=b_shape),
                        units=meta['b_units'])

        # Make a dummy version of A so we can figure out the nonzero indices
        A = np.ones(shape=(vec_size,) + meta['A_shape'])
        x = np.ones(shape=(vec_size,) + (n_cols,))
        bd_A = spla.block_diag(*A)
        x_repeat = np.repeat(x, A.shape[1], axis=0)
        bd_x_repeat = spla.block_diag(*x_repeat)
        db_dx_rows, db_dx_cols = np.nonzero(bd_A)
        db_dA_rows, db_dA_cols = np.nonzero(bd_x_repeat)

        self.declare_partials(of=meta['b_name'], wrt=meta['A_name'],
                              rows=db_dA_rows, cols=db_dA_cols)
        self.declare_partials(of=meta['b_name'], wrt=meta['x_name'],
                              rows=db_dx_rows, cols=db_dx_cols)

    def compute(self, inputs, outputs):
        meta = self.metadata
        A = inputs[meta['A_name']]
        x = inputs[meta['x_name']]

        # ... here allows b to be shaped either (n, i) or (i,)
        outputs[meta['b_name']][...] = np.einsum('nij,nj->ni', A, x)

    def compute_partials(self, inputs, partials):
        meta = self.metadata
        A_name = meta['A_name']
        x_name = meta['x_name']
        b_name = meta['b_name']
        A = inputs[A_name]
        x = inputs[x_name]

        # Use the following for sparse partials
        partials[b_name, A_name] = np.repeat(x, A.shape[1], axis=0).ravel()
        partials[b_name, x_name] = A.ravel()


def _for_docs():  # pragma: no cover
    return MatrixVectorProductComp()
