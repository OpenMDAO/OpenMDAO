"""Definition of the Matrix Vector Product Component."""


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

    def __init__(self, **kwargs):
        """
        Initialize the Matrix Vector Product component.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super(MatrixVectorProductComp, self).__init__(**kwargs)

        self._products = []

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('vec_size', types=int, default=1,
                             desc='The number of points at which the matrix vector product '
                                  'is to be computed')
        self.options.declare('A_name', types=str, default='A',
                             desc='The variable name for the matrix.')
        self.options.declare('A_shape', types=tuple, default=(3, 3),
                             desc='The shape of the input matrix at a single point.')
        self.options.declare('A_units', types=str, allow_none=True, default=None,
                             desc='The units of the input matrix.')
        self.options.declare('x_name', types=str, default='x',
                             desc='The name of the input vector.')
        self.options.declare('x_units', types=str, default=None, allow_none=True,
                             desc='The units of the input vector.')
        self.options.declare('b_name', types=str, default='b',
                             desc='The variable name of the output vector.')
        self.options.declare('b_units', types=str, allow_none=True, default=None,
                             desc='The units of the output vector.')

    def add_product(self, output, matrix, vector, output_units=None, matrix_units=None,
                    vector_units=None, vec_size=1, shape=(3, 3)):
        """
        Adds a new output product to the matrix vector product component.

        Parameters
        ----------
        output : str
            The name of the vector product output.
        matrix : str
            The name of the matrix input.
        vector : str
            The name of the vector input.
        output_units : str or None
            The units of the output matrix.
        matrix_units : str or None
            The units of the input matrix.
        vector_units : str or None
            The units of the input vector.
        vec_size : int
            The number of points at which the matrix vector product
            should be computed simultaneously.
        shape : tuple of (int, int)
            The shape of the matrix at each point.
            The first element also specifies the size of the output at each point.
            The second element specifies the size of the input vector at each point.
            For example, if vec_size=10 and shape is (5, 3), then
            the input matrix will have a shape of (10, 5, 3),
            the input vector will have a shape of (10, 3), and
            the output vector will have shape of (10, 5).
        """
        self._products.append({
            'output': output,
            'matrix': matrix,
            'vector': vector,
            'output_units': output_units,
            'matrix_units': matrix_units,
            'vector_units': vector_units,
            'output_shape': (vec_size, shape[0]) if vec_size > 1 else (shape[0], ),
            'matrix_shape': (vec_size, ) + shape,
            'vector_shape': (vec_size, shape[1])
        })

    def setup(self):
        """
        Declare inputs, outputs, and derivatives for the matrix vector product component.
        """
        products = self._products

        # add product specified in component options
        opts = self.options
        vec_size = opts['vec_size']
        n_rows, n_cols = opts['A_shape']

        products.append({
            'output': opts['b_name'],
            'matrix': opts['A_name'],
            'vector': opts['x_name'],
            'output_units': opts['b_units'],
            'matrix_units': opts['A_units'],
            'vector_units': opts['x_units'],
            'output_shape': (vec_size, n_rows) if vec_size > 1 else (n_rows, ),
            'matrix_shape': (vec_size, n_rows, n_cols),
            'vector_shape': (vec_size, n_cols)
        })

        # add inputs and outputs for all products
        var_rel2meta = self._var_rel2meta

        for product in products:
            output = product['output']
            matrix = product['matrix']
            vector = product['vector']

            output_units = product['output_units']
            matrix_units = product['matrix_units']
            vector_units = product['vector_units']

            output_shape = product['output_shape']
            matrix_shape = product['matrix_shape']
            vector_shape = product['vector_shape']

            if output not in var_rel2meta:
                self.add_output(name=output, shape=output_shape, units=output_units)
            else:
                raise NameError(f"{self.msginfo}: Multiple definition of output '{output}'")

            if matrix not in var_rel2meta:
                self.add_input(name=matrix, shape=matrix_shape, units=matrix_units)
            else:
                meta = var_rel2meta[matrix]
                if matrix_shape != meta['shape']:
                    raise ValueError(f"{self.msginfo}: Conflicting shapes specified for matrix "
                                     f"{matrix}, {matrix_shape} and {meta['shape']}")

                elif matrix_units != meta['units']:
                    raise ValueError(f"{self.msginfo}: Conflicting units specified for matrix "
                                     f"{matrix}, '{matrix_units}' and '{meta['units']}'")

            if vector not in var_rel2meta:
                self.add_input(name=vector, shape=vector_shape, units=vector_units)
            else:
                meta = var_rel2meta[vector]
                if vector_shape != meta['shape']:
                    raise ValueError(f"{self.msginfo}: Conflicting shapes specified for vector "
                                     f"{vector}, {vector_shape} and {meta['shape']}")

                elif vector_units != meta['units']:
                    raise ValueError(f"{self.msginfo}: Conflicting units specified for vector "
                                     f"{vector}, '{vector_units}' and '{meta['units']}'")

            # Make a dummy version of A so we can figure out the nonzero indices
            A = np.ones(product['matrix_shape'])
            x = np.ones(product['vector_shape'])
            bd_A = spla.block_diag(*A)
            x_repeat = np.repeat(x, A.shape[1], axis=0)
            bd_x_repeat = spla.block_diag(*x_repeat)
            db_dx_rows, db_dx_cols = np.nonzero(bd_A)
            db_dA_rows, db_dA_cols = np.nonzero(bd_x_repeat)

            self.declare_partials(of=product['output'], wrt=product['matrix'],
                                  rows=db_dA_rows, cols=db_dA_cols)
            self.declare_partials(of=product['output'], wrt=product['vector'],
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
        for product in self._products:
            A = inputs[product['matrix']]
            x = inputs[product['vector']]

            # ... here allows b to be shaped either (n, i) or (i,)
            outputs[product['output']][...] = np.einsum('nij,nj->ni', A, x)

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
        for product in self._products:
            A_name = product['matrix']
            x_name = product['vector']
            b_name = product['output']

            A = inputs[A_name]
            x = inputs[x_name]

            # Use the following for sparse partials
            partials[b_name, A_name] = np.repeat(x, A.shape[1], axis=0).ravel()
            partials[b_name, x_name] = A.ravel()
