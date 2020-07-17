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

    Attributes
    ----------
    _products : list
        Cache the data provided during `add_product`
        so everything can be saved until setup is called.
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

        opt = self.options
        self.add_product(b_name=opt['b_name'], A_name=opt['A_name'], x_name=opt['x_name'],
                         b_units=opt['b_units'], A_units=opt['A_units'], x_units=opt['x_units'],
                         vec_size=opt['vec_size'], A_shape=opt['A_shape'])

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
        self.options.declare('A_units', types=str, default=None, allow_none=True,
                             desc='The units of the input matrix.')
        self.options.declare('x_name', types=str, default='x',
                             desc='The name of the input vector.')
        self.options.declare('x_units', types=str, default=None, allow_none=True,
                             desc='The units of the input vector.')
        self.options.declare('b_name', types=str, default='b',
                             desc='The variable name of the output vector.')
        self.options.declare('b_units', types=str, default=None, allow_none=True,
                             desc='The units of the output vector.')

    def add_product(self, b_name, A_name='A', x_name='x', A_units=None, x_units=None, b_units=None,
                    vec_size=1, A_shape=(3, 3)):
        """
        Add a new output product to the matrix vector product component.

        Parameters
        ----------
        A_name : str
            The name of the matrix input.
        x_name : str
            The name of the vector input.
        b_name : str
            The name of the vector product output.
        A_units : str or None
            The units of the input matrix.
        x_units : str or None
            The units of the input vector.
        b_units : str or None
            The units of the output matrix.
        vec_size : int
            The number of points at which the matrix vector product
            should be computed simultaneously.
        A_shape : tuple of (int, int)
            The shape of the matrix at each point.
            The first element also specifies the size of the output at each point.
            The second element specifies the size of the input vector at each point.
            For example, if vec_size=10 and shape is (5, 3), then
            the input matrix will have a shape of (10, 5, 3),
            the input vector will have a shape of (10, 3), and
            the output vector will have shape of (10, 5).
        """
        self._products.append({
            'A_name': A_name,
            'x_name': x_name,
            'b_name': b_name,
            'A_units': A_units,
            'x_units': x_units,
            'b_units': b_units,
            'A_shape': A_shape,
            'vec_size': vec_size
        })

        # add inputs and outputs for all products
        if self._static_mode:
            var_rel2meta = self._static_var_rel2meta
            var_rel_names = self._static_var_rel_names
        else:
            var_rel2meta = self._var_rel2meta
            var_rel_names = self._var_rel_names

        n_rows, n_cols = A_shape

        A_shape = (vec_size, n_rows, n_cols)
        b_shape = (vec_size, n_rows) if vec_size > 1 else (n_rows, )
        x_shape = (vec_size, n_cols)

        if b_name not in var_rel2meta:
            self.add_output(name=b_name, shape=b_shape, units=b_units)
        elif b_name in var_rel_names['input']:
            raise NameError(f"{self.msginfo}: '{b_name}' specified as an output, "
                            "but it has already been defined as an input.")
        else:
            raise NameError(f"{self.msginfo}: Multiple definition of output '{b_name}'.")

        if A_name not in var_rel2meta:
            self.add_input(name=A_name, shape=A_shape, units=A_units)
        elif A_name in var_rel_names['output']:
            raise NameError(f"{self.msginfo}: '{A_name}' specified as an input, "
                            "but it has already been defined as an output.")
        else:
            meta = var_rel2meta[A_name]
            if vec_size != meta['shape'][0]:
                raise ValueError(f"{self.msginfo}: Conflicting vec_size={x_shape[0]} "
                                 f"specified for matrix '{A_name}', which has already "
                                 f"been defined with vec_size={meta['shape'][0]}.")

            elif (n_rows, n_cols) != meta['shape'][1:]:
                raise ValueError(f"{self.msginfo}: Conflicting shape {A_shape[1:]} specified "
                                 f"for matrix '{A_name}', which has already been defined "
                                 f"with shape {meta['shape'][1:]}.")

            elif A_units != meta['units']:
                raise ValueError(f"{self.msginfo}: Conflicting units '{A_units}' specified "
                                 f"for matrix '{A_name}', which has already been defined "
                                 f"with units '{meta['units']}'.")

        if x_name not in var_rel2meta:
            self.add_input(name=x_name, shape=x_shape, units=x_units)
        elif x_name in var_rel_names['output']:
            raise NameError(f"{self.msginfo}: '{x_name}' specified as an input, "
                            "but it has already been defined as an output.")
        else:
            meta = var_rel2meta[x_name]
            if vec_size != meta['shape'][0]:
                raise ValueError(f"{self.msginfo}: Conflicting vec_size={x_shape[0]} "
                                 f"specified for vector '{x_name}', which has already "
                                 f"been defined with vec_size={meta['shape'][0]}.")

            elif n_cols != meta['shape'][1]:
                raise ValueError(f"{self.msginfo}: Matrix shape {A_shape[1:]} is incompatible "
                                 f"with vector '{x_name}', which has already been defined "
                                 f"with {meta['shape'][1]} column(s).")

            elif x_units != meta['units']:
                raise ValueError(f"{self.msginfo}: Conflicting units '{x_units}' specified "
                                 f"for vector '{x_name}', which has already been defined "
                                 f"with units '{meta['units']}'.")

        # Make a dummy version of A so we can figure out the nonzero indices
        A = np.ones(A_shape)
        x = np.ones(x_shape)
        bd_A = spla.block_diag(*A)
        x_repeat = np.repeat(x, A.shape[1], axis=0)
        bd_x_repeat = spla.block_diag(*x_repeat)
        db_dx_rows, db_dx_cols = np.nonzero(bd_A)
        db_dA_rows, db_dA_cols = np.nonzero(bd_x_repeat)

        self.declare_partials(of=b_name, wrt=A_name,
                              rows=db_dA_rows, cols=db_dA_cols)
        self.declare_partials(of=b_name, wrt=x_name,
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
            A = inputs[product['A_name']]
            x = inputs[product['x_name']]

            # ... here allows b to be shaped either (n, i) or (i,)
            outputs[product['b_name']][...] = np.einsum('nij,nj->ni', A, x)

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
            A_name = product['A_name']
            x_name = product['x_name']
            b_name = product['b_name']

            A = inputs[A_name]
            x = inputs[x_name]

            # Use the following for sparse partials
            partials[b_name, A_name] = np.repeat(x, A.shape[1], axis=0).ravel()
            partials[b_name, x_name] = A.ravel()
