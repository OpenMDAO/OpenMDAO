"""Definition of the Dot Product Component."""


from six import string_types

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class DotProductComp(ExplicitComponent):
    """
    Computes a vectorized dot product.

    math::
        c = np.dot(a, b)

    where a is of shape (vec_size, n)
          b is of shape (vec_size, n)
          c is of shape (vec_size,)

    Vectors a and b must be of the same length, specified by the option 'length'.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('vec_size', types=int, default=1,
                             desc='The number of points at which the dot product is computed')
        self.options.declare('length', types=int, default=3,
                             desc='The length of vectors a and b')
        self.options.declare('a_name', types=string_types, default='a',
                             desc='The variable name for input vector a.')
        self.options.declare('b_name', types=string_types, default='b',
                             desc='The variable name for input vector b.')
        self.options.declare('c_name', types=string_types, default='c',
                             desc='The variable name for output vector c.')
        self.options.declare('a_units', types=string_types, default=None, allow_none=True,
                             desc='The units for vector a.')
        self.options.declare('b_units', types=string_types, default=None, allow_none=True,
                             desc='The units for vector b.')
        self.options.declare('c_units', types=string_types, default=None, allow_none=True,
                             desc='The units for vector c.')

    def setup(self):
        """
        Declare inputs, outputs, and derivatives for the dot product component.
        """
        opts = self.options
        vec_size = opts['vec_size']
        m = opts['length']

        self.add_input(name=opts['a_name'],
                       shape=(vec_size, m),
                       units=opts['a_units'])

        self.add_input(name=opts['b_name'],
                       shape=(vec_size, m),
                       units=opts['b_units'])

        self.add_output(name=opts['c_name'],
                        val=np.zeros(shape=(vec_size,)),
                        units=opts['c_units'])

        row_idxs = np.repeat(np.arange(vec_size), m)
        col_idxs = np.arange(vec_size * m)
        self.declare_partials(of=opts['c_name'], wrt=opts['a_name'], rows=row_idxs, cols=col_idxs)
        self.declare_partials(of=opts['c_name'], wrt=opts['b_name'], rows=row_idxs, cols=col_idxs)

    def compute(self, inputs, outputs):
        """
        Compute the dot product of inputs `a` and `b` using np.einsum.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        opts = self.options
        a = inputs[opts['a_name']]
        b = inputs[opts['b_name']]
        outputs[opts['c_name']] = np.einsum('ni,ni->n', a, b)

    def compute_partials(self, inputs, partials):
        """
        Compute the sparse partials for the dot product w.r.t. the inputs.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        opts = self.options
        a = inputs[opts['a_name']]
        b = inputs[opts['b_name']]

        # Use the following for sparse partials
        partials[opts['c_name'], opts['a_name']] = b.ravel()
        partials[opts['c_name'], opts['b_name']] = a.ravel()
