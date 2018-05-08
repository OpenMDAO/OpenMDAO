"""Definition of the Cross Product Component."""

from six import string_types

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class CrossProductComp(ExplicitComponent):
    """
    Compute a vectorized dot product.

    math::
        c = np.cross(a, b)

    where a is of shape (vec_size, 3)
          b is of shape (vec_size, 3)
          c is of shape (vec_size, 3)

    if vec_size > 1 and

    where a is of shape (3,)
          b is of shape (3,)
          c is of shape (3,)

    otherwise.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('vec_size', types=int, default=1,
                             desc='The number of points at which the dot product is computed')
        self.options.declare('a_name', types=string_types, default='a',
                             desc='The variable name for vector a.')
        self.options.declare('b_name', types=string_types, default='b',
                             desc='The variable name for vector b.')
        self.options.declare('c_name', types=string_types, default='c',
                             desc='The variable name for vector c.')
        self.options.declare('a_units', types=string_types, default=None, allow_none=True,
                             desc='The units for vector a.')
        self.options.declare('b_units', types=string_types, default=None, allow_none=True,
                             desc='The units for vector b.')
        self.options.declare('c_units', types=string_types, default=None, allow_none=True,
                             desc='The units for vector c.')

        self._k = np.array([[0, 0, 0, -1, 0, 1],
                            [0, 1, 0, 0, -1, 0],
                            [-1, 0, 1, 0, 0, 0]], dtype=np.float64)

    def setup(self):
        """
        Declare inputs, outputs, and derivatives for the cross product component.
        """
        opts = self.options
        vec_size = opts['vec_size']

        shape = (vec_size, 3) if vec_size > 1 else (3,)

        self.add_input(name=opts['a_name'],
                       shape=shape,
                       units=opts['a_units'])

        self.add_input(name=opts['b_name'],
                       shape=shape,
                       units=opts['b_units'])

        self.add_output(name=opts['c_name'],
                        val=np.ones(shape=shape),
                        units=opts['c_units'])

        row_idxs = np.repeat(np.arange(vec_size * 3, dtype=int), 2)
        col_idxs = np.empty((0,), dtype=int)
        M = np.array([1, 2, 0, 2, 0, 1], dtype=int)
        for i in range(vec_size):
            col_idxs = np.concatenate((col_idxs, M + i * 3))

        self.declare_partials(of=opts['c_name'], wrt=opts['a_name'],
                              rows=row_idxs, cols=col_idxs, val=0)

        self.declare_partials(of=opts['c_name'], wrt=opts['b_name'],
                              rows=row_idxs, cols=col_idxs, val=0)

    def compute(self, inputs, outputs):
        """
        Compute the dot product of inputs `a` and `b` using np.cross.

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
        outputs[opts['c_name']] = np.cross(a, b)

    def compute_partials(self, inputs, partials):
        """
        Compute the sparse partials for the cross product w.r.t. the inputs.

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
        partials[opts['c_name'], opts['a_name']] = \
            np.einsum('...j,ji->...i', b, self._k * -1).ravel()
        partials[opts['c_name'], opts['b_name']] = \
            np.einsum('...j,ji->...i', a, self._k).ravel()
