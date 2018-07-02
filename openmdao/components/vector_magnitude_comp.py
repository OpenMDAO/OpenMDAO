"""Definition of the Vector Magnitude Component."""


from six import string_types

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class VectorMagnitudeComp(ExplicitComponent):
    """
    Computes a vectorized magnitude.

    math::
        a_mag = np.sqrt(np.dot(a, a))

    where a is of shape (vec_size, n)

    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('vec_size', types=int, default=1,
                             desc='The number of points at which the dot product is computed')
        self.options.declare('length', types=int, default=3,
                             desc='The length of the input vector at each point')
        self.options.declare('in_name', types=string_types, default='a',
                             desc='The variable name for input vector.')
        self.options.declare('units', types=string_types, default=None, allow_none=True,
                             desc='The units for vector a.')
        self.options.declare('mag_name', types=string_types, default='a_mag',
                             desc='The variable name for output vector magnitude.')

    def setup(self):
        """
        Declare inputs, outputs, and derivatives for the dot product component.
        """
        opts = self.options
        vec_size = opts['vec_size']
        m = opts['length']

        self.add_input(name=opts['in_name'],
                       shape=(vec_size, m),
                       units=opts['units'])

        self.add_output(name=opts['mag_name'],
                        val=np.zeros(shape=(vec_size,)),
                        units=opts['units'])

        row_idxs = np.repeat(np.arange(vec_size), m)
        col_idxs = np.arange(vec_size * m)
        self.declare_partials(of=opts['mag_name'], wrt=opts['in_name'],
                              rows=row_idxs, cols=col_idxs)

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
        a = inputs[opts['in_name']]
        outputs[opts['mag_name']] = np.sqrt(np.einsum('ni,ni->n', a, a))

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
        a = inputs[opts['in_name']]

        # Use the following for sparse partials
        partials[opts['mag_name'], opts['in_name']] = \
            a.ravel() / np.repeat(np.sqrt(np.einsum('ni,ni->n', a, a)), opts['length'])
