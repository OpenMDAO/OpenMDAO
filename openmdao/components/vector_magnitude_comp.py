"""Definition of the Vector Magnitude Component."""


import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class VectorMagnitudeComp(ExplicitComponent):
    """
    Computes a vectorized magnitude.

    math::
        a_mag = np.sqrt(np.dot(a, a))

    where a is of shape (vec_size, n)

    Attributes
    ----------
    _magnitudes : list
        Cache the data provided during `add_magnitude`
        so everything can be saved until setup is called.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Vector Magnitude component.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super(VectorMagnitudeComp, self).__init__(**kwargs)

        self._magnitudes = []

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('vec_size', types=int, default=1,
                             desc='The number of points at which the vector magnitude is computed')
        self.options.declare('length', types=int, default=3,
                             desc='The length of the input vector at each point')
        self.options.declare('in_name', types=str, default='a',
                             desc='The variable name for input vector.')
        self.options.declare('units', types=str, default=None, allow_none=True,
                             desc='The units for vector a.')
        self.options.declare('mag_name', types=str, default='a_mag',
                             desc='The variable name for output vector magnitude.')

    def add_magnitude(self, in_name, mag_name, units=None, vec_size=1, length=3):
        """
        Add a new output product to the dot product component.

        Parameters
        ----------
        in_name : str
            The name of the first vector input.
        mag_name : str
            The name of the second input.
        units : str or None
            The units of input a.
        vec_size : int
            The number of points at which the dot vector product
            should be computed simultaneously.  The shape of
            the output is (vec_size,).
        length : int
            The length of the vectors a and b.  Their shapes are
            (vec_size, length)
        """
        self._magnitudes.append({
            'in_name': in_name,
            'mag_name': mag_name,
            'units': units,
            'vec_size': vec_size,
            'length': length
        })

    def setup(self):
        """
        Declare inputs, outputs, and derivatives for the vector magnitude component.
        """
        if len(self._magnitudes) == 0:
            magnitudes = self._magnitudes = [self.options]
        else:
            # prepend the product specified in component options
            opts = self.options
            magnitudes = self._magnitudes
            magnitudes.insert(0, {
                'in_name': opts['in_name'],
                'mag_name': opts['mag_name'],
                'units': opts['units'],
                'vec_size': opts['vec_size'],
                'length': opts['length']
            })

        # add inputs and outputs for all products
        var_rel2meta = self._var_rel2meta

        for magnitude in magnitudes:
            in_name = magnitude['in_name']
            mag_name = magnitude['mag_name']
            units = magnitude['units']
            vec_size = magnitude['vec_size']
            m = magnitude['length']

            if in_name not in var_rel2meta:
                self.add_input(name=in_name, shape=(vec_size, m), units=units)
            else:
                meta = var_rel2meta[in_name]
                if units != meta['units']:
                    raise ValueError(f"{self.msginfo}: Conflicting units specified for input "
                                     f"'{in_name}', '{meta['units']}' and '{units}'.")
                if vec_size != meta['shape'][0]:
                    raise ValueError(f"{self.msginfo}: Conflicting vec_size specified for input "
                                     f"'{in_name}', {meta['shape'][0]} versus {vec_size}.")
                if m != meta['shape'][1]:
                    raise ValueError(f"{self.msginfo}: Conflicting length specified for input "
                                     f"'{in_name}', {meta['shape'][1]} versus {m}.")

            if mag_name not in var_rel2meta:
                self.add_output(name=mag_name, shape=(vec_size,), units=units)
            else:
                raise NameError(f"{self.msginfo}: Multiple definition of output '{mag_name}'.")

            row_idxs = np.repeat(np.arange(vec_size), m)
            col_idxs = np.arange(vec_size * m)
            self.declare_partials(of=mag_name, wrt=in_name,
                                  rows=row_idxs, cols=col_idxs)

    def compute(self, inputs, outputs):
        """
        Compute the vector magnitude of input.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        for magnitude in self._magnitudes:
            a = inputs[magnitude['in_name']]
            outputs[magnitude['mag_name']] = np.sqrt(np.einsum('ni,ni->n', a, a))

    def compute_partials(self, inputs, partials):
        """
        Compute the sparse partials for the vector magnitude w.r.t. the inputs.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        for magnitude in self._magnitudes:
            a = inputs[magnitude['in_name']]

            # Use the following for sparse partials
            partials[magnitude['mag_name'], magnitude['in_name']] = \
                a.ravel() / np.repeat(np.sqrt(np.einsum('ni,ni->n', a, a)), magnitude['length'])
