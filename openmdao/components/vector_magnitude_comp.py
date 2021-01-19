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
        super().__init__(**kwargs)

        self._magnitudes = []

        opt = self.options
        self.add_magnitude(mag_name=opt['mag_name'], in_name=opt['in_name'], units=opt['units'],
                           vec_size=opt['vec_size'], length=opt['length'])

        self._no_check_partials = True

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
                             desc='The units of the input vector.')
        self.options.declare('mag_name', types=str, default='a_mag',
                             desc='The variable name for output vector magnitude.')

    def add_magnitude(self, mag_name, in_name, units=None, vec_size=1, length=3):
        """
        Add a new output magnitude to the vector magnitude component.

        Parameters
        ----------
        mag_name : str
            The name of the output vector magnitude.
        in_name : str
            The name of the input vector.
        units : str or None
            The units of the input vector.
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

        # add inputs and outputs for all products
        if self._static_mode:
            var_rel2meta = self._static_var_rel2meta
            var_rel_names = self._static_var_rel_names
        else:
            var_rel2meta = self._var_rel2meta
            var_rel_names = self._var_rel_names

        if mag_name not in var_rel2meta:
            self.add_output(name=mag_name, shape=(vec_size,), units=units)
        elif mag_name in var_rel_names['input']:
            raise NameError(f"{self.msginfo}: '{mag_name}' specified as an output, "
                            "but it has already been defined as an input.")
        else:
            raise NameError(f"{self.msginfo}: Multiple definition of output '{mag_name}'.")

        if in_name not in var_rel2meta:
            self.add_input(name=in_name, shape=(vec_size, length), units=units)
        elif in_name in var_rel_names['output']:
            raise NameError(f"{self.msginfo}: '{in_name}' specified as an input, "
                            "but it has already been defined as an output.")
        else:
            # declaring a duplicate magnitude with a different output name?  okay...
            meta = var_rel2meta[in_name]
            if units != meta['units']:
                raise ValueError(f"{self.msginfo}: Conflicting units '{units}' specified for "
                                 f"input '{in_name}', which has already been defined with units "
                                 f"'{meta['units']}'.")
            if vec_size != meta['shape'][0]:
                raise ValueError(f"{self.msginfo}: Conflicting vec_size={vec_size} specified "
                                 f"for input '{in_name}', which has already been defined with "
                                 f"vec_size={meta['shape'][0]}.")
            if length != meta['shape'][1]:
                raise ValueError(f"{self.msginfo}: Conflicting length={length} specified "
                                 f"for input '{in_name}', which has already been defined with "
                                 f"length={meta['shape'][0]}.")

        row_idxs = np.repeat(np.arange(vec_size), length)
        col_idxs = np.arange(vec_size * length)
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
