"""Definition of the Dot Product Component."""

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
        super(DotProductComp, self).__init__(**kwargs)

        self._products = []

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('vec_size', types=int, default=1,
                             desc='The number of points at which the dot product is computed')
        self.options.declare('length', types=int, default=3,
                             desc='The length of vectors a and b')
        self.options.declare('a_name', types=str, default='a',
                             desc='The variable name for input vector a.')
        self.options.declare('b_name', types=str, default='b',
                             desc='The variable name for input vector b.')
        self.options.declare('c_name', types=str, default='c',
                             desc='The variable name for output vector c.')
        self.options.declare('a_units', types=str, default=None, allow_none=True,
                             desc='The units for vector a.')
        self.options.declare('b_units', types=str, default=None, allow_none=True,
                             desc='The units for vector b.')
        self.options.declare('c_units', types=str, default=None, allow_none=True,
                             desc='The units for vector c.')

    def add_product(self, a_name, b_name, c_name, a_units=None, b_units=None, c_units=None,
                    vec_size=1, length=3):
        """
        Add a new output product to the dot product component.

        Parameters
        ----------
        a_name : str
            The name of the first vector input.
        b_name : str
            The name of the second input.
        c_name : str
            The name of the vector product output.
        a_units : str or None
            The units of input a.
        b_units : str or None
            The units of input b.
        c_units : str or None
            The units of the output.
        vec_size : int
            The number of points at which the dot vector product
            should be computed simultaneously.  The shape of
            the output is (vec_size,).
        length : int
            The length of the vectors a and b.  Their shapes are
            (vec_size, length)
        """
        self._products.append({
            'a_name': a_name,
            'b_name': b_name,
            'c_name': c_name,
            'a_units': a_units,
            'b_units': b_units,
            'c_units': c_units,
            'vec_size': vec_size,
            'length': length
        })

    def setup(self):
        """
        Declare inputs, outputs, and derivatives for the dot product component.
        """
        products = self._products

        # prepend the product specified in component options
        opts = self.options

        products.insert(0, {
            'a_name': opts['a_name'],
            'b_name': opts['b_name'],
            'c_name': opts['c_name'],
            'a_units': opts['a_units'],
            'b_units': opts['b_units'],
            'c_units': opts['c_units'],
            'vec_size': opts['vec_size'],
            'length': opts['length']
        })

        # add inputs and outputs for all products
        var_rel2meta = self._var_rel2meta

        for product in products:
            a_name = product['a_name']
            b_name = product['b_name']
            c_name = product['c_name']
            a_units = product['a_units']
            b_units = product['b_units']
            c_units = product['c_units']
            vec_size = product['vec_size']
            length = product['length']

            if a_name not in var_rel2meta:
                self.add_input(name=a_name, shape=(vec_size, length), units=a_units)
            else:
                meta = var_rel2meta[a_name]
                if a_units != meta['units']:
                    raise ValueError(f"{self.msginfo}: Conflicting units specified for input "
                                     f"'{a_name}', '{meta['units']}' and '{a_units}'.")
                if vec_size != meta['shape'][0]:
                    raise ValueError(f"{self.msginfo}: Conflicting vec_size specified for input "
                                     f"'{a_name}', {meta['shape'][0]} versus {vec_size}.")
                if length != meta['shape'][1]:
                    raise ValueError(f"{self.msginfo}: Conflicting length specified for input "
                                     f"'{a_name}', {meta['shape'][1]} versus {length}.")

            if b_name not in var_rel2meta:
                self.add_input(name=b_name, shape=(vec_size, length), units=b_units)
            else:
                meta = var_rel2meta[b_name]
                if b_units != meta['units']:
                    raise ValueError(f"{self.msginfo}: Conflicting units specified for input "
                                     f"'{b_name}', '{meta['units']}' and '{b_units}'.")
                if vec_size != meta['shape'][0]:
                    raise ValueError(f"{self.msginfo}: Conflicting vec_size specified for input "
                                     f"'{b_name}', {meta['shape'][0]} versus {vec_size}.")
                if length != meta['shape'][1]:
                    raise ValueError(f"{self.msginfo}: Conflicting length specified for input "
                                     f"'{b_name}', {meta['shape'][1]} versus {length}.")

            if c_name not in var_rel2meta:
                self.add_output(name=c_name, shape=(vec_size,), units=c_units)
            else:
                raise NameError(f"{self.msginfo}: Multiple definition of output '{c_name}'.")

            row_idxs = np.repeat(np.arange(vec_size), length)
            col_idxs = np.arange(vec_size * length)

            self.declare_partials(of=c_name, wrt=a_name, rows=row_idxs, cols=col_idxs)
            self.declare_partials(of=c_name, wrt=b_name, rows=row_idxs, cols=col_idxs)

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
        for product in self._products:
            a = inputs[product['a_name']]
            b = inputs[product['b_name']]
            outputs[product['c_name']] = np.einsum('ni,ni->n', a, b)

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
        for product in self._products:
            a = inputs[product['a_name']]
            b = inputs[product['b_name']]

            # Use the following for sparse partials
            partials[product['c_name'], product['a_name']] = b.ravel()
            partials[product['c_name'], product['b_name']] = a.ravel()
