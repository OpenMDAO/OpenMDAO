"""Definition of the Mux Component."""


import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class MuxComp(ExplicitComponent):
    """
    Mux one or more inputs along a given axis.

    Attributes
    ----------
    _vars : dict
        Container mapping name of variables to be muxed with additional data.
    _input_names : dict
        Container mapping name of variables to be muxed with associated inputs.
    """

    def __init__(self, **kwargs):
        """
        Instantiate MuxComp and populate private members.

        Parameters
        ----------
        **kwargs : dict
            Arguments to be passed to the component initialization method.
        """
        super(MuxComp, self).__init__(**kwargs)

        self._vars = {}
        self._input_names = {}

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('vec_size', types=int, default=2,
                             desc='The number of elements to be combined into an output.')

    def add_var(self, name, val=1.0, shape=None, units=None, desc='', axis=0):
        """
        Add an output variable to be muxed, and all associated input variables.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray or Iterable
            The initial value of the variable being added in user-defined units.
            Default is 1.0.
        shape : int or tuple or list or None
            Shape of the input variables to be muxed, only required if val is not an array.
            Default is None.
        units : str or None
            Units in which this input variable will be provided to the component
            during execution. Default is None, which means it is unitless.
        desc : str
            description of the variable
        axis : int
            The axis along which the elements will be stacked.  Note that N-dimensional inputs
            cannot be stacked along an axis greater than N.
        """
        self._vars[name] = {'val': val, 'shape': shape, 'units': units, 'desc': desc, 'axis': axis}

        opts = self.options
        vec_size = opts['vec_size']

        options = self._vars[name]

        kwgs = dict(options)
        in_shape = np.asarray(options['val']).shape \
            if options['shape'] is None else options['shape']
        in_size = np.prod(in_shape)
        out_shape = list(in_shape)
        out_shape.insert(options['axis'], vec_size)
        kwgs.pop('shape')
        ax = kwgs.pop('axis')

        in_dimension = len(in_shape)

        if ax > in_dimension:
            raise ValueError('{3}: Cannot mux a {0}D inputs for {2} along axis greater '
                             'than {0} ({1})'.format(in_dimension, ax, name, self.msginfo))

        self.add_output(name=name,
                        val=options['val'],
                        shape=out_shape,
                        units=options['units'],
                        desc=options['desc'])

        self._input_names[name] = []

        for i in range(vec_size):
            in_name = '{0}_{1}'.format(name, i)
            self._input_names[name].append(in_name)

            self.add_input(name=in_name, shape=in_shape, **kwgs)

            in_templates = [np.zeros(in_shape, dtype=int) for _ in range(vec_size)]

            rs = []
            cs = []

            for j in range(in_size):
                in_templates[i].flat[:] = 0
                in_templates[i].flat[j] = 1
                temp_out = np.stack(in_templates, axis=ax)
                cs.append(j)
                rs.append(int(np.nonzero(temp_out.ravel())[0]))

            self.declare_partials(of=name, wrt=in_name, rows=rs, cols=cs, val=1.0)

    def compute(self, inputs, outputs):
        """
        Mux the inputs into the appropriate outputs.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        opts = self.options
        vec_size = opts['vec_size']

        for var in self._vars:
            ax = self._vars[var]['axis']
            invar = self._input_names[var]
            vals = [inputs[invar[i]] for i in range(vec_size)]
            outputs[var][...] = np.stack(vals, axis=ax)
