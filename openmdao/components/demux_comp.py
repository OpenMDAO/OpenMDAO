"""Definition of the Demux Component."""


from six import iteritems

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class DemuxComp(ExplicitComponent):
    """
    Demux one or more inputs along a given axis.
    """

    def __init__(self, **kwargs):
        super(DemuxComp, self).__init__(**kwargs)

        self._vars = {}
        self._output_names = {}

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('vec_size', types=int, default=2,
                             desc='The number of elements to be extracted from each input.')

    def add_var(self, name, val=1.0, shape=None, src_indices=None, flat_src_indices=None,
                units=None, desc='', axis=0):
        """
        Add an input variable to be demuxed, and all associated output variables

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray or Iterable
            The initial value of the variable being added in user-defined units.
            Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if src_indices not provided and
            val is not an array. Default is None.
        src_indices : int or list of ints or tuple of ints or int ndarray or Iterable or None
            The global indices of the source variable to transfer data from.
            A value of None implies this input depends on all entries of source.
            Default is None. The shapes of the target and src_indices must match,
            and form of the entries within is determined by the value of 'flat_src_indices'.
        flat_src_indices : bool
            If True, each entry of src_indices is assumed to be an index into the
            flattened source.  Otherwise each entry must be a tuple or list of size equal
            to the number of dimensions of the source.
        units : str or None
            Units in which this input variable will be provided to the component
            during execution. Default is None, which means it is unitless.
        desc : str
            description of the variable
        axis : int
            The axis along which the elements will be selected.  Note the axis must have length
            vec_size, otherwise a RuntimeError is raised at setup.
        """
        self._vars[name] = {'val': val, 'shape': shape, 'src_indices': src_indices,
                            'flat_src_indices': flat_src_indices, 'units': units,
                            'desc': desc, 'axis': axis}

    def setup(self):
        """
        Declare inputs, outputs, and derivatives for the demux component.
        """
        opts = self.options
        vec_size = opts['vec_size']

        for var, options in iteritems(self._vars):
            kwgs = dict(options)
            shape = options['shape']
            size = np.prod(shape)
            axis = kwgs.pop('axis')

            if axis >= len(shape):
                raise RuntimeError('Invalid axis ({0}) for variable of shape {1}'.format(axis,
                                                                                         shape))

            if options['shape'][axis] != vec_size:
                raise RuntimeError('Variable {0} cannot be demuxed along axis {1}. Axis size is {2}'
                                   ' but vec_size is {3}.'.format(var, axis, shape[axis], vec_size))

            self.add_input(var, **kwgs)

            template = np.reshape(np.arange(size), shape)

            self._output_names[var] = []

            out_shape = list(shape)
            out_shape.pop(axis)
            if len(out_shape) == 0:
                out_shape = [1]

            for i in range(vec_size):
                out_name = '{0}_{1}'.format(var, i)
                self._output_names[var].append(out_name)
                self.add_output(name=out_name,
                                val=options['val'],
                                shape=out_shape,
                                units=options['units'],
                                desc=options['desc'])

                rs = np.arange(np.prod(out_shape))
                cs = np.atleast_1d(np.take(template, indices=i, axis=axis))

                self.declare_partials(of=out_name, wrt=var, rows=rs, cols=cs, val=1.0)

    def compute(self, inputs, outputs):
        """
        Demux the inputs into the appropriate outputs using numpy.take.

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
            for i in range(vec_size):
                out_name = self._output_names[var][i]
                outputs[out_name] = np.reshape(np.take(inputs[var], indices=i, axis=ax),
                                               newshape=outputs[out_name].shape)
