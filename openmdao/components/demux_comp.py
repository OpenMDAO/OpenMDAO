"""Definition of the Demux Component."""


import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.om_warnings import warn_deprecation
from openmdao.utils.array_utils import shape_to_len


class DemuxComp(ExplicitComponent):
    """
    Demux one or more inputs along a given axis.

    Parameters
    ----------
    **kwargs : dict
        Arguments to be passed to the component initialization method.

    Attributes
    ----------
    _vars : dict
        Container mapping name of variables to be demuxed with additional data.
    _output_names : dict
        Container mapping name of variables to be demuxed with associated outputs.
    """

    def __init__(self, **kwargs):
        """
        Instantiate DemuxComp and populate private members.
        """
        super().__init__(**kwargs)

        self._vars = {}
        self._output_names = {}

        self._no_check_partials = True

        warn_deprecation("DemuxComp is being deprecated. This same functionality can be achieved "
                         "directly in the connect/promotes indices arg using om.slicer.")

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('vec_size', types=int, default=2,
                             desc='The number of elements to be extracted from each input.')

    def add_var(self, name, val=1.0, shape=None, units=None, desc='', axis=0):
        """
        Add an input variable to be demuxed, and all associated output variables.

        Parameters
        ----------
        name : str
            Name of the variable in this component's namespace.
        val : float or list or tuple or ndarray or Iterable
            The initial value of the variable being added in user-defined units.
            Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if val is not an array. Default is None.
        units : str or None
            Units in which this input variable will be provided to the component
            during execution. Default is None, which means it is unitless.
        desc : str
            Description of the variable.
        axis : int
            The axis along which the elements will be selected.  Note the axis must have length
            vec_size, otherwise a RuntimeError is raised at setup.
        """
        self._vars[name] = {'val': val, 'shape': shape, 'units': units, 'desc': desc, 'axis': axis}

        opts = self.options
        vec_size = opts['vec_size']

        # for var, options in self._vars.items():
        options = self._vars[name]
        kwgs = dict(options)
        shape = options['shape']
        size = shape_to_len(shape)
        axis = kwgs.pop('axis')

        if axis >= len(shape):
            raise RuntimeError("{}: Invalid axis ({}) for variable '{}' of "
                               "shape {}".format(self.msginfo, axis, name, shape))

        if shape[axis] != vec_size:
            raise RuntimeError("{}: Variable '{}' cannot be demuxed along axis {}. Axis size "
                               "is {} but vec_size is {}.".format(self.msginfo, name, axis,
                                                                  shape[axis], vec_size))

        self.add_input(name, **kwgs)

        template = np.reshape(np.arange(size), shape)

        self._output_names[name] = []

        out_shape = list(shape)
        out_shape.pop(axis)
        if len(out_shape) == 0:
            out_shape = [1]

        for i in range(vec_size):
            out_name = '{0}_{1}'.format(name, i)
            self._output_names[name].append(out_name)
            self.add_output(name=out_name,
                            val=options['val'],
                            shape=out_shape,
                            units=options['units'],
                            desc=options['desc'])

            rs = np.arange(shape_to_len(out_shape))
            cs = np.atleast_1d(np.take(template, indices=i, axis=axis)).flatten()

            self.declare_partials(of=out_name, wrt=name, rows=rs, cols=cs, val=1.0)

    def compute(self, inputs, outputs):
        """
        Demux the inputs into the appropriate outputs using numpy.take.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        """
        opts = self.options
        vec_size = opts['vec_size']

        for var in self._vars:
            ax = self._vars[var]['axis']
            for i in range(vec_size):
                out_name = self._output_names[var][i]
                outputs[out_name] = np.reshape(np.take(inputs[var], indices=i, axis=ax),
                                               newshape=outputs[out_name].shape)
