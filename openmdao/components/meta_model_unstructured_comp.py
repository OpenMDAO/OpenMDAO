"""MetaModel provides basic meta modeling capability."""
from six.moves import range
from copy import deepcopy
from itertools import chain

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.surrogate_models.surrogate_model import SurrogateModel
from openmdao.utils.class_util import overrides_method
from openmdao.utils.general_utils import warn_deprecation


class MetaModelUnStructuredComp(ExplicitComponent):
    """
    Class that creates a reduced order model for outputs from inputs.

    Each output may have its own surrogate model.
    Training inputs and outputs are automatically created with 'train:' prepended to the
    corresponding input/output name.

    For a Float variable, the training data is an array of length m,
    where m is the number of training points.

    Attributes
    ----------
    train : bool
        If True, training will occur on the next execution.
    _input_size : int
        Keeps track of the cumulative size of all inputs.
    _surrogate_input_names : [str, ..]
        List of inputs that are not the training vars.
    _surrogate_output_names : [str, ..]
        List of outputs that are not the training vars.
    _training_input : dict
        Training data for inputs.
    _training_output : dict
        Training data for outputs.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super(MetaModelUnStructuredComp, self).__init__(**kwargs)

        # keep list of inputs and outputs that are not the training vars
        self._surrogate_input_names = []
        self._surrogate_output_names = []

        # training will occur on first execution
        self.train = True
        self._training_input = np.empty(0)
        self._training_output = {}

        self._input_size = 0

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('default_surrogate', types=(SurrogateModel, type(None)), default=None,
                             desc="Surrogate that will be used for all outputs that don't have a "
                                  "specific surrogate assigned to them.")
        self.options.declare('vec_size', types=int, default=1, lower=1,
                             desc='Number of points that will be simultaneously predicted by '
                                  'the surrogate.')

    def add_input(self, name, val=1.0, training_data=None, **kwargs):
        """
        Add an input to this component and a corresponding training input.

        Parameters
        ----------
        name : string
            Name of the input.
        val : float or ndarray
            Initial value for the input.
        training_data : float or ndarray
            training data for this variable. Optional, can be set by the problem later.
        **kwargs : dict
            Additional agruments for add_input.

        Returns
        -------
        dict
            metadata for added variable
        """
        metadata = super(MetaModelUnStructuredComp, self).add_input(name, val, **kwargs)
        vec_size = self.options['vec_size']

        if vec_size > 1:
            if metadata['shape'][0] != vec_size:
                raise RuntimeError("Metamodel: First dimension of input '%s' must be %d"
                                   % (name, vec_size))
            input_size = metadata['value'][0].size
        else:
            input_size = metadata['value'].size

        self._surrogate_input_names.append((name, input_size))
        self._input_size += input_size

        train_name = 'train:%s' % name
        self.options.declare(train_name, default=None, desc='Training data for %s' % name)
        if training_data is not None:
            self.options[train_name] = training_data

        return metadata

    def add_output(self, name, val=1.0, training_data=None, surrogate=None, **kwargs):
        """
        Add an output to this component and a corresponding training output.

        Parameters
        ----------
        name : string
            Name of the variable output.
        val : float or ndarray
            Initial value for the output. While the value is overwritten during execution, it is
            useful for inferring size.
        training_data : float or ndarray
            Training data for this variable. Optional, can be set by the problem later.
        surrogate : <SurrogateModel>, optional
            Surrogate model to use for this output; if None, use default surrogate.
        **kwargs : dict
            Additional arguments for add_output.

        Returns
        -------
        dict
            metadata for added variable
        """
        metadata = super(MetaModelUnStructuredComp, self).add_output(name, val, **kwargs)
        vec_size = self.options['vec_size']

        if vec_size > 1:
            if metadata['shape'][0] != vec_size:
                raise RuntimeError("Metamodel: First dimension of output '%s' must be %d"
                                   % (name, vec_size))
            output_shape = metadata['shape'][1:]
            if len(output_shape) == 0:
                output_shape = 1
        else:
            output_shape = metadata['shape']

        self._surrogate_output_names.append((name, output_shape))
        self._training_output[name] = np.zeros(0)

        # Note: the default_surrogate flag is stored in metadata so that we can reconfigure
        # with a new default by rerunning setup.
        if surrogate:
            metadata['surrogate'] = surrogate
            metadata['default_surrogate'] = False
        else:
            metadata['default_surrogate'] = True

        train_name = 'train:%s' % name
        self.options.declare(train_name, default=None, desc='Training data for %s' % name)
        if training_data is not None:
            self.options[train_name] = training_data

        return metadata

    def _setup_vars(self, recurse=True):
        """
        Count total variables.

        Also instantiates surrogates for the output variables that use the default surrogate.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        # Create an instance of the default surrogate for outputs that did not have a surrogate
        # specified.
        default_surrogate = self.options['default_surrogate']
        if default_surrogate is not None:
            for name, shape in self._surrogate_output_names:
                metadata = self._metadata(name)
                if metadata.get('default_surrogate'):
                    surrogate = deepcopy(default_surrogate)
                    metadata['surrogate'] = surrogate

        # training will occur on first execution after setup
        self.train = True

        super(MetaModelUnStructuredComp, self)._setup_vars()

    def _setup_partials(self, recurse=True):
        """
        Process all partials and approximations that the user declared.

        Metamodel needs to declare its partials after inputs and outputs are known.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(MetaModelUnStructuredComp, self)._setup_partials()

        vec_size = self.options['vec_size']
        if vec_size > 1:
            # Sparse specification of partials for vectorized models.
            for wrt, n_wrt in self._surrogate_input_names:
                for of, shape_of in self._surrogate_output_names:

                    n_of = np.prod(shape_of)
                    rows = np.repeat(np.arange(n_of), n_wrt)
                    cols = np.tile(np.arange(n_wrt), n_of)
                    nnz = len(rows)
                    rows = np.tile(rows, vec_size) + np.repeat(np.arange(vec_size), nnz) * n_of
                    cols = np.tile(cols, vec_size) + np.repeat(np.arange(vec_size), nnz) * n_wrt

                    self._declare_partials(of=of, wrt=wrt, rows=rows, cols=cols)

        else:
            # Dense specification of partials for non-vectorized models.
            self._declare_partials(of=[name[0] for name in self._surrogate_output_names],
                                   wrt=[name[0] for name in self._surrogate_input_names])

    def check_config(self, logger):
        """
        Perform optional error checks.

        Parameters
        ----------
        logger : object
            The object that manages logging output.
        """
        # All outputs must have surrogates assigned either explicitly or through the default
        # surrogate.
        if self.options['default_surrogate'] is None:
            no_sur = []
            for name, shape in self._surrogate_output_names:
                surrogate = self._metadata(name).get('surrogate')
                if surrogate is None:
                    no_sur.append(name)
            if len(no_sur) > 0:
                msg = ("No default surrogate model is defined and the following"
                       " outputs do not have a surrogate model:\n%s\n"
                       "Either specify a default_surrogate, or specify a "
                       "surrogate model for all outputs."
                       % no_sur)
                logger.error(msg)

    def compute(self, inputs, outputs):
        """
        Predict outputs.

        If the training flag is set, train the metamodel first.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        vec_size = self.options['vec_size']

        # train first
        if self.train:
            self._train()

        # predict for current inputs
        if vec_size > 1:
            flat_inputs = self._vec_to_array_vectorized(inputs)
        else:
            flat_inputs = self._vec_to_array(inputs)

        for name, shape in self._surrogate_output_names:
            surrogate = self._metadata(name).get('surrogate')

            if vec_size == 1:
                # Non vectorized.
                predicted = surrogate.predict(flat_inputs)
                if isinstance(predicted, tuple):  # rmse option
                    self._metadata(name)['rmse'] = predicted[1]
                    predicted = predicted[0]
                outputs[name] = np.reshape(predicted, shape)

            elif overrides_method('vectorized_predict', surrogate, SurrogateModel):
                # Vectorized; surrogate provides vectorized computation.
                # TODO: This code is untested because no surrogates provide this option.
                predicted = surrogate.vectorized_predict(flat_inputs.flat)
                if isinstance(predicted, tuple):  # rmse option
                    self._metadata(name)['rmse'] = predicted[1]
                    predicted = predicted[0]
                outputs[name] = np.reshape(predicted, shape)

            else:
                # Vectorized; must call surrogate multiple times.
                if isinstance(shape, tuple):
                    output_shape = (vec_size, ) + shape
                else:
                    output_shape = (vec_size, )
                predicted = np.zeros(output_shape)
                rmse = self._metadata(name)['rmse'] = []
                for i in range(vec_size):
                    pred_i = surrogate.predict(flat_inputs[i])
                    if isinstance(pred_i, tuple):  # rmse option
                        rmse.append(pred_i[1])
                        pred_i = pred_i[0]
                    predicted[i] = np.reshape(pred_i, shape)

                outputs[name] = np.reshape(predicted, output_shape)

    def _vec_to_array(self, vec):
        """
        Convert from a dictionary of inputs to a flat ndarray.

        Parameters
        ----------
        vec : <Vector>
            pointer to the input vector.

        Returns
        -------
        ndarray
            flattened array of input data
        """
        array_real = True

        arr = np.zeros(self._input_size)

        idx = 0
        for name, sz in self._surrogate_input_names:
            val = vec[name]
            if array_real and np.issubdtype(val.dtype, np.complexfloating):
                array_real = False
                arr = arr.astype(np.complexfloating)
            arr[idx:idx + sz] = val.flat
            idx += sz

        return arr

    def _vec_to_array_vectorized(self, vec):
        """
        Convert from a dictionary of inputs to a 2d ndarray with vec_size rows.

        Parameters
        ----------
        vec : <Vector>
            pointer to the input vector.

        Returns
        -------
        ndarray
            2d array, self._vectorize rows of flattened input data.
        """
        vec_size = self.options['vec_size']
        array_real = True

        arr = np.zeros((vec_size, self._input_size))

        for row in range(vec_size):
            idx = 0
            for name, sz in self._surrogate_input_names:
                val = vec[name]
                if array_real and np.issubdtype(val.dtype, np.complexfloating):
                    array_real = False
                    arr = arr.astype(np.complexfloating)
                arr[row][idx:idx + sz] = val[row].flat
                idx += sz

        return arr

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        vec_size = self.options['vec_size']

        if vec_size > 1:
            flat_inputs = self._vec_to_array_vectorized(inputs)
        else:
            flat_inputs = self._vec_to_array(inputs)

        for out_name, out_shape in self._surrogate_output_names:
            surrogate = self._metadata(out_name).get('surrogate')
            if vec_size > 1:
                out_size = np.prod(out_shape)
                for j in range(vec_size):
                    flat_input = flat_inputs[j]
                    if overrides_method('linearize', surrogate, SurrogateModel):
                        derivs = surrogate.linearize(flat_input)
                        idx = 0
                        for in_name, sz in self._surrogate_input_names:
                            j1 = j * out_size * sz
                            j2 = j1 + out_size * sz
                            partials[out_name, in_name][j1:j2] = derivs[:, idx:idx + sz].flat
                            idx += sz

            else:
                if overrides_method('linearize', surrogate, SurrogateModel):
                    # print( "calling {}".format(surrogate.linearize))
                    sjac = surrogate.linearize(flat_inputs)

                    idx = 0
                    for in_name, sz in self._surrogate_input_names:
                        partials[(out_name, in_name)] = sjac[:, idx:idx + sz]
                        idx += sz

    def _train(self):
        """
        Train the metamodel, if necessary, using the provided training data.
        """
        missing_training_data = []
        num_sample = None
        for name, _ in chain(self._surrogate_input_names, self._surrogate_output_names):
            train_name = 'train:' + name
            val = self.options[train_name]
            if val is None:
                missing_training_data.append(train_name)
                continue

            if num_sample is None:
                num_sample = len(val)
            elif len(val) != num_sample:
                msg = "MetaModelUnStructuredComp: Each variable must have the same number"\
                      " of training points. Expected {0} but found {1} "\
                      "points for '{2}'."\
                      .format(num_sample, len(val), name)
                raise RuntimeError(msg)

        if len(missing_training_data) > 0:
            msg = "MetaModelUnStructuredComp: The following training data sets must be " \
                  "provided as options for %s: " % self.pathname + \
                  str(missing_training_data)
            raise RuntimeError(msg)

        inputs = np.zeros((num_sample, self._input_size))
        self._training_input = inputs

        # Assemble input data.
        idx = 0
        for name, sz in self._surrogate_input_names:
            val = self.options['train:' + name]
            if isinstance(val[0], float):
                inputs[:, idx] = val
                idx += 1
            else:
                for row_idx, v in enumerate(val):
                    v = np.asarray(v)
                    inputs[row_idx, idx:idx + sz] = v.flat

        # Assemble output data and train each output.
        for name, shape in self._surrogate_output_names:
            output_size = np.prod(shape)

            outputs = np.zeros((num_sample, output_size))
            self._training_output[name] = outputs

            val = self.options['train:' + name]

            if isinstance(val[0], float):
                outputs[:, 0] = val
            else:
                for row_idx, v in enumerate(val):
                    v = np.asarray(v)
                    outputs[row_idx, :] = v.flat

            surrogate = self._metadata(name).get('surrogate')
            if surrogate is None:
                raise RuntimeError("Metamodel '%s': No surrogate specified for output '%s'"
                                   % (self.pathname, name))
            else:
                surrogate.train(self._training_input,
                                self._training_output[name])

        self.train = False

    def _metadata(self, name):
        return self._var_rel2data_io[name]['metadata']

    @property
    def default_surrogate(self):
        """
        Get the default surrogate for this MetaModel.
        """
        warn_deprecation("The 'default_surrogate' attribute provides backwards compatibility "
                         "with earlier version of OpenMDAO; use options['default_surrogate'] "
                         "instead.")
        return self.options['default_surrogate']

    @default_surrogate.setter
    def default_surrogate(self, value):
        warn_deprecation("The 'default_surrogate' attribute provides backwards compatibility "
                         "with earlier version of OpenMDAO; use options['default_surrogate'] "
                         "instead.")
        self.options['default_surrogate'] = value


class MetaModel(MetaModelUnStructuredComp):
    """
    Deprecated.
    """

    def __init__(self, *args, **kwargs):
        """
        Capture Initialize to throw warning.

        Parameters
        ----------
        *args : list
            Deprecated arguments.
        **kwargs : dict
            Deprecated arguments.
        """
        warn_deprecation("'MetaModel' has been deprecated. Use "
                         "'MetaModelUnStructuredComp' instead.")
        super(MetaModel, self).__init__(*args, **kwargs)


class MetaModelUnStructured(MetaModelUnStructuredComp):
    """
    Deprecated.
    """

    def __init__(self, *args, **kwargs):
        """
        Capture Initialize to throw warning.

        Parameters
        ----------
        *args : list
            Deprecated arguments.
        **kwargs : dict
            Deprecated arguments.
        """
        warn_deprecation("'MetaModelUnStructured' has been deprecated. Use "
                         "'MetaModelUnStructuredComp' instead.")
        super(MetaModelUnStructured, self).__init__(*args, **kwargs)
