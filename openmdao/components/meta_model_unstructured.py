"""MetaModel provides basic meta modeling capability."""

from six.moves import range

import numpy as np
from copy import deepcopy

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.surrogate_models.surrogate_model import SurrogateModel
from openmdao.utils.class_util import overrides_method
from openmdao.utils.general_utils import warn_deprecation


class MetaModelUnStructured(ExplicitComponent):
    """
    Class that creates a reduced order model for outputs from inputs.

    Each output may have it's own surrogate model.
    Training inputs and outputs are automatically created with
    'train:' prepended to the corresponding inputeter/output name.

    For a Float variable, the training data is an array of length m.

    Attributes
    ----------
    _surrogate_input_names : [str, ..]
        List of inputs that are not the training vars
    _surrogate_output_names : [str, ..]
        List of outputs that are not the training vars
    _vectorize : None or int
        First dimension of all inputs and outputs for case where data is vectorized
    warm_restart : bool
        When set to False (default), the metamodel retrains with the new
        dataset whenever the training data values are changed. When set to
        True, the new data is appended to the old data and all of the data
        is used to train.
    _surrogate_overrides : set
        keeps track of which sur_<name> slots are full.
    _training_input : dict
        Training data for inputs.
    _training_output : dict
        Training data for outputs.
    _input_size : int
        Keeps track of the cumulative size of all inputs.
    default_surrogate : str
        This surrogate will be used for all outputs that don't have
        a specific surrogate assigned to them
    train : bool
        If True, training will occur on the first execution.


    """

    def __init__(self, default_surrogate=None, vectorize=None):
        """
        Initialize all attributes.

        Parameters
        ----------
        default_surrogate : SurrogateModel
            Default surrogate model to use.
        vectorize : None or int
            First dimension of all inputs and outputs for case where data is vectorized, optional.
        """
        super(MetaModelUnStructured, self).__init__()

        # This surrogate will be used for all outputs that don't have
        # a specific surrogate assigned to them
        self.default_surrogate = default_surrogate

        # all inputs and outputs will have this many independent rows
        if vectorize and (not isinstance(vectorize, int) or vectorize < 2):
            raise RuntimeError("Metamodel: The value of the 'vectorize' "
                               "argument must be an integer greater than "
                               "one, found '%s'."
                               % vectorize)

        self._vectorize = vectorize

        # keep list of inputs and outputs that are not the training vars
        self._surrogate_input_names = []
        self._surrogate_output_names = []

        # training will occur on first execution
        self.train = True
        self._training_input = np.zeros(0)
        self._training_output = {}

        # When set to False (default), the metamodel retrains with the new
        # dataset whenever the training data values are changed. When set to
        # True, the new data is appended to the old data and all of the data
        # is used to train.
        self.warm_restart = False

        # keeps track of which sur_<name> slots are full
        self._surrogate_overrides = set()

        self._input_size = 0

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
            training data for this variable. Optional, can be set
            by the problem later.
        **kwargs : dict
            Additional agruments for add_input.

        Returns
        -------
        dict
            metadata for added variable
        """
        metadata = super(MetaModelUnStructured, self).add_input(name, val, **kwargs)

        if self._vectorize is not None:
            if metadata['shape'][0] != self._vectorize:
                raise RuntimeError("Metamodel: First dimension of input '%s' must be %d"
                                   % (name, self._vectorize))
            input_size = metadata['value'][0].size
        else:
            input_size = metadata['value'].size

        self._surrogate_input_names.append((name, input_size))
        self._input_size += input_size

        train_name = 'train:%s' % name
        self.metadata.declare(train_name, default=None, desc='Training data for %s' % name)
        if training_data is not None:
            self.metadata[train_name] = training_data

        return metadata

    def add_output(self, name, val=1.0, training_data=None, **kwargs):
        """
        Add an output to this component and a corresponding training output.

        Parameters
        ----------
        name : string
            Name of the variable output.
        val : float or ndarray
            Initial value for the output. While the value is overwritten during
            execution, it is useful for inferring size.
        training_data : float or ndarray
            training data for this variable. Optional, can be set
            by the problem later.
        **kwargs : dict
            Additional arguments for add_output.

        Returns
        -------
        dict
            metadata for added variable
        """
        surrogate = kwargs.pop('surrogate', None)

        metadata = super(MetaModelUnStructured, self).add_output(name, val, **kwargs)

        if self._vectorize is not None:
            if metadata['shape'][0] != self._vectorize:
                raise RuntimeError("Metamodel: First dimension of output '%s' must be %d"
                                   % (name, self._vectorize))
            output_shape = metadata['shape'][1:]
            if len(output_shape) == 0:
                output_shape = 1
        else:
            output_shape = metadata['shape']

        self._surrogate_output_names.append((name, output_shape))
        self._training_output[name] = np.zeros(0)

        if surrogate:
            metadata['surrogate'] = surrogate
            metadata['default_surrogate'] = False
        else:
            metadata['default_surrogate'] = True

        train_name = 'train:%s' % name
        self.metadata.declare(train_name, default=None, desc='Training data for %s' % name)
        if training_data is not None:
            self.metadata[train_name] = training_data

        return metadata

    def _setup_vars(self, recurse=True):
        """
        Call setup in components and count variables, total and by var_set.

        Also instantiates surrogates for the output variables that use the default surrogate.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        # create an instance of the default surrogate for outputs that
        # did not have a surrogate specified
        if self.default_surrogate is not None:
            for name, shape in self._surrogate_output_names:
                metadata = self._metadata(name)
                if metadata.get('default_surrogate'):
                    surrogate = deepcopy(self.default_surrogate)
                    metadata['surrogate'] = surrogate

        # training will occur on first execution after setup
        self.train = True

        super(MetaModelUnStructured, self)._setup_vars()

    def check_config(self, logger):
        """
        Perform optional error checks.

        Parameters
        ----------
        logger : object
            The object that manages logging output.
        """
        # All outputs must have surrogates assigned
        # either explicitly or through the default surrogate
        if self.default_surrogate is None:
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
        if self.train:  # train first
            self._train()

        # predict for current inputs
        if self._vectorize is None:
            inputs = self._vec_to_array(inputs)
        else:
            inputs = self._vec_to_array2d(inputs)

        for name, shape in self._surrogate_output_names:
            surrogate = self._metadata(name).get('surrogate')
            if surrogate is None:
                raise RuntimeError("Metamodel '%s': No surrogate specified for output '%s'"
                                   % (self.pathname, name))
            else:
                if self._vectorize is None:
                    # one input, one prediction
                    predicted = surrogate.predict(inputs)
                    if isinstance(predicted, tuple):  # rmse option
                        self._metadata(name)['rmse'] = predicted[1]
                        predicted = predicted[0]
                    outputs[name] = np.reshape(predicted, outputs[name].shape)
                elif overrides_method('vectorized_predict', surrogate, SurrogateModel):
                    # multiple inputs flattened, one prediction of multiple outputs
                    predicted = surrogate.vectorized_predict(inputs.flat)
                    if isinstance(predicted, tuple):  # rmse option
                        self._metadata(name)['rmse'] = predicted[1]
                        predicted = predicted[0]
                    outputs[name] = np.reshape(predicted, outputs[name].shape)
                else:
                    # multiple inputs, multiple predictions
                    if isinstance(shape, tuple):
                        output_shape = (self._vectorize,) + shape
                    else:
                        output_shape = (self._vectorize,)
                    predicted = np.zeros(output_shape)
                    rmse = self._metadata(name)['rmse'] = []
                    for i in range(self._vectorize):
                        pred_i = surrogate.predict(inputs[i])
                        predicted[i] = np.reshape(pred_i, shape)
                        if isinstance(predicted[i], tuple):  # rmse option
                            rmse.append(predicted[i][1])
                            predicted[i] = predicted[i][0]
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
            if isinstance(val, np.ndarray):
                if array_real and np.issubdtype(val.dtype, complex):
                    array_real = False
                    arr = arr.astype(complex)
                arr[idx:idx + sz] = val.flat
                idx += sz
            else:
                arr[idx] = val
                idx += 1

        return arr

    def _vec_to_array2d(self, vec):
        """
        Convert from a dictionary of inputs to a 2d ndarray with self._vectorize rows.

        Parameters
        ----------
        vec : <Vector>
            pointer to the input vector.

        Returns
        -------
        ndarray
            2d array, self._vectorize rows of flattened input data.
        """
        array_real = True

        arr = np.zeros((self._vectorize, self._input_size))

        for row in range(self._vectorize):
            idx = 0
            for name, sz in self._surrogate_input_names:
                val = vec[name]
                if isinstance(val, np.ndarray):
                    if array_real and np.issubdtype(val.dtype, complex):
                        array_real = False
                        arr = arr.astype(complex)
                    arr[row][idx:idx + sz] = val[row].flat
                    idx += sz
                else:
                    arr[row][idx] = val
                    idx += 1

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
        arr = self._vec_to_array(inputs)

        for uname, _ in self._surrogate_output_names:
            surrogate = self._metadata(uname).get('surrogate')
            sjac = surrogate.linearize(arr)

            idx = 0
            for pname, sz in self._surrogate_input_names:
                partials[(uname, pname)] = sjac[:, idx:idx + sz]
                idx += sz

    def _setup_partials(self, recurse=True):
        """
        Process all partials and approximations that the user declared.

        Metamodel needs to declare its partials after inputs and outputs are known.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(MetaModelUnStructured, self)._setup_partials()
        self._declare_partials(of=[name[0] for name in self._surrogate_output_names],
                               wrt=[name[0] for name in self._surrogate_input_names])

    def _train(self):
        """
        Train the metamodel, if necessary, using the provided training data.
        """
        missing_training_data = []
        num_sample = None
        for name, sz in self._surrogate_input_names:
            train_name = 'train:' + name
            val = self.metadata[train_name]
            if val is None:
                missing_training_data.append(train_name)
                continue
            if num_sample is None:
                num_sample = len(val)
            elif len(val) != num_sample:
                msg = "MetaModelUnStructured: Each variable must have the same number"\
                      " of training points. Expected {0} but found {1} "\
                      "points for '{2}'."\
                      .format(num_sample, len(val), name)
                raise RuntimeError(msg)

        for name, shape in self._surrogate_output_names:
            train_name = 'train:' + name
            val = self.metadata[train_name]
            if val is None:
                missing_training_data.append(train_name)
                continue
            if len(val) != num_sample:
                msg = "MetaModelUnStructured: Each variable must have the same number" \
                      " of training points. Expected {0} but found {1} " \
                      "points for '{2}'." \
                    .format(num_sample, len(val), name)
                raise RuntimeError(msg)

        if len(missing_training_data) > 0:
            msg = "MetaModelUnStructured: The following training data sets must be " \
                  "provided as metadata for %s: " % self.pathname + \
                  str(missing_training_data)
            raise RuntimeError(msg)

        if self.warm_restart:
            num_old_pts = self._training_input.shape[0]
            inputs = np.zeros((num_sample + num_old_pts, self._input_size))
            if num_old_pts > 0:
                inputs[:num_old_pts, :] = self._training_input
            new_input = inputs[num_old_pts:, :]
        else:
            inputs = np.zeros((num_sample, self._input_size))
            new_input = inputs

        self._training_input = inputs

        # add training data for each input
        if num_sample > 0:
            idx = 0
            for name, sz in self._surrogate_input_names:
                val = self.metadata['train:' + name]
                if isinstance(val[0], float):
                    new_input[:, idx] = val
                    idx += 1
                else:
                    for row_idx, v in enumerate(val):
                        if not isinstance(v, np.ndarray):
                            v = np.array(v)
                        new_input[row_idx, idx:idx + sz] = v.flat

        # add training data for each output
        for name, shape in self._surrogate_output_names:
            if num_sample > 0:
                output_size = np.prod(shape)

                if self.warm_restart:
                    outputs = np.zeros((num_sample + num_old_pts,
                                        output_size))
                    if num_old_pts > 0:
                        outputs[:num_old_pts, :] = self._training_output[name]
                    self._training_output[name] = outputs
                    new_output = outputs[num_old_pts:, :]
                else:
                    outputs = np.zeros((num_sample, output_size))
                    self._training_output[name] = outputs
                    new_output = outputs

                val = self.metadata['train:' + name]

                if isinstance(val[0], float):
                    new_output[:, 0] = val
                else:
                    for row_idx, v in enumerate(val):
                        if not isinstance(v, np.ndarray):
                            v = np.array(v)
                        new_output[row_idx, :] = v.flat

            surrogate = self._metadata(name).get('surrogate')
            if surrogate is not None:
                surrogate.train(self._training_input,
                                self._training_output[name])

        self.train = False

    def _metadata(self, name):
        return self._var_rel2data_io[name]['metadata']


class MetaModel(MetaModelUnStructured):
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
        warn_deprecation("'MetaModel' component has been deprecated. Use"
                         "'MetaModelUnStructured' instead.")
        super(Metamodel, self).__init__(*args, **kwargs)
