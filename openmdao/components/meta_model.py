"""Metamodel provides basic Meta Modeling capability."""

import numpy as np
from copy import deepcopy

from openmdao.api import ExplicitComponent
from openmdao.core.component import _NotSet


class MetaModel(ExplicitComponent):
    """
    Class that creates a reduced order model for outputs from inputs.

    Each output may have it's own surrogate model.
    Training inputs and outputs are automatically created with
    'train:' prepended to the corresponding inputeter/output name.

    For a Float variable, the training data is an array of length m.

    Options
    -------
    deriv_options['type'] :  str('user')
        Derivative calculation type ('user', 'fd', 'cs')
        Default is 'user', where derivative is calculated from
        user-supplied derivatives. Set to 'fd' to finite difference
        this system. Set to 'cs' to perform the complex step
        if your components support it.
    deriv_options['form'] :  str('forward')
        Finite difference mode. (forward, backward, central)
    deriv_options['step_size'] :  float(1e-06)
        Default finite difference stepsize
    deriv_options['step_calc'] :  str('absolute')
        Set to absolute, relative
    deriv_options['check_type'] :  str('fd')
        Type of derivative check for check_partial_derivatives. Set
        to 'fd' to finite difference this system. Set to
        'cs' to perform the complex step method if
        your components support it.
    deriv_options['check_form'] :  str('forward')
        Finite difference mode: ("forward", "backward", "central")
        During check_partial_derivatives, the difference form that is used
        for the check.
    deriv_options['check_step_calc'] : str('absolute',)
        Set to 'absolute' or 'relative'. Default finite difference
        step calculation for the finite difference check in check_partial_derivatives.
    deriv_options['check_step_size'] :  float(1e-06)
        Default finite difference stepsize for the finite difference check
        in check_partial_derivatives"
    deriv_options['linearize'] : bool(False)
        Set to True if you want linearize to be called even though you are using FD.
    """

    def __init__(self):
        """
        Initialize all attributes.
        """
        super(MetaModel, self).__init__()

        # This surrogate will be used for all outputs that don't have
        # a specific surrogate assigned to them
        self.default_surrogate = None

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
        """
        metadata = super(MetaModel, self).add_input(name, val, **kwargs)
        input_size = metadata['value'].size

        self._surrogate_input_names.append((name, input_size))
        self._input_size += input_size

        train_name = 'train:%s' % name
        self.metadata.declare(train_name, desc='Training data for %s' % name)
        if training_data is not None:
            self.metadata[train_name] = training_data

        return metadata

    def add_output(self, name, val=_NotSet, training_data=None, num_training_points=None, **kwargs):
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
        """
        surrogate = kwargs.pop('surrogate', None)

        metadata = super(MetaModel, self).add_output(name, val, **kwargs)

        output_shape = metadata['shape']
        self._surrogate_output_names.append((name, output_shape))
        self._training_output[name] = np.zeros(0)

        if surrogate:
            metadata['surrogate'] = surrogate
            metadata['default_surrogate'] = False
        else:
            metadata['default_surrogate'] = True

        train_name = 'train:%s' % name
        self.metadata.declare(train_name, desc='Training data for %s' % name)
        if training_data is not None:
            self.metadata[train_name] = training_data

        return metadata

    def _setup_vars(self, recurse=True):
        """
        Return our inputs and outputs dictionaries re-keyed to use absolute variable names.

        Also instantiates surrogates for the output variables that use the default surrogate.
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

        return super(MetaModel, self)._setup_vars()

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
        # Train first
        if self.train:
            self._train()

        # Now Predict for current inputs
        inputs = self._vec_to_array(inputs)

        for name, shape in self._surrogate_output_names:
            surrogate = self._metadata(name).get('surrogate')
            if surrogate:
                predicted = surrogate.predict(inputs)
                if isinstance(predicted, np.ndarray) and len(predicted.shape) > 1:
                    outputs[name] = predicted[0]
                else:
                    outputs[name] = predicted
            else:
                raise RuntimeError("Metamodel '%s': No surrogate specified for output '%s'"
                                   % (self.pathname, name))

    def _vec_to_array(self, vec, out=None):
        """
        Convert from a dictionary of inputs to the ndarray input.
        """
        array_real = True

        if out is None:
            arr = np.zeros(self._input_size)
        else:
            arr = out

        idx = 0
        for name, sz in self._surrogate_input_names:
            val = vec[name]
            if isinstance(val, list):
                val = np.array(val)
            if isinstance(val, np.ndarray):
                if array_real and np.issubdtype(val.dtype, complex):
                    array_real = False
                    inputs = inputs.astype(complex)
                arr[idx:idx + sz] = val.flat
                idx += sz
            else:
                arr[idx] = val
                idx += 1

        return arr

    def compute_partial_derivs(self, inputs, outputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
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
                msg = "MetaModel: Each variable must have the same number"\
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
                msg = "MetaModel: Each variable must have the same number" \
                      " of training points. Expected {0} but found {1} " \
                      "points for '{2}'." \
                    .format(num_sample, len(val), name)
                raise RuntimeError(msg)

        if len(missing_training_data) > 0:
            msg = "MetaModel: The following training data sets must be " \
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
        return self._static_var_rel2data_io[name]['metadata']
