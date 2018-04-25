"""Define the MultiFiMetaModel class."""

from six.moves import range

import numpy as np

from openmdao.components.meta_model_unstructured_comp import MetaModelUnStructuredComp
from openmdao.utils.general_utils import warn_deprecation


def _get_name_fi(name, fi_index):
    """
    Generate variable name taking into account fidelity level.

    Parameters
    ----------
    name : str
        base name
    fi_index : int
        fidelity level

    Returns
    -------
    str
        variable name
    """
    if fi_index > 0:
        return "%s_fi%d" % (name, fi_index + 1)
    else:
        return name


class MultiFiMetaModelUnStructuredComp(MetaModelUnStructuredComp):
    """
    Generalize MetaModel to be able to train surrogates with multi-fidelity training inputs.

    For a given number of levels of fidelity **nfi** (given at initialization)
    the corresponding training input variables *train:[invar]_fi[2..nfi]* and
    *train:[outvar]_fi[2..nfi]* are automatically created
    besides the given *train:[invar]* and *train:[outvar]* variables.
    Note the index starts at 2, the index 1 is omitted considering
    the simple name *var* is equivalent to *var_fi1* which is intended
    to be the data of highest fidelity.

    The surrogate models are trained with a list of (m samples, n dim)
    ndarrays built from the various training input data. By convention,
    the fidelities are intended to be ordered from highest to lowest fidelity.
    Obviously for a given level of fidelity corresponding lists
    *train:[var]_fi[n]* have to be of the same size.

    Thus given the initialization::

    >>> mm = MultiFiMetaModelUnStructuredComp(nfi=2)`
    >>> mm.add_input('x1', 0.)
    >>> mm.add_input('x2', 0.)
    >>> mm.add_ouput('y1', 0.)
    >>> mm.add_ouput('y2', 0.)

    the following supplementary training input variables
    ``train:x1_fi2`` and ``train:x2_fi2`` are created together with the classic
    ones ``train:x1`` and ``train:x2`` and the output variables ``train:y1_fi2``
    and ``train:y2_fi2`` are created as well.
    The embedded surrogate for y1 will be trained with a couple (X, Y).

    Where X is the list [X_fi1, X_fi2] where X_fi1 is an (m1, 2) ndarray
    filled with the m1 samples [x1 value, x2 value], X_fi2 is an (m2, 2) ndarray
    filled with the m2 samples [x1_fi2 value, x2_fi2 value]

    Where Y is a list [Y1_fi1, Y1_fi2] where Y1_fi1 is a (m1, 1) ndarray of
    y1 values and Y1_fi2 a (m2, 1) ndarray y1_fi2 values.

    .. note:: when *nfi* ==1 a :class:`MultiFiMetaModelUnStructuredComp` object behaves as
        a :class:`MetaModelUnStructured` object.

    Attributes
    ----------
    _input_sizes : list
        Stores the size of the inputs at each level.
    _nfi : float
        number of levels of fidelity
    """

    def __init__(self, nfi=1):
        """
        Initialize all attributes.

        Parameters
        ----------
        nfi : float
            number of levels of fidelity
        """
        super(MultiFiMetaModelUnStructuredComp, self).__init__()

        self._nfi = nfi

        # generalize MetaModelUnStructured training inputs to a list of training inputs
        self._training_input = nfi * [np.zeros(0)]
        self._input_sizes = nfi * [0]

    def add_input(self, name, val=1.0, shape=None, src_indices=None, flat_src_indices=None,
                  units=None, desc='', var_set=0):
        """
        Add an input variable to the component.

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
            If val is given as an array_like object, the shapes of val and
            src_indices must match. A value of None implies this input depends
            on all entries of source. Default is None.
        flat_src_indices : bool
            If True, each entry of src_indices is assumed to be an index into the
            flattened source.  Otherwise each entry must be a tuple or list of size equal
            to the number of dimensions of the source.
        units : str or None
            Units in which this input variable will be provided to the component
            during execution. Default is None, which means it is unitless.
        desc : str
            description of the variable
        var_set : hashable object
            For advanced users only. ID or color for this variable, relevant for
            reconfigurability. Default is 0.
        """
        item = MultiFiMetaModelUnStructuredComp
        metadata = super(item, self).add_input(name, val, shape=shape, src_indices=src_indices,
                                               flat_src_indices=flat_src_indices, units=units,
                                               desc=desc, var_set=var_set)
        input_size = metadata['value'].size

        self._input_sizes[0] = self._input_size

        # Add train:<invar>_fi<n>
        for fi in range(self._nfi):
            if fi > 0:
                name_with_fi = 'train:' + _get_name_fi(name, fi)
                self.metadata.declare(
                    name_with_fi, default=None, desc='Training data for %s' % name_with_fi)
                self._input_sizes[fi] += input_size

    def add_output(self, name, val=1.0, surrogate=None, shape=None, units=None, res_units=None,
                   desc='', lower=None, upper=None, ref=1.0, ref0=0.0, res_ref=1.0, var_set=0):
        """
        Add an output variable to the component.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        surrogate : SurrogateModel
            Surrogate model to use.
        shape : int or tuple or list or None
            Shape of this variable, only required if val is not an array.
            Default is None.
        units : str or None
            Units in which the output variables will be provided to the component during execution.
            Default is None, which means it has no units.
        res_units : str or None
            Units in which the residuals of this output will be given to the user when requested.
            Default is None, which means it has no units.
        desc : str
            description of the variable.
        lower : float or list or tuple or ndarray or Iterable or None
            lower bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no lower bound.
            Default is None.
        upper : float or list or tuple or ndarray or or Iterable None
            upper bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no upper bound.
            Default is None.
        ref : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 1. Default is 1.
        ref0 : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 0. Default is 0.
        res_ref : float
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 1. Default is 1.
        var_set : hashable object
            For advanced users only. ID or color for this variable, relevant for reconfigurability.
            Default is 0.
        """
        super(MultiFiMetaModelUnStructuredComp, self).add_output(name, val, shape=shape,
                                                                 units=units, res_units=res_units,
                                                                 desc=desc, lower=lower,
                                                                 upper=upper, ref=ref,
                                                                 ref0=ref0, res_ref=res_ref,
                                                                 var_set=var_set,
                                                                 surrogate=surrogate)
        self._training_output[name] = self._nfi * [np.zeros(0)]

        # Add train:<outvar>_fi<n>
        for fi in range(self._nfi):
            if fi > 0:
                name_with_fi = 'train:' + _get_name_fi(name, fi)
                self.metadata.declare(
                    name_with_fi, default=None, desc='Training data for %s' % name_with_fi)

    def _train(self):
        """
        Override MetaModelUnStructured _train method to take into account multi-fidelity input data.
        """
        if self._nfi == 1:
            # shortcut: fallback to base class behaviour immediatly
            super(MultiFiMetaModelUnStructuredComp, self)._train()
            return

        num_sample = self._nfi * [None]
        for name, sz in self._surrogate_input_names:
            for fi in range(self._nfi):
                name = _get_name_fi(name, fi)
                val = self.metadata['train:' + name]
                if num_sample[fi] is None:
                    num_sample[fi] = len(val)
                elif len(val) != num_sample[fi]:
                    msg = "MetaModelUnStructured: Each variable must have the same number"\
                          " of training points. Expected {0} but found {1} "\
                          "points for '{2}'."\
                          .format(num_sample[fi], len(val), name)
                    raise RuntimeError(msg)

        for name, shape in self._surrogate_output_names:
            for fi in range(self._nfi):
                name = _get_name_fi(name, fi)
                val = self.metadata['train:' + name]
                if len(val) != num_sample[fi]:
                    msg = "MetaModelUnStructured: Each variable must have the same number" \
                          " of training points. Expected {0} but found {1} " \
                          "points for '{2}'." \
                        .format(num_sample[fi], len(val), name)
                    raise RuntimeError(msg)

        if self.warm_restart:
            inputs = []
            new_inputs = self._nfi * [None]
            num_old_pts = self._nfi * [0]
            for fi in range(self._nfi):
                num_old_pts[fi] = self._training_input[fi].shape[0]
                inputs.append(np.zeros((num_sample[fi] + num_old_pts[fi],
                                        self._input_sizes[fi])))
                if num_old_pts[fi] > 0:
                    inputs[fi][:num_old_pts[fi], :] = self._training_input[fi]
                new_inputs[fi] = inputs[fi][num_old_pts[fi]:, :]
        else:
            inputs = [np.zeros((num_sample[fi], self._input_sizes[fi]))
                      for fi in range(self._nfi)]
            new_inputs = inputs

        self._training_input = inputs

        # add training data for each input
        idx = self._nfi * [0]
        for name, sz in self._surrogate_input_names:
            for fi in range(self._nfi):
                if num_sample[fi] > 0:
                    name = _get_name_fi(name, fi)
                    val = self.metadata['train:' + name]
                    if isinstance(val[0], float):
                        new_inputs[fi][:, idx[fi]] = val
                        idx[fi] += 1
                    else:
                        for row_idx, v in enumerate(val):
                            if not isinstance(v, np.ndarray):
                                v = np.array(v)
                            new_inputs[fi][row_idx, idx[
                                fi]:idx[fi] + sz] = v.flat

        # add training data for each output
        outputs = self._nfi * [None]
        new_outputs = self._nfi * [None]
        for name, shape in self._surrogate_output_names:
            for fi in range(self._nfi):
                name_fi = _get_name_fi(name, fi)
                if num_sample[fi] > 0:
                    output_size = np.prod(shape)
                    if self.warm_restart:
                        outputs[fi] = np.zeros((num_sample[fi] + num_old_pts[fi],
                                                output_size))
                        if num_old_pts[fi] > 0:
                            outputs[fi][:num_old_pts[fi],
                                        :] = self._training_output[name][fi]
                        self._training_output[name][fi] = outputs[fi]
                        new_outputs[fi] = outputs[fi][num_old_pts[fi]:, :]
                    else:
                        outputs[fi] = np.zeros((num_sample[fi], output_size))
                        self._training_output[name] = []
                        self._training_output[name].extend(outputs)
                        new_outputs = outputs

                    val = self.metadata['train:' + name_fi]

                    if isinstance(val[0], float):
                        new_outputs[fi][:, 0] = val
                    else:
                        for row_idx, v in enumerate(val):
                            if not isinstance(v, np.ndarray):
                                v = np.array(v)
                            new_outputs[fi][row_idx, :] = v.flat

            surrogate = self._metadata(name).get('surrogate')
            if surrogate is not None:
                surrogate.train_multifi(self._training_input,
                                        self._training_output[name])

        self.train = False


class MultiFiMetaModel(MultiFiMetaModelUnStructuredComp):
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
        warn_deprecation("'MultiFiMetaModel' component has been deprecated. Use"
                         "'MultiFiMetaModelUnStructuredComp' instead.")
        super(MultiFiMetaModel, self).__init__(*args, **kwargs)


class MultiFiMetaModelUnStructured(MultiFiMetaModelUnStructuredComp):
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
        warn_deprecation("'MultiFiMetaModelUnStructured' component has been deprecated. Use"
                         "'MultiFiMetaModelUnStructuredComp' instead.")
        super(MultiFiMetaModelUnStructured, self).__init__(*args, **kwargs)
