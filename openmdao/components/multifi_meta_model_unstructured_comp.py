"""Define the MultiFiMetaModel class."""
from itertools import chain

import numpy as np

from openmdao.components.meta_model_unstructured_comp import MetaModelUnStructuredComp


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
    >>> mm.add_output('y1', 0.)
    >>> mm.add_output('y2', 0.)

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
    _static_input_sizes : list
        Stores the size of the inputs at each level for inputs added outside of setup.
    _nfi : float
        number of levels of fidelity
    _training_input : dict
        Training data for inputs.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super().__init__(**kwargs)

        nfi = self._nfi = self.options['nfi']

        # generalize MetaModelUnStructured training inputs to a list of training inputs
        self._training_input = nfi * [np.empty(0)]
        self._input_sizes = nfi * [0]

        self._static_input_sizes = nfi * [0]

        self._no_check_partials = True

    def initialize(self):
        """
        Declare options.
        """
        super().initialize()

        self.options.declare('nfi', types=int, default=1, lower=1,
                             desc='Number of levels of fidelity.')

    def _setup_procs(self, pathname, comm, mode, prob_meta):
        """
        Execute first phase of the setup process.

        Distribute processors, assign pathnames, and call setup on the component.

        Parameters
        ----------
        pathname : str
            Global name of the system, including the path.
        comm : MPI.Comm or <FakeComm>
            MPI communicator object.
        mode : str
            Derivatives calculation mode, 'fwd' for forward, and 'rev' for
            reverse (adjoint).
        prob_meta : dict
            Problem level options.
        """
        self._input_sizes = list(self._static_input_sizes)

        super()._setup_procs(pathname, comm, mode, prob_meta)

    def add_input(self, name, val=1.0, shape=None, src_indices=None, flat_src_indices=None,
                  units=None, desc=''):
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
        """
        metadata = super().add_input(name, val, shape=shape, src_indices=src_indices,
                                     flat_src_indices=flat_src_indices, units=units,
                                     desc=desc)
        if self.options['vec_size'] > 1:
            input_size = metadata['value'][0].size
        else:
            input_size = metadata['value'].size

        if self._static_mode:
            self._static_input_sizes[0] += input_size
        else:
            self._input_sizes[0] += input_size

        # Add train:<invar>_fi<n>
        for fi in range(self._nfi):
            if fi > 0:
                name_with_fi = 'train:' + _get_name_fi(name, fi)
                self.options.declare(
                    name_with_fi, default=None, desc='Training data for %s' % name_with_fi)
                if self._static_mode:
                    self._static_input_sizes[fi] += input_size
                else:
                    self._input_sizes[fi] += input_size

    def add_output(self, name, val=1.0, surrogate=None, shape=None, units=None, res_units=None,
                   desc='', lower=None, upper=None, ref=1.0, ref0=0.0, res_ref=1.0, tags=None,
                   shape_by_conn=False, copy_shape=None):
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
        tags : str or list of strs or set of strs
            User defined tags that can be used to filter what gets listed when calling
            list_inputs and list_outputs.
        shape_by_conn : bool
            If True, shape this output to match its connected input(s).
        copy_shape : str or None
            If a str, that str is the name of a variable. Shape this output to match that of
            the named variable.
        """
        super().add_output(name, val, shape=shape,
                           units=units, res_units=res_units,
                           desc=desc, lower=lower,
                           upper=upper, ref=ref,
                           ref0=ref0, res_ref=res_ref,
                           surrogate=surrogate, tags=tags,
                           shape_by_conn=shape_by_conn,
                           copy_shape=copy_shape)
        self._training_output[name] = self._nfi * [np.empty(0)]

        # Add train:<outvar>_fi<n>
        for fi in range(self._nfi):
            if fi > 0:
                name_with_fi = 'train:' + _get_name_fi(name, fi)
                self.options.declare(
                    name_with_fi, default=None, desc='Training data for %s' % name_with_fi)

    def _train(self):
        """
        Override MetaModelUnStructured _train method to take into account multi-fidelity input data.
        """
        if self._nfi == 1:
            # shortcut: fallback to base class behaviour immediatly
            super()._train()
            return

        num_sample = self._nfi * [None]
        for name_root, _ in chain(self._surrogate_input_names, self._surrogate_output_names):
            for fi in range(self._nfi):
                name = _get_name_fi(name_root, fi)
                val = self.options['train:' + name]
                if num_sample[fi] is None:
                    num_sample[fi] = len(val)
                elif len(val) != num_sample[fi]:
                    msg = f"{self.msginfo}: Each variable must have the same number " \
                          f"of training points. Expected {num_sample[fi]} but found {len(val)} " \
                          f"points for '{name}'."
                    raise RuntimeError(msg)

        inputs = [np.zeros((num_sample[fi], self._input_sizes[fi]))
                  for fi in range(self._nfi)]

        # add training data for each input
        idx = self._nfi * [0]
        for name_root, sz in self._surrogate_input_names:
            for fi in range(self._nfi):
                name = _get_name_fi(name_root, fi)
                val = self.options['train:' + name]
                if isinstance(val[0], float):
                    inputs[fi][:, idx[fi]] = val
                    idx[fi] += 1
                else:
                    for row_idx, v in enumerate(val):
                        v = np.asarray(v)
                        inputs[fi][row_idx, idx[fi]:idx[fi] + sz] = v.flat

        # add training data for each output
        outputs = self._nfi * [None]
        for name_root, shape in self._surrogate_output_names:
            output_size = np.prod(shape)
            for fi in range(self._nfi):
                name_fi = _get_name_fi(name_root, fi)
                outputs[fi] = np.zeros((num_sample[fi], output_size))

                val = self.options['train:' + name_fi]

                if isinstance(val[0], float):
                    outputs[fi][:, 0] = val
                else:
                    for row_idx, v in enumerate(val):
                        v = np.asarray(v)
                        outputs[fi][row_idx, :] = v.flat

            self._training_output[name] = []
            self._training_output[name].extend(outputs)

            surrogate = self._metadata(name_root).get('surrogate')
            if surrogate is None:
                msg = f"{self.msginfo}: No surrogate specified for output '{name_root}'"
                raise RuntimeError(msg)
            else:
                surrogate.train_multifi(inputs, self._training_output[name])

        self._training_input = inputs
        self.train = False
