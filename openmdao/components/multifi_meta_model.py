import numpy as np

from openmdao.components.meta_model import MetaModel
from openmdao.core.component import _NotSet

# generate variable names taking into account fidelity level
def _get_name_fi(name, fi_index):
    if fi_index>0:
        return "%s_fi%d" % (name, fi_index+1)
    else:
        return name

class MultiFiMetaModel(MetaModel):
    """ Class that generalizes the MetaModel class to be able to train surrogates
    with multi-fidelity training inputs. For a given number of levels of fidelity
    **nfi** (given at initialization) the corresponding training input variables
    *train:<invar>_fi<2..nfi>* and *train:<outvar>_fi<2..nfi>* are automatically created
    besides the given *train:<invar>* and *train:<outvar>* variables.
    Note the index starts at 2, the index 1 is omitted considering
    the simple name *<var>* is equivalent to *<var>_fi1* which is intended
    to be the data of highest fidelity.

    The surrogate models are trained with a list of (m samples, n dim)
    ndarrays built from the various training input data. By convention,
    the fidelities are intended to be ordered from highest to lowest fidelity.
    Obviously for a given level of fidelity corresponding lists
    *train:<var>_fi<n>* have to be of the same size.

    Thus given the initialization::

    >>> mm = MultiFiMetaModel(nfi=2)`
    >>> mm.add_param('x1', 0.)
    >>> mm.add_param('x2', 0.)
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

    .. note:: when *nfi* ==1 a :class:`MultiFiMetaModel` object behaves as
        a :class:`MetaModel` object.

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

    def __init__(self, nfi=1):
        super(MultiFiMetaModel, self).__init__()

        self._nfi = nfi

        # generalize MetaModel training inputs to a list of training inputs
        self._training_input = nfi*[np.zeros(0)]
        self._input_sizes = nfi*[0]

    def add_param(self, name, val=_NotSet, **kwargs):
        super(MultiFiMetaModel, self).add_param(name, val, **kwargs)
        self._input_sizes[0]=self._input_size

        # Add train:<invar>_fi<n>
        for fi in range(self._nfi):
            if fi > 0:
                name_with_fi = 'train:'+_get_name_fi(name, fi)
                super(MetaModel, self).add_param(name_with_fi, val=[], pass_by_obj=True)
                self._input_sizes[fi]+=self._init_params_dict[name]['size']


    def add_output(self, name, val=_NotSet, **kwargs):
        super(MultiFiMetaModel, self).add_output(name, val, **kwargs)
        self._training_output[name]=self._nfi*[np.zeros(0)]

        # Add train:<outvar>_fi<n>
        for fi in range(self._nfi):
            if fi > 0:
                name_with_fi = 'train:'+_get_name_fi(name, fi)
                super(MetaModel, self).add_param(name_with_fi, val=[], pass_by_obj=True)

    def _train(self):
        """Override MetaModel _train method to take into account multi-fidelity
        input data. Basicall
        """

        if self._nfi==1:
            # shortcut: fallback to base class behaviour immediatly
            super(MultiFiMetaModel, self)._train()
            return

        num_sample = self._nfi*[None]
        for name, sz in self._surrogate_param_names:
            for fi in range(self._nfi):
                name = _get_name_fi(name, fi)
                val = self.params['train:' + name]
                if num_sample[fi] is None:
                    num_sample[fi] = len(val)
                elif len(val) != num_sample[fi]:
                    msg = "MetaModel: Each variable must have the same number"\
                          " of training points. Expected {0} but found {1} "\
                          "points for '{2}'."\
                          .format(num_sample[fi], len(val), name)
                    raise RuntimeError(msg)

        for name, shape in self._surrogate_output_names:
            for fi in range(self._nfi):
                name = _get_name_fi(name, fi)
                val = self.params['train:' + name]
                if len(val) != num_sample[fi]:
                    msg = "MetaModel: Each variable must have the same number" \
                          " of training points. Expected {0} but found {1} " \
                          "points for '{2}'." \
                        .format(num_sample[fi], len(val), name)
                    raise RuntimeError(msg)

        if self.warm_restart:
            inputs = []
            new_inputs = self._nfi*[None]
            num_old_pts = self._nfi*[0]
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
        idx = self._nfi*[0]
        for name, sz in self._surrogate_param_names:
            for fi in range(self._nfi):
                if num_sample[fi] > 0:
                    name = _get_name_fi(name, fi)
                    val = self.params['train:' + name]
                    if isinstance(val[0], float):
                        new_inputs[fi][:, idx[fi]] = val
                        idx[fi] += 1
                    else:
                        for row_idx, v in enumerate(val):
                            if not isinstance(v, np.ndarray):
                                v = np.array(v)
                            new_inputs[fi][row_idx, idx[fi]:idx[fi]+sz] = v.flat

        # add training data for each output
        outputs=self._nfi*[None]
        new_outputs=self._nfi*[None]
        for name, shape in self._surrogate_output_names:
            for fi in range(self._nfi):
                name_fi = _get_name_fi(name, fi)
                if num_sample[fi] > 0:
                    output_size = np.prod(shape)
                    if self.warm_restart:
                        outputs[fi] = np.zeros((num_sample[fi] + num_old_pts[fi],
                                                output_size))
                        if num_old_pts[fi] > 0:
                            outputs[fi][:num_old_pts[fi], :] = self._training_output[name][fi]
                        self._training_output[name][fi] = outputs[fi]
                        new_outputs[fi] = outputs[fi][num_old_pts[fi]:, :]
                    else:
                        outputs[fi] = np.zeros((num_sample[fi], output_size))
                        self._training_output[name] = []
                        self._training_output[name].extend(outputs)
                        new_outputs = outputs

                    val = self.params['train:' + name_fi]

                    if isinstance(val[0], float):
                        new_outputs[fi][:, 0] = val
                    else:
                        for row_idx, v in enumerate(val):
                            if not isinstance(v, np.ndarray):
                                v = np.array(v)
                            new_outputs[fi][row_idx, :] = v.flat

            surrogate = self._init_unknowns_dict[name].get('surrogate')
            if surrogate is not None:
                surrogate.train_multifi(self._training_input,
                                        self._training_output[name])

        self.train = False
