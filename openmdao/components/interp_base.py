from six import raise_from, iteritems, itervalues
from six.moves import range

import numpy as np

from openmdao.components.structured_metamodel_util.outofbounds_error import OutOfBoundsError
from openmdao.components.structured_metamodel_util.python_interp import PythonGridInterp
from openmdao.components.structured_metamodel_util.scipy_interp import ScipyGridInterp
from openmdao.core.analysis_error import AnalysisError
from openmdao.core.explicitcomponent import ExplicitComponent

ALL_METHODS = ('cubic', 'slinear', 'lagrange2', 'lagrange3', 'akima',
               'scipy_cubic', 'scipy_slinear', 'scipy_quintic')

class InterpBase(ExplicitComponent):

    def __init__(self, **kwargs):

        super(InterpBase, self).__init__(**kwargs)
        self.pnames = []
        self.params = []
        self.training_outputs = {}
        self.interps = {}
        self.grad_shape = ()

    def _declare_options(self):
        """
        Initialize the component.
        """
        super(InterpBase, self)._declare_options()
        self.options.declare('extrapolate', types=bool, default=False,
                             desc='Sets whether extrapolation should be performed '
                                  'when an input is out of bounds.')
        self.options.declare('training_data_gradients', types=bool, default=False,
                             desc='Sets whether gradients with respect to output '
                                  'training data should be computed.')
        self.options.declare('vec_size', types=int, default=1,
                             desc='Number of points to evaluate at once.')
        self.options.declare('method', values=ALL_METHODS, default='scipy_cubic',
                             desc='Spline interpolation method to use for all outputs.')


    def _setup_var_data(self, recurse=True):
        """
        Instantiate surrogates for the output variables that use the default surrogate.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        interp_method = self.options['method']
        if interp_method.startswith('scipy'):
            interp = ScipyGridInterp
            interp_method = interp_method[6:]
        else:
            interp = PythonGridInterp

        for name, train_data in iteritems(self.training_outputs):
            # self.params is equal to the x_cp_val data
            # train_data is equal to y_cp_val data
            self.interps[name] = interp(self.params, train_data,
                                        interp_method=interp_method,
                                        bounds_error=not self.options['extrapolate'])

        if self.options['training_data_gradients']:
            self.grad_shape = tuple([self.options['vec_size']] + [i.size for i in self.params])

        super(InterpBase, self)._setup_var_data(recurse=recurse)


    def _setup_partials(self, recurse=True):
        """
        Process all partials and approximations that the user declared.

        Metamodel needs to declare its partials after inputs and outputs are known.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(InterpBase, self)._setup_partials()
        arange = np.arange(self.options['vec_size'])
        pnames = tuple(self.pnames)
        dct = {
            'rows': arange,
            'cols': arange,
            'dependent': True,
        }

        for name in self._outputs:
            self._declare_partials(of=name, wrt=pnames, dct=dct)
            if self.options['training_data_gradients']:
                self._declare_partials(of=name, wrt="%s_train" % name, dct={'dependent': True})

        # The scipy methods do not support complex step.
        if self.options['method'].startswith('scipy'):
            self.set_check_partial_options('*', method='fd')


    def compute(self, inputs, outputs):
        """
        Perform the interpolation at run time.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        pt = np.array([inputs[pname].flatten() for pname in self.pnames]).T
        for out_name, interp in iteritems(self.interps):
            if self.options['training_data_gradients']:
                # Training point values may have changed every time we compute.
                interp.values = inputs["%s_train" % out_name]
                interp.training_data_gradients = True

            try:
                val = interp.interpolate(pt)

            except OutOfBoundsError as err:
                varname_causing_error = '.'.join((self.pathname, self.pnames[err.idx]))
                errmsg = "{}: Error interpolating output '{}' because input '{}' " \
                    "was out of bounds ('{}', '{}') with " \
                    "value '{}'".format(self.msginfo, out_name, varname_causing_error,
                                        err.lower, err.upper, err.value)
                raise_from(AnalysisError(errmsg), None)

            except ValueError as err:
                raise ValueError("{}: Error interpolating output '{}':\n{}".format(self.msginfo,
                                                                                   out_name,
                                                                                   str(err)))
            outputs[out_name] = val


    def compute_partials(self, inputs, partials):
        """
        Collect computed partial derivatives and return them.

        Checks if the needed derivatives are cached already based on the
        inputs vector. Refreshes the cache by re-computing the current point
        if necessary.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        pt = np.array([inputs[pname].flatten() for pname in self.pnames]).T
        if self.options['training_data_gradients']:
            dy_ddata = np.zeros(self.grad_shape)
            interp = next(itervalues(self.interps))
            for j in range(self.options['vec_size']):
                val = interp.training_gradients(pt[j, :])
                dy_ddata[j] = val.reshape(self.grad_shape[1:])

        for out_name in self.interps:
            dval = self.interps[out_name].gradient(pt).T
            for i, p in enumerate(self.pnames):
                partials[out_name, p] = dval[i, :]

            if self.options['training_data_gradients']:
                partials[out_name, "%s_train" % out_name] = dy_ddata