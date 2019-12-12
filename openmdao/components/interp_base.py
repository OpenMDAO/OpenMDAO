from six import raise_from, iteritems, itervalues
from six.moves import range

import numpy as np

from openmdao.components.interp_util.outofbounds_error import OutOfBoundsError
from openmdao.components.interp_util.python_interp import PythonGridInterp
from openmdao.components.interp_util.scipy_interp import ScipyGridInterp
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

        opts = {}
        if 'interp_options' in self.options:
            opts = self.options['interp_options']
        for name, train_data in iteritems(self.training_outputs):
            # self.params is equal to the x_cp_val data
            # train_data is equal to y_cp_val data
            self.interps[name] = interp(self.params, train_data,
                                        interp_method=interp_method,
                                        bounds_error=not self.options['extrapolate'],
                                        **opts)

        if self.options['training_data_gradients']:
            self.grad_shape = tuple([self.options['vec_size']] + [i.size for i in self.params])

        super(InterpBase, self)._setup_var_data(recurse=recurse)


