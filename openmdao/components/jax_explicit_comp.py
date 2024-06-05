
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.jax_utils import jax, jit, ExplicitCompJaxify, jax_deriv_shape


class JaxExplicitComponent(ExplicitComponent):
    def __init__(self, fallback_deriv_method='fd', **kwargs):
        super().__init__(**kwargs)
        self.options['derivs_method'] = 'jax' if jax else fallback_deriv_method
        self.options.declare('use_jit', types=bool, default=True,
                             desc='If True, use jit on the function')


