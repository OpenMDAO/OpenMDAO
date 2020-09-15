
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

class BrokenDerivComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x1', 3.0)
        self.add_input('x2', 5.0)

        self.add_output('y', 5.5)

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        """ Compute outputs. """
        outputs['y'] = 3.0 * inputs['x1'] + 4.0 * inputs['x2']

    def compute_partials(self, inputs, partials):
        """Intentionally incorrect derivative."""
        J = partials
        J['y', 'x1'] = np.array([4.0])
        J['y', 'x2'] = np.array([40])


if __name__ == '__main__':
    prob = om.Problem()
    prob.model = BrokenDerivComp()

    prob.set_solver_print(level=0)

    prob.setup()
    prob.run_model()

    data = prob.check_partials(out_stream=None)

    try:
        assert_check_partials(data, atol=1.e-6, rtol=1.e-6)
    except ValueError as err:
        print(str(err))

