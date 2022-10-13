
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class IC(om.ImplicitComponent):

    def setup(self):
        self.add_input('x')
        self.add_output('y')

    def guess_nonlinear(self, inputs, outputs, residuals):
        outputs['y'] = 0.

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['y'] = inputs['x'] + outputs['y']

    def solve_nonlinear(self, inputs, outputs):
        outputs['y'] = -inputs['x']


class EC(om.ExplicitComponent):

    def setup(self):
        self.add_input('y')
        self.add_output('z')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs['z'] = inputs['y']



class ImplicitCompMPITestCase(unittest.TestCase):

    N_PROCS = 2

    def test_compute_and_derivs(self):
        prob = om.Problem()
        pg = om.ParallelGroup()
        pg.add_subsystem('ic', IC(), promotes=['*'])
        pg.add_subsystem('ec', EC(), promotes=['*'])
        prob.model.add_subsystem('pg', pg, promotes=['*'])
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.setup()
        prob.set_val('x', 1.)
        prob.run_model()

        assert_near_equal(prob.get_val('y', get_remote=True), [-1.])
        assert_near_equal(prob.get_val('z', get_remote=True), [-1.])

