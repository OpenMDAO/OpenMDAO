import unittest

import numpy as np

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_no_warning


SIZE = 5

class MyTestComp(om.ExplicitComponent):
    def __init__(self, neg_iter=4):
        super().__init__()
        self.neg_iter = neg_iter
        self.count = 0

    def setup(self):
        self.add_input('x', np.ones(SIZE))
        self.add_output('y', np.ones(SIZE))

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        self.count += 1
        if self.count == self.neg_iter:
            outputs['y'] = -99.
        else:
            outputs['y'] = 3.0*inputs['x']


class MyLogComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', np.ones(SIZE))
        self.add_output('y', np.ones(SIZE))

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['y'] = 3.0*np.log(inputs['x']) + inputs['x']**3


class TestSolverFeatures(unittest.TestCase):

    def test_simple_caching(self):
        p = om.Problem()
        p.model.nonlinear_solver = om.NonlinearBlockGS()
        p.model.nonlinear_solver.options['err_on_non_converge'] = True

        p.model.add_subsystem('C1', MyTestComp(neg_iter=9999))
        p.model.add_subsystem('C2', MyLogComp())
        p.model.connect('C1.y', 'C2.x')
        p.model.connect('C2.y', 'C1.x')

        p.setup()
        p.run_model()

        print("iters:", p.model.nonlinear_solver._iter_count)

