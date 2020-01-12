"""
Tests connections with Reconfigurable Model Execution.

Tests for absolute and promoted connections, for different nonlinear solvers.
"""
# FIXME Reconfiguration does not work with Newton and NLBGS solvers.

from __future__ import division

import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error


class ReconfComp1(om.ExplicitComponent):

    def initialize(self):
        self.size = 1
        self.counter = 0

    def reconfigure(self):
        self.counter += 1

        if self.counter % 2 == 0:
            self.size += 1
            flag = True
        else:
            flag = False
        return flag

    def setup(self):
        self.add_input('x', val=1.0)
        self.add_output('y', val=np.zeros(self.size))
        # All derivatives are defined.
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['y'] = 2 * inputs['x']

    def compute_partials(self, inputs, jacobian):
        jacobian['y', 'x'] = 2 * np.ones((self.size, 1))


class ReconfComp2(ReconfComp1):
    """The size of the y input changes the same as way as in ReconfComp"""

    def setup(self):
        self.add_input('y', val=np.zeros(self.size))
        self.add_output('f', val=np.zeros(self.size))
        # All derivatives are defined.
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['f'] = 2 * inputs['y']

    def compute_partials(self, inputs, jacobian):
        jacobian['f', 'y'] = 2 * np.ones((self.size, 1))


class TestReconfConnections(unittest.TestCase):

    def test_promoted_connections(self):
        p = om.Problem()

        p.model = model = om.Group()
        model.add_subsystem('c1', om.IndepVarComp('x', 1.0), promotes_outputs=['x'])
        model.add_subsystem('c2', ReconfComp1(), promotes_inputs=['x'], promotes_outputs=['y'])
        model.add_subsystem('c3', ReconfComp2(), promotes_inputs=['y'],
                            promotes_outputs=['f'])

        p.setup()
        p['x'] = 3.

        self.assertEqual(len(p['y']), 1)
        # First run the model once; counter = 1, size of y = 1
        p.run_model()

        totals = p.compute_totals(wrt=['x'], of=['y'])
        assert_rel_error(self, p['x'], 3.0)
        assert_rel_error(self, p['y'], 6.0)
        assert_rel_error(self, totals['y', 'x'], [[2.0]])

        # Run the model again, which will trigger reconfiguration; counter = 2, size of y = 2
        p.run_model()  # Fails with ValueError
        self.assertEqual(len(p['y']), 2)

    def test_abs_connections(self):
        p = om.Problem()

        p.model = model = om.Group()
        model.add_subsystem('c1', om.IndepVarComp('x', 1.0), promotes_outputs=['x'])
        model.add_subsystem('c2', ReconfComp1(), promotes_inputs=['x'])
        model.add_subsystem('c3', ReconfComp2(), promotes_outputs=['f'])
        model.connect('c2.y', 'c3.y')
        p.setup()
        p['x'] = 3.

        self.assertEqual(len(p['c2.y']), 1)
        self.assertEqual(len(p['c3.y']), 1)
        # Run the model again, which will trigger reconfiguration; counter = 2, size of y = 2
        p.run_model()

        # Run the model again, which will trigger reconfiguration; counter = 2, size of y = 2
        p.run_model()  # Fails with ValueError
        self.assertEqual(len(p['c2.y']), 2)
        self.assertEqual(len(p['c3.y']), 2)

    @unittest.expectedFailure
    def test_reconf_comp_connections_newton_solver(self):
        p = om.Problem()

        p.model = model = om.Group()
        model.linear_solver = om.DirectSolver()
        model.nonlinear_solver = nl = om.NewtonSolver()
        nl.linesearch = om.BoundsEnforceLS()
        model.add_subsystem('c1', om.IndepVarComp('x', 1.0), promotes_outputs=['x'])
        model.add_subsystem('c2', ReconfComp1(), promotes_inputs=['x'])
        model.add_subsystem('c3', ReconfComp2(), promotes_outputs=['f'])
        model.connect('c2.y', 'c3.y')
        p.setup()
        p['x'] = 3.

        # First run the model once; counter = 1, size of y = 1
        p.run_model()

        self.assertEqual(len(p['c2.y']), 1)
        self.assertEqual(len(p['c3.y']), 1)
        # Run the model again, which will trigger reconfiguration; counter = 2, size of y = 2
        p.run_model()

        self.assertEqual(len(p['c2.y']), 2)
        self.assertEqual(len(p['c3.y']), 2)
        assert_rel_error(self, p['c3.y'], [6., 6.])

    @unittest.expectedFailure
    def test_reconf_comp_connections_nlbgs_solver(self):
        p = om.Problem()

        p.model = model = om.Group()
        model.linear_solver = om.DirectSolver()
        model.nonlinear_solver = om.NonlinearBlockGS()
        model.add_subsystem('c1', om.IndepVarComp('x', 1.0), promotes_outputs=['x'])
        model.add_subsystem('c2', ReconfComp1(), promotes_inputs=['x'])
        model.add_subsystem('c3', ReconfComp2(), promotes_outputs=['f'])
        model.connect('c2.y', 'c3.y')
        p.setup()
        p['x'] = 3.

        # First run the model once; counter = 1, size of y = 1
        p.run_model()

        self.assertEqual(len(p['c2.y']), 1)
        self.assertEqual(len(p['c3.y']), 1)
        # Run the model again, which will trigger reconfiguration; counter = 2, size of y = 2
        p.run_model()

        self.assertEqual(len(p['c2.y']), 2)
        self.assertEqual(len(p['c3.y']), 2)
        assert_rel_error(self, p['c3.y'], [6., 6.])

    @unittest.expectedFailure
    def test_promoted_connections_newton_solver(self):
        p = om.Problem()

        p.model = model = om.Group()
        model.linear_solver = om.DirectSolver()
        model.nonlinear_solver = nl = om.NewtonSolver()
        nl.linesearch = om.BoundsEnforceLS()
        model.add_subsystem('c1', om.IndepVarComp('x', 1.0), promotes_outputs=['x'])
        model.add_subsystem('c2', ReconfComp1(), promotes_inputs=['x'], promotes_outputs=['y'])
        model.add_subsystem('c3', ReconfComp2(), promotes_inputs=['y'], promotes_outputs=['f'])
        p.setup()
        p['x'] = 3.

        # First run the model once; counter = 1, size of y = 1
        p.run_model()
        self.assertEqual(len(p['y']), 1)

        # Run the model again, which will trigger reconfiguration; counter = 2, size of y = 2
        p.run_model()
        self.assertEqual(len(p['y']), 2)
        assert_rel_error(self, p['y'], [6., 6.])

    @unittest.expectedFailure
    def test_test_promoted_connections_nlbgs_solver(self):
        p = om.Problem()

        p.model = model = om.Group()
        model.linear_solver = om.DirectSolver()
        model.nonlinear_solver = om.NonlinearBlockGS()
        model.add_subsystem('c1', om.IndepVarComp('x', 1.0), promotes_outputs=['x'])
        model.add_subsystem('c2', ReconfComp1(), promotes_inputs=['x'], promotes_outputs=['y'])
        model.add_subsystem('c3', ReconfComp2(), promotes_inputs=['y'], promotes_outputs=['f'])
        p.setup()
        p['x'] = 3.

        # First run the model once; counter = 1, size of y = 1
        p.run_model()
        self.assertEqual(len(p['y']), 1)

        # Run the model again, which will trigger reconfiguration; counter = 2, size of y = 2
        p.run_model()
        self.assertEqual(len(p['y']), 2)
        assert_rel_error(self, p['y'], [6., 6.])

    def test_reconf_comp_not_connected(self):
        p = om.Problem()

        p.model = model = om.Group()
        model.add_subsystem('c1', om.IndepVarComp('x', 1.0), promotes_outputs=['x'])
        model.add_subsystem('c2', ReconfComp1(), promotes_inputs=['x'])
        model.add_subsystem('c3', ReconfComp2(), promotes_outputs=['f'])
        # c2.y not connected to c3.y
        p.setup()
        p['x'] = 3.

        # First run the model once; counter = 1, size of y = 1
        p.run_model()

        self.assertEqual(len(p['c2.y']), 1)
        self.assertEqual(len(p['c3.y']), 1)

        # Run the model again, which will trigger reconfiguration; counter = 2, size of y = 2
        p.run_model()
        self.assertEqual(len(p['c3.y']), 2)
        self.assertEqual(len(p['c2.y']), 2)


if __name__ == '__main__':
    unittest.main()
