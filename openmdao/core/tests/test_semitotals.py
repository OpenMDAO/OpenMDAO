import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals

step = 1e-6
size = 3

class Mult(om.ExplicitComponent):

    def setup(self):

        self.add_input('x', np.ones(size))
        self.add_input('y', np.ones(size))

        self.add_output('z', np.ones(size))

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):

        outputs['z'] = inputs['x'] * inputs['y']
        print(self.pathname, 'x', inputs['x'], 'y', inputs['y'], 'out', outputs['z'])

    def compute_partials(self, inputs, partials):

        partials['z', 'x'] = inputs['y']
        partials['z', 'y'] = inputs['x']


class GeometryAndAero(om.Group):

    def setup(self):

        self.add_subsystem("comp1", Mult(), promotes_inputs=['y'])
        self.add_subsystem("comp2", Mult(), promotes_inputs=['y'])
        self.add_subsystem("comp3", Mult(), promotes_inputs=['y'])

        # self.connect('comp1.z', 'comp2.x')
        self.connect('comp2.z', 'comp3.x')

        if self.method == 'fd':
            self.approx_totals(step=step, step_calc="abs", method=self.method, form="forward")
        else:
            self.approx_totals(method=self.method)


class TestSemiTotals(unittest.TestCase):

    def test_semi_totals_fd(self):
        prob = om.Problem()

        sub = prob.model.add_subsystem('sub', GeometryAndAero(), promotes=['*'])
        sub.method = 'fd'

        prob.model.add_design_var("y")
        prob.model.add_objective("comp3.z", index=0)

        # prob.model.approx_totals(step=step, step_calc="abs", method="fd", form="forward")

        prob.setup(force_alloc_complex=True, check=False)
        prob.set_val("y", 5.0 * np.ones(size))

        prob.run_model()

        # Deriv should be 75. Analytic is wrong.
        data = prob.check_totals(method="fd", form="forward", step=step, step_calc="abs")
        assert_check_totals(data, atol=1e-5, rtol=1e-6)

    def test_semi_totals_cs(self):
        prob = om.Problem()

        sub = prob.model.add_subsystem('sub', GeometryAndAero(), promotes=['*'])
        sub.method = 'cs'

        prob.model.add_design_var("y")
        prob.model.add_objective("comp3.z", index=0)

        # prob.model.approx_totals(step=step, step_calc="abs", method="fd", form="forward")

        prob.setup(force_alloc_complex=True, check=False)
        prob.set_val("y", 5.0 * np.ones(size))

        prob.run_model()

        # Deriv should be 75. Analytic is wrong.
        data = prob.check_totals(method="cs")

        assert_check_totals(data, atol=1e-6, rtol=1e-6)

    def test_semi_totals_cs_indirect(self):
        prob = om.Problem()

        prob.model.add_subsystem('indeps', om.IndepVarComp('yy', np.ones(size)))
        prob.model.add_subsystem('comp', om.ExecComp('z=2*y', z=np.ones(size), y=np.ones(size)))
        sub = prob.model.add_subsystem('sub', GeometryAndAero(), promotes=['*'])
        sub.method = 'cs'

        prob.model.connect('indeps.yy', 'comp.y')
        prob.model.connect('comp.z', 'y')
        prob.model.add_design_var("indeps.yy")
        prob.model.add_objective("comp3.z", index=0)

        # prob.model.approx_totals(step=step, step_calc="abs", method="fd", form="forward")

        prob.setup(force_alloc_complex=True, check=False)
        prob.set_val("y", 5.0 * np.ones(size))

        prob.run_model()

        #from openmdao.api import n2
        #n2(prob)

        # Deriv should be 75. Analytic is wrong.
        data = prob.check_totals(method="cs")

        assert_check_totals(data, atol=1e-6, rtol=1e-6)

    def test_multi_conn_inputs_manual_connect(self):

        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x', 1.0))
        sub1 = prob.model.add_subsystem('sub1', om.Group())
        sub2 = prob.model.add_subsystem('sub2', om.Group())

        sub1.add_subsystem('src', om.ExecComp('y=x'))
        sub2.add_subsystem('comp1', om.ExecComp('z=x+y'))
        sub2.add_subsystem('comp2', om.ExecComp('z=x+y'))
        sub2.add_subsystem('comp3', om.ExecComp('z=x+y'))

        prob.model.connect('px1.x', 'sub1.src.x')
        prob.model.connect('sub1.src.y', 'sub2.comp1.y')
        prob.model.connect('sub1.src.y', 'sub2.comp2.y')
        prob.model.connect('sub1.src.y', 'sub2.comp3.y')

        sub2.approx_totals(method='cs')

        wrt = ['px1.x']
        of = ['sub2.comp1.z', 'sub2.comp2.z', 'sub2.comp3.z']

        prob.setup(mode='fwd')
        prob.run_model()

        #from openmdao.api import n2
        #n2(prob)

        #assert_near_equal(prob['sub1.src.y'], 100.0, 1e-6)
        #assert_near_equal(prob['sub2.comp1.z'], 101.0, 1e-6)
        #assert_near_equal(prob['sub2.comp2.z'], 201.0, 1e-6)
        #assert_near_equal(prob['sub2.comp3.z'], 101.0, 1e-6)

        #J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        data = prob.check_totals(of=of, wrt=wrt, method="fd")
        assert_check_totals(data, atol=1e-6, rtol=1e-6)

        # Check the total derivatives in reverse mode
        #prob.setup(mode='rev')
        #prob.run_model()
        #J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        #assert_near_equal(J['sub2.comp1.y']['px1.x'][0][0], 1.0, 1e-6)
        #assert_near_equal(J['sub2.comp2.y']['px1.x'][0][0], 1.0, 1e-6)
        #assert_near_equal(J['sub2.comp3.y']['px1.x'][0][0], 1.0, 1e-6)

    def test_multi_conn_inputs_promoted(self):

        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x', 1.0))
        sub1 = prob.model.add_subsystem('sub1', om.Group(), promotes=['y'])
        sub2 = prob.model.add_subsystem('sub2', om.Group(), promotes=['y'])

        sub1.add_subsystem('src', om.ExecComp('y=x'), promotes=['y'])
        sub2.add_subsystem('comp1', om.ExecComp('z=x+y'), promotes_inputs=['y'])
        sub2.add_subsystem('comp2', om.ExecComp('z=x+y'), promotes_inputs=['y'])
        sub2.add_subsystem('comp3', om.ExecComp('z=x+y'), promotes_inputs=['y'])

        prob.model.connect('px1.x', 'sub1.src.x')

        sub2.approx_totals(method='cs')

        wrt = ['px1.x']
        of = ['sub2.comp1.z', 'sub2.comp2.z', 'sub2.comp3.z']

        prob.setup(mode='fwd')
        prob.run_model()

        #from openmdao.api import n2
        #n2(prob)

        #assert_near_equal(prob['sub1.src.y'], 100.0, 1e-6)
        #assert_near_equal(prob['sub2.comp1.z'], 101.0, 1e-6)
        #assert_near_equal(prob['sub2.comp2.z'], 201.0, 1e-6)
        #assert_near_equal(prob['sub2.comp3.z'], 101.0, 1e-6)

        #J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        data = prob.check_totals(of=of, wrt=wrt, method="fd")
        assert_check_totals(data, atol=1e-6, rtol=1e-6)

        # Check the total derivatives in reverse mode
        #prob.setup(mode='rev')
        #prob.run_model()
        #J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        #assert_near_equal(J['sub2.comp1.y']['px1.x'][0][0], 1.0, 1e-6)
        #assert_near_equal(J['sub2.comp2.y']['px1.x'][0][0], 1.0, 1e-6)
        #assert_near_equal(J['sub2.comp3.y']['px1.x'][0][0], 1.0, 1e-6)
