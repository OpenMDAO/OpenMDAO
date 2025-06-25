import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_totals

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

        prob.setup(force_alloc_complex=True, check=False)
        prob.set_val("y", 5.0 * np.ones(size))

        prob.run_model()

        # Deriv should be 75. Analytic was wrong before the fix.
        data = prob.check_totals(method="fd", form="forward", step=step, step_calc="abs", out_stream=None)
        assert_check_totals(data, atol=1e-5, rtol=1e-6)

    def test_semi_totals_cs(self):
        prob = om.Problem()

        sub = prob.model.add_subsystem('sub', GeometryAndAero(), promotes=['*'])
        sub.method = 'cs'

        prob.model.add_design_var("y")
        prob.model.add_objective("comp3.z", index=0)

        prob.setup(force_alloc_complex=True, check=False)
        prob.set_val("y", 5.0 * np.ones(size))

        prob.run_model()

        # Deriv should be 75. Analytic was wrong before the fix.
        data = prob.check_totals(method="cs", out_stream=None)

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

        prob.setup(force_alloc_complex=True, check=False)
        prob.set_val("y", 5.0 * np.ones(size))

        prob.run_model()

        # Deriv should be 75. Analytic was wrong before the fix.
        data = prob.check_totals(method="cs", out_stream=None)

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

        assert_near_equal(prob['sub1.src.y'], 1.0, 1e-6)
        assert_near_equal(prob['sub2.comp1.z'], 2.0, 1e-6)
        assert_near_equal(prob['sub2.comp2.z'], 2.0, 1e-6)
        assert_near_equal(prob['sub2.comp3.z'], 2.0, 1e-6)

        data = prob.check_totals(of=of, wrt=wrt, method="fd", out_stream=None)
        assert_check_totals(data, atol=1e-6, rtol=1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(mode='rev')
        prob.run_model()

        assert_near_equal(prob['sub1.src.y'], 1.0, 1e-6)
        assert_near_equal(prob['sub2.comp1.z'], 2.0, 1e-6)
        assert_near_equal(prob['sub2.comp2.z'], 2.0, 1e-6)
        assert_near_equal(prob['sub2.comp3.z'], 2.0, 1e-6)

        data = prob.check_totals(of=of, wrt=wrt, method="fd", out_stream=None)
        assert_check_totals(data, atol=1e-6, rtol=1e-6)

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

        assert_near_equal(prob['y'], 1.0, 1e-6)
        assert_near_equal(prob['sub2.comp1.z'], 2.0, 1e-6)
        assert_near_equal(prob['sub2.comp2.z'], 2.0, 1e-6)
        assert_near_equal(prob['sub2.comp3.z'], 2.0, 1e-6)

        data = prob.check_totals(of=of, wrt=wrt, method="fd", out_stream=None)
        assert_check_totals(data, atol=1e-6, rtol=1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(mode='rev')
        prob.run_model()

        assert_near_equal(prob['y'], 1.0, 1e-6)
        assert_near_equal(prob['sub2.comp1.z'], 2.0, 1e-6)
        assert_near_equal(prob['sub2.comp2.z'], 2.0, 1e-6)
        assert_near_equal(prob['sub2.comp3.z'], 2.0, 1e-6)

        data = prob.check_totals(of=of, wrt=wrt, method="fd", out_stream=None)
        assert_check_totals(data, atol=1e-6, rtol=1e-6)


class FakeGeomComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("n", types=int)
        self.options.declare('declare_partials', types=(bool,), default=True)

    def setup(self):
        n = self.options["n"]
        self.add_input("x0", val=np.zeros(n), units="m")
        self.add_input("feather", val=0.0, units="deg")
        self.add_output("x", val=np.zeros(n), units="m")

        if self.options['declare_partials']:
            self.declare_partials("*", "*", method="fd")

        self._counter = 0

    def compute(self, inputs, outputs):
        self._counter += 1
        feather_rad = inputs["feather"][0]*np.pi/180
        x0 = inputs["x0"]

        outputs["x"][:] = 3*np.sin(feather_rad + 0.2)*x0 + 3*feather_rad**2


class FakeAeroComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("n", types=int)
        self.options.declare('declare_partials', types=(bool,), default=True)

    def setup(self):
        n = self.options["n"]

        self.add_input("x", val=np.arange(n), units="m")
        self.add_input("omega", val=7000*2*np.pi/60, units="rad/s")

        self.add_output("CT", val=0.5)
        self.add_output("CP", val=0.5)

        if self.options['declare_partials']:
            self.declare_partials("*", "*", method="fd")

        self._counter = 0

    def compute(self, inputs, outputs):
        self._counter += 1
        omega = inputs["omega"][0]
        x = inputs["x"]

        outputs["CT"][0] = 0.8*omega**2 + np.sum(x)
        outputs["CP"][0] = 0.1*omega**3 + np.sum(x**2)


class GeometryAndAero2(om.Group):

    def initialize(self):
        self.options.declare("n", types=int)
        self.options.declare("rho", types=float)
        self.options.declare("vinf", types=float)
        self.options.declare('declare_partials', types=(bool,), default=True)

    def setup(self):
        n = self.options["n"]
        dp = self.options['declare_partials']

        comp = self.add_subsystem("init_geom", om.IndepVarComp(), promotes_outputs=["x0"])
        comp.add_output("x0", val=np.arange(n) + 1.0, units="m")

        self.add_subsystem("geom", FakeGeomComp(n=n, declare_partials=dp),
                           promotes_inputs=["x0", "feather"], promotes_outputs=["x"])
        self.add_subsystem("aero", FakeAeroComp(n=n, declare_partials=dp),
                           promotes_inputs=["x", "omega"], promotes_outputs=["CT", "CP"])

    def reset_count(self):
        self.geom._counter = 0
        self.aero._counter = 0


class TestSemiTotalsNumCalls(unittest.TestCase):

    def test_call_counts(self):
        size = 10
        rho = 1.17573
        minf = 0.111078231621482
        speedofsound = 344.5760217432
        vinf = minf*speedofsound

        prob = om.Problem()

        omega = 7199.759242*2*np.pi/60
        ivc = prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes_outputs=["*"])
        ivc.add_output("feather", val=0.0, units="deg")
        ivc.add_output("omega", val=omega, units="rad/s")

        geom_and_aero = prob.model.add_subsystem('geom_and_aero',
                                                GeometryAndAero2(n=size, rho=rho, vinf=vinf),
                                                promotes_inputs=["feather", "omega"],
                                                promotes_outputs=["CT", "CP"])
        geom_and_aero.approx_totals(step=step, step_calc="abs", method="fd", form="forward")

        prob.model.add_design_var("feather", lower=-5.0, upper=25.0, units="deg", ref=1.0)
        prob.model.add_design_var("omega", lower=3000*2*np.pi/60, upper=7500*2*np.pi/60, units="rad/s", ref=1.0)
        prob.model.add_objective("CP", ref=1e0)

        prob.setup(force_alloc_complex=False)

        omega = (6245.096992023524*2*np.pi/60) + step
        feather = 0.6362159381168669
        prob.set_val("omega", omega, units="rad/s")
        prob.set_val("feather", feather, units="deg")
        prob.run_model()
        geom_and_aero.reset_count()
        prob.compute_totals(of=["CT"], wrt=["feather", "omega"])

        self.assertEqual(geom_and_aero.geom._counter, 2)
        self.assertEqual(geom_and_aero.aero._counter, 2)

        geom_and_aero.reset_count()
        data = prob.check_totals(method="fd", form="forward", step=step, step_calc="abs", out_stream=None)
        assert_check_totals(data, atol=1e-6, rtol=1e-6)

        self.assertEqual(geom_and_aero.geom._counter, 3)
        self.assertEqual(geom_and_aero.aero._counter, 4)

    def test_check_relevance_approx_totals(self):

            size = 10
            rho = 1.17573
            minf = 0.111078231621482
            speedofsound = 344.5760217432
            vinf = minf*speedofsound

            prob = om.Problem()
            prob.driver = om.ScipyOptimizeDriver()
            prob.driver.options["optimizer"] = "SLSQP"

            omega = 7199.759242*2*np.pi/60
            ivc = prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes_outputs=["*"])
            ivc.add_output("feather", val=0.0, units="deg")
            ivc.add_output("omega", val=omega, units="rad/s")

            geom_and_aero = prob.model.add_subsystem('geom_and_aero',
                                                    GeometryAndAero2(n=size, rho=rho, vinf=vinf, declare_partials=False),
                                                    promotes_inputs=["feather", "omega"],
                                                    promotes_outputs=["CT", "CP"])
            geom_and_aero.approx_totals(step=step, step_calc="abs", method="fd", form="forward")

            prob.model.add_design_var("feather", lower=-5.0, upper=25.0, units="deg", ref=1.0)
            prob.model.add_design_var("omega", lower=3000*2*np.pi/60, upper=7500*2*np.pi/60, units="rad/s", ref=1.0)
            prob.model.add_objective("CP", ref=1e0)
            prob.model.add_constraint("CT", lower=0.0)

            prob.setup(force_alloc_complex=False)

            omega = (6245.096992023524*2*np.pi/60) + step
            feather = 0.6362159381168669
            prob.set_val("omega", omega, units="rad/s")
            prob.set_val("feather", feather, units="deg")
            result = prob.run_driver()

            self.assertTrue(result.success)


if __name__ == '__main__':
    unittest.main()
