import unittest
from itertools import product

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

    def test_approx_totals_internal_wrt_raises(self):
        prob = om.Problem()

        group = prob.model.add_subsystem('G', om.Group())
        group.add_subsystem('ivc', om.IndepVarComp('x', 1.0))
        group.add_subsystem('comp', om.ExecComp('y = 2.0 * x'))
        group.connect('ivc.x', 'comp.x')
        group.approx_totals(method='fd')

        prob.setup(mode='fwd')
        prob.run_model()

        msg = r"'G.ivc.x'.*approx_totals.*internal"
        with self.assertRaisesRegex(RuntimeError, msg):
            prob.compute_totals(of=['G.comp.y'], wrt=['G.ivc.x'])

        with self.assertRaisesRegex(RuntimeError, msg):
            prob.check_totals(of=['G.comp.y'], wrt=['G.ivc.x'], out_stream=None)

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


class SharedInputComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', types=int, default=4)
        self.options.declare('k', types=float, default=1.0)
        self.options.declare('units', default=None, allow_none=True)

    def setup(self):
        n = self.options['n']
        self.add_input('x', np.ones(n), units=self.options['units'])
        self.add_input('y_in', 1.0)
        self.add_output('y', 1.0)
        self.declare_partials('y', ['x', 'y_in'])

        self._counter = 0

    def compute(self, inputs, outputs):
        self._counter += 1
        k = self.options['k']
        outputs['y'] = k * np.sum(np.sin(inputs['x']) * np.arange(1, inputs['x'].size + 1)) + \
            inputs['y_in'] ** 2

    def compute_partials(self, inputs, partials):
        k = self.options['k']
        partials['y', 'x'] = k * np.cos(inputs['x']) * np.arange(1, inputs['x'].size + 1)
        partials['y', 'y_in'] = 2.0 * inputs['y_in']


class TestSemiTotalsSharedInputs(unittest.TestCase):

    def build(self, ncomps, method, mode, chain=True, src_indices=None, units=None,
              approx='sub', assembled=False):
        prob = om.Problem()
        model = prob.model

        ivc = model.add_subsystem('ivc', om.IndepVarComp())
        if src_indices is None:
            ivc.add_output('x', np.array([0.3, 0.7, 1.1, 1.9]))
        else:
            ivc.add_output('x', np.array([0.3, 0.7, 1.1, 1.9, 2.3]))

        sub = model.add_subsystem('sub', om.Group())
        for i in range(ncomps):
            comp_units = None if units is None else units[i]
            comp = SharedInputComp(n=4, k=1.0 + i, units=comp_units)
            if src_indices is None:
                sub.add_subsystem(f'c{i}', comp, promotes_inputs=['x'])
            else:
                sub.add_subsystem(f'c{i}', comp)
                model.connect('ivc.x', f'sub.c{i}.x', src_indices=src_indices[i])
            if chain and i > 0:
                sub.connect(f'c{i - 1}.y', f'c{i}.y_in')

        if src_indices is None:
            model.connect('ivc.x', 'sub.x')

        if chain:
            model.add_subsystem('obj', om.ExecComp('f = 2.0 * y ** 2'))
            model.connect(f'sub.c{ncomps - 1}.y', 'obj.y')
        else:
            terms = ' + '.join(f'y{i} ** 2' for i in range(ncomps))
            model.add_subsystem('obj', om.ExecComp(f'f = {terms}'))
            for i in range(ncomps):
                model.connect(f'sub.c{i}.y', f'obj.y{i}')

        model.add_design_var('ivc.x')
        model.add_objective('obj.f')
        model.add_constraint('sub.c0.y', upper=100.0)

        if approx == 'sub':
            sub.approx_totals(method=method)
        else:
            model.approx_totals(method=method)

        if assembled:
            model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.setup(mode=mode, force_alloc_complex=True)
        prob.run_model()

        return prob

    def get_calls(self, prob, ncomps):
        return [prob.model.sub._get_subsystem(f'c{i}')._counter for i in range(ncomps)]

    def reset_calls(self, prob, ncomps):
        for i in range(ncomps):
            prob.model.sub._get_subsystem(f'c{i}')._counter = 0

    def check_totals(self, prob, method):
        tol = 1e-5 if method == 'fd' else 1e-10
        data = prob.check_totals(method='cs', out_stream=None)
        assert_check_totals(data, atol=tol, rtol=tol)

    def test_promoted_input_call_counts(self):
        # each promoted copy of x should not be perturbed separately, so the number of
        # compute calls should match the number made when approximating at the top level.
        for ncomps, method, mode in product([2, 5], ['fd', 'cs'], ['fwd', 'rev']):
            with self.subTest(ncomps=ncomps, method=method, mode=mode):
                calls = {}
                for approx in ('model', 'sub'):
                    prob = self.build(ncomps, method, mode, approx=approx)
                    self.reset_calls(prob, ncomps)
                    prob.compute_totals()
                    calls[approx] = self.get_calls(prob, ncomps)

                self.assertEqual(calls['sub'], [4] * ncomps)
                self.assertEqual(calls['sub'], calls['model'])

                self.check_totals(prob, method)

    def test_copies_with_different_src_indices_or_units(self):
        src_indices = [[0, 1, 2, 3], [1, 2, 3, 4], [0, 1, 2, 3], [1, 2, 3, 4]]
        units = ['m', 'cm', 'm', 'cm']
        for method, mode in product(['fd', 'cs'], ['fwd', 'rev']):
            with self.subTest(method=method, mode=mode, case='src_indices'):
                prob = self.build(4, method, mode, src_indices=src_indices)
                self.reset_calls(prob, 4)
                prob.compute_totals()
                self.assertEqual(self.get_calls(prob, 4), [8] * 4)
                self.check_totals(prob, method)

            with self.subTest(method=method, mode=mode, case='units'):
                prob = self.build(4, method, mode, units=units)
                self.reset_calls(prob, 4)
                prob.compute_totals()
                self.assertEqual(self.get_calls(prob, 4), [8] * 4)
                self.check_totals(prob, method)

    def test_outputs_depend_on_one_copy(self):
        # Each output depends on only one of the promoted copies of x.
        for method, mode, assembled in product(['fd', 'cs'], ['fwd', 'rev'], [False, True]):
            with self.subTest(method=method, mode=mode, assembled=assembled):
                prob = self.build(3, method, mode, chain=False, assembled=assembled)
                self.reset_calls(prob, 3)
                prob.compute_totals()
                if not assembled:
                    self.assertEqual(self.get_calls(prob, 3), [4] * 3)
                self.check_totals(prob, method)

    def test_issue_3426(self):
        # y = f1(x; alpha), z = f2(x, y; beta), q = f3(z) with f1 and f2 in an approximated group.
        nx, ny = 3, 50

        class F1(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones(nx))
                self.add_input('alpha', np.ones(4))
                self.add_output('y', np.ones(ny))
                self.declare_partials('*', '*', method='fd')
                self._counter = 0

            def compute(self, inputs, outputs):
                self._counter += 1
                outputs['y'] = np.sin(np.arange(ny) * inputs['x'].sum()) * inputs['alpha'].sum()

        class F2(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.ones(nx))
                self.add_input('y', np.ones(ny))
                self.add_input('beta', np.ones(4))
                self.add_output('z', 1.0)
                self.declare_partials('*', '*', method='fd')
                self._counter = 0

            def compute(self, inputs, outputs):
                self._counter += 1
                outputs['z'] = (inputs['x'] ** 2).sum() + \
                    (inputs['y'] ** 2).sum() * inputs['beta'].sum()

        calls = {}
        for approx in ('sub', 'model'):
            prob = om.Problem()
            sub = prob.model.add_subsystem('sub', om.Group(), promotes=['*'])
            sub.add_subsystem('f1', F1(), promotes=['*'])
            sub.add_subsystem('f2', F2(), promotes=['*'])
            prob.model.add_subsystem('f3', om.ExecComp('q = 2.0 * z'), promotes=['*'])
            if approx == 'sub':
                sub.approx_totals(method='fd')
            else:
                prob.model.approx_totals(method='fd')
            prob.model.add_design_var('x')
            prob.model.add_objective('q')
            prob.setup(force_alloc_complex=True)
            prob.set_val('x', np.array([0.1, 0.2, 0.3]))
            prob.run_model()

            sub.f1._counter = sub.f2._counter = 0
            prob.compute_totals()
            calls[approx] = (sub.f1._counter, sub.f2._counter)

            data = prob.check_totals(method='cs', out_stream=None)
            assert_check_totals(data, atol=1e-6, rtol=1e-4)

        self.assertEqual(calls['sub'], (nx, nx))
        self.assertEqual(calls['sub'], calls['model'])


if __name__ == '__main__':
    unittest.main()
