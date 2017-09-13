""" Testing for Problem.check_partials and check_total_derivatives."""

import unittest
from six import iteritems

import numpy as np

from openmdao.api import Group, ExplicitComponent, IndepVarComp, Problem, NonLinearRunOnce, \
                         ImplicitComponent, NonlinearBlockGS
from openmdao.devtools.testutil import assert_rel_error, TestLogger
from openmdao.test_suite.components.impl_comp_array import TestImplCompArrayMatVec
from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec
from openmdao.test_suite.components.sellar import SellarDerivatives


class ParaboloidTricky(ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.scale = 1e-7

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """
        sc = self.scale
        x = inputs['x']*sc
        y = inputs['y']*sc

        outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    def compute_partials(self, inputs, partials):
        """
        Jacobian for our paraboloid.
        """
        sc = self.scale
        x = inputs['x']
        y = inputs['y']

        partials['f_xy', 'x'] = 2.0*x*sc*sc - 6.0*sc + y*sc*sc
        partials['f_xy', 'y'] = 2.0*y*sc*sc + 8.0*sc + x*sc*sc


class TestProblemCheckPartials(unittest.TestCase):

    def test_incorrect_jacobian(self):
        class MyComp(ExplicitComponent):
            def setup(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)

                self.add_output('y', 5.5)

            def compute(self, inputs, outputs):
                """ Doesn't do much. """
                outputs['y'] = 3.0*inputs['x1'] + 4.0*inputs['x2']

            def compute_partials(self, inputs, partials):
                """Intentionally incorrect derivative."""
                J = partials
                J['y', 'x1'] = np.array([4.0])
                J['y', 'x2'] = np.array([40])

        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('comp', MyComp())

        prob.model.connect('p1.x1', 'comp.x1')
        prob.model.connect('p2.x2', 'comp.x2')

        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        testlogger = TestLogger()
        prob.check_partials(logger=testlogger)

        lines = testlogger.get('info')

        y_wrt_x1_line = lines.index("  comp: 'y' wrt 'x1'\n")

        self.assertTrue(lines[y_wrt_x1_line+4].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        self.assertTrue(lines[y_wrt_x1_line+5].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        self.assertFalse(lines[y_wrt_x1_line+6].endswith('*'),
                        msg='Error flag not expected in output but displayed')

    def test_component_only(self):
        class MyComp(ExplicitComponent):
            def setup(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)

                self.add_output('y', 5.5)

            def compute(self, inputs, outputs):
                """ Doesn't do much. """
                outputs['y'] = 3.0*inputs['x1'] + 4.0*inputs['x2']

            def compute_partials(self, inputs, partials):
                """Intentionally incorrect derivative."""
                J = partials
                J['y', 'x1'] = np.array([4.0])
                J['y', 'x2'] = np.array([40])

        prob = Problem()
        prob.model = MyComp()

        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        testlogger = TestLogger()
        prob.check_partials(logger=testlogger)

        lines = testlogger.get('info')

        y_wrt_x1_line = lines.index("  : 'y' wrt 'x1'\n")

        self.assertTrue(lines[y_wrt_x1_line+4].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        self.assertTrue(lines[y_wrt_x1_line+5].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        self.assertFalse(lines[y_wrt_x1_line+6].endswith('*'),
                        msg='Error flag not expected in output but displayed')

    def test_component_only_suppress(self):
        class MyComp(ExplicitComponent):
            def setup(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)

                self.add_output('y', 5.5)

            def compute(self, inputs, outputs):
                """ Doesn't do much. """
                outputs['y'] = 3.0*inputs['x1'] + 4.0*inputs['x2']

            def compute_partials(self, inputs, partials):
                """Intentionally incorrect derivative."""
                J = partials
                J['y', 'x1'] = np.array([4.0])
                J['y', 'x2'] = np.array([40])

        prob = Problem()
        prob.model = MyComp()

        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        testlogger = TestLogger()
        data = prob.check_partials(logger=testlogger, suppress_output=True)

        subheads = data[''][('y', 'x1')]
        self.assertTrue('J_fwd' in subheads)
        self.assertTrue('rel error' in subheads)
        self.assertTrue('abs error' in subheads)
        self.assertTrue('magnitude' in subheads)

        lines = testlogger.get('info')
        self.assertEqual(len(lines), 0)

    def test_missing_entry(self):
        class MyComp(ExplicitComponent):
            def setup(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)

                self.add_output('y', 5.5)

            def compute(self, inputs, outputs):
                """ Doesn't do much. """
                outputs['y'] = 3.0*inputs['x1'] + 4.0*inputs['x2']

            def compute_partials(self, inputs, partials):
                """Intentionally left out derivative."""
                J = partials
                J['y', 'x1'] = np.array([3.0])

        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('p1', IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('comp', MyComp())

        prob.model.connect('p1.x1', 'comp.x1')
        prob.model.connect('p2.x2', 'comp.x2')

        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(suppress_output=True)

        abs_error = data['comp']['y', 'x1']['abs error']
        rel_error = data['comp']['y', 'x1']['rel error']
        self.assertAlmostEqual(abs_error.forward, 0.)
        self.assertAlmostEqual(abs_error.reverse, 0.)
        self.assertAlmostEqual(rel_error.forward, 0.)
        self.assertAlmostEqual(rel_error.reverse, 0.)
        self.assertAlmostEqual(np.linalg.norm(data['comp']['y', 'x1']['J_fd'] - 3.), 0.,
                               delta=1e-6)

        abs_error = data['comp']['y', 'x2']['abs error']
        rel_error = data['comp']['y', 'x2']['rel error']
        self.assertAlmostEqual(abs_error.forward, 4.)
        self.assertAlmostEqual(abs_error.reverse, 4.)
        self.assertAlmostEqual(rel_error.forward, 1.)
        self.assertAlmostEqual(rel_error.reverse, 1.)
        self.assertAlmostEqual(np.linalg.norm(data['comp']['y', 'x2']['J_fd'] - 4.), 0.,
                               delta=1e-6)

    def test_nested_fd_units(self):
        class UnitCompBase(ExplicitComponent):
            def setup(self):
                self.add_input('T', val=284., units="degR", desc="Temperature")
                self.add_input('P', val=1., units='lbf/inch**2', desc="Pressure")

                self.add_output('flow:T', val=284., units="degR", desc="Temperature")
                self.add_output('flow:P', val=1., units='lbf/inch**2', desc="Pressure")

                # Finite difference everything
                self.approx_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                outputs['flow:T'] = inputs['T']
                outputs['flow:P'] = inputs['P']

        p = Problem()
        model = p.model = Group()
        indep = model.add_subsystem('indep', IndepVarComp(), promotes=['*'])

        indep.add_output('T', val=100., units='degK')
        indep.add_output('P', val=1., units='bar')

        units = model.add_subsystem('units', UnitCompBase(), promotes=['*'])

        p.setup()
        data = p.check_partials(suppress_output=True)

        for comp_name, comp in iteritems(data):
            for partial_name, partial in iteritems(comp):
                forward = partial['J_fwd']
                reverse = partial['J_rev']
                fd = partial['J_fd']
                self.assertAlmostEqual(np.linalg.norm(forward - reverse), 0.)
                self.assertAlmostEqual(np.linalg.norm(forward - fd), 0., delta=1e-6)

    def test_units(self):
        class UnitCompBase(ExplicitComponent):
            def setup(self):
                self.add_input('T', val=284., units="degR", desc="Temperature")
                self.add_input('P', val=1., units='lbf/inch**2', desc="Pressure")

                self.add_output('flow:T', val=284., units="degR", desc="Temperature")
                self.add_output('flow:P', val=1., units='lbf/inch**2', desc="Pressure")

                self.run_count = 0

            def compute_partials(self, inputs, partials):
                partials['flow:T', 'T'] = 1.
                partials['flow:P', 'P'] = 1.

            def compute(self, inputs, outputs):
                outputs['flow:T'] = inputs['T']
                outputs['flow:P'] = inputs['P']

                self.run_count += 1

        p = Problem()
        model = p.model = Group()
        indep = model.add_subsystem('indep', IndepVarComp(), promotes=['*'])

        indep.add_output('T', val=100., units='degK')
        indep.add_output('P', val=1., units='bar')

        units = model.add_subsystem('units', UnitCompBase(), promotes=['*'])

        model.nonlinear_solver = NonLinearRunOnce()

        p.setup()
        data = p.check_partials(suppress_output=True)

        for comp_name, comp in iteritems(data):
            for partial_name, partial in iteritems(comp):
                abs_error = partial['abs error']
                self.assertAlmostEqual(abs_error.forward, 0.)
                self.assertAlmostEqual(abs_error.reverse, 0.)
                self.assertAlmostEqual(abs_error.forward_reverse, 0.)

        # Make sure we only FD this twice.
        # The count is 5 because in check_partials, there are two calls to apply_nonlinear
        # when compute the fwd and rev analytic derivatives, then one call to apply_nonlinear
        # to compute the reference point for FD, then two additional calls for the two inputs.
        comp = model.get_subsystem('units')
        self.assertEqual(comp.run_count, 5)

    def test_scalar_val(self):
        class PassThrough(ExplicitComponent):
            """
            Helper component that is needed when variables must be passed
            directly from input to output
            """

            def __init__(self, i_var, o_var, val, units=None):
                super(PassThrough, self).__init__()
                self.i_var = i_var
                self.o_var = o_var
                self.units = units
                self.val = val

                if isinstance(val, (float, int)) or np.isscalar(val):
                    size=1
                else:
                    size = np.prod(val.shape)

                self.size = size

            def setup(self):
                if self.units is None:
                    self.add_input(self.i_var, self.val)
                    self.add_output(self.o_var, self.val)
                else:
                    self.add_input(self.i_var, self.val, units=self.units)
                    self.add_output(self.o_var, self.val, units=self.units)

                row_col = np.arange(self.size)
                self.declare_partials(of=self.o_var, wrt=self.i_var,
                                      val=1, rows=row_col, cols=row_col)

            def compute(self, inputs, outputs):
                outputs[self.o_var] = inputs[self.i_var]

            def linearize(self, inputs, outputs, J):
                pass

        p = Problem()

        indeps = p.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('foo', val=np.ones(4))
        indeps.add_output('foo2', val=np.ones(4))

        p.model.add_subsystem('pt', PassThrough("foo", "bar", val=np.ones(4)), promotes=['*'])
        p.model.add_subsystem('pt2', PassThrough("foo2", "bar2", val=np.ones(4)), promotes=['*'])

        p.set_solver_print(level=0)

        p.setup()
        p.run_model()

        data = p.check_partials(suppress_output=True)
        identity = np.eye(4)
        assert_rel_error(self, data['pt'][('bar', 'foo')]['J_fwd'], identity, 1e-15)
        assert_rel_error(self, data['pt'][('bar', 'foo')]['J_rev'], identity, 1e-15)
        assert_rel_error(self, data['pt'][('bar', 'foo')]['J_fd'], identity, 1e-9)

        assert_rel_error(self, data['pt2'][('bar2', 'foo2')]['J_fwd'], identity, 1e-15)
        assert_rel_error(self, data['pt2'][('bar2', 'foo2')]['J_rev'], identity, 1e-15)
        assert_rel_error(self, data['pt2'][('bar2', 'foo2')]['J_fd'], identity, 1e-9)

    def test_matrix_free_explicit(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidMatVec())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(suppress_output=True)

        for comp_name, comp in iteritems(data):
            for partial_name, partial in iteritems(comp):
                abs_error = partial['abs error']
                rel_error = partial['rel error']
                assert_rel_error(self, abs_error.forward, 0., 1e-5)
                assert_rel_error(self, abs_error.reverse, 0., 1e-5)
                assert_rel_error(self, abs_error.forward_reverse, 0., 1e-5)
                assert_rel_error(self, rel_error.forward, 0., 1e-5)
                assert_rel_error(self, rel_error.reverse, 0., 1e-5)
                assert_rel_error(self, rel_error.forward_reverse, 0., 1e-5)

        assert_rel_error(self, data['comp'][('f_xy', 'x')]['J_fwd'][0][0], 5.0, 1e-6)
        assert_rel_error(self, data['comp'][('f_xy', 'x')]['J_rev'][0][0], 5.0, 1e-6)
        assert_rel_error(self, data['comp'][('f_xy', 'y')]['J_fwd'][0][0], 21.0, 1e-6)
        assert_rel_error(self, data['comp'][('f_xy', 'y')]['J_rev'][0][0], 21.0, 1e-6)

    def test_matrix_free_implicit(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('rhs', np.ones((2, ))))
        prob.model.add_subsystem('comp', TestImplCompArrayMatVec())

        prob.model.connect('p1.rhs', 'comp.rhs')

        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(suppress_output=True)

        for comp_name, comp in iteritems(data):
            for partial_name, partial in iteritems(comp):
                abs_error = partial['abs error']
                rel_error = partial['rel error']
                assert_rel_error(self, abs_error.forward, 0., 1e-5)
                assert_rel_error(self, abs_error.reverse, 0., 1e-5)
                assert_rel_error(self, abs_error.forward_reverse, 0., 1e-5)
                assert_rel_error(self, rel_error.forward, 0., 1e-5)
                assert_rel_error(self, rel_error.reverse, 0., 1e-5)
                assert_rel_error(self, rel_error.forward_reverse, 0., 1e-5)

    def test_implicit_undeclared(self):
        # Test to see that check_partials works when state_wrt_input and state_wrt_state
        # partials are missing.

        class ImplComp4Test(ImplicitComponent):

            def setup(self):
                self.add_input('x', np.ones(2))
                self.add_input('dummy', np.ones(2))
                self.add_output('y', np.ones(2))
                self.add_output('extra', np.ones(2))
                self.mtx = np.array([
                    [ 3., 4.],
                    [ 2., 3.],
                ])
            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['y'] = self.mtx.dot(outputs['y']) - inputs['x']

            def linearize(self, inputs, outputs, partials):
                partials['y', 'x'] = -np.eye(2)
                partials['y', 'y'] = self.mtx

        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', np.ones((2, ))))
        prob.model.add_subsystem('p2', IndepVarComp('dummy', np.ones((2, ))))
        prob.model.add_subsystem('comp', ImplComp4Test())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.dummy', 'comp.dummy')

        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(suppress_output=True)

        assert_rel_error(self, data['comp']['y', 'extra']['J_fwd'], np.zeros((2, 2)))
        assert_rel_error(self, data['comp']['y', 'extra']['J_rev'], np.zeros((2, 2)))
        assert_rel_error(self, data['comp']['y', 'dummy']['J_fwd'], np.zeros((2, 2)))
        assert_rel_error(self, data['comp']['y', 'dummy']['J_rev'], np.zeros((2, 2)))

    def test_dependent_false_hide(self):
        # Test that we omit derivs declared with dependent=False

        class SimpleComp1(ExplicitComponent):
            def setup(self):
                self.add_input('z', shape=(2, 2))
                self.add_input('x', shape=(2, 2))
                self.add_output('g', shape=(2, 2))

                self.declare_partials('g', 'z', dependent=False)

            def compute(self, inputs, outputs):
                outputs['g'] = 3.0*inputs['x']

            def compute_partials(self, inputs, partials):
                partials['g', 'x'] = 3.


        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('z', np.ones((2, 2))))
        prob.model.add_subsystem('p2', IndepVarComp('x', np.ones((2, 2))))
        prob.model.add_subsystem('comp', SimpleComp1())
        prob.model.connect('p1.z', 'comp.z')
        prob.model.connect('p2.x', 'comp.x')

        prob.setup(check=False)

        testlogger = TestLogger()
        data = prob.check_partials(logger=testlogger)
        lines = testlogger.get('info')

        self.assertTrue("  comp: 'g' wrt 'z'\n" not in lines)
        self.assertTrue(('g', 'z') not in data['comp'])
        self.assertTrue("  comp: 'g' wrt 'x'\n"  in lines)
        self.assertTrue(('g', 'x') in data['comp'])

    def test_dependent_false_show(self):
        # Test that we show derivs declared with dependent=False if the fd is not
        # ~zero.

        class SimpleComp2(ExplicitComponent):
            def setup(self):
                self.add_input('z', shape=(2, 2))
                self.add_input('x', shape=(2, 2))
                self.add_output('g', shape=(2, 2))

                self.declare_partials('g', 'z', dependent=False)

            def compute(self, inputs, outputs):
                outputs['g'] = 2.0*inputs['z'] + 3.0*inputs['x']

            def compute_partials(self, inputs, partials):
                partials['g', 'x'] = 3.


        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('z', np.ones((2, 2))))
        prob.model.add_subsystem('p2', IndepVarComp('x', np.ones((2, 2))))
        prob.model.add_subsystem('comp', SimpleComp2())
        prob.model.connect('p1.z', 'comp.z')
        prob.model.connect('p2.x', 'comp.x')

        prob.setup(check=False)

        testlogger = TestLogger()
        data = prob.check_partials(logger=testlogger)
        lines = testlogger.get('info')

        self.assertTrue("  comp: 'g' wrt 'z'\n" in lines)
        self.assertTrue(('g', 'z') in data['comp'])
        self.assertTrue("  comp: 'g' wrt 'x'\n"  in lines)
        self.assertTrue(('g', 'x') in data['comp'])

    def test_set_step_on_comp(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.metadata['check_step'] = 1e-2

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(suppress_output=True)

        # This will fail unless you set the check_step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-5)
        self.assertLess(x_error.reverse, 1e-5)

    def test_set_step_global(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        opts = {'step' : 1e-2}

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(suppress_output=True, global_options=opts)

        # This will fail unless you set the global step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-5)
        self.assertLess(x_error.reverse, 1e-5)

    def test_complex_step_not_allocated(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.metadata['check_method'] = 'cs'

        prob.setup(check=False)
        prob.run_model()

        with self.assertRaises(RuntimeError) as context:
            data = prob.check_partials(suppress_output=True)

        msg = 'In order to check partials with complex step, you need to set ' + \
            '"force_alloc_complex" to True during setup.'
        self.assertEqual(str(context.exception), msg)

    def test_set_method_on_comp(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.metadata['check_method'] = 'cs'

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        data = prob.check_partials(suppress_output=True)

        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-5)
        self.assertLess(x_error.reverse, 1e-5)

    def test_set_method_global(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        opts = {'method' : 'cs'}

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        data = prob.check_partials(suppress_output=True, global_options=opts)

        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-5)
        self.assertLess(x_error.reverse, 1e-5)

    def test_set_form_on_comp(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.metadata['check_form'] = 'central'

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(suppress_output=True)

        # This will fail unless you set the check_step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-3)
        self.assertLess(x_error.reverse, 1e-3)

    def test_set_form_global(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        opts = {'form' : 'central'}

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(suppress_output=True, global_options=opts)

        # This will fail unless you set the check_step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-3)
        self.assertLess(x_error.reverse, 1e-3)

    def test_set_step_calc_on_comp(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.metadata['check_step_calc'] = 'rel'

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(suppress_output=True)

        # This will fail unless you set the check_step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 3e-3)
        self.assertLess(x_error.reverse, 3e-3)

    def test_set_step_calc_global(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        opts = {'step_calc' : 'rel'}

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(suppress_output=True, global_options=opts)

        # This will fail unless you set the global step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 3e-3)
        self.assertLess(x_error.reverse, 3e-3)


class TestCheckPartialsFeature(unittest.TestCase):

    def test_feature_incorrect_jacobian(self):
        class MyComp(ExplicitComponent):
            def setup(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)

                self.add_output('y', 5.5)

            def compute(self, inputs, outputs):
                """ Doesn't do much. """
                outputs['y'] = 3.0*inputs['x1'] + 4.0*inputs['x2']

            def compute_partials(self, inputs, partials):
                """Intentionally incorrect derivative."""
                J = partials
                J['y', 'x1'] = np.array([4.0])
                J['y', 'x2'] = np.array([40])

        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('comp', MyComp())

        prob.model.connect('p1.x1', 'comp.x1')
        prob.model.connect('p2.x2', 'comp.x2')

        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials()

        x1_error = data['comp']['y', 'x1']['abs error']
        assert_rel_error(self, x1_error.forward, 1., 1e-8)
        assert_rel_error(self, x1_error.reverse, 1., 1e-8)

        x2_error = data['comp']['y', 'x2']['rel error']
        assert_rel_error(self, x2_error.forward, 9., 1e-8)
        assert_rel_error(self, x2_error.reverse, 9., 1e-8)

    def test_feature_check_partials_suppress(self):
        class MyComp(ExplicitComponent):
            def setup(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)

                self.add_output('y', 5.5)

            def compute(self, inputs, outputs):
                """ Doesn't do much. """
                outputs['y'] = 3.0*inputs['x1'] + 4.0*inputs['x2']

            def compute_partials(self, inputs, partials):
                """Intentionally incorrect derivative."""
                J = partials
                J['y', 'x1'] = np.array([4.0])
                J['y', 'x2'] = np.array([40])

        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('comp', MyComp())

        prob.model.connect('p1.x1', 'comp.x1')
        prob.model.connect('p2.x2', 'comp.x2')

        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(suppress_output=True)
        print(data)

    def test_set_step_on_comp(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')
        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        comp.metadata['check_step'] = 1e-2

        prob.setup()
        prob.run_model()

        prob.check_partials()

    def test_set_step_global(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')
        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        opts = {'step' : 1e-2}

        prob.setup()
        prob.run_model()

        prob.check_partials(global_options=opts)

    def test_set_method_on_comp(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')
        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        comp.metadata['check_method'] = 'cs'

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        prob.check_partials()

    def test_set_method_global(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')
        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        opts = {'method' : 'cs'}

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        prob.check_partials(global_options=opts)

    def test_set_form_on_comp(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')
        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        comp.metadata['check_form'] = 'central'

        prob.setup()
        prob.run_model()

        prob.check_partials()

    def test_set_form_global(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')
        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        opts = {'form' : 'central'}

        prob.setup()
        prob.run_model()

        prob.check_partials(global_options=opts)

    def test_set_step_calc_on_comp(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')
        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        comp.metadata['check_step_calc'] = 'rel'

        prob.setup()
        prob.run_model()

        prob.check_partials()

    def test_set_step_calc_global(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        opts = {'step_calc' : 'rel'}

        prob.setup()
        prob.run_model()

        prob.check_partials(global_options=opts)



class TestProblemCheckTotals(unittest.TestCase):

    def test_cs(self):
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(force_alloc_complex=True)

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives with complex step and a larger step size.
        testlogger = TestLogger()
        totals = prob.check_total_derivatives(method='cs', step=1.0e-1, logger=testlogger)

        lines = testlogger.get('info')

        self.assertTrue('9.80614' in lines[4], "'9.80614' not found in '%s'" % lines[4])
        self.assertTrue('9.80614' in lines[5], "'9.80614' not found in '%s'" % lines[5])

        assert_rel_error(self, totals['con_cmp2.con2', 'px.x']['J_fwd'], [[0.09692762]], 1e-5)
        assert_rel_error(self, totals['con_cmp2.con2', 'px.x']['J_fd'], [[0.09692762]], 1e-5)

    def test_desvar_as_obj(self):
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_objective('x')

        prob.set_solver_print(level=0)

        prob.setup(force_alloc_complex=True)

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives with complex step and a larger step size.
        testlogger = TestLogger()
        totals = prob.check_total_derivatives(method='cs', step=1.0e-1, logger=testlogger)

        lines = testlogger.get('info')

        self.assertTrue('1.000' in lines[4])
        self.assertTrue('1.000' in lines[5])
        self.assertTrue('0.000' in lines[6])
        self.assertTrue('0.000' in lines[8])

        assert_rel_error(self, totals['px.x', 'px.x']['J_fwd'], [[1.0]], 1e-5)
        assert_rel_error(self, totals['px.x', 'px.x']['J_fd'], [[1.0]], 1e-5)

    def test_desvar_and_response_with_indices(self):

        class ArrayComp2D(ExplicitComponent):
            """
            A fairly simple array component.
            """

            def setup(self):

                self.JJ = np.array([[1.0, 3.0, -2.0, 7.0],
                                    [6.0, 2.5, 2.0, 4.0],
                                    [-1.0, 0.0, 8.0, 1.0],
                                    [1.0, 4.0, -5.0, 6.0]])

                # Params
                self.add_input('x1', np.zeros([4]))

                # Unknowns
                self.add_output('y1', np.zeros([4]))

            def compute(self, inputs, outputs):
                """
                Execution.
                """
                outputs['y1'] = self.JJ.dot(inputs['x1'])

            def compute_partials(self, inputs, partials):
                """
                Analytical derivatives.
                """
                partials[('y1', 'x1')] = self.JJ

        prob = Problem()
        prob.model = model = Group()
        model.add_subsystem('x_param1', IndepVarComp('x1', np.ones((4))),
                            promotes=['x1'])
        model.add_subsystem('mycomp', ArrayComp2D(), promotes=['x1', 'y1'])

        model.add_design_var('x1', indices=[1, 3])
        model.add_constraint('y1', indices=[0, 2])

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        Jbase = model.get_subsystem('mycomp').JJ
        of = ['y1']
        wrt = ['x1']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['y1', 'x1'][0][0], Jbase[0, 1], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][0][1], Jbase[0, 3], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][1][0], Jbase[2, 1], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][1][1], Jbase[2, 3], 1e-8)

        totals = prob.check_total_derivatives()
        jac = totals[('mycomp.y1', 'x_param1.x1')]['J_fd']
        assert_rel_error(self, jac[0][0], Jbase[0, 1], 1e-8)
        assert_rel_error(self, jac[0][1], Jbase[0, 3], 1e-8)
        assert_rel_error(self, jac[1][0], Jbase[2, 1], 1e-8)
        assert_rel_error(self, jac[1][1], Jbase[2, 3], 1e-8)

        # Objective instead

        prob = Problem()
        prob.model = model = Group()
        model.add_subsystem('x_param1', IndepVarComp('x1', np.ones((4))),
                            promotes=['x1'])
        model.add_subsystem('mycomp', ArrayComp2D(), promotes=['x1', 'y1'])

        model.add_design_var('x1', indices=[1, 3])
        model.add_objective('y1', index=1)

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        Jbase = model.get_subsystem('mycomp').JJ
        of = ['y1']
        wrt = ['x1']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['y1', 'x1'][0][0], Jbase[1, 1], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][0][1], Jbase[1, 3], 1e-8)

        totals = prob.check_total_derivatives()
        jac = totals[('mycomp.y1', 'x_param1.x1')]['J_fd']
        assert_rel_error(self, jac[0][0], Jbase[1, 1], 1e-8)
        assert_rel_error(self, jac[0][1], Jbase[1, 3], 1e-8)

    def test_cs_suppress(self):
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(force_alloc_complex=True)

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives with complex step and a larger step size.
        testlogger = TestLogger()
        totals = prob.check_total_derivatives(method='cs', step=1.0e-1, logger=testlogger,
                                              suppress_output=True)

        data = totals['con_cmp2.con2', 'px.x']
        self.assertTrue('J_fwd' in data)
        self.assertTrue('rel error' in data)
        self.assertTrue('abs error' in data)
        self.assertTrue('magnitude' in data)

        lines = testlogger.get('info')
        self.assertEqual(len(lines), 0)

    def test_two_desvar_as_con(self):
        #
        # TODO: This tests a bug that omitted the finite differencing of cross derivatives for
        # cases where a design variable is also a constraint or objective. It currently
        # fails because of another bug which will be fixed on Bret's revelance branch.

        raise unittest.SkipTest('Waiting for a bug fix.')

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_constraint('x', upper=0.0)
        prob.model.add_constraint('z', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(check=False)

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # XXXX
        z = prob._compute_total_derivs()
        print(z)

        testlogger = TestLogger()
        totals = prob.check_total_derivatives(method='fd', step=1.0e-1, logger=testlogger)

        lines = testlogger.get('info')

        assert_rel_error(self, totals['px.x', 'px.x']['J_fwd'], [[1.0]], 1e-5)
        assert_rel_error(self, totals['px.x', 'px.x']['J_fd'], [[1.0]], 1e-5)
        assert_rel_error(self, totals['pz.z', 'pz.z']['J_fwd'], [[1.0]], 1e-5)
        assert_rel_error(self, totals['pz.z', 'pz.z']['J_fd'], [[1.0]], 1e-5)
        assert_rel_error(self, totals['px.x', 'pz.z']['J_fwd'], [[0.0]], 1e-5)
        assert_rel_error(self, totals['px.x', 'pz.z']['J_fd'], [[0.0]], 1e-5)
        assert_rel_error(self, totals['pz.z', 'px.x']['J_fwd'], [[0.0]], 1e-5)
        assert_rel_error(self, totals['pz.z', 'px.x']['J_fd'], [[0.0]], 1e-5)

if __name__ == "__main__":
    unittest.main()
