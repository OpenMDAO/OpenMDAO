""" Testing for Problem.check_partials and check_totals."""

from six import iteritems
from six.moves import cStringIO

import unittest
import warnings

import numpy as np

from openmdao.api import Problem, Group, ExplicitComponent, ImplicitComponent, \
    IndepVarComp, ExecComp, NonlinearRunOnce, NonlinearBlockGS, ScipyKrylov, NewtonSolver, \
    DirectSolver, LinearBlockGS, BroydenSolver
from openmdao.core.tests.test_impl_comp import QuadraticLinearize, QuadraticJacVec
from openmdao.core.tests.test_matmat import MultiJacVec
from openmdao.test_suite.components.impl_comp_array import TestImplCompArrayMatVec
from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec
from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDis1withDerivatives, \
     SellarDis2withDerivatives
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.test_suite.groups.parallel_groups import FanInSubbedIDVC
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class ParaboloidTricky(ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.scale = 1e-7

        self.declare_partials(of='*', wrt='*')

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


class MyCompGoodPartials(ExplicitComponent):
    def setup(self):
        self.add_input('x1', 3.0)
        self.add_input('x2', 5.0)
        self.add_output('y', 5.5)
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['y'] = 3.0 * inputs['x1'] + 4.0 * inputs['x2']

    def compute_partials(self, inputs, partials):
        """Correct derivative."""
        J = partials
        J['y', 'x1'] = np.array([3.0])
        J['y', 'x2'] = np.array([4.0])


class MyCompBadPartials(ExplicitComponent):
    def setup(self):
        self.add_input('y1', 3.0)
        self.add_input('y2', 5.0)
        self.add_output('z', 5.5)
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['z'] = 3.0 * inputs['y1'] + 4.0 * inputs['y2']

    def compute_partials(self, inputs, partials):
        """Intentionally incorrect derivative."""
        J = partials
        J['z', 'y1'] = np.array([33.0])
        J['z', 'y2'] = np.array([40.0])


class MyComp(ExplicitComponent):
    def setup(self):
        self.add_input('x1', 3.0)
        self.add_input('x2', 5.0)

        self.add_output('y', 5.5)

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['y'] = 3.0*inputs['x1'] + 4.0*inputs['x2']

    def compute_partials(self, inputs, partials):
        """Intentionally incorrect derivative."""
        J = partials
        J['y', 'x1'] = np.array([4.0])
        J['y', 'x2'] = np.array([40])


class TestProblemCheckPartials(unittest.TestCase):

    def test_incorrect_jacobian(self):

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

        stream = cStringIO()
        prob.check_partials(out_stream=stream)
        lines = stream.getvalue().splitlines()

        y_wrt_x1_line = lines.index("  comp: 'y' wrt 'x1'")

        self.assertTrue(lines[y_wrt_x1_line+3].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        self.assertTrue(lines[y_wrt_x1_line+5].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        y_wrt_x2_line = lines.index("  comp: 'y' wrt 'x2'")
        self.assertTrue(lines[y_wrt_x2_line+3].endswith('*'),
                        msg='Error flag not expected in output but displayed')
        self.assertTrue(lines[y_wrt_x2_line+5].endswith('*'),
                        msg='Error flag not expected in output but displayed')

    def test_component_only(self):

        prob = Problem()
        prob.model = MyComp()

        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        stream = cStringIO()
        prob.check_partials(out_stream=stream)
        lines = stream.getvalue().splitlines()

        y_wrt_x1_line = lines.index("  : 'y' wrt 'x1'")
        self.assertTrue(lines[y_wrt_x1_line+3].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        self.assertTrue(lines[y_wrt_x1_line+5].endswith('*'),
                        msg='Error flag expected in output but not displayed')

    def test_component_only_suppress(self):

        prob = Problem()
        prob.model = MyComp()

        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        stream = cStringIO()
        data = prob.check_partials(out_stream=None)

        subheads = data[''][('y', 'x1')]
        self.assertTrue('J_fwd' in subheads)
        self.assertTrue('rel error' in subheads)
        self.assertTrue('abs error' in subheads)
        self.assertTrue('magnitude' in subheads)

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 0)

    def test_component_has_no_outputs(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem("indep", IndepVarComp('x', 5.))
        model.add_subsystem("comp1", ExecComp("y=2*x"))

        comp2 = model.add_subsystem("comp2", ExplicitComponent())
        comp2.add_input('x', val=0.)

        model.connect('indep.x', ['comp1.x', 'comp2.x'])

        prob.setup()
        prob.run_model()

        with warnings.catch_warnings(record=True) as w:
            data = prob.check_partials(out_stream=None)

        # warning about 'comp2'
        self.assertEqual(len(w), 1)
        expected = "No derivative data found for Component 'comp2'."
        self.assertEqual(str(w[0].message), expected)

        # and no derivative data for 'comp2'
        self.assertFalse('comp2' in data)

        # but we still get good derivative data for 'comp1'
        self.assertTrue('comp1' in data)

        assert_rel_error(self, data['comp1'][('y', 'x')]['J_fd'][0][0], 2., 1e-9)
        assert_rel_error(self, data['comp1'][('y', 'x')]['J_fwd'][0][0], 2., 1e-15)
        assert_rel_error(self, data['comp1'][('y', 'x')]['J_rev'][0][0], 2., 1e-15)

    def test_missing_entry(self):
        class MyComp(ExplicitComponent):
            def setup(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)

                self.add_output('y', 5.5)

                self.declare_partials(of='*', wrt='*')

                self.lin_count = 0

            def compute(self, inputs, outputs):
                outputs['y'] = 3.0*inputs['x1'] + 4.0*inputs['x2']

            def compute_partials(self, inputs, partials):
                """Intentionally left out derivative."""
                J = partials
                J['y', 'x1'] = np.array([3.0])
                self.lin_count += 1

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

        data = prob.check_partials(out_stream=None)

        self.assertEqual(prob.model.comp.lin_count, 1)

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
                self.declare_partials(of='*', wrt='*', method='fd')

            def compute(self, inputs, outputs):
                outputs['flow:T'] = inputs['T']
                outputs['flow:P'] = inputs['P']

        p = Problem()
        model = p.model = Group()
        indep = model.add_subsystem('indep', IndepVarComp(), promotes=['*'])

        indep.add_output('T', val=100., units='degK')
        indep.add_output('P', val=1., units='bar')

        model.add_subsystem('units', UnitCompBase(), promotes=['*'])

        p.setup()
        data = p.check_partials(out_stream=None)

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

                self.declare_partials(of='*', wrt='*')

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

        model.nonlinear_solver = NonlinearRunOnce()

        p.setup()
        data = p.check_partials(out_stream=None)

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
        self.assertEqual(units.run_count, 5)

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
                    size = 1
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

        data = p.check_partials(out_stream=None)
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

        data = prob.check_partials(out_stream=None)

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

        data = prob.check_partials(out_stream=None)

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
                    [3., 4.],
                    [2., 3.],
                ])

                self.declare_partials(of='*', wrt='*')

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

        data = prob.check_partials(out_stream=None)

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

                self.declare_partials(of='g', wrt='x')
                self.declare_partials(of='g', wrt='z', dependent=False)

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

        stream = cStringIO()
        data = prob.check_partials(out_stream=stream)
        lines = stream.getvalue().splitlines()

        self.assertTrue("  comp: 'g' wrt 'z'" not in lines)
        self.assertTrue(('g', 'z') not in data['comp'])
        self.assertTrue("  comp: 'g' wrt 'x'" in lines)
        self.assertTrue(('g', 'x') in data['comp'])

    def test_dependent_false_show(self):
        # Test that we show derivs declared with dependent=False if the fd is not
        # ~zero.

        class SimpleComp2(ExplicitComponent):
            def setup(self):
                self.add_input('z', shape=(2, 2))
                self.add_input('x', shape=(2, 2))
                self.add_output('g', shape=(2, 2))

                self.declare_partials(of='g', wrt='x')
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

        stream = cStringIO()
        data = prob.check_partials(out_stream=stream)
        lines = stream.getvalue().splitlines()

        self.assertTrue("  comp: 'g' wrt 'z'" in lines)
        self.assertTrue(('g', 'z') in data['comp'])
        self.assertTrue("  comp: 'g' wrt 'x'" in lines)
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

        comp.set_check_partial_options(wrt='*', step=1e-2)

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        # This will fail unless you set the check_step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-5)
        self.assertLess(x_error.reverse, 1e-5)

    def test_set_step_global(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(out_stream=None, step=1e-2)

        # This will fail unless you set the global step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-5)
        self.assertLess(x_error.reverse, 1e-5)

    def test_complex_step_not_allocated(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidMatVec())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.set_check_partial_options(wrt='*', method='cs')

        prob.setup(check=False)
        prob.run_model()

        with warnings.catch_warnings(record=True) as w:
            data = prob.check_partials(out_stream=None)

        self.assertEqual(len(w), 1)

        msg = "The following components requested complex step, but force_alloc_complex " + \
              "has not been set to True, so finite difference was used: ['comp']\n" + \
              "To enable complex step, specify 'force_alloc_complex=True' when calling " + \
              "setup on the problem, e.g. 'problem.setup(force_alloc_complex=True)'"
        self.assertEqual(str(w[0].message), msg)

        # Derivative still calculated, but with fd instead.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-5)
        self.assertLess(x_error.reverse, 1e-5)

    def test_set_method_on_comp(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.set_check_partial_options(wrt='*', method='cs')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-5)
        self.assertLess(x_error.reverse, 1e-5)

    def test_set_method_global(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        data = prob.check_partials(out_stream=None, method='cs')

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

        comp.set_check_partial_options(wrt='*', form='central')

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        # This will fail unless you set the check_step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-3)
        self.assertLess(x_error.reverse, 1e-3)

    def test_set_form_global(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(out_stream=None, form='central')

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

        comp.set_check_partial_options(wrt='*', step_calc='rel')

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        # This will fail unless you set the check_step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 3e-3)
        self.assertLess(x_error.reverse, 3e-3)

    def test_set_step_calc_global(self):
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partials(out_stream=None, step_calc='rel')

        # This will fail unless you set the global step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 3e-3)
        self.assertLess(x_error.reverse, 3e-3)

    def test_set_check_option_precedence(self):
        # Test that we omit derivs declared with dependent=False

        class SimpleComp1(ExplicitComponent):
            def setup(self):
                self.add_input('ab', 13.0)
                self.add_input('aba', 13.0)
                self.add_input('ba', 13.0)
                self.add_output('y', 13.0)

                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                ab = inputs['ab']
                aba = inputs['aba']
                ba = inputs['ba']

                outputs['y'] = ab**3 + aba**3 + ba**3

            def compute_partials(self, inputs, partials):
                ab = inputs['ab']
                aba = inputs['aba']
                ba = inputs['ba']

                partials['y', 'ab'] = 3.0*ab**2
                partials['y', 'aba'] = 3.0*aba**2
                partials['y', 'ba'] = 3.0*ba**2

        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('ab', 13.0))
        prob.model.add_subsystem('p2', IndepVarComp('aba', 13.0))
        prob.model.add_subsystem('p3', IndepVarComp('ba', 13.0))
        comp = prob.model.add_subsystem('comp', SimpleComp1())

        prob.model.connect('p1.ab', 'comp.ab')
        prob.model.connect('p2.aba', 'comp.aba')
        prob.model.connect('p3.ba', 'comp.ba')

        prob.setup(check=False)

        comp.set_check_partial_options(wrt='a*', step=1e-2)
        comp.set_check_partial_options(wrt='*a', step=1e-4)

        prob.run_model()

        data = prob.check_partials(out_stream=None)

        # Note 'aba' gets the better value from the second options call with the *a wildcard.
        assert_rel_error(self, data['comp']['y', 'ab']['J_fd'][0][0], 507.3901, 1e-4)
        assert_rel_error(self, data['comp']['y', 'aba']['J_fd'][0][0], 507.0039, 1e-4)
        assert_rel_error(self, data['comp']['y', 'ba']['J_fd'][0][0], 507.0039, 1e-4)

    def test_option_printing(self):
        # Make sure we print the approximation type for each variable.
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.set_check_partial_options(wrt='x', method='cs')
        comp.set_check_partial_options(wrt='y', form='central')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        stream = cStringIO()
        prob.check_partials(out_stream=stream)

        lines = stream.getvalue().splitlines()
        self.assertTrue('cs' in lines[5],
                        msg='Did you change the format for printing check derivs?')
        self.assertTrue('fd' in lines[19],
                        msg='Did you change the format for printing check derivs?')

    def test_set_check_partial_options_invalid(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

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

        prob.setup()
        prob.run_model()

        # check invalid wrt
        with self.assertRaises(ValueError) as cm:
            comp.set_check_partial_options(wrt=np.array([1.0]))

        self.assertEqual(str(cm.exception),
                         "The value of 'wrt' must be a string or list of strings, but a "
                         "type of 'ndarray' was provided.")

        # check invalid method
        with self.assertRaises(ValueError) as cm:
            comp.set_check_partial_options(wrt=['*'], method='foo')

        self.assertEqual(str(cm.exception),
                         "Method 'foo' is not supported, method must be one of ('fd', 'cs')")

        # check invalid form
        comp._declared_partial_checks = []
        comp.set_check_partial_options(wrt=['*'], form='foo')

        with self.assertRaises(ValueError) as cm:
            prob.check_partials()

        # The form options sometimes print out in different order.
        msg = str(cm.exception)
        self.assertTrue("'foo' is not a valid form of finite difference; "
                         "must be one of [" in msg, 'error message not correct.')
        self.assertTrue('forward' in msg, 'error message not correct.')
        self.assertTrue('backward' in msg, 'error message not correct.')
        self.assertTrue('central' in msg, 'error message not correct.')

        # check invalid step
        with self.assertRaises(ValueError) as cm:
            comp.set_check_partial_options(wrt=['*'], step='foo')

        self.assertEqual(str(cm.exception),
                         "The value of 'step' must be numeric, but 'foo' was specified.")

        # check invalid step_calc
        with self.assertRaises(ValueError) as cm:
            comp.set_check_partial_options(wrt=['*'], step_calc='foo')

        self.assertEqual(str(cm.exception),
                         "The value of 'step_calc' must be one of ('abs', 'rel'), "
                         "but 'foo' was specified.")

        # check invalid wrt
        comp._declared_partial_checks = []
        comp.set_check_partial_options(wrt=['x*', 'y', 'z', 'a*'])

        with self.assertRaises(ValueError) as cm:
            prob.check_partials()

        self.assertEqual(str(cm.exception), "Invalid 'wrt' variable specified "
                         "for check_partial options on Component 'comp': 'z'.")

        # check multiple invalid wrt
        comp._declared_partial_checks = []
        comp.set_check_partial_options(wrt=['a', 'b', 'c'])

        with self.assertRaises(ValueError) as cm:
            prob.check_partials()

        self.assertEqual(str(cm.exception), "Invalid 'wrt' variables specified "
                         "for check_partial options on Component 'comp': ['a', 'b', 'c'].")

    def test_compact_print_formatting(self):
        class MyCompShortVarNames(ExplicitComponent):
            def setup(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)
                self.add_output('y', 5.5)
                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                outputs['y'] = 3.0*inputs['x1'] + 4.0*inputs['x2']

            def compute_partials(self, inputs, partials):
                """Intentionally incorrect derivative."""
                J = partials
                J['y', 'x1'] = np.array([4.0])
                J['y', 'x2'] = np.array([40])

        class MyCompLongVarNames(ExplicitComponent):
            def setup(self):
                self.add_input('really_long_variable_name_x1', 3.0)
                self.add_input('x2', 5.0)
                self.add_output('really_long_variable_name_y', 5.5)
                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                outputs['really_long_variable_name_y'] = \
                    3.0*inputs['really_long_variable_name_x1'] + 4.0*inputs['x2']

            def compute_partials(self, inputs, partials):
                """Intentionally incorrect derivative."""
                J = partials
                J['really_long_variable_name_y', 'really_long_variable_name_x1'] = np.array([4.0])
                J['really_long_variable_name_y', 'x2'] = np.array([40])

        # First short var names
        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('p1', IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('comp', MyCompShortVarNames())
        prob.model.connect('p1.x1', 'comp.x1')
        prob.model.connect('p2.x2', 'comp.x2')
        prob.set_solver_print(level=0)
        prob.setup(check=False)
        prob.run_model()
        stream = cStringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        lines = stream.getvalue().splitlines()
        # Check to make sure all the header and value lines have their columns lined up
        header_locations_of_bars = None
        sep = '|'
        for line in lines:
            if sep in line:
                if header_locations_of_bars:
                    value_locations_of_bars = [i for i, ltr in enumerate(line) if ltr == sep]
                    self.assertEqual(value_locations_of_bars, header_locations_of_bars,
                                     msg="Column separators should all be aligned")
                else:
                    header_locations_of_bars = [i for i, ltr in enumerate(line) if ltr == sep]

        # Then long var names
        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('p1', IndepVarComp('really_long_variable_name_x1', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('comp', MyCompLongVarNames())
        prob.model.connect('p1.really_long_variable_name_x1', 'comp.really_long_variable_name_x1')
        prob.model.connect('p2.x2', 'comp.x2')
        prob.set_solver_print(level=0)
        prob.setup(check=False)
        prob.run_model()
        stream = cStringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        lines = stream.getvalue().splitlines()
        # Check to make sure all the header and value lines have their columns lined up
        header_locations_of_bars = None
        sep = '|'
        for line in lines:
            if sep in line:
                if header_locations_of_bars:
                    value_locations_of_bars = [i for i, ltr in enumerate(line) if ltr == sep]
                    self.assertEqual(value_locations_of_bars, header_locations_of_bars,
                                     msg="Column separators should all be aligned")
                else:
                    header_locations_of_bars = [i for i, ltr in enumerate(line) if ltr == sep]

    def test_compact_print_exceed_tol(self):

        prob = Problem()
        prob.model = MyCompGoodPartials()
        prob.set_solver_print(level=0)
        prob.setup(check=False)
        prob.run_model()
        stream = cStringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('>ABS_TOL'), 0)
        self.assertEqual(stream.getvalue().count('>REL_TOL'), 0)

        prob = Problem()
        prob.model = MyCompBadPartials()
        prob.set_solver_print(level=0)
        prob.setup(check=False)
        prob.run_model()
        stream = cStringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('>ABS_TOL'), 2)
        self.assertEqual(stream.getvalue().count('>REL_TOL'), 2)

    def test_check_partials_display_rev(self):

        # 1: Check display of revs for implicit comp for compact and non-compact display
        group = Group()
        comp1 = group.add_subsystem('comp1', IndepVarComp())
        comp1.add_output('a', 1.0)
        comp1.add_output('b', -4.0)
        comp1.add_output('c', 3.0)
        group.add_subsystem('comp2', QuadraticLinearize())
        group.add_subsystem('comp3', QuadraticJacVec())
        group.connect('comp1.a', 'comp2.a')
        group.connect('comp1.b', 'comp2.b')
        group.connect('comp1.c', 'comp2.c')
        group.connect('comp1.a', 'comp3.a')
        group.connect('comp1.b', 'comp3.b')
        group.connect('comp1.c', 'comp3.c')
        prob = Problem(model=group)
        prob.setup(check=False)

        stream = cStringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('n/a'), 25)
        self.assertEqual(stream.getvalue().count('rev'), 15)
        self.assertEqual(stream.getvalue().count('Component'), 2)
        self.assertEqual(stream.getvalue().count('wrt'), 12)

        stream = cStringIO()
        prob.check_partials(out_stream=stream, compact_print=False)
        self.assertEqual(stream.getvalue().count('Reverse Magnitude'), 4)
        self.assertEqual(stream.getvalue().count('Raw Reverse Derivative'), 4)
        self.assertEqual(stream.getvalue().count('Jrev'), 16)

        # 2: Explicit comp, all comps define Jacobians for compact and non-compact display
        class MyComp(ExplicitComponent):
            def setup(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)
                self.add_output('z', 5.5)
                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                outputs['z'] = 3.0 * inputs['x1'] + -4444.0 * inputs['x2']

            def compute_partials(self, inputs, partials):
                """Correct derivative."""
                J = partials
                J['z', 'x1'] = np.array([3.0])
                J['z', 'x2'] = np.array([-4444.0])

        prob = Problem()
        prob.model = MyComp()
        prob.set_solver_print(level=0)
        prob.setup(check=False)
        prob.run_model()
        stream = cStringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('rev'), 0)

        stream = cStringIO()
        prob.check_partials(out_stream=stream, compact_print=False)
        # So for this case, they do all provide them, so rev should not be shown
        self.assertEqual(stream.getvalue().count('Forward Magnitude'), 2)
        self.assertEqual(stream.getvalue().count('Reverse Magnitude'), 0)
        self.assertEqual(stream.getvalue().count('Absolute Error'), 2)
        self.assertEqual(stream.getvalue().count('Relative Error'), 2)
        self.assertEqual(stream.getvalue().count('Raw Forward Derivative'), 2)
        self.assertEqual(stream.getvalue().count('Raw Reverse Derivative'), 0)
        self.assertEqual(stream.getvalue().count('Raw FD Derivative'), 2)

        # 3: Explicit comp that does not define Jacobian. It defines compute_jacvec_product
        #      For both compact and non-compact display
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
        stream = cStringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('rev'), 10)

        stream = cStringIO()
        prob.check_partials(out_stream=stream, compact_print=False)
        self.assertEqual(stream.getvalue().count('Reverse'), 4)
        self.assertEqual(stream.getvalue().count('Jrev'), 8)

        # 4: Mixed comps. Some with jacobians. Some not
        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('p0', IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p1', IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('c0', MyComp())  # in x1,x2, out is z
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidMatVec())
        prob.model.connect('p0.x1', 'c0.x1')
        prob.model.connect('p1.x2', 'c0.x2')
        prob.model.connect('c0.z', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')
        prob.set_solver_print(level=0)
        prob.setup(check=False)
        prob.run_model()

        stream = cStringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('n/a'), 10)
        self.assertEqual(stream.getvalue().count('rev'), 15)
        self.assertEqual(stream.getvalue().count('Component'), 2)
        self.assertEqual(stream.getvalue().count('wrt'), 8)

        stream = cStringIO()
        prob.check_partials(out_stream=stream, compact_print=False)
        self.assertEqual(stream.getvalue().count('Forward Magnitude'), 4)
        self.assertEqual(stream.getvalue().count('Reverse Magnitude'), 2)
        self.assertEqual(stream.getvalue().count('Absolute Error'), 8)
        self.assertEqual(stream.getvalue().count('Relative Error'), 8)
        self.assertEqual(stream.getvalue().count('Raw Forward Derivative'), 4)
        self.assertEqual(stream.getvalue().count('Raw Reverse Derivative'), 2)
        self.assertEqual(stream.getvalue().count('Raw FD Derivative'), 4)

        # 5: One comp defines compute_multi_jacvec_product
        size = 6
        prob = Problem()
        model = prob.model
        model.add_subsystem('px', IndepVarComp('x', val=(np.arange(size, dtype=float) + 1.) * 3.0))
        model.add_subsystem('py', IndepVarComp('y', val=(np.arange(size, dtype=float) + 1.) * 2.0))
        model.add_subsystem('comp', MultiJacVec(size))

        model.connect('px.x', 'comp.x')
        model.connect('py.y', 'comp.y')

        model.add_design_var('px.x', vectorize_derivs=False)
        model.add_design_var('py.y', vectorize_derivs=False)
        model.add_constraint('comp.f_xy', vectorize_derivs=False)

        prob.setup(check=False)
        prob.run_model()
        stream = cStringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('rev'), 10)

    def test_check_partials_worst_subjac(self):
        # The first is printing the worst subjac at the bottom of the output. Worst is defined by
        # looking at the fwd and rev columns of the relative error (i.e., the 2nd and 3rd last
        # columns) of the compact_print=True output. We should print the component name, then
        # repeat the full row for the worst-case subjac (i.e., output-input pair).
        # This should only occur in the compact_print=True case.

        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('p0', IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p1', IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('p2', IndepVarComp('y2', 6.0))
        prob.model.add_subsystem('good', MyCompGoodPartials())
        prob.model.add_subsystem('bad', MyCompBadPartials())
        prob.model.connect('p0.x1', 'good.x1')
        prob.model.connect('p1.x2', 'good.x2')
        prob.model.connect('good.y', 'bad.y1')
        prob.model.connect('p2.y2', 'bad.y2')
        prob.set_solver_print(level=0)
        prob.setup(check=False)
        prob.run_model()

        stream = cStringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count("'z'        wrt 'y1'"), 2)

    def test_check_partials_show_only_incorrect(self):
        # The second is adding an option to show only the incorrect subjacs
        # (according to abs_err_tol and rel_err_tol), called
        # show_only_incorrect. This should be False by default, but when True,
        # it should print only the subjacs found to be incorrect. This applies
        # to both compact_print=True and False.

        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('p0', IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p1', IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('p2', IndepVarComp('y2', 6.0))
        prob.model.add_subsystem('good', MyCompGoodPartials())
        prob.model.add_subsystem('bad', MyCompBadPartials())
        prob.model.connect('p0.x1', 'good.x1')
        prob.model.connect('p1.x2', 'good.x2')
        prob.model.connect('good.y', 'bad.y1')
        prob.model.connect('p2.y2', 'bad.y2')
        prob.set_solver_print(level=0)
        prob.setup(check=False)
        prob.run_model()

        stream = cStringIO()
        # prob.check_partials(compact_print=True,show_only_incorrect=False)
        prob.check_partials(out_stream=stream, compact_print=True, show_only_incorrect=True)
        self.assertEqual(stream.getvalue().count("MyCompBadPartials"), 2)
        self.assertEqual(stream.getvalue().count("'z'        wrt 'y1'"), 2)
        self.assertEqual(stream.getvalue().count("MyCompGoodPartials"), 0)

        stream = cStringIO()
        prob.check_partials(compact_print=False, show_only_incorrect=False)
        prob.check_partials(out_stream=stream, compact_print=False, show_only_incorrect=True)
        self.assertEqual(stream.getvalue().count("MyCompGoodPartials"), 0)
        self.assertEqual(stream.getvalue().count("MyCompBadPartials"), 1)

    def test_includes_excludes(self):

        prob = Problem()
        model = prob.model

        sub = model.add_subsystem('c1c', Group())
        sub.add_subsystem('d1', ExecComp('y=2*x'))
        sub.add_subsystem('e1', ExecComp('y=2*x'))

        sub2 = model.add_subsystem('sss', Group())
        sub3 = sub2.add_subsystem('sss2', Group())
        sub2.add_subsystem('d1', ExecComp('y=2*x'))
        sub3.add_subsystem('e1', ExecComp('y=2*x'))

        model.add_subsystem('abc1cab', ExecComp('y=2*x'))

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None, includes='*c*c*')
        self.assertEqual(len(data), 3)
        self.assertTrue('c1c.d1' in data)
        self.assertTrue('c1c.e1' in data)
        self.assertTrue('abc1cab' in data)

        data = prob.check_partials(out_stream=None, includes=['*d1', '*e1'])
        self.assertEqual(len(data), 4)
        self.assertTrue('c1c.d1' in data)
        self.assertTrue('c1c.e1' in data)
        self.assertTrue('sss.d1' in data)
        self.assertTrue('sss.sss2.e1' in data)

        data = prob.check_partials(out_stream=None, includes=['abc1cab'])
        self.assertEqual(len(data), 1)
        self.assertTrue('abc1cab' in data)

        data = prob.check_partials(out_stream=None, includes='*c*c*', excludes=['*e*'])
        self.assertEqual(len(data), 2)
        self.assertTrue('c1c.d1' in data)
        self.assertTrue('abc1cab' in data)


class TestCheckPartialsFeature(unittest.TestCase):

    def test_feature_incorrect_jacobian(self):
        import numpy as np

        from openmdao.api import Group, ExplicitComponent, IndepVarComp, Problem

        class MyComp(ExplicitComponent):
            def setup(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)

                self.add_output('y', 5.5)

                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
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
        import numpy as np

        from openmdao.api import Group, ExplicitComponent, IndepVarComp, Problem

        class MyComp(ExplicitComponent):
            def setup(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)

                self.add_output('y', 5.5)

                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
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

        data = prob.check_partials(out_stream=None)
        print(data)

    def test_set_step_on_comp(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

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

        comp.set_check_partial_options(wrt='*', step=1e-2)

        prob.setup()
        prob.run_model()

        prob.check_partials()

    def test_set_step_global(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')
        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        prob.check_partials(step=1e-2)

    def test_set_method_on_comp(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

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

        comp.set_check_partial_options(wrt='*', method='cs')

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        prob.check_partials()

    def test_set_method_global(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')
        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        prob.check_partials(method='cs')

    def test_set_form_on_comp(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

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

        comp.set_check_partial_options(wrt='*', form='central')

        prob.setup()
        prob.run_model()

        prob.check_partials()

    def test_set_form_global(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')
        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        prob.check_partials(form='central')

    def test_set_step_calc_on_comp(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

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

        comp.set_check_partial_options(wrt='*', step_calc='rel')

        prob.setup()
        prob.run_model()

        prob.check_partials()

    def test_set_step_calc_global(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky

        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        prob.check_partials(step_calc='rel')

    def test_feature_compact_print_formatting(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

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

        comp.set_check_partial_options(wrt='*', step_calc='rel')

        prob.setup()
        prob.run_model()

        prob.check_partials(compact_print=True)

    def test_feature_check_partials_show_only_incorrect(self):
        from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent

        class MyCompGoodPartials(ExplicitComponent):
            def setup(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)
                self.add_output('y', 5.5)
                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                outputs['y'] = 3.0 * inputs['x1'] + 4.0 * inputs['x2']

            def compute_partials(self, inputs, partials):
                """Correct derivative."""
                J = partials
                J['y', 'x1'] = np.array([3.0])
                J['y', 'x2'] = np.array([4.0])

        class MyCompBadPartials(ExplicitComponent):
            def setup(self):
                self.add_input('y1', 3.0)
                self.add_input('y2', 5.0)
                self.add_output('z', 5.5)
                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                outputs['z'] = 3.0 * inputs['y1'] + 4.0 * inputs['y2']

            def compute_partials(self, inputs, partials):
                """Intentionally incorrect derivative."""
                J = partials
                J['z', 'y1'] = np.array([33.0])
                J['z', 'y2'] = np.array([40.0])

        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('p0', IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p1', IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('p2', IndepVarComp('y2', 6.0))
        prob.model.add_subsystem('good', MyCompGoodPartials())
        prob.model.add_subsystem('bad', MyCompBadPartials())
        prob.model.connect('p0.x1', 'good.x1')
        prob.model.connect('p1.x2', 'good.x2')
        prob.model.connect('good.y', 'bad.y1')
        prob.model.connect('p2.y2', 'bad.y2')
        prob.set_solver_print(level=0)
        prob.setup(check=False)
        prob.run_model()

        prob.check_partials(compact_print=True, show_only_incorrect=True)
        prob.check_partials(compact_print=False, show_only_incorrect=True)

    def test_includes_excludes(self):
        from openmdao.api import Problem, Group, ExecComp

        prob = Problem()
        model = prob.model

        sub = model.add_subsystem('c1c', Group())
        sub.add_subsystem('d1', ExecComp('y=2*x'))
        sub.add_subsystem('e1', ExecComp('y=2*x'))

        sub2 = model.add_subsystem('sss', Group())
        sub3 = sub2.add_subsystem('sss2', Group())
        sub2.add_subsystem('d1', ExecComp('y=2*x'))
        sub3.add_subsystem('e1', ExecComp('y=2*x'))

        model.add_subsystem('abc1cab', ExecComp('y=2*x'))

        prob.setup()
        prob.run_model()

        prob.check_partials(compact_print=True, includes='*c*c*')

        prob.check_partials(compact_print=True, includes=['*d1', '*e1'])

        prob.check_partials(compact_print=True, includes=['abc1cab'])

        prob.check_partials(compact_print=True, includes='*c*c*', excludes=['*e*'])


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

        prob.model.nonlinear_solver.options['atol'] = 1e-15
        prob.model.nonlinear_solver.options['rtol'] = 1e-15

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives with complex step and a larger step size.
        stream = cStringIO()
        totals = prob.check_totals(method='cs', out_stream=stream)

        lines = stream.getvalue().splitlines()

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
        stream = cStringIO()
        totals = prob.check_totals(method='cs', out_stream=stream)

        lines = stream.getvalue().splitlines()

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

                self.declare_partials(of='*', wrt='*')

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
        mycomp = model.add_subsystem('mycomp', ArrayComp2D(), promotes=['x1', 'y1'])

        model.add_design_var('x1', indices=[1, 3])
        model.add_constraint('y1', indices=[0, 2])

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        Jbase = mycomp.JJ
        of = ['y1']
        wrt = ['x1']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['y1', 'x1'][0][0], Jbase[0, 1], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][0][1], Jbase[0, 3], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][1][0], Jbase[2, 1], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][1][1], Jbase[2, 3], 1e-8)

        totals = prob.check_totals()
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
        mycomp = model.add_subsystem('mycomp', ArrayComp2D(), promotes=['x1', 'y1'])

        model.add_design_var('x1', indices=[1, 3])
        model.add_objective('y1', index=1)

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        Jbase = mycomp.JJ
        of = ['y1']
        wrt = ['x1']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['y1', 'x1'][0][0], Jbase[1, 1], 1e-8)
        assert_rel_error(self, J['y1', 'x1'][0][1], Jbase[1, 3], 1e-8)

        totals = prob.check_totals()
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
        totals = prob.check_totals(method='cs', out_stream=None)

        data = totals['con_cmp2.con2', 'px.x']
        self.assertTrue('J_fwd' in data)
        self.assertTrue('rel error' in data)
        self.assertTrue('abs error' in data)
        self.assertTrue('magnitude' in data)

    def test_two_desvar_as_con(self):
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

        totals = prob.check_totals(method='fd', step=1.0e-1, out_stream=None)

        assert_rel_error(self, totals['px.x', 'px.x']['J_fwd'], [[1.0]], 1e-5)
        assert_rel_error(self, totals['px.x', 'px.x']['J_fd'], [[1.0]], 1e-5)
        assert_rel_error(self, totals['pz.z', 'pz.z']['J_fwd'], np.eye(2), 1e-5)
        assert_rel_error(self, totals['pz.z', 'pz.z']['J_fd'], np.eye(2), 1e-5)
        assert_rel_error(self, totals['px.x', 'pz.z']['J_fwd'], [[0.0, 0.0]], 1e-5)
        assert_rel_error(self, totals['px.x', 'pz.z']['J_fd'], [[0.0, 0.0]], 1e-5)
        assert_rel_error(self, totals['pz.z', 'px.x']['J_fwd'], [[0.0], [0.0]], 1e-5)
        assert_rel_error(self, totals['pz.z', 'px.x']['J_fd'], [[0.0], [0.0]], 1e-5)

    def test_full_con_with_index_desvar(self):
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('z', lower=-100, upper=100, indices=[1])
        prob.model.add_constraint('z', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(check=False)

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        totals = prob.check_totals(method='fd', step=1.0e-1, out_stream=None)

        assert_rel_error(self, totals['pz.z', 'pz.z']['J_fwd'], [[0.0], [1.0]], 1e-5)
        assert_rel_error(self, totals['pz.z', 'pz.z']['J_fd'], [[0.0], [1.0]], 1e-5)

    def test_full_desvar_with_index_con(self):
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_constraint('z', upper=0.0, indices=[1])

        prob.set_solver_print(level=0)

        prob.setup(check=False)

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        totals = prob.check_totals(method='fd', step=1.0e-1, out_stream=None)

        assert_rel_error(self, totals['pz.z', 'pz.z']['J_fwd'], [[0.0, 1.0]], 1e-5)
        assert_rel_error(self, totals['pz.z', 'pz.z']['J_fd'], [[0.0, 1.0]], 1e-5)

    def test_full_desvar_with_index_obj(self):
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('z', index=1)

        prob.set_solver_print(level=0)

        prob.setup(check=False)

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        totals = prob.check_totals(method='fd', step=1.0e-1, out_stream=None)

        assert_rel_error(self, totals['pz.z', 'pz.z']['J_fwd'], [[0.0, 1.0]], 1e-5)
        assert_rel_error(self, totals['pz.z', 'pz.z']['J_fd'], [[0.0, 1.0]], 1e-5)

    def test_bug_fd_with_sparse(self):
        # This bug was found via the x57 model in pointer.

        class TimeComp(ExplicitComponent):

            def setup(self):
                self.node_ptau = node_ptau = np.array([-1., 0., 1.])

                self.add_input('t_duration', val=1.)
                self.add_output('time', shape=len(node_ptau))

                # Setup partials
                nn = 3
                rs = np.arange(nn)
                cs = np.zeros(nn)

                self.declare_partials(of='time', wrt='t_duration', rows=rs, cols=cs, val=1.0)

            def compute(self, inputs, outputs):
                node_ptau = self.node_ptau
                t_duration = inputs['t_duration']

                outputs['time'][:] = 0.5 * (node_ptau + 33) * t_duration

            def compute_partials(self, inputs, jacobian):
                node_ptau = self.node_ptau

                jacobian['time', 't_duration'] = 0.5 * (node_ptau + 33)

        class CellComp(ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                n = self.options['num_nodes']

                self.add_input('I_Li', val=3.25*np.ones(n))
                self.add_output('zSOC', val=np.ones(n))

                # Partials
                ar = np.arange(n)
                self.declare_partials(of='zSOC', wrt='I_Li', rows=ar, cols=ar)

            def compute(self, inputs, outputs):
                I_Li = inputs['I_Li']
                outputs['zSOC'] = -I_Li / (3600.0)

            def compute_partials(self, inputs, partials):
                partials['zSOC', 'I_Li'] = -1./(3600.0)

        class GaussLobattoPhase(Group):

            def setup(self):
                self.connect('t_duration', 'time.t_duration')

                indep = IndepVarComp()
                indep.add_output('t_duration', val=1.0)
                self.add_subsystem('time_extents', indep, promotes_outputs=['*'])
                self.add_design_var('t_duration', 5.0, 25.0)

                time_comp = TimeComp()
                self.add_subsystem('time', time_comp, promotes_outputs=['time'])

                self.add_subsystem(name='cell', subsys=CellComp(num_nodes=3))

                self.linear_solver = ScipyKrylov()
                self.nonlinear_solver = NewtonSolver()
                self.nonlinear_solver.options['maxiter'] = 1

            def initialize(self):
                self.options.declare('ode_class', desc='System defining the ODE.')

        p = Problem(model=GaussLobattoPhase())

        p.model.add_objective('time', index=-1)

        p.model.linear_solver = ScipyKrylov(assemble_jac=True)

        p.setup(mode='fwd')
        p.set_solver_print(level=0)
        p.run_model()

        # Make sure we don't bomb out with an error.
        J = p.check_totals(out_stream=None)

        assert_rel_error(self, J[('time.time', 'time_extents.t_duration')]['J_fwd'][0], 17.0, 1e-5)
        assert_rel_error(self, J[('time.time', 'time_extents.t_duration')]['J_fd'][0], 17.0, 1e-5)

        # Try again with a direct solver and sparse assembled hierarchy.

        p = Problem()
        p.model.add_subsystem('sub', GaussLobattoPhase())

        p.model.sub.add_objective('time', index=-1)

        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.setup(mode='fwd')
        p.set_solver_print(level=0)
        p.run_model()

        # Make sure we don't bomb out with an error.
        J = p.check_totals(out_stream=None)

        assert_rel_error(self, J[('sub.time.time', 'sub.time_extents.t_duration')]['J_fwd'][0], 17.0, 1e-5)
        assert_rel_error(self, J[('sub.time.time', 'sub.time_extents.t_duration')]['J_fd'][0], 17.0, 1e-5)

    def test_vector_scaled_derivs(self):

        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('px', IndepVarComp(name="x", val=np.ones((2, ))))
        comp = model.add_subsystem('comp', DoubleArrayComp())
        model.connect('px.x', 'comp.x1')

        model.add_design_var('px.x', ref=np.array([2.0, 3.0]), ref0=np.array([0.5, 1.5]))
        model.add_objective('comp.y1', ref=np.array([[7.0, 11.0]]), ref0=np.array([5.2, 6.3]))
        model.add_constraint('comp.y2', lower=0.0, upper=1.0,
                             ref=np.array([[2.0, 4.0]]), ref0=np.array([1.2, 2.3]))

        prob.setup(check=False)
        prob.run_driver()

        # First, test that we get scaled results in compute and check totals.

        derivs = prob.compute_totals(of=['comp.y1'], wrt=['px.x'], return_format='dict',
                                     driver_scaling=True)

        oscale = np.array([1.0/(7.0-5.2), 1.0/(11.0-6.3)])
        iscale = np.array([2.0-0.5, 3.0-1.5])
        J = np.zeros((2, 2))
        J[:] = comp.JJ[0:2, 0:2]

        # doing this manually so that I don't inadvertantly make an error in
        # the vector math in both the code and test.
        J[0, 0] *= oscale[0]*iscale[0]
        J[0, 1] *= oscale[0]*iscale[1]
        J[1, 0] *= oscale[1]*iscale[0]
        J[1, 1] *= oscale[1]*iscale[1]
        assert_rel_error(self, J, derivs['comp.y1']['px.x'], 1.0e-3)

        cderiv = prob.check_totals(driver_scaling=True, out_stream=None)
        assert_rel_error(self, cderiv['comp.y1', 'px.x']['J_fwd'], J, 1.0e-3)

        # cleanup after FD
        prob.run_model()

        # Now, test that default is unscaled.

        derivs = prob.compute_totals(of=['comp.y1'], wrt=['px.x'], return_format='dict')

        J = comp.JJ[0:2, 0:2]
        assert_rel_error(self, J, derivs['comp.y1']['px.x'], 1.0e-3)

        cderiv = prob.check_totals(out_stream=None)
        assert_rel_error(self, cderiv['comp.y1', 'px.x']['J_fwd'], J, 1.0e-3)

    def test_cs_around_newton(self):
        # Basic sellar test.

        prob = Problem()
        model = prob.model
        sub = model.add_subsystem('sub', Group(), promotes=['*'])

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = NewtonSolver()
        sub.linear_solver = DirectSolver()

        # Need this.
        model.linear_solver = LinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(level=0)

        prob.run_model()

        totals = prob.check_totals(method='cs', out_stream=None)

        for key, val in iteritems(totals):
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-10)

    def test_cs_around_broyden(self):
        # Basic sellar test.

        prob = Problem()
        model = prob.model
        sub = model.add_subsystem('sub', Group(), promotes=['*'])

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = BroydenSolver()
        sub.linear_solver = DirectSolver()

        # Need this.
        model.linear_solver = LinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(level=0)

        prob.run_model()

        totals = prob.check_totals(method='cs', out_stream=None)

        for key, val in iteritems(totals):
            assert_rel_error(self, val['rel error'][0], 0.0, 1e-6)

    def test_cs_error_allocate(self):
        prob = Problem()
        model = prob.model
        model.add_subsystem('p', IndepVarComp('x', 3.0), promotes=['*'])
        model.add_subsystem('comp', ParaboloidTricky(), promotes=['*'])
        prob.setup(check=False)
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            prob.check_totals(method='cs')

        msg = "\nTo enable complex step, specify 'force_alloc_complex=True' when calling " + \
                "setup on the problem, e.g. 'problem.setup(force_alloc_complex=True)'"
        self.assertEqual(str(cm.exception), msg)


@unittest.skipUnless(MPI and PETScVector, "only run under MPI with PETSc.")
class TestProblemCheckTotalsMPI(unittest.TestCase):

    N_PROCS = 2

    def test_indepvarcomp_under_par_sys(self):

        prob = Problem()
        prob.model = FanInSubbedIDVC()

        prob.setup(check=False, mode='rev')
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.check_totals(out_stream=None)
        assert_rel_error(self, J['sum.y', 'sub.sub1.p1.x']['J_fwd'], [[2.0]], 1.0e-6)
        assert_rel_error(self, J['sum.y', 'sub.sub2.p2.x']['J_fwd'], [[4.0]], 1.0e-6)
        assert_rel_error(self, J['sum.y', 'sub.sub1.p1.x']['J_fd'], [[2.0]], 1.0e-6)
        assert_rel_error(self, J['sum.y', 'sub.sub2.p2.x']['J_fd'], [[4.0]], 1.0e-6)


if __name__ == "__main__":
    unittest.main()
