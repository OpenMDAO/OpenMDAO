""" Testing for Problem.check_partials and check_totals."""

from io import StringIO


import unittest

import numpy as np

import openmdao.api as om
from openmdao.core.tests.test_impl_comp import QuadraticLinearize, QuadraticJacVec
from openmdao.core.tests.test_matmat import MultiJacVec
from openmdao.test_suite.components.impl_comp_array import TestImplCompArrayMatVec
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec
from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDis1withDerivatives, \
     SellarDis2withDerivatives
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.test_suite.components.array_comp import ArrayComp
from openmdao.test_suite.groups.parallel_groups import FanInSubbedIDVC
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_check_partials
from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class ParaboloidTricky(om.ExplicitComponent):
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


class MyCompGoodPartials(om.ExplicitComponent):
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


class MyCompBadPartials(om.ExplicitComponent):
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


class MyComp(om.ExplicitComponent):
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

        prob = om.Problem()
        prob.model.add_subsystem('comp', MyComp())

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        stream = StringIO()
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

        prob = om.Problem()
        prob.model = MyComp()

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        stream = StringIO()
        prob.check_partials(out_stream=stream)
        lines = stream.getvalue().splitlines()

        y_wrt_x1_line = lines.index("  : 'y' wrt 'x1'")
        self.assertTrue(lines[y_wrt_x1_line+3].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        self.assertTrue(lines[y_wrt_x1_line+5].endswith('*'),
                        msg='Error flag expected in output but not displayed')

    def test_component_only_suppress(self):

        prob = om.Problem()
        prob.model = MyComp()

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        stream = StringIO()
        data = prob.check_partials(out_stream=None)

        subheads = data[''][('y', 'x1')]
        self.assertTrue('J_fwd' in subheads)
        self.assertTrue('rel error' in subheads)
        self.assertTrue('abs error' in subheads)
        self.assertTrue('magnitude' in subheads)

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 0)

    def test_component_has_no_outputs(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem("indep", om.IndepVarComp('x', 5.))
        model.add_subsystem("comp1", om.ExecComp("y=2*x"))

        comp2 = model.add_subsystem("comp2", om.ExplicitComponent())
        comp2.add_input('x', val=0.)

        model.connect('indep.x', ['comp1.x', 'comp2.x'])

        prob.setup()
        prob.run_model()

        # warning about 'comp2'
        msg = "No derivative data found for Component 'comp2'."

        with assert_warning(UserWarning, msg):
            data = prob.check_partials(out_stream=None)

        # and no derivative data for 'comp2'
        self.assertFalse('comp2' in data)

        # but we still get good derivative data for 'comp1'
        self.assertTrue('comp1' in data)

        assert_near_equal(data['comp1'][('y', 'x')]['J_fd'][0][0], 2., 1e-9)
        assert_near_equal(data['comp1'][('y', 'x')]['J_fwd'][0][0], 2., 1e-15)

    def test_missing_entry(self):
        class MyComp(om.ExplicitComponent):
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

        prob = om.Problem()
        prob.model.add_subsystem('p1', om.IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('comp', MyComp())

        prob.model.connect('p1.x1', 'comp.x1')
        prob.model.connect('p2.x2', 'comp.x2')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        self.assertEqual(prob.model.comp.lin_count, 1)

        abs_error = data['comp']['y', 'x1']['abs error']
        rel_error = data['comp']['y', 'x1']['rel error']
        self.assertAlmostEqual(abs_error.forward, 0.)
        self.assertAlmostEqual(rel_error.forward, 0.)
        self.assertAlmostEqual(np.linalg.norm(data['comp']['y', 'x1']['J_fd'] - 3.), 0.,
                               delta=1e-6)

        abs_error = data['comp']['y', 'x2']['abs error']
        rel_error = data['comp']['y', 'x2']['rel error']
        self.assertAlmostEqual(abs_error.forward, 4.)
        self.assertAlmostEqual(rel_error.forward, 1.)
        self.assertAlmostEqual(np.linalg.norm(data['comp']['y', 'x2']['J_fd'] - 4.), 0.,
                               delta=1e-6)

    def test_nested_fd_units(self):
        class UnitCompBase(om.ExplicitComponent):
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

        p = om.Problem()
        model = p.model
        indep = model.add_subsystem('indep', om.IndepVarComp(), promotes=['*'])

        indep.add_output('T', val=100., units='degK')
        indep.add_output('P', val=1., units='bar')

        model.add_subsystem('units', UnitCompBase(), promotes=['*'])

        p.setup()
        data = p.check_partials(out_stream=None)

        for comp_name, comp in data.items():
            for partial_name, partial in comp.items():
                forward = partial['J_fwd']
                fd = partial['J_fd']
                self.assertAlmostEqual(np.linalg.norm(forward - fd), 0., delta=1e-6)

    def test_units(self):
        class UnitCompBase(om.ExplicitComponent):
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

        p = om.Problem()
        model = p.model
        indep = model.add_subsystem('indep', om.IndepVarComp(), promotes=['*'])

        indep.add_output('T', val=100., units='degK')
        indep.add_output('P', val=1., units='bar')

        units = model.add_subsystem('units', UnitCompBase(), promotes=['*'])

        model.nonlinear_solver = om.NonlinearRunOnce()

        p.setup()
        data = p.check_partials(out_stream=None)

        for comp_name, comp in data.items():
            for partial_name, partial in comp.items():
                abs_error = partial['abs error']
                self.assertAlmostEqual(abs_error.forward, 0.)

        # Make sure we only FD this twice.
        # The count is 5 because in check_partials, there are two calls to apply_nonlinear
        # when compute the fwd and rev analytic derivatives, then one call to apply_nonlinear
        # to compute the reference point for FD, then two additional calls for the two inputs.
        self.assertEqual(units.run_count, 5)

    def test_scalar_val(self):
        class PassThrough(om.ExplicitComponent):
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

        p = om.Problem()

        indeps = p.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('foo', val=np.ones(4))
        indeps.add_output('foo2', val=np.ones(4))

        p.model.add_subsystem('pt', PassThrough("foo", "bar", val=np.ones(4)), promotes=['*'])
        p.model.add_subsystem('pt2', PassThrough("foo2", "bar2", val=np.ones(4)), promotes=['*'])

        p.set_solver_print(level=0)

        p.setup()
        p.run_model()

        data = p.check_partials(out_stream=None)
        identity = np.eye(4)
        assert_near_equal(data['pt'][('bar', 'foo')]['J_fwd'], identity, 1e-15)
        assert_near_equal(data['pt'][('bar', 'foo')]['J_fd'], identity, 1e-9)

        assert_near_equal(data['pt2'][('bar2', 'foo2')]['J_fwd'], identity, 1e-15)
        assert_near_equal(data['pt2'][('bar2', 'foo2')]['J_fd'], identity, 1e-9)

    def test_matrix_free_explicit(self):
        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidMatVec())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        for comp_name, comp in data.items():
            for partial_name, partial in comp.items():
                abs_error = partial['abs error']
                rel_error = partial['rel error']
                assert_near_equal(abs_error.forward, 0., 1e-5)
                assert_near_equal(abs_error.reverse, 0., 1e-5)
                assert_near_equal(abs_error.forward_reverse, 0., 1e-5)
                assert_near_equal(rel_error.forward, 0., 1e-5)
                assert_near_equal(rel_error.reverse, 0., 1e-5)
                assert_near_equal(rel_error.forward_reverse, 0., 1e-5)

        assert_near_equal(data['comp'][('f_xy', 'x')]['J_fwd'][0][0], 5.0, 1e-6)
        assert_near_equal(data['comp'][('f_xy', 'x')]['J_rev'][0][0], 5.0, 1e-6)
        assert_near_equal(data['comp'][('f_xy', 'y')]['J_fwd'][0][0], 21.0, 1e-6)
        assert_near_equal(data['comp'][('f_xy', 'y')]['J_rev'][0][0], 21.0, 1e-6)

    def test_matrix_free_implicit(self):
        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('rhs', np.ones((2, ))))
        prob.model.add_subsystem('comp', TestImplCompArrayMatVec())

        prob.model.connect('p1.rhs', 'comp.rhs')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        for comp_name, comp in data.items():
            for partial_name, partial in comp.items():
                abs_error = partial['abs error']
                rel_error = partial['rel error']
                assert_near_equal(abs_error.forward, 0., 1e-5)
                assert_near_equal(abs_error.reverse, 0., 1e-5)
                assert_near_equal(abs_error.forward_reverse, 0., 1e-5)
                assert_near_equal(rel_error.forward, 0., 1e-5)
                assert_near_equal(rel_error.reverse, 0., 1e-5)
                assert_near_equal(rel_error.forward_reverse, 0., 1e-5)

    def test_implicit_undeclared(self):
        # Test to see that check_partials works when state_wrt_input and state_wrt_state
        # partials are missing.

        class ImplComp4Test(om.ImplicitComponent):

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

        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', np.ones((2, ))))
        prob.model.add_subsystem('p2', om.IndepVarComp('dummy', np.ones((2, ))))
        prob.model.add_subsystem('comp', ImplComp4Test())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.dummy', 'comp.dummy')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        assert_near_equal(data['comp']['y', 'extra']['J_fwd'], np.zeros((2, 2)))
        assert_near_equal(data['comp']['y', 'dummy']['J_fwd'], np.zeros((2, 2)))

    def test_dependent_false_hide(self):
        # Test that we omit derivs declared with dependent=False

        class SimpleComp1(om.ExplicitComponent):
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

        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('z', np.ones((2, 2))))
        prob.model.add_subsystem('p2', om.IndepVarComp('x', np.ones((2, 2))))
        prob.model.add_subsystem('comp', SimpleComp1())
        prob.model.connect('p1.z', 'comp.z')
        prob.model.connect('p2.x', 'comp.x')

        prob.setup()

        stream = StringIO()
        data = prob.check_partials(out_stream=stream)
        lines = stream.getvalue().splitlines()

        self.assertTrue("  comp: 'g' wrt 'z'" not in lines)
        self.assertTrue(('g', 'z') not in data['comp'])
        self.assertTrue("  comp: 'g' wrt 'x'" in lines)
        self.assertTrue(('g', 'x') in data['comp'])

    def test_dependent_false_compact_print_never_hide(self):
        # API Change: we no longer omit derivatives for compact_print, even when declared as not
        # dependent.

        class SimpleComp1(om.ExplicitComponent):
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

        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('z', np.ones((2, 2))))
        prob.model.add_subsystem('p2', om.IndepVarComp('x', np.ones((2, 2))))
        prob.model.add_subsystem('comp', SimpleComp1())
        prob.model.connect('p1.z', 'comp.z')
        prob.model.connect('p2.x', 'comp.x')

        prob.setup()

        stream = StringIO()
        data = prob.check_partials(out_stream=stream, compact_print=True)
        txt = stream.getvalue()

        self.assertTrue("'g'        wrt 'z'" in txt)
        self.assertTrue(('g', 'z') in data['comp'])
        self.assertTrue("'g'        wrt 'x'" in txt)
        self.assertTrue(('g', 'x') in data['comp'])

    def test_dependent_false_show(self):
        # Test that we show derivs declared with dependent=False if the fd is not
        # ~zero.

        class SimpleComp2(om.ExplicitComponent):
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

        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('z', np.ones((2, 2))))
        prob.model.add_subsystem('p2', om.IndepVarComp('x', np.ones((2, 2))))
        prob.model.add_subsystem('comp', SimpleComp2())
        prob.model.connect('p1.z', 'comp.z')
        prob.model.connect('p2.x', 'comp.x')

        prob.setup()

        stream = StringIO()
        data = prob.check_partials(out_stream=stream)
        lines = stream.getvalue().splitlines()

        self.assertTrue("  comp: 'g' wrt 'z'" in lines)
        self.assertTrue(('g', 'z') in data['comp'])
        self.assertTrue("  comp: 'g' wrt 'x'" in lines)
        self.assertTrue(('g', 'x') in data['comp'])

    def test_set_step_on_comp(self):
        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.set_check_partial_options(wrt='*', step=1e-2)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None, compact_print=True)

        # This will fail unless you set the check_step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-5)

    def test_set_step_global(self):
        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None, step=1e-2)

        # This will fail unless you set the global step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-5)

    def test_complex_step_not_allocated(self):
        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidMatVec())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.set_check_partial_options(wrt='*', method='cs')

        prob.setup()
        prob.run_model()

        msg = "The following components requested complex step, but force_alloc_complex " + \
              "has not been set to True, so finite difference was used: ['comp']\n" + \
              "To enable complex step, specify 'force_alloc_complex=True' when calling " + \
              "setup on the problem, e.g. 'problem.setup(force_alloc_complex=True)'"

        with assert_warning(UserWarning, msg):
            data = prob.check_partials(out_stream=None)

        # Derivative still calculated, but with fd instead.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-5)
        self.assertLess(x_error.reverse, 1e-5)

    def test_set_method_on_comp(self):
        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.set_check_partial_options(wrt='*', method='cs')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        data = prob.check_partials(out_stream=None, compact_print=True)

        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-5)

    def test_set_method_global(self):
        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        data = prob.check_partials(out_stream=None, method='cs')

        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-5)

    def test_set_form_on_comp(self):
        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.set_check_partial_options(wrt='*', form='central')

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None, compact_print=True)

        # This will fail unless you set the check_step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-3)

    def test_set_form_global(self):
        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None, form='central')

        # This will fail unless you set the check_step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 1e-3)

    def test_set_step_calc_on_comp(self):
        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.set_check_partial_options(wrt='*', step_calc='rel')

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None, compact_print=True)

        # This will fail unless you set the check_step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 3e-3)

    def test_set_step_calc_global(self):
        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None, step_calc='rel')

        # This will fail unless you set the global step.
        x_error = data['comp']['f_xy', 'x']['rel error']
        self.assertLess(x_error.forward, 3e-3)

    def test_set_check_option_precedence(self):
        # Test that we omit derivs declared with dependent=False

        class SimpleComp1(om.ExplicitComponent):
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

        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('ab', 13.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('aba', 13.0))
        prob.model.add_subsystem('p3', om.IndepVarComp('ba', 13.0))
        comp = prob.model.add_subsystem('comp', SimpleComp1())

        prob.model.connect('p1.ab', 'comp.ab')
        prob.model.connect('p2.aba', 'comp.aba')
        prob.model.connect('p3.ba', 'comp.ba')

        prob.setup()

        comp.set_check_partial_options(wrt='a*', step=1e-2)
        comp.set_check_partial_options(wrt='*a', step=1e-4)

        prob.run_model()

        data = prob.check_partials(out_stream=None)

        # Note 'aba' gets the better value from the second options call with the *a wildcard.
        assert_near_equal(data['comp']['y', 'ab']['J_fd'][0][0], 507.3901, 1e-4)
        assert_near_equal(data['comp']['y', 'aba']['J_fd'][0][0], 507.0039, 1e-4)
        assert_near_equal(data['comp']['y', 'ba']['J_fd'][0][0], 507.0039, 1e-4)

    def test_option_printing(self):
        # Make sure we print the approximation type for each variable.
        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.set_check_partial_options(wrt='x', method='cs')
        comp.set_check_partial_options(wrt='y', form='central')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        stream = StringIO()
        prob.check_partials(out_stream=stream)

        lines = stream.getvalue().splitlines()
        self.assertTrue('cs' in lines[5],
                        msg='Did you change the format for printing check derivs?')
        self.assertTrue('fd' in lines[19],
                        msg='Did you change the format for printing check derivs?')

    def test_set_check_partial_options_invalid(self):
        import openmdao.api as om
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
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
                         "ParaboloidTricky (comp): The value of 'wrt' must be a string or list of strings, but a "
                         "type of 'ndarray' was provided.")

        # check invalid method
        with self.assertRaises(ValueError) as cm:
            comp.set_check_partial_options(wrt=['*'], method='foo')

        self.assertEqual(str(cm.exception),
                         "ParaboloidTricky (comp): Method 'foo' is not supported, method must be one of ('fd', 'cs')")

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
                         "ParaboloidTricky (comp): The value of 'step' must be numeric, but 'foo' was specified.")

        # check invalid step_calc
        with self.assertRaises(ValueError) as cm:
            comp.set_check_partial_options(wrt=['*'], step_calc='foo')

        self.assertEqual(str(cm.exception),
                         "ParaboloidTricky (comp): The value of 'step_calc' must be one of ('abs', 'rel'), "
                         "but 'foo' was specified.")

        # check invalid wrt
        comp._declared_partial_checks = []
        comp.set_check_partial_options(wrt=['x*', 'y', 'z', 'a*'])

        with self.assertRaises(ValueError) as cm:
            prob.check_partials()

        self.assertEqual(str(cm.exception), "ParaboloidTricky (comp): Invalid 'wrt' variables specified "
                         "for check_partial options: ['z'].")

        # check multiple invalid wrt
        comp._declared_partial_checks = []
        comp.set_check_partial_options(wrt=['a', 'b', 'c'])

        with self.assertRaises(ValueError) as cm:
            prob.check_partials()

        self.assertEqual(str(cm.exception), "ParaboloidTricky (comp): Invalid 'wrt' variables specified "
                         "for check_partial options: ['a', 'b', 'c'].")

    def test_compact_print_formatting(self):
        class MyCompShortVarNames(om.ExplicitComponent):
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

        class MyCompLongVarNames(om.ExplicitComponent):
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
        prob = om.Problem()
        prob.model.add_subsystem('p1', om.IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('comp', MyCompShortVarNames())
        prob.model.connect('p1.x1', 'comp.x1')
        prob.model.connect('p2.x2', 'comp.x2')
        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()
        stream = StringIO()
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
        prob = om.Problem()
        prob.model.add_subsystem('p1', om.IndepVarComp('really_long_variable_name_x1', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('comp', MyCompLongVarNames())
        prob.model.connect('p1.really_long_variable_name_x1', 'comp.really_long_variable_name_x1')
        prob.model.connect('p2.x2', 'comp.x2')
        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()
        stream = StringIO()
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

        prob = om.Problem()
        prob.model = MyCompGoodPartials()
        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()
        stream = StringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('>ABS_TOL'), 0)
        self.assertEqual(stream.getvalue().count('>REL_TOL'), 0)

        prob = om.Problem()
        prob.model = MyCompBadPartials()
        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()
        stream = StringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('>ABS_TOL'), 2)
        self.assertEqual(stream.getvalue().count('>REL_TOL'), 2)

    def test_check_partials_display_rev(self):

        # 1: Check display of revs for implicit comp for compact and non-compact display
        group = om.Group()
        comp1 = group.add_subsystem('comp1', om.IndepVarComp())
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
        prob = om.Problem(model=group)
        prob.setup()

        stream = StringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('n/a'), 25)
        self.assertEqual(stream.getvalue().count('rev'), 15)
        self.assertEqual(stream.getvalue().count('Component'), 2)
        self.assertEqual(stream.getvalue().count('wrt'), 12)

        stream = StringIO()
        prob.check_partials(out_stream=stream, compact_print=False)
        self.assertEqual(stream.getvalue().count('Reverse Magnitude'), 4)
        self.assertEqual(stream.getvalue().count('Raw Reverse Derivative'), 4)
        self.assertEqual(stream.getvalue().count('Jrev'), 20)

        # 2: Explicit comp, all comps define Jacobians for compact and non-compact display
        class MyComp(om.ExplicitComponent):
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

        prob = om.Problem()
        prob.model = MyComp()
        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()
        stream = StringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('rev'), 0)

        stream = StringIO()
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
        prob = om.Problem()
        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidMatVec())
        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')
        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()
        stream = StringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('rev'), 10)

        stream = StringIO()
        prob.check_partials(out_stream=stream, compact_print=False)
        self.assertEqual(stream.getvalue().count('Reverse'), 4)
        self.assertEqual(stream.getvalue().count('Jrev'), 10)

        # 4: Mixed comps. Some with jacobians. Some not
        prob = om.Problem()
        prob.model.add_subsystem('p0', om.IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p1', om.IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('c0', MyComp())  # in x1,x2, out is z
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
        prob.model.add_subsystem('comp', ParaboloidMatVec())
        prob.model.connect('p0.x1', 'c0.x1')
        prob.model.connect('p1.x2', 'c0.x2')
        prob.model.connect('c0.z', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')
        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()

        stream = StringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('n/a'), 10)
        self.assertEqual(stream.getvalue().count('rev'), 15)
        self.assertEqual(stream.getvalue().count('Component'), 2)
        self.assertEqual(stream.getvalue().count('wrt'), 8)

        stream = StringIO()
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
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('px', om.IndepVarComp('x', val=(np.arange(size, dtype=float) + 1.) * 3.0))
        model.add_subsystem('py', om.IndepVarComp('y', val=(np.arange(size, dtype=float) + 1.) * 2.0))
        model.add_subsystem('comp', MultiJacVec(size))

        model.connect('px.x', 'comp.x')
        model.connect('py.y', 'comp.y')

        model.add_design_var('px.x', vectorize_derivs=False)
        model.add_design_var('py.y', vectorize_derivs=False)
        model.add_constraint('comp.f_xy', vectorize_derivs=False)

        prob.setup()
        prob.run_model()
        stream = StringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('rev'), 10)

    def test_check_partials_worst_subjac(self):
        # The first is printing the worst subjac at the bottom of the output. Worst is defined by
        # looking at the fwd and rev columns of the relative error (i.e., the 2nd and 3rd last
        # columns) of the compact_print=True output. We should print the component name, then
        # repeat the full row for the worst-case subjac (i.e., output-input pair).
        # This should only occur in the compact_print=True case.

        prob = om.Problem()
        prob.model.add_subsystem('p0', om.IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p1', om.IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y2', 6.0))
        prob.model.add_subsystem('good', MyCompGoodPartials())
        prob.model.add_subsystem('bad', MyCompBadPartials())
        prob.model.connect('p0.x1', 'good.x1')
        prob.model.connect('p1.x2', 'good.x2')
        prob.model.connect('good.y', 'bad.y1')
        prob.model.connect('p2.y2', 'bad.y2')
        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()

        stream = StringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count("'z'        wrt 'y1'"), 2)

    def test_check_partials_show_only_incorrect(self):
        # The second is adding an option to show only the incorrect subjacs
        # (according to abs_err_tol and rel_err_tol), called
        # show_only_incorrect. This should be False by default, but when True,
        # it should print only the subjacs found to be incorrect. This applies
        # to both compact_print=True and False.

        prob = om.Problem()
        prob.model.add_subsystem('p0', om.IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p1', om.IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y2', 6.0))
        prob.model.add_subsystem('good', MyCompGoodPartials())
        prob.model.add_subsystem('bad', MyCompBadPartials())
        prob.model.connect('p0.x1', 'good.x1')
        prob.model.connect('p1.x2', 'good.x2')
        prob.model.connect('good.y', 'bad.y1')
        prob.model.connect('p2.y2', 'bad.y2')
        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()

        stream = StringIO()
        # prob.check_partials(compact_print=True,show_only_incorrect=False)
        prob.check_partials(out_stream=stream, compact_print=True, show_only_incorrect=True)
        self.assertEqual(stream.getvalue().count("MyCompBadPartials"), 2)
        self.assertEqual(stream.getvalue().count("'z'        wrt 'y1'"), 2)
        self.assertEqual(stream.getvalue().count("MyCompGoodPartials"), 0)

        stream = StringIO()
        prob.check_partials(compact_print=False, show_only_incorrect=False)
        prob.check_partials(out_stream=stream, compact_print=False, show_only_incorrect=True)
        self.assertEqual(stream.getvalue().count("MyCompGoodPartials"), 0)
        self.assertEqual(stream.getvalue().count("MyCompBadPartials"), 1)

    def test_includes_excludes(self):

        prob = om.Problem()
        model = prob.model

        sub = model.add_subsystem('c1c', om.Group())
        sub.add_subsystem('d1', om.ExecComp('y=2*x'))
        sub.add_subsystem('e1', om.ExecComp('y=2*x'))

        sub2 = model.add_subsystem('sss', om.Group())
        sub3 = sub2.add_subsystem('sss2', om.Group())
        sub2.add_subsystem('d1', om.ExecComp('y=2*x'))
        sub3.add_subsystem('e1', om.ExecComp('y=2*x'))

        model.add_subsystem('abc1cab', om.ExecComp('y=2*x'))

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

    def test_directional_derivative_option(self):

        prob = om.Problem()
        model = prob.model
        mycomp = model.add_subsystem('mycomp', ArrayComp(), promotes=['*'])

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None)

        # Note on why we run 10 times:
        # 1    - Initial execution
        # 2~3  - Called apply_nonlinear at the start of fwd and rev analytic deriv calculations
        # 4    - Called apply_nonlinear to clean up before starting FD
        # 5~8  - FD wrt bb, non-directional
        # 9    - FD wrt x1, directional
        # 10   - FD wrt x2, directional
        self.assertEqual(mycomp.exec_count, 10)

        assert_check_partials(data, atol=1.0E-8, rtol=1.0E-8)

        stream = StringIO()
        J = prob.check_partials(out_stream=stream, compact_print=True)
        output = stream.getvalue()
        self.assertTrue("(d)'x1'" in output)
        self.assertTrue("(d)'x2'" in output)

    def test_directional_derivative_option_complex_step(self):

        class ArrayCompCS(ArrayComp):
            def setup(self):
                super(ArrayCompCS, self).setup()
                self.set_check_partial_options('x*', directional=True, method='cs')

        prob = om.Problem()
        model = prob.model
        mycomp = model.add_subsystem('mycomp', ArrayCompCS(), promotes=['*'])

        np.random.seed(1)

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        data = prob.check_partials(method='cs', out_stream=None)

        # Note on why we run 10 times:
        # 1    - Initial execution
        # 2~3  - Called apply_nonlinear at the start of fwd and rev analytic deriv calculations
        # 4    - Called apply_nonlinear to clean up before starting FD
        # 5~8  - FD wrt bb, non-directional
        # 9    - FD wrt x1, directional
        # 10   - FD wrt x2, directional
        self.assertEqual(mycomp.exec_count, 10)

        assert_check_partials(data, atol=1.0E-8, rtol=1.0E-8)

    def test_directional_vectorized_matrix_free(self):

        class TestDirectional(om.ExplicitComponent):
            def initialize(self):
                self.options.declare('n',default=1, desc='vector size')

                self.n_compute = 0
                self.n_fwd = 0
                self.n_rev = 0

            def setup(self):
                self.add_input('in',shape=self.options['n'])
                self.add_output('out',shape=self.options['n'])

                self.set_check_partial_options(wrt='*', directional=True, method='cs')

            def compute(self,inputs,outputs):
                self.n_compute += 1
                fac = 2.0 + np.arange(self.options['n'])
                outputs['out'] = fac * inputs['in']

            def compute_jacvec_product(self,inputs,d_inputs,d_outputs, mode):
                fac = 2.0 + np.arange(self.options['n'])
                if mode == 'fwd':
                    if 'out' in d_outputs:
                        if 'in' in d_inputs:
                            d_outputs['out'] = fac * d_inputs['in']
                            self.n_fwd += 1

                if mode == 'rev':
                    if 'out' in d_outputs:
                        if 'in' in d_inputs:
                            d_inputs['in'] = fac * d_outputs['out']
                            self.n_rev += 1

        prob = om.Problem()
        model = prob.model

        np.random.seed(1)

        comp = TestDirectional(n=5)
        model.add_subsystem('comp', comp)

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        J = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(J)
        self.assertEqual(comp.n_fwd, 1)
        self.assertEqual(comp.n_rev, 1)

        # Compact print needs to print the dot-product test.
        stream = StringIO()
        J = prob.check_partials(method='cs', out_stream=stream, compact_print=True)
        lines = stream.getvalue().splitlines()

        self.assertEqual(lines[6][43:46], 'n/a')
        assert_near_equal(float(lines[6][95:105]), 0.0, 1e-15)

    def test_directional_mixed_matrix_free(self):

        class ArrayCompMatrixFree(om.ExplicitComponent):

            def setup(self):

                J1 = np.array([[1.0, 3.0, -2.0, 7.0],
                                [6.0, 2.5, 2.0, 4.0],
                                [-1.0, 0.0, 8.0, 1.0],
                                [1.0, 4.0, -5.0, 6.0]])

                self.J1 = J1
                self.J2 = J1 * 3.3
                self.Jb = J1.T

                # Inputs
                self.add_input('x1', np.zeros([4]))
                self.add_input('x2', np.zeros([4]))
                self.add_input('bb', np.zeros([4]))

                # Outputs
                self.add_output('y1', np.zeros([4]))

                self.declare_partials(of='*', wrt='*')
                self.set_check_partial_options('*', directional=True, method='fd')

            def compute(self, inputs, outputs):
                """
                Execution.
                """
                outputs['y1'] = self.J1.dot(inputs['x1']) + self.J2.dot(inputs['x2']) + self.Jb.dot(inputs['bb'])

            def compute_jacvec_product(self, inputs, dinputs, doutputs, mode):
                """Returns the product of the incoming vector with the Jacobian."""

                if mode == 'fwd':
                    if 'x1' in dinputs:
                        doutputs['y1'] += self.J1.dot(dinputs['x1'])
                    if 'x2' in dinputs:
                        doutputs['y1'] += self.J2.dot(dinputs['x2'])
                    if 'bb' in dinputs:
                        doutputs['y1'] += self.Jb.dot(dinputs['bb'])

                elif mode == 'rev':
                    if 'x1' in dinputs:
                        dinputs['x1'] += self.J1.T.dot(doutputs['y1'])
                    if 'x2' in dinputs:
                        dinputs['x2'] += self.J2.T.dot(doutputs['y1'])
                    if 'bb' in dinputs:
                        dinputs['bb'] += self.Jb.T.dot(doutputs['y1'])

        prob = om.Problem()
        model = prob.model
        mycomp = model.add_subsystem('mycomp', ArrayCompMatrixFree(), promotes=['*'])

        np.random.seed(1)

        prob.setup()
        prob.run_model()

        J = prob.check_partials(method='fd', out_stream=None)
        assert_check_partials(J)

    def test_directional_mixed_matrix_free_central_diff(self):

        class ArrayCompMatrixFree(om.ExplicitComponent):

            def setup(self):

                J1 = np.array([[1.0, 3.0, -2.0, 7.0],
                                [6.0, 2.5, 2.0, 4.0],
                                [-1.0, 0.0, 8.0, 1.0],
                                [1.0, 4.0, -5.0, 6.0]])

                self.J1 = J1
                self.J2 = J1 * 3.3
                self.Jb = J1.T

                # Inputs
                self.add_input('x1', np.zeros([4]))
                self.add_input('x2', np.zeros([4]))
                self.add_input('bb', np.zeros([4]))

                # Outputs
                self.add_output('y1', np.zeros([4]))

                self.declare_partials(of='*', wrt='*')
                self.set_check_partial_options('*', directional=True, method='fd', form='central')

            def compute(self, inputs, outputs):
                """
                Execution.
                """
                outputs['y1'] = self.J1.dot(inputs['x1']) + self.J2.dot(inputs['x2']) + self.Jb.dot(inputs['bb'])

            def compute_jacvec_product(self, inputs, dinputs, doutputs, mode):
                """Returns the product of the incoming vector with the Jacobian."""

                if mode == 'fwd':
                    if 'x1' in dinputs:
                        doutputs['y1'] += self.J1.dot(dinputs['x1'])
                    if 'x2' in dinputs:
                        doutputs['y1'] += self.J2.dot(dinputs['x2'])
                    if 'bb' in dinputs:
                        doutputs['y1'] += self.Jb.dot(dinputs['bb'])

                elif mode == 'rev':
                    if 'x1' in dinputs:
                        dinputs['x1'] += self.J1.T.dot(doutputs['y1'])
                    if 'x2' in dinputs:
                        dinputs['x2'] += self.J2.T.dot(doutputs['y1'])
                    if 'bb' in dinputs:
                        dinputs['bb'] += self.Jb.T.dot(doutputs['y1'])

        prob = om.Problem()
        model = prob.model
        mycomp = model.add_subsystem('mycomp', ArrayCompMatrixFree(), promotes=['*'])

        np.random.seed(1)

        prob.setup()
        prob.run_model()

        J = prob.check_partials(method='fd', out_stream=None)
        assert_check_partials(J)

    def test_directional_vectorized(self):

        class TestDirectional(om.ExplicitComponent):
            def initialize(self):
                self.options.declare('n',default=1, desc='vector size')

                self.n_compute = 0
                self.n_fwd = 0
                self.n_rev = 0

            def setup(self):
                self.add_input('in',shape=self.options['n'])
                self.add_output('out',shape=self.options['n'])

                self.declare_partials('out', 'in')
                self.set_check_partial_options(wrt='*', directional=True, method='cs')

            def compute(self,inputs,outputs):
                self.n_compute += 1
                fac = 2.0 + np.arange(self.options['n'])
                outputs['out'] = fac * inputs['in']

            def compute_partials(self, inputs, partials):
                partials['out', 'in'] = np.diag(2.0 + np.arange(self.options['n']))

        prob = om.Problem()
        model = prob.model

        np.random.seed(1)

        comp = TestDirectional(n=5)
        model.add_subsystem('comp', comp)

        prob.setup(force_alloc_complex=True)
        prob.run_model()
        J = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(J)

    def test_directional_mixed_error_message(self):
        import openmdao.api as om

        class ArrayCompMatrixFree(om.ExplicitComponent):

            def setup(self):

                J1 = np.array([[1.0, 3.0, -2.0, 7.0],
                                [6.0, 2.5, 2.0, 4.0],
                                [-1.0, 0.0, 8.0, 1.0],
                                [1.0, 4.0, -5.0, 6.0]])

                self.J1 = J1
                self.J2 = J1 * 3.3
                self.Jb = J1.T

                # Inputs
                self.add_input('x1', np.zeros([4]))
                self.add_input('x2', np.zeros([4]))
                self.add_input('bb', np.zeros([4]))

                # Outputs
                self.add_output('y1', np.zeros([4]))

                self.declare_partials(of='*', wrt='*')
                self.set_check_partial_options('x*', directional=True, method='fd')

            def compute(self, inputs, outputs):
                """
                Execution.
                """
                outputs['y1'] = self.J1.dot(inputs['x1']) + self.J2.dot(inputs['x2']) + self.Jb.dot(inputs['bb'])

            def compute_jacvec_product(self, inputs, dinputs, doutputs, mode):
                """Returns the product of the incoming vector with the Jacobian."""

                if mode == 'fwd':
                    if 'x1' in dinputs:
                        doutputs['y1'] += self.J1.dot(dinputs['x1'])
                    if 'x2' in dinputs:
                        doutputs['y1'] += self.J2.dot(dinputs['x2'])
                    if 'bb' in dinputs:
                        doutputs['y1'] += self.Jb.dot(dinputs['bb'])

                elif mode == 'rev':
                    if 'x1' in dinputs:
                        dinputs['x1'] += self.J1.T.dot(doutputs['y1'])
                    if 'x2' in dinputs:
                        dinputs['x2'] += self.J2.T.dot(doutputs['y1'])
                    if 'bb' in dinputs:
                        dinputs['bb'] += self.Jb.T.dot(doutputs['y1'])

        prob = om.Problem()
        model = prob.model
        mycomp = model.add_subsystem('mycomp', ArrayCompMatrixFree(), promotes=['*'])

        prob.setup()
        prob.run_model()

        with self.assertRaises(ValueError) as cm:
            J = prob.check_partials(method='fd', out_stream=None)

        msg = "ArrayCompMatrixFree (mycomp): For matrix free components, directional should be set to True for all inputs."
        self.assertEqual(str(cm.exception), msg)

    def test_directional_mimo(self):

        class DirectionalComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('n', default=1, desc='vector size')

            def setup(self):
                n = self.options['n']
                self.add_input('in', shape=n)
                self.add_input('in2', shape=n)
                self.add_output('out', shape=n)
                self.add_output('out2', shape=n)

                self.set_check_partial_options(wrt='*', directional=True, method='cs')
                self.mat = np.random.rand(n, n)
                self.mat2 = np.random.rand(n, n)

            def compute(self,inputs,outputs):
                outputs['out'] = self.mat.dot(inputs['in']) + self.mat2.dot(inputs['in2'])
                outputs['out2'] = 2.0 * self.mat.dot(inputs['in']) - self.mat2.dot(inputs['in2'])

            def compute_jacvec_product(self,inputs,d_inputs,d_outputs, mode):
                if mode == 'fwd':
                    if 'out' in d_outputs:
                        if 'in' in d_inputs:
                            d_outputs['out'] += self.mat.dot(d_inputs['in'])
                        if 'in2' in d_inputs:
                            d_outputs['out'] += self.mat2.dot(d_inputs['in2'])
                    if 'out2' in d_outputs:
                        if 'in' in d_inputs:
                            d_outputs['out2'] += 2.0 * self.mat.dot(d_inputs['in'])
                        if 'in2' in d_inputs:
                            d_outputs['out2'] += -1.0 * self.mat2.dot(d_inputs['in2'])

                if mode == 'rev':
                    if 'out' in d_outputs:
                        if 'in' in d_inputs:
                            d_inputs['in'] += self.mat.transpose().dot(d_outputs['out'])
                        if 'in2' in d_inputs:
                            d_inputs['in2'] += self.mat2.transpose().dot(d_outputs['out'])
                    if 'out2' in d_outputs:
                        if 'in' in d_inputs:
                            # This one is wrong in reverse.
                            d_inputs['in'] += 999.0 * self.mat.transpose().dot(d_outputs['out2'])
                        if 'in2' in d_inputs:
                            d_inputs['in2'] += -1.0 * self.mat2.transpose().dot(d_outputs['out2'])

        prob = om.Problem()
        comp = DirectionalComp(n=2)
        prob.model.add_subsystem('comp', comp)

        prob.setup(force_alloc_complex=True)
        prob.run_model()
        partials = prob.check_partials(method='cs', out_stream=None)

        self.assertGreater(np.abs(partials['comp']['out2', 'in']['directional_fwd_rev']),
                           1e-3, msg='Reverse deriv is supposed to be wrong.')
        assert_near_equal(np.abs(partials['comp']['out', 'in']['directional_fwd_rev']),
                          0.0, 1e-12)
        assert_near_equal(np.abs(partials['comp']['out', 'in2']['directional_fwd_rev']),
                          0.0, 1e-12)
        assert_near_equal(np.abs(partials['comp']['out2', 'in2']['directional_fwd_rev']),
                          0.0, 1e-12)

    def test_bug_local_method(self):
        # This fixes a bug setting the check method on a component overrode the requested method for
        # subsequent components.
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp1', Paraboloid())
        fdcomp = model.add_subsystem('comp2', Paraboloid())
        model.add_subsystem('comp3', Paraboloid())

        fdcomp.set_check_partial_options(wrt='*', method='fd')

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(level=0)
        prob.run_model()

        data = prob.check_partials(method='cs', out_stream=None)

        # Comp1 and Comp3 are complex step, so have tighter tolerances.
        for key, val in data['comp1'].items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-15)
        for key, val in data['comp2'].items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-6)
        for key, val in data['comp3'].items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-15)

    def test_rel_error_fd_zero(self):
        # When the fd turns out to be zero, test that we switch the definition of relative
        # to divide by the forward derivative instead of reporting NaN.

        class SimpleComp2(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', val=3.0)
                self.add_output('y', val=4.0)

                self.declare_partials(of='y', wrt='x')

            def compute(self, inputs, outputs):
                # Mimics forgetting to set a variable.
                pass

            def compute_partials(self, inputs, partials):
                partials['y', 'x'] = 3.0

        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.5))
        prob.model.add_subsystem('comp', SimpleComp2())
        prob.model.connect('p1.x', 'comp.x')

        prob.setup()

        stream = StringIO()
        data = prob.check_partials(out_stream=stream)
        lines = stream.getvalue().splitlines()

        self.assertTrue("Relative Error (Jfor  - Jfd) : 1." in lines[8])


class TestCheckPartialsFeature(unittest.TestCase):

    def test_feature_incorrect_jacobian(self):
        import numpy as np

        import openmdao.api as om

        class MyComp(om.ExplicitComponent):
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

        prob = om.Problem()

        prob.model.add_subsystem('comp', MyComp())

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        data = prob.check_partials()

        x1_error = data['comp']['y', 'x1']['abs error']

        assert_near_equal(x1_error.forward, 1., 1e-8)

        x2_error = data['comp']['y', 'x2']['rel error']

        assert_near_equal(x2_error.forward, 9., 1e-8)

    def test_feature_check_partials_suppress(self):
        import numpy as np

        import openmdao.api as om

        class MyComp(om.ExplicitComponent):
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

        prob = om.Problem()

        prob.model.add_subsystem('comp', MyComp())

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(out_stream=None, compact_print=True)
        print(data)

    def test_set_step_on_comp(self):
        import openmdao.api as om
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

        prob = om.Problem()

        comp = prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        comp.set_check_partial_options(wrt='*', step=1e-2)

        prob.setup()
        prob.run_model()

        prob.check_partials(compact_print=True)

    def test_set_step_global(self):
        import openmdao.api as om
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

        prob = om.Problem()

        prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        prob.check_partials(step=1e-2, compact_print=True)

    def test_set_method_on_comp(self):
        import openmdao.api as om
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

        prob = om.Problem()

        comp = prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        comp.set_check_partial_options(wrt='*', method='cs')

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        prob.check_partials(compact_print=True)

    def test_set_method_global(self):
        import openmdao.api as om
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

        prob = om.Problem()

        prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        prob.check_partials(method='cs', compact_print=True)

    def test_set_form_global(self):
        import openmdao.api as om
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky
        from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec

        prob = om.Problem()

        prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        prob.check_partials(form='central', compact_print=True)

    def test_set_step_calc_global(self):
        import openmdao.api as om
        from openmdao.core.tests.test_check_derivs import ParaboloidTricky

        prob = om.Problem()

        prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        prob.check_partials(step_calc='rel', compact_print=True)

    def test_feature_check_partials_show_only_incorrect(self):
        import numpy as np
        import openmdao.api as om

        class MyCompGoodPartials(om.ExplicitComponent):
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

        class MyCompBadPartials(om.ExplicitComponent):
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

        prob = om.Problem()
        prob.model.add_subsystem('good', MyCompGoodPartials())
        prob.model.add_subsystem('bad', MyCompBadPartials())
        prob.model.connect('good.y', 'bad.y1')

        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()

        prob.check_partials(compact_print=True, show_only_incorrect=True)
        prob.check_partials(compact_print=False, show_only_incorrect=True)

    def test_includes_excludes(self):
        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        sub = model.add_subsystem('c1c', om.Group())
        sub.add_subsystem('d1', om.ExecComp('y=2*x'))
        sub.add_subsystem('e1', om.ExecComp('y=2*x'))

        sub2 = model.add_subsystem('sss', om.Group())
        sub3 = sub2.add_subsystem('sss2', om.Group())
        sub2.add_subsystem('d1', om.ExecComp('y=2*x'))
        sub3.add_subsystem('e1', om.ExecComp('y=2*x'))

        model.add_subsystem('abc1cab', om.ExecComp('y=2*x'))

        prob.setup()
        prob.run_model()

        prob.check_partials(compact_print=True, includes='*c*c*')

        prob.check_partials(compact_print=True, includes=['*d1', '*e1'])

        prob.check_partials(compact_print=True, includes=['abc1cab'])

        prob.check_partials(compact_print=True, includes='*c*c*', excludes=['*e*'])

    def test_directional(self):
        import openmdao.api as om
        from openmdao.test_suite.components.array_comp import ArrayComp

        prob = om.Problem()
        model = prob.model
        mycomp = model.add_subsystem('mycomp', ArrayComp(), promotes=['*'])

        prob.setup()
        prob.run_model()

        data = prob.check_partials()

    def test_directional_matrix_free(self):
        import numpy as np

        import openmdao.api as om

        class ArrayCompMatrixFree(om.ExplicitComponent):

            def setup(self):

                J1 = np.array([[1.0, 3.0, -2.0, 7.0],
                                [6.0, 2.5, 2.0, 4.0],
                                [-1.0, 0.0, 8.0, 1.0],
                                [1.0, 4.0, -5.0, 6.0]])

                self.J1 = J1
                self.J2 = J1 * 3.3
                self.Jb = J1.T

                # Inputs
                self.add_input('x1', np.zeros([4]))
                self.add_input('x2', np.zeros([4]))
                self.add_input('bb', np.zeros([4]))

                # Outputs
                self.add_output('y1', np.zeros([4]))

                self.declare_partials(of='*', wrt='*')
                self.set_check_partial_options('*', directional=True)

            def compute(self, inputs, outputs):
                """
                Execution.
                """
                outputs['y1'] = self.J1.dot(inputs['x1']) + self.J2.dot(inputs['x2']) + self.Jb.dot(inputs['bb'])

            def compute_jacvec_product(self, inputs, dinputs, doutputs, mode):
                """Returns the product of the incoming vector with the Jacobian."""

                if mode == 'fwd':
                    if 'x1' in dinputs:
                        doutputs['y1'] += self.J1.dot(dinputs['x1'])
                    if 'x2' in dinputs:
                        doutputs['y1'] += self.J2.dot(dinputs['x2'])
                    if 'bb' in dinputs:
                        doutputs['y1'] += self.Jb.dot(dinputs['bb'])

                elif mode == 'rev':
                    if 'x1' in dinputs:
                        dinputs['x1'] += self.J1.T.dot(doutputs['y1'])
                    if 'x2' in dinputs:
                        dinputs['x2'] += self.J2.T.dot(doutputs['y1'])
                    if 'bb' in dinputs:
                        dinputs['bb'] += self.Jb.T.dot(doutputs['y1'])

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('mycomp', ArrayCompMatrixFree(), promotes=['*'])

        prob.setup()
        prob.run_model()

        data = prob.check_partials()

    def test_set_method_and_step_bug(self):
        # If a model-builder set his a component to fd, and the global method is cs with a specified
        # step size, that size is probably unusable, and can lead to false error in the check.

        prob = om.Problem()

        prob.model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        prob.model.add_subsystem('p2', om.IndepVarComp('y', 5.0))
        comp = prob.model.add_subsystem('comp', Paraboloid())

        prob.model.connect('p1.x', 'comp.x')
        prob.model.connect('p2.y', 'comp.y')

        prob.set_solver_print(level=0)

        comp.set_check_partial_options(wrt='*', method='fd')

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        J = prob.check_partials(compact_print=True, method='cs', step=1e-40, out_stream=None)

        assert_check_partials(J, atol=1e-5, rtol=1e-5)


class TestProblemCheckTotals(unittest.TestCase):

    def test_cs(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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
        stream = StringIO()
        totals = prob.check_totals(method='cs', out_stream=stream)

        lines = stream.getvalue().splitlines()

        self.assertTrue('9.80614' in lines[4], "'9.80614' not found in '%s'" % lines[4])
        self.assertTrue('9.80614' in lines[5], "'9.80614' not found in '%s'" % lines[5])
        self.assertTrue('cs:None' in lines[5], "'cs:None not found in '%s'" % lines[5])

        assert_near_equal(totals['con_cmp2.con2', 'x']['J_fwd'], [[0.09692762]], 1e-5)
        assert_near_equal(totals['con_cmp2.con2', 'x']['J_fd'], [[0.09692762]], 1e-5)

        # Test compact_print output
        compact_stream = StringIO()
        compact_totals = prob.check_totals(method='fd', out_stream=compact_stream,
            compact_print=True)

        compact_lines = compact_stream.getvalue().splitlines()

        self.assertTrue('<output>' in compact_lines[3],
            "'<output>' not found in '%s'" % compact_lines[4])
        self.assertTrue('9.7743e+00' in compact_lines[11],
            "'9.7743e+00' not found in '%s'" % compact_lines[11])

    def test_desvar_as_obj(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_objective('x')

        prob.set_solver_print(level=0)

        prob.setup(force_alloc_complex=True)

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives with complex step and a larger step size.
        stream = StringIO()
        totals = prob.check_totals(method='cs', out_stream=stream)

        lines = stream.getvalue().splitlines()

        self.assertTrue('1.000' in lines[4])
        self.assertTrue('1.000' in lines[5])
        self.assertTrue('0.000' in lines[6])
        self.assertTrue('0.000' in lines[8])

        assert_near_equal(totals['x', 'x']['J_fwd'], [[1.0]], 1e-5)
        assert_near_equal(totals['x', 'x']['J_fd'], [[1.0]], 1e-5)

    def test_desvar_and_response_with_indices(self):

        class ArrayComp2D(om.ExplicitComponent):
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

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('x_param1', om.IndepVarComp('x1', np.ones((4))),
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
        assert_near_equal(J['y1', 'x1'][0][0], Jbase[0, 1], 1e-8)
        assert_near_equal(J['y1', 'x1'][0][1], Jbase[0, 3], 1e-8)
        assert_near_equal(J['y1', 'x1'][1][0], Jbase[2, 1], 1e-8)
        assert_near_equal(J['y1', 'x1'][1][1], Jbase[2, 3], 1e-8)

        totals = prob.check_totals()
        jac = totals[('mycomp.y1', 'x_param1.x1')]['J_fd']
        assert_near_equal(jac[0][0], Jbase[0, 1], 1e-8)
        assert_near_equal(jac[0][1], Jbase[0, 3], 1e-8)
        assert_near_equal(jac[1][0], Jbase[2, 1], 1e-8)
        assert_near_equal(jac[1][1], Jbase[2, 3], 1e-8)

        # Objective instead

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('x_param1', om.IndepVarComp('x1', np.ones((4))),
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
        assert_near_equal(J['y1', 'x1'][0][0], Jbase[1, 1], 1e-8)
        assert_near_equal(J['y1', 'x1'][0][1], Jbase[1, 3], 1e-8)

        totals = prob.check_totals()
        jac = totals[('mycomp.y1', 'x_param1.x1')]['J_fd']
        assert_near_equal(jac[0][0], Jbase[1, 1], 1e-8)
        assert_near_equal(jac[0][1], Jbase[1, 3], 1e-8)

    def test_cs_suppress(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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

        data = totals['con_cmp2.con2', 'x']
        self.assertTrue('J_fwd' in data)
        self.assertTrue('rel error' in data)
        self.assertTrue('abs error' in data)
        self.assertTrue('magnitude' in data)

    def test_two_desvar_as_con(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_constraint('x', upper=0.0)
        prob.model.add_constraint('z', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        totals = prob.check_totals(method='fd', step=1.0e-1, out_stream=None)

        assert_near_equal(totals['x', 'x']['J_fwd'], [[1.0]], 1e-5)
        assert_near_equal(totals['x', 'x']['J_fd'], [[1.0]], 1e-5)
        assert_near_equal(totals['z', 'z']['J_fwd'], np.eye(2), 1e-5)
        assert_near_equal(totals['z', 'z']['J_fd'], np.eye(2), 1e-5)
        assert_near_equal(totals['x', 'z']['J_fwd'], [[0.0, 0.0]], 1e-5)
        assert_near_equal(totals['x', 'z']['J_fd'], [[0.0, 0.0]], 1e-5)
        assert_near_equal(totals['z', 'x']['J_fwd'], [[0.0], [0.0]], 1e-5)
        assert_near_equal(totals['z', 'x']['J_fd'], [[0.0], [0.0]], 1e-5)

    def test_full_con_with_index_desvar(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('z', lower=-100, upper=100, indices=[1])
        prob.model.add_constraint('z', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        totals = prob.check_totals(method='fd', step=1.0e-1, out_stream=None)

        assert_near_equal(totals['z', 'z']['J_fwd'], [[0.0], [1.0]], 1e-5)
        assert_near_equal(totals['z', 'z']['J_fd'], [[0.0], [1.0]], 1e-5)

    def test_full_desvar_with_index_con(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_constraint('z', upper=0.0, indices=[1])

        prob.set_solver_print(level=0)

        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        totals = prob.check_totals(method='fd', step=1.0e-1, out_stream=None)

        assert_near_equal(totals['z', 'z']['J_fwd'], [[0.0, 1.0]], 1e-5)
        assert_near_equal(totals['z', 'z']['J_fd'], [[0.0, 1.0]], 1e-5)

    def test_full_desvar_with_index_obj(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('z', index=1)

        prob.set_solver_print(level=0)

        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        totals = prob.check_totals(method='fd', step=1.0e-1, out_stream=None)

        assert_near_equal(totals['z', 'z']['J_fwd'], [[0.0, 1.0]], 1e-5)
        assert_near_equal(totals['z', 'z']['J_fd'], [[0.0, 1.0]], 1e-5)

    def test_bug_fd_with_sparse(self):
        # This bug was found via the x57 model in pointer.

        class TimeComp(om.ExplicitComponent):

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

        class CellComp(om.ExplicitComponent):

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

        class GaussLobattoPhase(om.Group):

            def setup(self):
                self.connect('t_duration', 'time.t_duration')

                indep = om.IndepVarComp()
                indep.add_output('t_duration', val=1.0)
                self.add_subsystem('time_extents', indep, promotes_outputs=['*'])
                self.add_design_var('t_duration', 5.0, 25.0)

                time_comp = TimeComp()
                self.add_subsystem('time', time_comp, promotes_outputs=['time'])

                self.add_subsystem(name='cell', subsys=CellComp(num_nodes=3))

                self.linear_solver = om.ScipyKrylov()
                self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
                self.nonlinear_solver.options['maxiter'] = 1

            def initialize(self):
                self.options.declare('ode_class', desc='System defining the ODE.')

        p = om.Problem(model=GaussLobattoPhase())

        p.model.add_objective('time', index=-1)

        p.model.linear_solver = om.ScipyKrylov(assemble_jac=True)

        p.setup(mode='fwd')
        p.set_solver_print(level=0)
        p.run_model()

        # Make sure we don't bomb out with an error.
        J = p.check_totals(out_stream=None)

        assert_near_equal(J[('time.time', 'time_extents.t_duration')]['J_fwd'][0], 17.0, 1e-5)
        assert_near_equal(J[('time.time', 'time_extents.t_duration')]['J_fd'][0], 17.0, 1e-5)

        # Try again with a direct solver and sparse assembled hierarchy.

        p = om.Problem()
        p.model.add_subsystem('sub', GaussLobattoPhase())

        p.model.sub.add_objective('time', index=-1)

        p.model.linear_solver = om.DirectSolver(assemble_jac=True)

        p.setup(mode='fwd')
        p.set_solver_print(level=0)
        p.run_model()

        # Make sure we don't bomb out with an error.
        J = p.check_totals(out_stream=None)

        assert_near_equal(J[('sub.time.time', 'sub.time_extents.t_duration')]['J_fwd'][0], 17.0, 1e-5)
        assert_near_equal(J[('sub.time.time', 'sub.time_extents.t_duration')]['J_fd'][0], 17.0, 1e-5)

        # Make sure check_totals cleans up after itself by running it a second time.
        J = p.check_totals(out_stream=None)

        assert_near_equal(J[('sub.time.time', 'sub.time_extents.t_duration')]['J_fwd'][0], 17.0, 1e-5)
        assert_near_equal(J[('sub.time.time', 'sub.time_extents.t_duration')]['J_fd'][0], 17.0, 1e-5)

    def test_vector_scaled_derivs(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp(name="x", val=np.ones((2, ))))
        comp = model.add_subsystem('comp', DoubleArrayComp())
        model.connect('px.x', 'comp.x1')

        model.add_design_var('px.x', ref=np.array([2.0, 3.0]), ref0=np.array([0.5, 1.5]))
        model.add_objective('comp.y1', ref=np.array([[7.0, 11.0]]), ref0=np.array([5.2, 6.3]))
        model.add_constraint('comp.y2', lower=0.0, upper=1.0,
                             ref=np.array([[2.0, 4.0]]), ref0=np.array([1.2, 2.3]))

        prob.setup()
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
        assert_near_equal(J, derivs['comp.y1']['px.x'], 1.0e-3)

        cderiv = prob.check_totals(driver_scaling=True, out_stream=None)
        assert_near_equal(cderiv['comp.y1', 'px.x']['J_fwd'], J, 1.0e-3)

        # cleanup after FD
        prob.run_model()

        # Now, test that default is unscaled.

        derivs = prob.compute_totals(of=['comp.y1'], wrt=['px.x'], return_format='dict')

        J = comp.JJ[0:2, 0:2]
        assert_near_equal(J, derivs['comp.y1']['px.x'], 1.0e-3)

        cderiv = prob.check_totals(out_stream=None)
        assert_near_equal(cderiv['comp.y1', 'px.x']['J_fwd'], J, 1.0e-3)

    def test_cs_around_newton(self):
        # Basic sellar test.

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        sub.linear_solver = om.DirectSolver(assemble_jac=False)

        # Need this.
        model.linear_solver = om.LinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(level=0)

        prob.run_model()

        totals = prob.check_totals(method='cs', out_stream=None)

        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-10)

    def test_cs_around_broyden(self):
        # Basic sellar test.

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.BroydenSolver()
        sub.linear_solver = om.DirectSolver()

        # Need this.
        model.linear_solver = om.LinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(level=0)

        prob.run_model()

        totals = prob.check_totals(method='cs', out_stream=None)

        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-6)

    def test_cs_error_allocate(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p', om.IndepVarComp('x', 3.0), promotes=['*'])
        model.add_subsystem('comp', ParaboloidTricky(), promotes=['*'])
        prob.setup()
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            prob.check_totals(method='cs')

        msg = "\nProblem: To enable complex step, specify 'force_alloc_complex=True' when calling " + \
                "setup on the problem, e.g. 'problem.setup(force_alloc_complex=True)'"
        self.assertEqual(str(cm.exception), msg)

    def test_fd_zero_check(self):

        class BadComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', 3.0)
                self.add_output('y', 3.0)

                self.declare_partials('y', 'x')

            def compute(self, inputs, outputs):
                pass

            def compute_partials(self, inputs, partials):
                partials['y', 'x'] = 3.0 * inputs['x'] + 5

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', om.IndepVarComp('x', 3.0))
        model.add_subsystem('comp', BadComp())
        model.connect('p.x', 'comp.x')

        model.add_design_var('p.x')
        model.add_objective('comp.y')

        prob.setup()
        prob.run_model()

        # This test verifies fix of a TypeError (division by None)
        J = prob.check_totals(out_stream=None)
        assert_near_equal(J['comp.y', 'p.x']['J_fwd'], [[14.0]], 1e-6)
        assert_near_equal(J['comp.y', 'p.x']['J_fd'], [[0.0]], 1e-6)

    def test_response_index(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', om.IndepVarComp('x', np.ones(2)), promotes=['*'])
        model.add_subsystem('comp', om.ExecComp('y=2*x', x=np.ones(2), y=np.ones(2)),
                            promotes=['*'])

        model.add_design_var('x')
        model.add_constraint('y', indices=[1], lower=0.0)

        prob.setup()
        prob.run_model()

        stream = StringIO()
        prob.check_totals(out_stream=stream)
        lines = stream.getvalue().splitlines()
        self.assertTrue('index size: 1' in lines[3])


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestProblemCheckTotalsMPI(unittest.TestCase):

    N_PROCS = 2

    def test_indepvarcomp_under_par_sys(self):

        prob = om.Problem()
        prob.model = FanInSubbedIDVC()

        prob.setup(check=False, mode='rev')
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.check_totals(out_stream=None)
        assert_near_equal(J['sum.y', 'sub.sub1.p1.x']['J_fwd'], [[2.0]], 1.0e-6)
        assert_near_equal(J['sum.y', 'sub.sub2.p2.x']['J_fwd'], [[4.0]], 1.0e-6)
        assert_near_equal(J['sum.y', 'sub.sub1.p1.x']['J_fd'], [[2.0]], 1.0e-6)
        assert_near_equal(J['sum.y', 'sub.sub2.p2.x']['J_fd'], [[4.0]], 1.0e-6)


if __name__ == "__main__":
    unittest.main()
