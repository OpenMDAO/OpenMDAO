""" Testing for Problem.check_partials and check_totals."""

from io import StringIO


import unittest

import numpy as np

import openmdao.api as om
from openmdao.approximation_schemes.finite_difference import FiniteDifference
from openmdao.approximation_schemes.complex_step import ComplexStep
from openmdao.core.tests.test_impl_comp import QuadraticLinearize, QuadraticJacVec
from openmdao.test_suite.components.impl_comp_array import TestImplCompArrayMatVec
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.paraboloid_mat_vec import ParaboloidMatVec
from openmdao.test_suite.components.array_comp import ArrayComp
from openmdao.test_suite.scripts.circle_opt import CircleOpt
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_no_warning, \
     assert_check_partials
from openmdao.utils.om_warnings import DerivativesWarning, OMInvalidCheckDerivativesOptionsWarning
from openmdao.utils.testing_utils import set_env_vars_context

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

class DirectionalVectorizedComp(om.ExplicitComponent):
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


class DirectionalVectorizedMatFreeComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n', default=1, desc='vector size')

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

        self.assertTrue(lines[y_wrt_x1_line+4].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        self.assertTrue(lines[y_wrt_x1_line+6].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        y_wrt_x2_line = lines.index("  comp: 'y' wrt 'x2'")
        self.assertTrue(lines[y_wrt_x2_line+4].endswith('*'),
                        msg='Error flag not expected in output but displayed')
        self.assertTrue(lines[y_wrt_x2_line+6].endswith('*'),
                        msg='Error flag not expected in output but displayed')

    def test_component_has_no_outputs(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem("indep", om.IndepVarComp('x', 5.))
        model.add_subsystem("comp1", Paraboloid())

        comp2 = model.add_subsystem("comp2", om.ExplicitComponent())
        comp2.add_input('x', val=0.)

        model.connect('indep.x', ['comp1.x', 'comp2.x'])

        prob.setup()
        prob.run_model()

        # no warnings about 'comp2' having no derivatives
        with assert_no_warning(DerivativesWarning):
            data = prob.check_partials(out_stream=None)

        # and no derivative data for 'comp2'
        self.assertFalse('comp2' in data)

        # but we still get good derivative data for 'comp1'
        self.assertTrue('comp1' in data)

        assert_near_equal(data['comp1'][('f_xy', 'x')]['J_fd'][0][0], 4., 1e-6)
        assert_near_equal(data['comp1'][('f_xy', 'x')]['J_fwd'][0][0], 4., 1e-15)

    def test_component_has_no_inputs(self):
        prob = om.Problem()
        model = prob.model

        comp1 = model.add_subsystem("comp1", om.ExplicitComponent())
        comp1.add_output('x', val=5.)

        model.add_subsystem("comp2", Paraboloid())

        model.connect('comp1.x', 'comp2.x')

        prob.setup()
        prob.run_model()

        # no warnings about 'comp1' having no derivatives
        with assert_no_warning(DerivativesWarning):
            data = prob.check_partials(out_stream=None)

        # and no derivative data for 'comp1'
        self.assertFalse('comp1' in data)

        # but we still get good derivative data for 'comp1'
        self.assertTrue('comp2' in data)

        assert_near_equal(data['comp2'][('f_xy', 'x')]['J_fd'][0][0], 4., 1e-6)
        assert_near_equal(data['comp2'][('f_xy', 'x')]['J_fwd'][0][0], 4., 1e-15)

    def test_component_no_check_partials(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem("indep", om.IndepVarComp('x', 5.))
        model.add_subsystem("comp1", Paraboloid())

        comp2 = model.add_subsystem("comp2", Paraboloid())

        model.connect('indep.x', ['comp1.x', 'comp2.x'])

        prob.setup()
        prob.run_model()

        #
        # disable partials on comp2
        #
        comp2._no_check_partials = True

        # Make check_partials think we're not on CI so we'll get the expected  non-CI behavior
        with set_env_vars_context(OPENMDAO_CHECK_ALL_PARTIALS='0'):
            data = prob.check_partials(out_stream=None)

        # no derivative data for 'comp2'
        self.assertFalse('comp2' in data)

        # but we still get good derivative data for 'comp1'
        self.assertTrue('comp1' in data)

        assert_near_equal(data['comp1'][('f_xy', 'x')]['J_fd'][0][0], 4., 1e-6)
        assert_near_equal(data['comp1'][('f_xy', 'x')]['J_fwd'][0][0], 4., 1e-15)

        #
        # re-enable partials on comp2
        #
        comp2._no_check_partials = False
        data = prob.check_partials(out_stream=None)

        # now we should have derivative data for 'comp2'
        self.assertTrue('comp2' in data)

        assert_near_equal(data['comp2'][('f_xy', 'x')]['J_fd'][0][0], 4., 1e-6)
        assert_near_equal(data['comp2'][('f_xy', 'x')]['J_fwd'][0][0], 4., 1e-15)

        # and still get good derivative data for 'comp1'
        self.assertTrue('comp1' in data)

        assert_near_equal(data['comp1'][('f_xy', 'x')]['J_fd'][0][0], 4., 1e-6)
        assert_near_equal(data['comp1'][('f_xy', 'x')]['J_fwd'][0][0], 4., 1e-15)

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
        data = p.check_partials(step=1e-7, out_stream=None)

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
                super().__init__()
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

        self.assertTrue("'g'             | 'z'" in txt)
        self.assertTrue(('g', 'z') in data['comp'])
        self.assertTrue("'g'             | 'x'" in txt)
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
        self.assertTrue('cs' in lines[6],
                        msg='Did you change the format for printing check derivs?')
        self.assertTrue('fd' in lines[21],
                        msg='Did you change the format for printing check derivs?')

    def test_set_check_partial_options_invalid(self):
        from openmdao.core.tests.test_check_partials import ParaboloidTricky
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
                         "'comp' <class ParaboloidTricky>: The value of 'wrt' must be a string or list of strings, but a "
                         "type of 'ndarray' was provided.")

        # check invalid method
        with self.assertRaises(ValueError) as cm:
            comp.set_check_partial_options(wrt=['*'], method='foo')

        self.assertEqual(str(cm.exception),
                         "'comp' <class ParaboloidTricky>: Method 'foo' is not supported, method must be one of ('fd', 'cs')")

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
                         "'comp' <class ParaboloidTricky>: The value of 'step' must be numeric, but 'foo' was specified.")

        # check invalid step_calc
        with self.assertRaises(ValueError) as cm:
            comp.set_check_partial_options(wrt=['*'], step_calc='foo')

        self.assertEqual(str(cm.exception),
                         "'comp' <class ParaboloidTricky>: The value of 'step_calc' must be one of ('abs', 'rel', 'rel_legacy', 'rel_avg', 'rel_element'), "
                         "but 'foo' was specified.")

        # check invalid wrt
        comp._declared_partial_checks = []
        comp.set_check_partial_options(wrt=['x*', 'y', 'z', 'a*'])

        with self.assertRaises(ValueError) as cm:
            prob.check_partials()

        self.assertEqual(str(cm.exception), "'comp' <class ParaboloidTricky>: Invalid 'wrt' variables specified "
                         "for check_partial options: ['z'].")

        # check multiple invalid wrt
        comp._declared_partial_checks = []
        comp.set_check_partial_options(wrt=['a', 'b', 'c'])

        with self.assertRaises(ValueError) as cm:
            prob.check_partials()

        self.assertEqual(str(cm.exception), "'comp' <class ParaboloidTricky>: Invalid 'wrt' variables specified "
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
        tabnum = 0
        sep = '|'
        header_locations_of_bars = None
        for line in lines:
            if '#' in line and tabnum == 0:
                # transition to 2nd table (worst error table)
                tabnum = 1
                header_locations_of_bars = None

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
        tabnum = 0
        sep = '|'
        header_locations_of_bars = None
        for line in lines:
            if '#' in line and tabnum == 0:
                # transition to 2nd table (worst error table)
                tabnum = 1
                header_locations_of_bars = None

            if sep in line:
                if header_locations_of_bars:
                    value_locations_of_bars = [i for i, ltr in enumerate(line) if ltr == sep]
                    self.assertEqual(value_locations_of_bars, header_locations_of_bars,
                                     msg="Column separators should all be aligned")
                else:
                    header_locations_of_bars = [i for i, ltr in enumerate(line) if ltr == sep]

    def test_compact_print_exceed_tol(self):

        prob = om.Problem()
        prob.model.add_subsystem('comp', MyCompGoodPartials(), promotes=['*'])
        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()
        stream = StringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('>ABS_TOL'), 0)
        self.assertEqual(stream.getvalue().count('>REL_TOL'), 0)

        prob = om.Problem()
        prob.model.add_subsystem('comp', MyCompBadPartials(), promotes=['*'])
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
        self.assertEqual(len([l for l in stream.getvalue().splitlines() if l.startswith('| ')]), 12) # counts rows (including headers)

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
        prob.model.add_subsystem('comp', MyComp(), promotes=['*'])
        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()
        stream = StringIO()
        prob.check_partials(out_stream=stream, compact_print=True)
        self.assertEqual(stream.getvalue().count('rev'), 0)

        stream = StringIO()
        partials_data = prob.check_partials(out_stream=stream, compact_print=False)
        # So for this case, they do all provide them, so rev should not be shown
        self.assertEqual(stream.getvalue().count('Forward Magnitude'), 2)
        self.assertEqual(stream.getvalue().count('Reverse Magnitude'), 0)
        self.assertEqual(stream.getvalue().count('Absolute Error'), 2)
        self.assertEqual(stream.getvalue().count('Relative Error'), 2)
        self.assertEqual(stream.getvalue().count('Raw Forward Derivative'), 2)
        self.assertEqual(stream.getvalue().count('Raw Reverse Derivative'), 0)
        self.assertEqual(stream.getvalue().count('Raw FD Derivative'), 2)
        self.assertEqual(stream.getvalue().count(f"(Jfd)\n    {partials_data['comp'][('z', 'x1')]['J_fd']}"), 1)
        self.assertEqual(stream.getvalue().count(f"(Jfd)\n    {partials_data['comp'][('z', 'x2')]['J_fd']}"), 1)
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
        self.assertEqual(len([l for l in stream.getvalue().splitlines() if l.startswith('| ')]), 8)

        stream = StringIO()
        partials_data = prob.check_partials(out_stream=stream, compact_print=False)
        self.assertEqual(stream.getvalue().count('Forward Magnitude'), 4)
        self.assertEqual(stream.getvalue().count('Reverse Magnitude'), 2)
        self.assertEqual(stream.getvalue().count('Absolute Error'), 8)
        self.assertEqual(stream.getvalue().count('Relative Error'), 8)
        self.assertEqual(stream.getvalue().count('Raw Forward Derivative'), 4)
        self.assertEqual(stream.getvalue().count('Raw Reverse Derivative'), 2)
        self.assertEqual(stream.getvalue().count('Raw FD Derivative'), 4)
        self.assertEqual(stream.getvalue().count(f"(Jfd)\n    {partials_data['c0'][('z', 'x1')]['J_fd']}"), 1)
        self.assertEqual(stream.getvalue().count(f"(Jfd)\n    {partials_data['c0'][('z', 'x2')]['J_fd']}"), 1)
        self.assertEqual(stream.getvalue().count(f"(Jfd)\n    {partials_data['comp'][('f_xy', 'x')]['J_fd']}"), 1)
        self.assertEqual(stream.getvalue().count(f"(Jfd)\n    {partials_data['comp'][('f_xy', 'y')]['J_fd']}"), 1)

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
        self.assertEqual(stream.getvalue().count("'z'             | 'y1'"), 2)

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
        prob.check_partials(out_stream=stream, compact_print=True, show_only_incorrect=True)
        self.assertEqual(stream.getvalue().count("MyCompBadPartials"), 2)
        self.assertEqual(stream.getvalue().count("'z'             | 'y1'"), 2)
        self.assertEqual(stream.getvalue().count("MyCompGoodPartials"), 0)

        stream = StringIO()
        prob.check_partials(out_stream=stream, compact_print=False, show_only_incorrect=True)
        self.assertEqual(stream.getvalue().count("MyCompGoodPartials"), 0)
        self.assertEqual(stream.getvalue().count("MyCompBadPartials"), 1)

    def test_includes_excludes(self):

        prob = om.Problem()
        model = prob.model

        sub = model.add_subsystem('c1c', om.Group())
        sub.add_subsystem('d1', Paraboloid())
        sub.add_subsystem('e1', Paraboloid())

        sub2 = model.add_subsystem('sss', om.Group())
        sub3 = sub2.add_subsystem('sss2', om.Group())
        sub2.add_subsystem('d1', Paraboloid())
        sub3.add_subsystem('e1', Paraboloid())

        model.add_subsystem('abc1cab', Paraboloid())

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
                super().setup()
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

        prob = om.Problem()
        model = prob.model

        np.random.seed(1)

        comp = DirectionalVectorizedMatFreeComp(n=5)
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

        self.assertEqual(lines[7][53:56], 'n/a')
        assert_near_equal(float(lines[7][121:131]), 0.0, 1e-15)

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
        prob = om.Problem()
        model = prob.model

        np.random.seed(1)

        comp = DirectionalVectorizedComp(n=5)
        model.add_subsystem('comp', comp)

        prob.setup(force_alloc_complex=True)
        prob.run_model()
        J = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(J)

    def test_directional_mixed_error_message(self):
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

        msg = "'mycomp' <class ArrayCompMatrixFree>: For matrix free components, directional should be set to True for all inputs."
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
        partials = prob.check_partials(method='cs')#, out_stream=None)

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

        self.assertTrue("Relative Error (Jfor - Jfd) / Jfor : 1." in lines[10])

    def test_directional_bug_implicit(self):
        # Test for bug in directional derivative direction for implicit var and matrix-free.

        class Directional(om.ImplicitComponent):

            def setup(self):
                self.add_input('in',shape=3)
                self.add_input('in2',shape=3)
                self.add_output('out',shape=3)

                self.set_check_partial_options(wrt='*', directional=True, method='cs')

                self.mat = np.random.rand(3, 3)
                self.mat2 = np.random.rand(3, 3)

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['out'] = self.mat.dot(inputs['in']) + self.mat2.dot(inputs['in2']) - outputs['out']

            def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                if mode == 'fwd':
                    if 'out' in d_residuals:
                        if 'in' in d_inputs:
                            d_residuals['out'] += self.mat.dot(d_inputs['in'])
                        if 'in2' in d_inputs:
                            d_residuals['out'] += self.mat2.dot(d_inputs['in2'])
                        if 'out' in d_outputs:
                            d_residuals['out'] -= d_outputs['out']

                if mode == 'rev':
                    if 'out' in d_residuals:
                        if 'in' in d_inputs:
                            d_inputs['in'] += self.mat.transpose().dot(d_residuals['out'])
                        if 'in2' in d_inputs:
                            d_inputs['in2'] += self.mat2.transpose().dot(d_residuals['out'])
                        if 'out' in d_outputs:
                            d_outputs['out'] -= d_residuals['out']

        prob = om.Problem()
        comp = Directional()
        prob.model.add_subsystem('comp',comp)

        prob.setup(force_alloc_complex=True)
        prob.run_model()
        partials = prob.check_partials(method='cs', out_stream=None)

        assert_check_partials(partials)

    def test_no_linsolve_during_check(self):
        # ensure that solve_linear is not called during check partials
        from openmdao.solvers.solver import LinearSolver

        class ErrSolver(LinearSolver):
            def _linearize(self):
                raise RuntimeError("_linearize called")

            def solve(self, vec_names, mode, rel_systems=None):
                raise RuntimeError("solve called")

            def _run_apply(self):
                raise RuntimeError("_run_apply called")

        class SingularImplicitComp(om.ImplicitComponent):

            def setup(self):
                self.add_input('lhs', shape=10)
                self.add_input('rhs', shape=10)

                self.add_output('resid', shape=10)

                self.declare_partials('*', '*', method='cs')

            def apply_nonlinear(self, inputs, outputs, residuals):

                residuals['resid'] = inputs['lhs'] - inputs['rhs']


        p = om.Problem()

        p.model.add_subsystem('sing_check', SingularImplicitComp())
        p.model.sing_check.linear_solver = ErrSolver()
        p.setup()

        p.run_model()

        p.check_partials(out_stream=None)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestCheckPartialsDistribDirectional(unittest.TestCase):

    N_PROCS = 2

    def test_distrib_directional(self):
        class Sum(om.ExplicitComponent):
            def setup(self):
                self.add_input('u_g', shape_by_conn=True, distributed=True)
                self.add_output('sum')
                self.set_check_partial_options(wrt='*', directional=True, method='cs')

            def compute(self, inputs, outputs):
                outputs['sum'] = self.comm.allreduce(np.sum(inputs['u_g']))

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                if mode == 'fwd':
                    d_outputs['sum'] += self.comm.allreduce(np.sum(d_inputs['u_g']))
                if mode == 'rev':
                    d_inputs['u_g'] += d_outputs['sum']

        num_nodes = 5 + MPI.COMM_WORLD.rank
        prob = om.Problem()
        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('u_g', val= np.random.rand(3*num_nodes), distributed=True)

        prob.model.add_subsystem('adder', Sum(), promotes=['*'])

        prob.setup(force_alloc_complex=True, mode='rev')
        prob.run_model()
        # this test passes if this call doesn't raise an exception
        partials = prob.check_partials(compact_print=True, method='cs')


class TestCheckDerivativesOptionsDifferentFromComputeOptions(unittest.TestCase):
    # Ensure check_partials options differs from the compute partials options

    def test_partials_check_same_as_derivative_compute(self):
        def create_problem(force_alloc_complex=False):
            # Create a fresh model for each little test so tests don't interfere with each other
            prob = self.prob = om.Problem()
            model = prob.model
            model.add_subsystem('px', om.IndepVarComp('x', val=3.0))
            model.add_subsystem('py', om.IndepVarComp('y', val=5.0))
            parab = self.parab = Paraboloid()

            model.add_subsystem('parab', parab)
            model.connect('px.x', 'parab.x')
            model.connect('py.y', 'parab.y')
            model.add_design_var('px.x', lower=-50, upper=50)
            model.add_design_var('py.y', lower=-50, upper=50)
            model.add_objective('parab.f_xy')
            prob.setup(force_alloc_complex=force_alloc_complex)
            return prob, parab

        expected_check_partials_error = "Problem {prob._name}: Checking partials " \
              "with respect to variable '{var}' in component '{comp.pathname}' using the " \
              "same method and options as are used to compute the component's derivatives " \
              "will not provide any relevant information on the accuracy.\n" \
              "To correct this, change the options to do the \n" \
              "check_partials using either:\n" \
              "     - arguments to Problem.check_partials. \n" \
              "     - arguments to Component.set_check_partial_options"

        # Scenario 1:
        #    Compute partials: exact
        #    Check partials: fd using argument to check_partials, default options
        #    Expected result: no error
        prob, parab = create_problem()
        prob.check_partials(method='fd')

        # Scenario 2:
        #    Compute partials: fd, with default options
        #    Check partials: fd using argument to check_partials, default options
        #    Expected result: Error
        prob, parab = create_problem()
        parab.declare_partials(of='*', wrt='*', method='fd')
        with self.assertRaises(OMInvalidCheckDerivativesOptionsWarning) as cm:
            prob.check_partials(method='fd')
        self.assertEqual(str(cm.exception),
                         expected_check_partials_error.format(prob=prob, var='x', comp=parab))

        # Scenario 3:
        #    Compute partials: fd, with default options
        #    Check partials: fd using argument to check_partials, different step option
        #    Expected result: No error
        prob, parab = create_problem()
        parab.declare_partials(of='*', wrt='*', method='fd')
        prob.check_partials(method='fd', step=1e-4)

        # Scenario 4:
        #    Compute partials: fd, with default options
        #    Check partials: fd using argument to check_partials, different form option
        #    Expected result: No error
        prob, parab = create_problem()
        parab.declare_partials(of='*', wrt='*', method='fd')
        prob.check_partials(method='fd', form='backward')

        # Scenario 5:
        #    Compute partials: fd, with default options
        #    Check partials: fd using argument to check_partials, different step_calc option
        #    Expected result: No error
        prob, parab = create_problem()
        parab.declare_partials(of='*', wrt='*', method='fd')
        prob.check_partials(method='fd', step_calc='rel')

        # Scenario 5:
        #    Compute partials: fd, with default options
        #    Check partials: fd using argument to check_partials, different directional option
        #    Expected result: No error
        prob, parab = create_problem()
        parab.declare_partials(of='*', wrt='*', method='fd')
        parab.set_check_partial_options('*', directional=True)
        prob.check_partials(method='fd')

        # Scenario 6:
        #    Compute partials: fd, with default options
        #    Check partials: cs using argument to check_partials
        #    Other condition: haven't allocated a complex vector, so we fall back on fd
        #    Expected result: Error since using fd to check fd. All options the same
        prob, parab = create_problem()
        parab.declare_partials(of='*', wrt='*', method='fd')
        with self.assertRaises(OMInvalidCheckDerivativesOptionsWarning) as cm:
            prob.check_partials(method='cs')
        self.assertEqual(str(cm.exception),
                         expected_check_partials_error.format(prob=prob, var='x', comp=parab))

        # Scenario 7:
        #    Compute partials: fd, with default options
        #    Check partials: cs using argument to check_partials
        #    Other condition: setup with complex vector, so complex step can be used
        #    Expected result: No error
        prob, parab = create_problem(force_alloc_complex=True)
        parab.declare_partials(of='*', wrt='*', method='fd')
        prob.setup(force_alloc_complex=True)
        prob.run_model()
        prob.check_partials(method='cs')

        # Scenario 8:
        #    Compute partials: fd, with default options
        #    Check partials: fd using Component.set_check_partial_options argument, default options
        #    Expected result: Error
        prob, parab = create_problem()
        parab.declare_partials(of='*', wrt='*', method='fd')
        parab.set_check_partial_options('*')
        with self.assertRaises(OMInvalidCheckDerivativesOptionsWarning) as cm:
            prob.check_partials()
        self.assertEqual(str(cm.exception),
                         expected_check_partials_error.format(prob=prob, var='x', comp=parab))

        # Scenario 9:
        #    Compute partials: fd, with default options
        #    Check partials: fd using Component.set_check_partial_options argument, step set
        #    Expected result: No error
        prob, parab = create_problem()
        parab.declare_partials(of='*', wrt='*', method='fd')
        parab.set_check_partial_options('*', step=1e-4)
        prob.check_partials()

        # Scenario 10:
        #    Compute partials: fd, with default options
        #    Check partials: fd using Component.set_check_partial_options argument, form set
        #    Expected result: No error
        prob, parab = create_problem()
        parab.declare_partials(of='*', wrt='*', method='fd')
        parab.set_check_partial_options('*', form='backward',
                                        step= FiniteDifference.DEFAULT_OPTIONS['step'])
        prob.check_partials()

        # Scenario 11:
        #    Compute partials: fd, with default options
        #    Check partials: fd using Component.set_check_partial_options argument, step_calc set
        #    Expected result: No error
        prob, parab = create_problem()
        parab.declare_partials(of='*', wrt='*', method='fd')
        parab.set_check_partial_options('*', step_calc='rel',
                                        form=FiniteDifference.DEFAULT_OPTIONS['form'])
        prob.check_partials()

        # Scenario 12:
        #    Compute partials: fd, with default options
        #    Check partials: cs using Component.set_check_partial_options argument
        #    Expected result: No error
        prob, parab = create_problem(force_alloc_complex=True)
        parab.declare_partials(of='*', wrt='*', method='fd')
        parab.set_check_partial_options('*', method='cs')
        prob.check_partials()

        # Scenario 13:
        #    Compute partials: cs, with default options
        #    Check partials: fd
        #    Expected result: No error
        prob, parab = create_problem()
        parab.declare_partials(of='*', wrt='*', method='cs')
        parab.set_check_partial_options('*', method='fd')
        prob.check_partials()

        # Scenario 14:
        #    Compute partials: cs, with default options
        #    Check partials: cs
        #    Expected result: Error
        prob, parab = create_problem(force_alloc_complex=True)
        parab.declare_partials(of='*', wrt='*', method='cs')
        parab.set_check_partial_options('*', method='cs')
        with self.assertRaises(OMInvalidCheckDerivativesOptionsWarning) as cm:
            prob.check_partials()
        self.assertEqual(str(cm.exception),
                         expected_check_partials_error.format(prob=prob, var='x', comp=parab))

        # Scenario 15:
        #    Compute partials: cs, with default options
        #    Check partials: cs, with different step
        #    Expected result: No error
        prob, parab = create_problem(force_alloc_complex=True)
        parab.declare_partials(of='*', wrt='*', method='cs',
                               step=ComplexStep.DEFAULT_OPTIONS['step'])
        parab.set_check_partial_options('*', method='cs',
                                        step=2.0*ComplexStep.DEFAULT_OPTIONS['step'])
        prob.check_partials()

        # Now do similar checks for check_totals when approximations are used
        expected_check_totals_error_msg = "Problem {prob._name}: Checking totals using the " \
              "same method and options as are used to compute the totals will not provide " \
              "any relevant information on the accuracy.\n" \
              "To correct this, change the options to do the " \
              "check_totals or on the call to approx_totals for the model."

        # Scenario 16:
        #    Compute totals: no approx on totals
        #    Check totals: defaults
        #    Expected result: No error since no approx on computing totals
        prob, parab = create_problem()
        prob.setup()
        prob.run_model()
        prob.check_totals()

        # Scenario 17:
        #    Compute totals: approx on totals using defaults
        #    Check totals: using defaults
        #    Expected result: Error since compute and check have same method and options
        prob, parab = create_problem()
        prob.model.approx_totals()
        prob.setup()
        prob.run_model()
        with self.assertRaises(OMInvalidCheckDerivativesOptionsWarning) as cm:
            prob.check_totals()
        self.assertEqual(str(cm.exception),
                         expected_check_totals_error_msg.format(prob=prob))

        # Scenario 18:
        #    Compute totals: approx on totals using defaults
        #    Check totals: fd, non default step
        #    Expected result: No error
        prob, parab = create_problem()
        prob.model.approx_totals()
        prob.setup()
        prob.run_model()
        prob.check_totals(step=1e-7)

        # Scenario 18a:
        #    Compute totals: approx on totals using defaults
        #    Check totals: fd, non default steps
        #    Expected result: No error
        prob, parab = create_problem()
        prob.model.approx_totals()
        prob.setup()
        prob.run_model()
        prob.check_totals(step=[1e-7, 1e-8])

        # Scenario 19:
        #    Compute totals: approx on totals using defaults
        #    Check totals: fd, non default form
        #    Expected result: No error
        prob, parab = create_problem()
        prob.model.approx_totals()
        prob.setup()
        prob.run_model()
        prob.check_totals(form='central')


        # Scenario 20:
        #    Compute totals: approx on totals using defaults
        #    Check totals: fd, non default step_calc
        #    Expected result: No error
        prob, parab = create_problem()
        prob.model.approx_totals()
        prob.setup()
        prob.run_model()
        prob.check_totals(step_calc='rel')

        # Scenario 21:
        #    Compute totals: cs
        #    Check totals: cs
        #    Expected result: Error since both using the same method
        prob, parab = create_problem(force_alloc_complex=True)
        prob.model.approx_totals(method='cs')
        prob.setup()
        prob.run_model()
        with self.assertRaises(OMInvalidCheckDerivativesOptionsWarning) as cm:
            prob.check_totals(method='cs')
        self.assertEqual(str(cm.exception),
                         expected_check_totals_error_msg.format(prob=prob))

        # Scenario 22:
        #    Compute totals: fd, the default
        #    Check totals: cs
        #    Expected result: No error
        prob, parab = create_problem()
        prob.model.approx_totals()
        prob.setup(force_alloc_complex=True)
        prob.run_model()
        prob.check_totals(method='cs')

        # Scenario 23:
        #    Compute totals: cs
        #    Check totals: default
        #    Expected result: No error
        prob, parab = create_problem()
        prob.model.approx_totals(method='cs')
        prob.setup(force_alloc_complex=True)
        prob.run_model()
        prob.check_totals()


class TestCheckPartialsFeature(unittest.TestCase):

    def test_feature_incorrect_jacobian(self):

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

    def test_set_step_on_comp(self):

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

        prob = om.Problem()

        prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        prob.check_partials(step=1e-2, compact_print=True)

    def test_set_method_on_comp(self):

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

        prob = om.Problem()

        prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        prob.check_partials(method='cs', compact_print=True)

    def test_set_form_global(self):

        prob = om.Problem()

        prob.model.add_subsystem('comp', ParaboloidTricky())
        prob.model.add_subsystem('comp2', ParaboloidMatVec())

        prob.model.connect('comp.f_xy', 'comp2.x')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        stream = StringIO()
        prob.check_partials(form='central', compact_print=True, out_stream=stream)
        contents = stream.getvalue()

    def test_set_step_calc_global(self):

        prob = om.Problem()

        prob.model.add_subsystem('comp', ParaboloidTricky())

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        prob.check_partials(step_calc='rel', compact_print=True)

    def test_feature_check_partials_show_only_incorrect(self):

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

        prob = om.Problem()
        model = prob.model

        sub = model.add_subsystem('c1c', om.Group())
        sub.add_subsystem('d1', Paraboloid())
        sub.add_subsystem('e1', Paraboloid())

        sub2 = model.add_subsystem('sss', om.Group())
        sub3 = sub2.add_subsystem('sss2', om.Group())
        sub2.add_subsystem('d1', Paraboloid())
        sub3.add_subsystem('e1', Paraboloid())

        model.add_subsystem('abc1cab', Paraboloid())

        prob.setup()
        prob.run_model()

        prob.check_partials(compact_print=True, includes='*c*c*')

        prob.check_partials(compact_print=True, includes=['*d1', '*e1'])

        prob.check_partials(compact_print=True, includes=['abc1cab'])

        prob.check_partials(compact_print=True, includes='*c*c*', excludes=['*e*'])

    def test_directional(self):

        prob = om.Problem()
        model = prob.model
        mycomp = model.add_subsystem('mycomp', ArrayComp(), promotes=['*'])

        prob.setup()
        prob.run_model()

        data = prob.check_partials()

    def test_directional_matrix_free(self):

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

    def test_directional_sparse_deriv(self):

        class FDComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('vec_size', types=int, default=1)

            def setup(self):
                nn = self.options['vec_size']

                self.add_input('x_element', np.ones((nn, )))
                self.add_output('y', np.ones((nn, )))

            def setup_partials(self):
                nn = self.options['vec_size']
                self.declare_partials('*', 'x_element', rows=np.arange(nn), cols=np.arange(nn))

                self.set_check_partial_options('x_element', method='fd', step_calc='rel_avg', directional=True)

            def compute(self, inputs, outputs):
                x3 = inputs['x_element']
                outputs['y'] = 0.5 * x3 ** 2

            def compute_partials(self, inputs, partials):
                x3 = inputs['x_element']
                partials['y', 'x_element'] = x3


        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', FDComp(vec_size=3))

        prob.setup()

        x = np.array([3, 4, 5])
        prob.set_val('comp.x_element', x)

        prob.run_model()

        partials = prob.check_partials(out_stream=None)
        assert_check_partials(partials, atol=1e-5, rtol=1e-5)

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

        # J = prob.check_partials(compact_print=True, method='cs', step=1e-40, out_stream=None)
        J = prob.check_partials(compact_print=True, method='cs', step=1e-40)

        assert_check_partials(J, atol=1e-5, rtol=1e-5)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestCheckPartialsDistrib(unittest.TestCase):

    N_PROCS = 2

    def test_check_size_0(self):
        # this test passes if it doesn't raise an exception.  The actual partials are not
        # correct because compute_jacvec_product does nothing.
        class DistributedConverter(om.ExplicitComponent):
            "takes a non-distributed input and converts it to a distributed output that is only on rank 0"
            def setup(self):

                shape_switch = 1 if self.comm.rank == 0 else 0
                self.add_input('in1_serial', shape=5)
                self.add_output('out1', shape=5*shape_switch, tags=['mphys_coupling'], distributed=True)

                self.add_input('in2_serial', shape=15)
                self.add_output('out2', shape=15*shape_switch, tags=['mphys_result'], distributed=True)

            def compute(self,inputs, outputs):
                if self.comm.rank == 0:
                    outputs["out1"] = inputs["in1_serial"]
                    outputs["out2"] = inputs["in2_serial"]

            # this was broken
            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                pass # intentionally left empty because presence of this method triggers bug in check_partials

        prob = om.Problem()

        prob.model.add_subsystem('converter', DistributedConverter())

        prob.setup(force_alloc_complex=True)

        prob.check_partials(compact_print=True, method='cs')


class TestCheckPartialsMultipleSteps(unittest.TestCase):
    def setup_model(self, directional=False):
        class CompGoodPartials(om.ExplicitComponent):
            def setup(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)
                self.add_output('y', 5.5)
                self.declare_partials(of='*', wrt='*')
                if directional:
                    self.set_check_partial_options(wrt='*', directional=True)

            def compute(self, inputs, outputs):
                outputs['y'] = 3.0 * inputs['x1'] + 4.0 * inputs['x2']

            def compute_partials(self, inputs, partials):
                """Correct derivative."""
                J = partials
                J['y', 'x1'] = np.array([3.0])
                J['y', 'x2'] = np.array([4.0])

        class CompBadPartials(om.ExplicitComponent):
            def setup(self):
                self.add_input('y1', 3.0)
                self.add_input('y2', 5.0)
                self.add_output('z', 5.5)
                self.declare_partials(of='*', wrt='*')
                if directional:
                    self.set_check_partial_options(wrt='*', directional=True)

            def compute(self, inputs, outputs):
                outputs['z'] = 3.0 * inputs['y1'] + 4.0 * inputs['y2']

            def compute_partials(self, inputs, partials):
                """Intentionally incorrect derivative."""
                J = partials
                J['z', 'y1'] = np.array([33.0])
                J['z', 'y2'] = np.array([40.0])

        prob = om.Problem()
        prob.model.add_subsystem('good', CompGoodPartials())
        prob.model.add_subsystem('bad', CompBadPartials())
        prob.model.connect('good.y', 'bad.y1')

        prob.set_solver_print(level=0)
        prob.setup(force_alloc_complex=True)
        prob.run_model()

        return prob

    def get_tables(self, content):
        tables = []
        lines = content.splitlines()
        tlines = []
        skipto = 0
        for i, line in enumerate(lines):
            if i < skipto:
                continue
            if line.startswith('+-'):
                # start of a table
                for j in range(i, len(lines)):
                    if not lines[j].startswith('+') and not lines[j].startswith('|'):
                        break
                    tlines.append(lines[j])
                skipto = j + 1
                tables.append(tlines)
                tlines = []
        return tables

    def test_single_fd_step_fwd(self):
        p = self.setup_model()
        stream = StringIO()
        J = p.check_partials(step=[1e-6], out_stream=stream)
        contents = stream.getvalue()
        ncomps = 2
        nderivs = ncomps * 2
        self.assertEqual(contents.count("Component: CompGoodPartials 'good'"), 1)
        self.assertEqual(contents.count("Component: CompBadPartials 'bad'"), 1)
        self.assertEqual(contents.count("Fd Magnitude:"), nderivs)
        self.assertEqual(contents.count("Absolute Error (Jfor - Jfd), step="), 0)
        self.assertEqual(contents.count("Absolute Error (Jfor - Jfd)"), nderivs)
        self.assertEqual(contents.count("Relative Error (Jfor - Jfd) / Jf"), nderivs)
        self.assertEqual(contents.count("Raw FD Derivative (Jfd), step="), 0)
        self.assertEqual(contents.count("Raw FD Derivative (Jfd)"), nderivs)

    def test_single_fd_step_compact(self):
        p = self.setup_model()
        stream = StringIO()
        J = p.check_partials(step=[1e-6], compact_print=True, out_stream=stream)
        contents = stream.getvalue()
        self.assertEqual(contents.count("Component: CompGoodPartials 'good'"), 1)
        self.assertEqual(contents.count("Component: CompBadPartials 'bad'"), 1)
        self.assertEqual(contents.count("Sub Jacobian with Largest Relative Error: CompBadPartials 'bad'"), 1)
        self.assertEqual(contents.count(">ABS_TOL >REL_TOL"), 2)
        self.assertEqual(contents.count("step"), 0)
        tables = self.get_tables(contents)
        self.assertEqual(len(tables), 3)
        self.assertEqual(len(tables[0]), 7)
        self.assertEqual(len(tables[1]), 7)
        self.assertEqual(len(tables[2]), 5)
        # check cols
        self.assertEqual(tables[0][0].count('+'), 8)
        self.assertEqual(tables[1][0].count('+'), 8)
        self.assertEqual(tables[2][0].count('+'), 7)

    def test_single_cs_step_compact(self):
        p = self.setup_model()
        stream = StringIO()
        J = p.check_partials(method='cs', step=[1e-30], compact_print=True, out_stream=stream)
        contents = stream.getvalue()
        self.assertEqual(contents.count("Component: CompGoodPartials 'good'"), 1)
        self.assertEqual(contents.count("Component: CompBadPartials 'bad'"), 1)
        self.assertEqual(contents.count("Sub Jacobian with Largest Relative Error: CompBadPartials 'bad'"), 1)
        self.assertEqual(contents.count(">ABS_TOL >REL_TOL"), 2)
        self.assertEqual(contents.count("step"), 0)
        tables = self.get_tables(contents)
        self.assertEqual(len(tables), 3)
        self.assertEqual(len(tables[0]), 7)
        self.assertEqual(len(tables[1]), 7)
        self.assertEqual(len(tables[2]), 5)
        # check cols
        self.assertEqual(tables[0][0].count('+'), 8)
        self.assertEqual(tables[1][0].count('+'), 8)
        self.assertEqual(tables[2][0].count('+'), 7)

    def test_multi_fd_steps(self):
        p = self.setup_model()
        stream = StringIO()
        J = p.check_partials(step=[1e-6, 1e-7], out_stream=stream)
        contents = stream.getvalue()
        ncomps = 2
        nderivs = ncomps * 2
        self.assertEqual(contents.count("Component: CompGoodPartials 'good'"), 1)
        self.assertEqual(contents.count("Component: CompBadPartials 'bad'"), 1)
        self.assertEqual(contents.count("Fd Magnitude:"), nderivs * 2)
        self.assertEqual(contents.count("Absolute Error (Jfor - Jfd), step="), nderivs * 2)
        self.assertEqual(contents.count("Relative Error (Jfor - Jfd) / Jf"), nderivs * 2)
        self.assertEqual(contents.count("Raw FD Derivative (Jfd), step="), nderivs * 2)

    def test_multi_fd_steps_compact(self):
        p = self.setup_model()
        stream = StringIO()
        J = p.check_partials(step=[1e-6, 1e-7], compact_print=True, out_stream=stream)
        contents = stream.getvalue()
        self.assertEqual(contents.count("Component: CompGoodPartials 'good'"), 1)
        self.assertEqual(contents.count("Component: CompBadPartials 'bad'"), 1)
        self.assertEqual(contents.count("Sub Jacobian with Largest Relative Error: CompBadPartials 'bad'"), 1)
        self.assertEqual(contents.count(">ABS_TOL >REL_TOL"), 4)
        self.assertEqual(contents.count("step"), 3)
        tables = self.get_tables(contents)
        self.assertEqual(len(tables), 3)
        self.assertEqual(len(tables[0]), 11)
        self.assertEqual(len(tables[1]), 11)
        self.assertEqual(len(tables[2]), 5)
        # check cols
        self.assertEqual(tables[0][0].count('+'), 9)
        self.assertEqual(tables[1][0].count('+'), 9)
        self.assertEqual(tables[2][0].count('+'), 8)

    def test_multi_cs_steps_compact(self):
        p = self.setup_model()
        stream = StringIO()
        J = p.check_partials(method='cs', step=[1e-6, 1e-7], compact_print=True, out_stream=stream)
        contents = stream.getvalue()
        self.assertEqual(contents.count("Component: CompGoodPartials 'good'"), 1)
        self.assertEqual(contents.count("Component: CompBadPartials 'bad'"), 1)
        self.assertEqual(contents.count("Sub Jacobian with Largest Relative Error: CompBadPartials 'bad'"), 1)
        self.assertEqual(contents.count(">ABS_TOL >REL_TOL"), 4)
        self.assertEqual(contents.count("step"), 3)
        tables = self.get_tables(contents)
        self.assertEqual(len(tables), 3)
        self.assertEqual(len(tables[0]), 11)
        self.assertEqual(len(tables[1]), 11)
        self.assertEqual(len(tables[2]), 5)
        # check cols
        self.assertEqual(tables[0][0].count('+'), 9)
        self.assertEqual(tables[1][0].count('+'), 9)
        self.assertEqual(tables[2][0].count('+'), 8)

    def test_multi_fd_steps_compact_directional(self):
        p = self.setup_model(directional=True)
        stream = StringIO()
        J = p.check_partials(step=[1e-6, 1e-7], compact_print=True, out_stream=stream)
        contents = stream.getvalue()
        self.assertEqual(contents.count("Component: CompGoodPartials 'good'"), 1)
        self.assertEqual(contents.count("Component: CompBadPartials 'bad'"), 1)
        self.assertEqual(contents.count("Sub Jacobian with Largest Relative Error: CompBadPartials 'bad'"), 1)
        self.assertEqual(contents.count(">ABS_TOL >REL_TOL"), 4)
        self.assertEqual(contents.count("step"), 3)
        tables = self.get_tables(contents)
        self.assertEqual(len(tables), 3)
        self.assertEqual(len(tables[0]), 11)
        self.assertEqual(len(tables[1]), 11)
        self.assertEqual(len(tables[2]), 5)
        # check cols
        self.assertEqual(tables[0][0].count('+'), 9)
        self.assertEqual(tables[1][0].count('+'), 9)
        self.assertEqual(tables[2][0].count('+'), 8)


if __name__ == "__main__":
    unittest.main()
