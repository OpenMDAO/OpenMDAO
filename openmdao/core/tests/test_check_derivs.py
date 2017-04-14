""" Testing for Problem.check_partial_derivs and check_total_derivatives."""

import unittest
from six import iteritems
from six.moves import cStringIO as StringIO

import numpy as np

from openmdao.api import Group, ExplicitComponent, IndepVarComp, Problem, NLRunOnce
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.impl_comp_array import TestImplCompArrayMatVec
from openmdao.test_suite.components.paraboloid import ParaboloidMatVec


class TestProblemCheckPartials(unittest.TestCase):

    def test_incorrect_jacobian(self):
        class MyComp(ExplicitComponent):
            def initialize_variables(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)

                self.add_output('y', 5.5)

            def compute(self, inputs, outputs):
                """ Doesn't do much. """
                outputs['y'] = 3.0*inputs['x1'] + 4.0*inputs['x2']

            def compute_partial_derivs(self, inputs, outputs, partials):
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

        prob.model.suppress_solver_output = True

        prob.setup(check=False)
        prob.run_model()

        string_stream = StringIO()

        data = prob.check_partial_derivs(out_stream=string_stream)

        lines = string_stream.getvalue().split("\n")

        y_wrt_x1_line = lines.index("  comp: 'y' wrt 'x1'")

        self.assertTrue(lines[y_wrt_x1_line+6].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        self.assertTrue(lines[y_wrt_x1_line+7].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        self.assertFalse(lines[y_wrt_x1_line+8].endswith('*'),
                        msg='Error flag not expected in output but displayed')

    def test_missing_entry(self):
        class MyComp(ExplicitComponent):
            def initialize_variables(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)

                self.add_output('y', 5.5)

            def compute(self, inputs, outputs):
                """ Doesn't do much. """
                outputs['y'] = 3.0*inputs['x1'] + 4.0*inputs['x2']

            def compute_partial_derivs(self, inputs, outputs, partials):
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

        prob.model.suppress_solver_output = True

        prob.setup(check=False)
        prob.run_model()

        stream = StringIO()

        data = prob.check_partial_derivs(out_stream=stream)

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
            def initialize_variables(self):
                self.add_input('T', val=284., units="degR", desc="Temperature")
                self.add_input('P', val=1., units='lbf/inch**2', desc="Pressure")

                self.add_output('flow:T', val=284., units="degR", desc="Temperature")
                self.add_output('flow:P', val=1., units='lbf/inch**2', desc="Pressure")

            def initialize_partials(self):
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
        data = p.check_partial_derivs(out_stream=None)

        for comp_name, comp in iteritems(data):
            for partial_name, partial in iteritems(comp):
                forward = partial['J_fwd']
                reverse = partial['J_rev']
                fd = partial['J_fd']
                self.assertAlmostEqual(np.linalg.norm(forward - reverse), 0.)
                self.assertAlmostEqual(np.linalg.norm(forward - fd), 0., delta=1e-6)

    def test_units(self):
        class UnitCompBase(ExplicitComponent):
            def initialize_variables(self):
                self.add_input('T', val=284., units="degR", desc="Temperature")
                self.add_input('P', val=1., units='lbf/inch**2', desc="Pressure")

                self.add_output('flow:T', val=284., units="degR", desc="Temperature")
                self.add_output('flow:P', val=1., units='lbf/inch**2', desc="Pressure")

                self.run_count = 0

            def compute_partial_derivs(self, inputs, outputs, partials):
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

        model.nl_solver = NLRunOnce()

        p.setup()
        data = p.check_partial_derivs(out_stream=None)

        for comp_name, comp in iteritems(data):
            for partial_name, partial in iteritems(comp):
                abs_error = partial['abs error']
                self.assertAlmostEqual(abs_error.forward, 0.)
                self.assertAlmostEqual(abs_error.reverse, 0.)
                self.assertAlmostEqual(abs_error.forward_reverse, 0.)

        # Make sure we only FD this twice.
        # The count is 5 because in check_partial_derivs, there are two calls to apply_nonlinear
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

            def initialize_variables(self):
                if self.units is None:
                    self.add_input(self.i_var, self.val)
                    self.add_output(self.o_var, self.val)
                else:
                    self.add_input(self.i_var, self.val, units=self.units)
                    self.add_output(self.o_var, self.val, units=self.units)

            def initialize_partials(self):
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

        p.model.suppress_solver_output = True

        p.setup()
        p.run_model()

        data = p.check_partial_derivs(out_stream=None)
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

        prob.model.suppress_solver_output = True

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partial_derivs(out_stream=None)

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

        prob.model.suppress_solver_output = True

        prob.setup(check=False)
        prob.run_model()

        data = prob.check_partial_derivs(out_stream=None)

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

if __name__ == "__main__":
    unittest.main()
