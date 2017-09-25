"""
Test the DeprecatedComponent which allows openmmdao 1.x components to run.
"""

import unittest
from six import iteritems
from six.moves import cStringIO

import numpy as np

from openmdao.api import Problem, IndepVarComp, Component, Group, NewtonSolver, NonLinearRunOnce
from openmdao.api import ScipyIterativeSolver as ScipyGMRES
from openmdao.devtools.testutil import assert_rel_error, TestLogger


class Paraboloid(Component):
    """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

    def __init__(self):
        super(Paraboloid, self).__init__()

        self.add_param('x', val=0.0)
        self.add_param('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.declare_partials(of='*', wrt='*')

    def solve_nonlinear(self, params, unknowns, resids):
        """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """

        x = params['x']
        y = params['y']

        unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    def linearize(self, params, unknowns, resids):
        """ Jacobian for our paraboloid."""

        x = params['x']
        y = params['y']

        J = {}

        J['f_xy','x'] = 2.0*x - 6.0 + y
        J['f_xy','y'] = 2.0*y + 8.0 + x

        return J


class ParaboloidApply(Paraboloid):
    """ Use apply_linear instead."""

    def linearize(self, params, unknowns, resids):
        """Analytical derivatives."""
        pass

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids,
                     mode):
        """Returns the product of the incoming vector with the Jacobian."""

        x = params['x']
        y = params['y']

        if mode == 'fwd':
            if 'x' in dparams:
                dresids['f_xy'] += (2.0*x - 6.0 + y)*dparams['x']
            if 'y' in dparams:
                dresids['f_xy'] += (2.0*y + 8.0 + x)*dparams['y']

        elif mode == 'rev':
            if 'x' in dparams:
                dparams['x'] += (2.0*x - 6.0 + y)*dresids['f_xy']
            if 'y' in dparams:
                dparams['y'] += (2.0*y + 8.0 + x)*dresids['f_xy']


class SimpleImplicitComp(Component):
    """ A Simple Implicit Component with an additional output equation.

    f(x,z) = xz + z - 4
    y = x + 2z

    Sol: when x = 0.5, z = 2.666

    Coupled derivs:
    y = x + 8/(x+1)
    dy_dx = 1 - 8/(x+1)**2 = -2.5555555555555554
    z = 4/(x+1)
    dz_dx = -4/(x+1)**2 = -1.7777777777777777
    """

    def __init__(self, resid_scaler=1.0):
        super(SimpleImplicitComp, self).__init__()

        # Params
        self.add_param('x', 0.5)

        # Unknowns
        self.add_output('y', 0.0)

        # States
        self.add_state('z', 0.0, resid_scaler=resid_scaler)

        self.maxiter = 10
        self.atol = 1.0e-12
        self.self_solve = False

        # Derivatives
        self.declare_partials(of='*', wrt='*')

    def solve_nonlinear(self, params, unknowns, resids):
        """ Simple iterative solve. (Babylonian method)."""

        if self.self_solve:
            x = params['x']
            z = unknowns['z']
            znew = z

            iter = 0
            eps = 1.0e99
            while iter < self.maxiter and abs(eps) > self.atol:
                z = znew
                znew = 4.0 - x*z

                eps = x*znew + znew - 4.0

            unknowns['z'] = znew
            unknowns['y'] = x + 2.0*znew

            resids['z'] = eps

    def apply_nonlinear(self, params, unknowns, resids):
        """ Don't solve; just calculate the residual."""

        x = params['x']
        z = unknowns['z']
        resids['z'] = x*z + z - 4.0

        # Output equations need to evaluate a residual just like an explicit comp.
        resids['y'] = x + 2.0*z - unknowns['y']

    def linearize(self, params, unknowns, resids):
        """Analytical derivatives."""

        J = {}

        # Output equation
        J[('y', 'x')] = np.array([1.0])
        J[('y', 'z')] = np.array([2.0])

        # State equation
        J[('z', 'z')] = np.array([params['x'] + 1.0])
        J[('z', 'x')] = np.array([unknowns['z']])

        return J


class SimpleImplicitCompApply(SimpleImplicitComp):
    """ Use apply_linear instead."""

    def linearize(self, params, unknowns, resids):
        """Analytical derivatives."""
        pass

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids,
                     mode):
        """Returns the product of the incoming vector with the Jacobian."""

        if mode == 'fwd':
            if 'y' in dresids:
                if 'x' in dparams:
                    dresids['y'] += dparams['x']
                if 'z' in dunknowns:
                    dresids['y'] += 2.0*dunknowns['z']

            if 'z' in dresids:
                if 'x' in dparams:
                    dresids['z'] += (np.array([unknowns['z']])).dot(dparams['x'])
                if 'z' in dunknowns:
                    dresids['z'] += (np.array([params['x'] + 1.0])).dot(dunknowns['z'])

        elif mode == 'rev':
            if 'y' in dresids:
                if 'x' in dparams:
                    dparams['x'] += dresids['y']
                if 'z' in dunknowns:
                    dunknowns['z'] += 2.0*dresids['y']

            if 'z' in dresids:
                if 'x' in dparams:
                    dparams['x'] += (np.array([unknowns['z']])).dot(dresids['z'])
                if 'z' in dunknowns:
                    dunknowns['z'] += (np.array([params['x'] + 1.0])).dot(dresids['z'])


class DepCompTestCase(unittest.TestCase):

    def test_simple_explicit(self):
        prob = Problem(Group())

        prob.model.add_subsystem('px', IndepVarComp('x', 1.0))
        prob.model.add_subsystem('py', IndepVarComp('y', 1.0))
        prob.model.add_subsystem('comp', Paraboloid())

        prob.model.connect('px.x', 'comp.x')
        prob.model.connect('py.y', 'comp.y')
        prob.model.linear_solver = ScipyGMRES()
        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_rel_error(self, prob['comp.f_xy'], 27.0, 1e-6)

        J = prob.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])
        assert_rel_error(self, J[('comp.f_xy', 'px.x')][0][0], -3.0, 1e-5)
        assert_rel_error(self, J[('comp.f_xy', 'py.y')][0][0], 11.0, 1e-5)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        assert_rel_error(self, prob['comp.f_xy'], 27.0, 1e-6)

        J = prob.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])
        assert_rel_error(self, J[('comp.f_xy', 'px.x')][0][0], -3.0, 1e-5)
        assert_rel_error(self, J[('comp.f_xy', 'py.y')][0][0], 11.0, 1e-5)

        # Check partials
        data = prob.check_partials()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_explicit_apply(self):
        prob = Problem(Group())

        prob.model.add_subsystem('px', IndepVarComp('x', 1.0))
        prob.model.add_subsystem('py', IndepVarComp('y', 1.0))
        prob.model.add_subsystem('comp', ParaboloidApply())

        prob.model.connect('px.x', 'comp.x')
        prob.model.connect('py.y', 'comp.y')
        prob.model.linear_solver = ScipyGMRES()
        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_rel_error(self, prob['comp.f_xy'], 27.0, 1e-6)

        J = prob.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])
        assert_rel_error(self, J[('comp.f_xy', 'px.x')][0][0], -3.0, 1e-5)
        assert_rel_error(self, J[('comp.f_xy', 'py.y')][0][0], 11.0, 1e-5)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        assert_rel_error(self, prob['comp.f_xy'], 27.0, 1e-6)

        J = prob.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])
        assert_rel_error(self, J[('comp.f_xy', 'px.x')][0][0], -3.0, 1e-5)
        assert_rel_error(self, J[('comp.f_xy', 'py.y')][0][0], 11.0, 1e-5)

        # Check partials
        data = prob.check_partials()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

    def test_simple_implicit(self):
        prob = Problem(Group())
        prob.model.add_subsystem('p1', IndepVarComp('x', 0.5))
        prob.model.add_subsystem('comp', SimpleImplicitComp())

        prob.model.linear_solver = ScipyGMRES()
        prob.model.nonlinear_solver = NewtonSolver()
        prob.set_solver_print(level=0)

        prob.model.connect('p1.x', 'comp.x')

        # fwd mode
        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_rel_error(self, prob['comp.z'], 2.666, 1e-3)
        self.assertLess(prob.model.nonlinear_solver._iter_count, 5)

        J = prob.compute_totals(of=['comp.y', 'comp.z'], wrt=['p1.x'])
        assert_rel_error(self, J[('comp.y', 'p1.x')][0][0], -2.5555511, 1e-5)
        assert_rel_error(self, J[('comp.z', 'p1.x')][0][0], -1.77777777, 1e-5)

        # rev mode
        prob.setup(check=False, mode='rev')
        prob.run_model()

        assert_rel_error(self, prob['comp.z'], 2.666, 1e-3)
        self.assertLess(prob.model.nonlinear_solver._iter_count, 5)

        J = prob.compute_totals(of=['comp.y', 'comp.z'], wrt=['p1.x'])
        assert_rel_error(self, J[('comp.y', 'p1.x')][0][0], -2.5555511, 1e-5)
        assert_rel_error(self, J[('comp.z', 'p1.x')][0][0], -1.77777777, 1e-5)

        # Check partials
        data = prob.check_partials()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)

        assert_rel_error(self, data['comp'][('y', 'x')]['J_fwd'][0][0], 1.0, 1e-6)
        assert_rel_error(self, data['comp'][('y', 'z')]['J_fwd'][0][0], 2.0, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'x')]['J_fwd'][0][0], 2.66666667, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'z')]['J_fwd'][0][0], 1.5, 1e-6)

        # list inputs
        inputs = prob.model.list_inputs(out_stream=None)
        self.assertEqual(sorted(inputs), [('comp.x', [0.5])])

        # list explicit outputs
        outputs = sorted(prob.model.list_outputs(implicit=False, out_stream=None))
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0][0], 'comp.y')
        assert_rel_error(self, outputs[0][1], 5.8333333, 1e-6)
        self.assertEqual(outputs[1][0], 'p1.x')
        assert_rel_error(self, outputs[1][1], 0.5, 1e-6)

        # list states
        states = prob.model.list_outputs(explicit=False, out_stream=None)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0][0], 'comp.z')
        assert_rel_error(self, states[0][1], 2.6666667, 1e-6)

        # list residuals
        resids = sorted(prob.model.list_residuals(out_stream=None))
        self.assertEqual(len(resids), 3)
        self.assertEqual(resids[0][0], 'comp.y')
        assert_rel_error(self, resids[0][1], 0., 1e-6)
        self.assertEqual(resids[1][0], 'comp.z')
        assert_rel_error(self, resids[1][1], 0., 1e-6)
        self.assertEqual(resids[2][0], 'p1.x')
        assert_rel_error(self, resids[2][1], 0., 1e-6)

    def test_simple_implicit_self_solve(self):

        prob = Problem(Group())
        prob.model.add_subsystem('p1', IndepVarComp('x', 0.5))
        comp = prob.model.add_subsystem('comp', SimpleImplicitComp())
        comp.self_solve = True

        prob.model.linear_solver = ScipyGMRES()
        prob.model.nonlinear_solver = NonLinearRunOnce()
        prob.set_solver_print(level=0)

        prob.model.connect('p1.x', 'comp.x')

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_rel_error(self, prob['comp.z'], 2.666, 1e-3)
        self.assertLess(prob.model.nonlinear_solver._iter_count, 5)

        J = prob.compute_totals(of=['comp.y', 'comp.z'], wrt=['p1.x'])
        assert_rel_error(self, J[('comp.y', 'p1.x')][0][0], -2.5555511, 1e-5)
        assert_rel_error(self, J[('comp.z', 'p1.x')][0][0], -1.77777777, 1e-5)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        assert_rel_error(self, prob['comp.z'], 2.666, 1e-3)
        self.assertLess(prob.model.nonlinear_solver._iter_count, 5)

        J = prob.compute_totals(of=['comp.y', 'comp.z'], wrt=['p1.x'])
        assert_rel_error(self, J[('comp.y', 'p1.x')][0][0], -2.5555511, 1e-5)
        assert_rel_error(self, J[('comp.z', 'p1.x')][0][0], -1.77777777, 1e-5)

        # Check partials
        data = prob.check_partials()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)

        assert_rel_error(self, data['comp'][('y', 'x')]['J_fwd'][0][0], 1.0, 1e-6)
        assert_rel_error(self, data['comp'][('y', 'z')]['J_fwd'][0][0], 2.0, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'x')]['J_fwd'][0][0], 2.66666667, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'z')]['J_fwd'][0][0], 1.5, 1e-6)

    def test_simple_implicit_resid(self):

        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('p1', IndepVarComp('x', 0.5))
        prob.model.add_subsystem('comp', SimpleImplicitComp(resid_scaler=0.001))

        prob.model.linear_solver = ScipyGMRES()
        prob.model.nonlinear_solver = NewtonSolver()
        prob.set_solver_print(level=0)

        prob.model.connect('p1.x', 'comp.x')

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_rel_error(self, prob['comp.z'], 2.666, 1e-3)
        self.assertLess(prob.model.nonlinear_solver._iter_count, 5)

        J = prob.compute_totals(of=['comp.y', 'comp.z'], wrt=['p1.x'])
        assert_rel_error(self, J[('comp.y', 'p1.x')][0][0], -2.5555511, 1e-5)
        assert_rel_error(self, J[('comp.z', 'p1.x')][0][0], -1.77777777, 1e-5)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        assert_rel_error(self, prob['comp.z'], 2.666, 1e-3)
        self.assertLess(prob.model.nonlinear_solver._iter_count, 5)

        J = prob.compute_totals(of=['comp.y', 'comp.z'], wrt=['p1.x'])
        assert_rel_error(self, J[('comp.y', 'p1.x')][0][0], -2.5555511, 1e-5)
        assert_rel_error(self, J[('comp.z', 'p1.x')][0][0], -1.77777777, 1e-5)

        # Check partials
        data = prob.check_partials()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-3)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-3)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-3)

        assert_rel_error(self, data['comp'][('y', 'x')]['J_fwd'][0][0], 1.0, 1e-6)
        assert_rel_error(self, data['comp'][('y', 'z')]['J_fwd'][0][0], 2.0, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'x')]['J_fwd'][0][0], 2.66666667, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'z')]['J_fwd'][0][0], 1.50, 1e-6)

    def test_simple_implicit_apply(self):

        prob = Problem(Group())
        prob.model.add_subsystem('p1', IndepVarComp('x', 0.5))
        prob.model.add_subsystem('comp', SimpleImplicitCompApply())

        prob.model.linear_solver = ScipyGMRES()
        prob.model.nonlinear_solver = NewtonSolver()
        prob.set_solver_print(level=0)

        prob.model.connect('p1.x', 'comp.x')

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_rel_error(self, prob['comp.z'], 2.666, 1e-3)
        self.assertLess(prob.model.nonlinear_solver._iter_count, 5)

        J = prob.compute_totals(of=['comp.y', 'comp.z'], wrt=['p1.x'])
        assert_rel_error(self, J[('comp.y', 'p1.x')][0][0], -2.5555511, 1e-5)
        assert_rel_error(self, J[('comp.z', 'p1.x')][0][0], -1.77777777, 1e-5)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        assert_rel_error(self, prob['comp.z'], 2.666, 1e-3)
        self.assertLess(prob.model.nonlinear_solver._iter_count, 5)

        J = prob.compute_totals(of=['comp.y', 'comp.z'], wrt=['p1.x'])
        assert_rel_error(self, J[('comp.y', 'p1.x')][0][0], -2.5555511, 1e-5)
        assert_rel_error(self, J[('comp.z', 'p1.x')][0][0], -1.77777777, 1e-5)

        # Check partials
        data = prob.check_partials()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)


if __name__ == "__main__":
    unittest.main()
