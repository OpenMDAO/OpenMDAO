from __future__ import print_function
from pprint import pprint

import unittest
from six import iteritems

import numpy as np

from openmdao.api import Problem, IndepVarComp, Component, Group

try:
    # OpenMDAO 2.x
    from openmdao.devtools.testutil import assert_rel_error
    openmdao_version = 2
except ImportError:
    # OpenMDAO 1.x
    from openmdao.test.util import assert_rel_error
    from openmdao.solvers.scipy_gmres import ScipyGMRES
    openmdao_version = 1


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

    def solve_nonlinear(self, params, unknowns, resids):
        """ Simple iterative solve. (Babylonian method)."""

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
        #print(unknowns['y'], unknowns['z'])

    def apply_nonlinear(self, params, unknowns, resids):
        """ Don't solve; just calculate the residual."""

        x = params['x']
        z = unknowns['z']
        resids['z'] = x*z + z - 4.0

        # Output equations need to evaluate a residual just like an explicit comp.
        resids['y'] = x + 2.0*z - unknowns['y']
        #print(x, unknowns['y'], z, resids['z'], resids['y'])

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
            #dparams['x'] = self.multiplier*dresids['y']
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

    def test_simple_implicit(self):

        prob = Problem(Group())
        if openmdao_version == 1:
            prob.root.ln_solver = ScipyGMRES()
        prob.root.add('comp', SimpleImplicitComp())
        prob.root.add('p1', IndepVarComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        if openmdao_version == 1:
            J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='fwd')
        else:
            J = prob.compute_total_derivs(of=['comp.y'], wrt=['p1.x'])
        print('J:')
        pprint(J)
        assert_rel_error(self, J[0][0], -2.5555511, 1e-5)

        # Check partials
        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

        assert_rel_error(self, data['comp'][('y', 'x')]['J_fwd'][0][0], 1.0, 1e-6)
        assert_rel_error(self, data['comp'][('y', 'z')]['J_fwd'][0][0], 2.0, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'x')]['J_fwd'][0][0], 2.66666667, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'z')]['J_fwd'][0][0], 1.5, 1e-6)

        print('Check Partials:')
        pprint(data)

        # Check total derivs
        data = prob.check_total_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            assert_rel_error(self, val1['abs error'][0], 0.0, 1e-5)
            assert_rel_error(self, val1['abs error'][1], 0.0, 1e-5)
            assert_rel_error(self, val1['abs error'][2], 0.0, 1e-5)
            assert_rel_error(self, val1['rel error'][0], 0.0, 1e-5)
            assert_rel_error(self, val1['rel error'][1], 0.0, 1e-5)
            assert_rel_error(self, val1['rel error'][2], 0.0, 1e-5)

        print('Check Totals:')
        pprint(data)

    def test_simple_implicit_resid(self):

        prob = Problem()
        prob.root = Group()
        if openmdao_version == 1:
            prob.root.ln_solver = ScipyGMRES()
        prob.root.add('comp', SimpleImplicitComp(resid_scaler=0.001))
        prob.root.add('p1', IndepVarComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        if openmdao_version == 1:
            J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='fwd')
        else:
            J = prob.compute_total_derivs(of=['comp.y'], wrt=['p1.x'])
        print('J:')
        pprint(J)
        assert_rel_error(self, J[0][0], -2.5555511, 1e-5)

        # Check partials
        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-3)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-3)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-3)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

        assert_rel_error(self, data['comp'][('y', 'x')]['J_fwd'][0][0], 1.0, 1e-6)
        assert_rel_error(self, data['comp'][('y', 'z')]['J_fwd'][0][0], 2.0, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'x')]['J_fwd'][0][0], 2.66666667/0.001, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'z')]['J_fwd'][0][0], 1.50/0.001, 1e-6)

    def test_simple_implicit_apply(self):

        prob = Problem(Group())
        if openmdao_version == 1:
            prob.root.ln_solver = ScipyGMRES()
        prob.root.add('comp', SimpleImplicitCompApply())
        prob.root.add('p1', IndepVarComp('x', 0.5))

        prob.root.connect('p1.x', 'comp.x')

        prob.setup(check=False)
        prob.run()

        if openmdao_version == 1:
            J = prob.calc_gradient(['p1.x'], ['comp.y'], mode='fwd')
        else:
            J = prob.compute_total_derivs(of=['comp.y'], wrt=['p1.x'])
        print('J:')
        pprint(J)
        assert_rel_error(self, J[0][0], -2.5555511, 1e-5)

        # Check partials
        data = prob.check_partial_derivatives(out_stream=None)

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-5)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-5)

        assert_rel_error(self, data['comp'][('y', 'x')]['J_fwd'][0][0], 1.0, 1e-6)
        assert_rel_error(self, data['comp'][('y', 'z')]['J_fwd'][0][0], 2.0, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'x')]['J_fwd'][0][0], 2.66666667, 1e-6)
        assert_rel_error(self, data['comp'][('z', 'z')]['J_fwd'][0][0], 1.5, 1e-6)


if __name__ == "__main__":
    unittest.main()
