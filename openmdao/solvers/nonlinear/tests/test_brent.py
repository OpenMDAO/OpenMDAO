"""Test the Brent nonlinear solver. """

import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class CompTest(om.ImplicitComponent):

    def setup(self):
        self.add_input('a', val=1.)
        self.add_input('b', val=1.)
        self.add_input('c', val=10.)
        self.add_input('n', val=77.0/27.0)

        self.add_output('x', val=2., lower=0, upper=100)

    def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        n = inputs['n']
        x = outputs['x']

        # Can't take fractional power of negative number
        if x >= 0.0:
            fact = x ** n
        else:
            fact = - (-x) ** n

        residuals['x'] = a * fact + b * x - c


class TestBrentSolver(unittest.TestCase):

    def setUp(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', CompTest(), promotes=['*'])
        model.nonlinear_solver = om.BrentSolver(
            state_target='x',
            maxiter=100,
            atol=1e-8,
            rtol=1e-8,
        )
        self.prob = prob

    #def test_deprecated_option(self):
        #with warnings.catch_warnings(record=True) as w:
            ## Cause all warnings to always be triggered.
            #warnings.simplefilter("always")

            ## Trigger a warning.
            #self.prob.root.nl_solver.options['max_iter'] = 100

            #self.assertEqual(len(w), 1)
            #self.assertEqual(str(w[0].message),
                     #"Option 'max_iter' is deprecated. Use 'maxiter' instead.")

    #def test_no_state_var_err(self):

        #try:
            #self.prob.setup(check=False)
        #except ValueError as err:
            #self.assertEqual(str(err), "'state_var' option in Brent solver of root must be specified")
        #else:
            #self.fail('ValueError Expected')

    def test_brent_converge(self):

        p = self.prob

        p.setup()
        p.set_solver_print(2)

        p.run_model()

        assert_near_equal(self, p.get_val('x'), 2.06720359226, 1e-6)

    #def test_brent_analysis_error(self):

        #p = self.prob
        #p.root.nl_solver.options['state_var'] = 'x'
        #p.root.nl_solver.options['err_on_maxiter'] = True
        #p.root.nl_solver.options['maxiter'] = 2

        #p.root.ln_solver=ScipyGMRES()
        #p.setup(check=False)

        #try:
            #p.run()
        #except AnalysisError as err:
            #self.assertEqual(str(err), "Failed to converge after 2 iterations.")
        #else:
            #self.fail("expected AnalysisError")

    #def test_brent_converge_index(self):

        #p = Problem()
        #p.root = Group()
        #p.root.add('comp', IndexCompTest(), promotes=['a','x','n','b','c'])
        #p.root.nl_solver = Brent()

        #p.root.nl_solver.options['state_var'] = 'x'
        #p.root.nl_solver.options['state_var_idx'] = 2
        #p.root.ln_solver = ScipyGMRES()
        #p.setup(check=False)

        #p.run()

        #assert_rel_error(self, p.root.unknowns['x'][2], 2.06720359226, .0001)
        #assert_rel_error(self, p.root.resids['x'][2], 0, .0001)

    #def test_data_pass_bounds(self):

        #p = Problem()
        #p.root = Group()

        #p.root.add('lower', ExecComp('low = 2*a'), promotes=['low', 'a'])
        #p.root.add('upper', ExecComp('high = 2*b'), promotes=['high', 'b'])

        #sub = p.root.add('sub', Group(), promotes=['x','low', 'high'])
        #sub.add('comp', CompTest(), promotes=['a','x','n','b','c'])
        #sub.add('dummy1', ExecComp('d=low'), promotes=['low'])
        #sub.add('dummy2', ExecComp('d=high'), promotes=['high'])
        #sub.nl_solver = Brent()

        #sub.nl_solver.options['state_var'] = 'x'
        ## sub.nl_solver.options['lower_bound'] = -10.
        ## sub.nl_solver.options['upper_bound'] = 110.
        #sub.nl_solver.options['var_lower_bound'] = 'flow' # bad value for testing error
        #sub.nl_solver.options['var_upper_bound'] = 'high'

        #try:
            #p.setup(check=False)
        #except ValueError as err:
            #self.assertEqual(str(err), "'var_lower_bound' variable 'flow' was not found as a parameter on any component in sub")
        #else:
            #self.fail('ValueError expected')
        #sub.ln_solver=ScipyGMRES()

        #sub.nl_solver.options['var_lower_bound'] = 'low' # correct value

        #p.setup(check=False)
        #p['a'] = -5.
        #p['b'] = 55.
        #p.run()

        #assert_rel_error(self, p.root.resids['x'], 0, .0001)
        #assert_rel_error(self, p.root.unknowns['x'], 2.06720359226, .0001)

    #def test_data_pass_bounds_idx(self):

        #p = Problem()
        #p.root = Group()

        #p.root.add('lower', ExecComp('low = 2*a'), promotes=['low', 'a'])
        #p.root.add('upper', ExecComp('high = 2*b'), promotes=['high', 'b'])

        #sub = p.root.add('sub', Group(), promotes=['x','low', 'high'])
        #sub.add('comp', IndexCompTest(), promotes=['a','x','n','b','c'])
        #sub.add('dummy1', ExecComp('d=low'), promotes=['low'])
        #sub.add('dummy2', ExecComp('d=high'), promotes=['high'])
        #sub.nl_solver = Brent()

        #sub.nl_solver.options['state_var'] = 'x'
        #sub.nl_solver.options['state_var_idx'] = 2
        #sub.nl_solver.options['var_lower_bound'] = 'low'
        #sub.nl_solver.options['var_upper_bound'] = 'high'
        #sub.ln_solver=ScipyGMRES()

        #p.setup(check=False)
        #p['a'] = -5.
        #p['b'] = 55.
        #p.run()

        #assert_rel_error(self, p.root.resids['x'][2], 0, .0001)
        #assert_rel_error(self, p.root.unknowns['x'][2], 2.06720359226, .0001)


#class BracketTestComponent(Component):

    #def __init__(self):
        #super(BracketTestComponent, self).__init__()

        ## in
        #self.add_param('a', .3)
        #self.add_param('ap', .01)
        #self.add_param('lambda_r', 7.)

        ## states
        #self.add_state('phi', 0.)

    #def solve_nonlinear(self, p, u, r):
        #pass

    #def apply_nonlinear(self, p, u, r):

        #r['phi'] = np.sin(u['phi'])/(1-p['a']) - np.cos(u['phi'])/p['lambda_r']/(1+p['ap'])
        ## print u['phi'], p['a'], p['lambda_r'], 1+p['ap'], r['phi']


#class TestBrentBracketFunc(unittest.TestCase):

    #def test_bracket(self):

        #p = Problem()
        #p.root = Group()
        #p.root.add('comp', BracketTestComponent(), promotes=['phi', 'a', 'ap', 'lambda_r'])
        #p.root.nl_solver = Brent()
        #p.root.ln_solver = ScipyGMRES()

        #eps = 1e-6
        #p.root.nl_solver.options['lower_bound'] = eps
        #p.root.nl_solver.options['upper_bound'] = np.pi/2 - eps
        #p.root.nl_solver.options['state_var'] = 'phi'

        ## def resize(lower, upper, iter):
        ##     if lower == eps and upper == np.pi/2 - eps:
        ##         return -np.pi/4, -eps, True
        ##     elif lower == -np.pi/4 and upper == -eps:
        ##         return np.pi/2+eps, np.pi-eps, True
        ##     else:
        ##         return lower, upper, False

        ## p.root.nl_solver.f_resize_bracket = resize

        #p.setup(check=False)
        #p.run()
        ## manually compute the right answer
        #def manual_f(phi, params):
            #r = np.sin(phi)/(1-p['a']) - np.cos(phi)/p['lambda_r']/(1+p['ap'])
            ## print phi, p['a'], p['lambda_r'], 1+p['ap'], r
            #return r

        ## run manually
        #phi_star = brentq(manual_f, eps, np.pi/2-eps, args=(p.root.params,))


        #assert_rel_error(self, p.root.unknowns['phi'], phi_star, 1e-10)


if __name__ == '__main__':
    unittest.main()