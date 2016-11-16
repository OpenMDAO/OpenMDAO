""" This tets out the Petsc KSP solver in Serial mode. """

import unittest

from math import isnan
import numpy as np

from openmdao.api import Group, Problem, ExplicitComponent, \
                         ExecComp, IndepVarComp, NewtonSolver

try:
    from openmdao.solvers.petsc_ksp import PetscKSP
    from openmdao.core.petsc_impl import PetscImpl as impl
except ImportError:
    impl = None

# from openmdao.solvers import LinearSolver

# from openmdao.api import Group, Problem, IndepVarComp, ExecComp, \
                         # LinearGaussSeidel, AnalysisError, NewtonSolver
# from openmdao.test.converge_diverge import ConvergeDiverge, SingleDiamond, \
#                                            ConvergeDivergeGroups, SingleDiamondGrouped
# from openmdao.test.sellar import SellarDerivativesGrouped
# from openmdao.test.simple_comps import SimpleCompDerivMatVec, FanOut, FanIn, \
#                                        FanOutGrouped, DoubleArrayComp, \
#                                        FanInGrouped, ArrayComp2D, FanOutAllGrouped
# from openmdao.test.util import assert_rel_error
# from openmdao.util.options import OptionsDictionary


def assert_rel_error(test_case, actual, desired, tolerance):
    """
    Determine that the relative error between `actual` and `desired`
    is within `tolerance`. If `desired` is zero, then use absolute error.
    Args
    ----
    test_case : :class:`unittest.TestCase`
        TestCase instance used for assertions.
    actual : float
        The value from the test.
    desired : float
        The value expected.
    tolerance : float
        Maximum relative error ``(actual - desired) / desired``.
    """
    try:
        actual[0]
    except (TypeError, IndexError):
        if isnan(actual) and not isnan(desired):
            test_case.fail('actual nan, desired %s, rel error nan, tolerance %s'
                           % (desired, tolerance))
        if desired != 0:
            error = (actual - desired) / desired
        else:
            error = actual
        if abs(error) > tolerance:
            test_case.fail('actual %s, desired %s, rel error %s, tolerance %s'
                           % (actual, desired, error, tolerance))
    else: #array values
        if not np.all(np.isnan(actual)==np.isnan(desired)):
            test_case.fail('actual and desired values have non-matching nan values')

        if np.linalg.norm(desired) == 0:
            error = np.linalg.norm(actual)
        else:
            error = np.linalg.norm(actual - desired) / np.linalg.norm(desired)

        if abs(error) > tolerance:
            test_case.fail('arrays do not match, rel error %.3e > tol (%.3e)'  % (error, tolerance))

    return error


class SimpleComp(ExplicitComponent):
    """ The simplest component you can imagine. """

    def __init__(self, multiplier=2.0):
        super(SimpleComp, self).__init__()

        self.multiplier = multiplier

        # Params
        self.add_param('x', 3.0)

        # Unknowns
        self.add_output('y', 5.5)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """
        unknowns['y'] = self.multiplier*params['x']


class SimpleCompDerivMatVec(SimpleComp):
    """ The simplest component you can imagine, this time with derivatives
    defined using apply_linear. """

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids,
                     mode):
        """Returns the product of the incoming vector with the Jacobian."""

        if mode == 'fwd':
            dresids['y'] += self.multiplier*dparams['x']

        elif mode == 'rev':
            dparams['x'] = self.multiplier*dresids['y']


class TestPetscKSPSerial(unittest.TestCase):

    def setUp(self):
        if impl is None:
            raise unittest.SkipTest("Can't run this test (even in serial) without mpi4py and petsc4py")

    def test_simple(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem(impl=impl)
        prob.root = group
        prob.root.ln_solver = PetscKSP()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_simple_choose_different_alg(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem(impl=impl)
        prob.root = group
        prob.root.ln_solver = PetscKSP()
        prob.root.ln_solver.options['ksp_type'] = 'gmres'
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_simple_matvec_subbed(self):
        group = Group()
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem(impl=impl)
        prob.root = Group()
        prob.root.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        prob.root.add('sub', group, promotes=['*'])

        prob.root.ln_solver = PetscKSP()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='fd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_simple_matvec_subbed_like_multipoint(self):
        group = Group()
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem(impl=impl)
        prob.root = Group()
        prob.root.add('sub', group, promotes=['*'])
        prob.root.sub.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])

        prob.root.ln_solver = PetscKSP()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='fd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='fd', return_format='array')
        assert_rel_error(self, J[0][0], 2.0, 1e-6)

    # def test_array2D(self):
    #     group = Group()
    #     group.add('x_param', IndepVarComp('x', np.ones((2, 2))), promotes=['*'])
    #     group.add('mycomp', ArrayComp2D(), promotes=['x', 'y'])

    #     prob = Problem(impl=impl)
    #     prob.root = group
    #     prob.root.ln_solver = PetscKSP()
    #     prob.setup(check=False)
    #     prob.run()

    #     J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
    #     Jbase = prob.root.mycomp._jacobian_cache
    #     diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x'])
    #     assert_rel_error(self, diff, 0.0, 1e-8)

    #     J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
    #     diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x'])
    #     assert_rel_error(self, diff, 0.0, 1e-8)

    # def test_double_arraycomp(self):
    #     # Mainly testing a bug in the array return for multiple arrays

    #     group = Group()
    #     group.add('x_param1', IndepVarComp('x1', np.ones((2))), promotes=['*'])
    #     group.add('x_param2', IndepVarComp('x2', np.ones((2))), promotes=['*'])
    #     group.add('mycomp', DoubleArrayComp(), promotes=['*'])

    #     prob = Problem(impl=impl)
    #     prob.root = group
    #     prob.root.ln_solver = PetscKSP()
    #     prob.setup(check=False)
    #     prob.run()

    #     Jbase = group.mycomp.JJ

    #     J = prob.calc_gradient(['x1', 'x2'], ['y1', 'y2'], mode='fwd',
    #                            return_format='array')
    #     diff = np.linalg.norm(J - Jbase)
    #     assert_rel_error(self, diff, 0.0, 1e-8)

    #     J = prob.calc_gradient(['x1', 'x2'], ['y1', 'y2'], mode='fd',
    #                            return_format='array')
    #     diff = np.linalg.norm(J - Jbase)
    #     assert_rel_error(self, diff, 0.0, 1e-8)

    #     J = prob.calc_gradient(['x1', 'x2'], ['y1', 'y2'], mode='rev',
    #                            return_format='array')
    #     diff = np.linalg.norm(J - Jbase)
    #     assert_rel_error(self, diff, 0.0, 1e-8)

    def test_simple_in_group_matvec(self):
        group = Group()
        sub = group.add('sub', Group(), promotes=['x', 'y'])
        group.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        sub.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem(impl=impl)
        prob.root = group
        prob.root.ln_solver = PetscKSP()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_simple_jac(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        group.add('mycomp', ExecComp(['y=2.0*x']), promotes=['x', 'y'])

        prob = Problem(impl=impl)
        prob.root = group
        prob.root.ln_solver = PetscKSP()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    # def test_fan_out(self):
    #     prob = Problem(impl=impl)
    #     prob.root = FanOut()
    #     prob.root.ln_solver = PetscKSP()
    #     prob.setup(check=False)
    #     prob.run()

    #     indep_list = ['p.x']
    #     unknown_list = ['comp2.y', "comp3.y"]

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
    #     assert_rel_error(self, J['comp2.y']['p.x'][0][0], -6.0, 1e-6)
    #     assert_rel_error(self, J['comp3.y']['p.x'][0][0], 15.0, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
    #     assert_rel_error(self, J['comp2.y']['p.x'][0][0], -6.0, 1e-6)
    #     assert_rel_error(self, J['comp3.y']['p.x'][0][0], 15.0, 1e-6)

    # def test_fan_out_grouped(self):
    #     prob = Problem(impl=impl)
    #     prob.root = FanOutGrouped()
    #     prob.root.ln_solver = PetscKSP()
    #     prob.setup(check=False)
    #     prob.run()

    #     indep_list = ['p.x']
    #     unknown_list = ['sub.comp2.y', "sub.comp3.y"]

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
    #     assert_rel_error(self, J['sub.comp2.y']['p.x'][0][0], -6.0, 1e-6)
    #     assert_rel_error(self, J['sub.comp3.y']['p.x'][0][0], 15.0, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
    #     assert_rel_error(self, J['sub.comp2.y']['p.x'][0][0], -6.0, 1e-6)
    #     assert_rel_error(self, J['sub.comp3.y']['p.x'][0][0], 15.0, 1e-6)

    # def test_fan_in(self):
    #     prob = Problem(impl=impl)
    #     prob.root = FanIn()
    #     prob.root.ln_solver = PetscKSP()
    #     prob.setup(check=False)
    #     prob.run()

    #     indep_list = ['p1.x1', 'p2.x2']
    #     unknown_list = ['comp3.y']

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
    #     assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
    #     assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
    #     assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
    #     assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    # def test_fan_in_grouped(self):
    #     prob = Problem(impl=impl)
    #     prob.root = FanInGrouped()
    #     prob.root.ln_solver = PetscKSP()

    #     indep_list = ['p1.x1', 'p2.x2']
    #     unknown_list = ['comp3.y']

    #     prob.setup(check=False)
    #     prob.run()

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
    #     assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
    #     assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
    #     assert_rel_error(self, J['comp3.y']['p1.x1'][0][0], -6.0, 1e-6)
    #     assert_rel_error(self, J['comp3.y']['p2.x2'][0][0], 35.0, 1e-6)

    # def test_converge_diverge(self):
    #     prob = Problem(impl=impl)
    #     prob.root = ConvergeDiverge()
    #     prob.root.ln_solver = PetscKSP()
    #     prob.setup(check=False)
    #     prob.run()

    #     indep_list = ['p.x']
    #     unknown_list = ['comp7.y1']

    #     prob.run()

    #     # Make sure value is fine.
    #     assert_rel_error(self, prob['comp7.y1'], -102.7, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
    #     assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
    #     assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
    #     assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    # def test_analysis_error(self):
    #     prob = Problem(impl=impl)
    #     prob.root = ConvergeDiverge()
    #     prob.root.ln_solver = PetscKSP()
    #     prob.root.ln_solver.options['maxiter'] = 2
    #     prob.root.ln_solver.options['err_on_maxiter'] = True

    #     prob.setup(check=False)
    #     prob.run()

    #     indep_list = ['p.x']
    #     unknown_list = ['comp7.y1']

    #     prob.run()

    #     # Make sure value is fine.
    #     assert_rel_error(self, prob['comp7.y1'], -102.7, 1e-6)

    #     try:
    #         J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
    #     except AnalysisError as err:
    #         self.assertEqual(str(err), "Solve in '': PetscKSP FAILED to converge in 3 iterations")
    #     else:
    #         self.fail("expected AnalysisError")

    # def test_converge_diverge_groups(self):

    #     prob = Problem(impl=impl)
    #     prob.root = ConvergeDivergeGroups()
    #     prob.root.ln_solver = PetscKSP()
    #     prob.setup(check=False)
    #     prob.run()

    #     # Make sure value is fine.
    #     assert_rel_error(self, prob['comp7.y1'], -102.7, 1e-6)

    #     indep_list = ['p.x']
    #     unknown_list = ['comp7.y1']

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
    #     assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
    #     assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
    #     assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    # def test_single_diamond(self):

    #     prob = Problem(impl=impl)
    #     prob.root = SingleDiamond()
    #     prob.root.ln_solver = PetscKSP()
    #     prob.setup(check=False)
    #     prob.run()

    #     indep_list = ['p.x']
    #     unknown_list = ['comp4.y1', 'comp4.y2']

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
    #     assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
    #     assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
    #     assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
    #     assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

    # def test_single_diamond_grouped(self):

    #     prob = Problem(impl=impl)
    #     prob.root = SingleDiamondGrouped()
    #     prob.root.ln_solver = PetscKSP()
    #     prob.setup(check=False)
    #     prob.run()

    #     indep_list = ['p.x']
    #     unknown_list = ['comp4.y1', 'comp4.y2']

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
    #     assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
    #     assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
    #     assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
    #     assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
    #     assert_rel_error(self, J['comp4.y1']['p.x'][0][0], 25, 1e-6)
    #     assert_rel_error(self, J['comp4.y2']['p.x'][0][0], -40.5, 1e-6)

    # def test_sellar_derivs_grouped(self):

    #     prob = Problem(impl=impl)
    #     prob.root = SellarDerivativesGrouped()
    #     prob.root.ln_solver = PetscKSP()

    #     prob.root.mda.nl_solver.options['atol'] = 1e-12
    #     prob.setup(check=False)
    #     prob.run()

    #     # Just make sure we are at the right answer
    #     assert_rel_error(self, prob['y1'], 25.58830273, .00001)
    #     assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    #     indep_list = ['x', 'z']
    #     unknown_list = ['obj', 'con1', 'con2']

    #     Jbase = {}
    #     Jbase['con1'] = {}
    #     Jbase['con1']['x'] = -0.98061433
    #     Jbase['con1']['z'] = np.array([-9.61002285, -0.78449158])
    #     Jbase['con2'] = {}
    #     Jbase['con2']['x'] = 0.09692762
    #     Jbase['con2']['z'] = np.array([1.94989079, 1.0775421 ])
    #     Jbase['obj'] = {}
    #     Jbase['obj']['x'] = 2.98061392
    #     Jbase['obj']['z'] = np.array([9.61001155, 1.78448534])

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
    #     for key1, val1 in Jbase.items():
    #         for key2, val2 in val1.items():
    #             assert_rel_error(self, J[key1][key2], val2, .00001)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
    #     for key1, val1 in Jbase.items():
    #         for key2, val2 in val1.items():
    #             assert_rel_error(self, J[key1][key2], val2, .00001)

    #     # Cheat a bit so I can twiddle mode
    #     OptionsDictionary.locked = False

    #     prob.root.deriv_options['form'] = 'central'
    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
    #     for key1, val1 in Jbase.items():
    #         for key2, val2 in val1.items():
    #             assert_rel_error(self, J[key1][key2], val2, .00001)

    def test_nested_relevancy_gmres(self):
        # This test is just to make sure that values in the dp vector from
        # higher scopes aren't sitting there converting themselves during sub
        # iterations.
        prob = Problem(impl=impl)
        root = prob.root = Group()
        root.add('p1', IndepVarComp('xx', 3.0))
        root.add('c1', ExecComp(['y1=0.5*x + 1.0*xx', 'y2=0.3*x - 1.0*xx'], units={'y2': 'km'}))
        root.add('c2', ExecComp(['y=0.5*x']))
        sub = root.add('sub', Group())
        sub.add('cc1', ExecComp(['y=1.01*x1 + 1.01*x2'], units={'x1': 'fm'}))
        sub.add('cc2', ExecComp(['y=1.01*x']))

        root.connect('p1.xx', 'c1.xx')
        root.connect('c1.y1', 'c2.x')
        root.connect('c2.y', 'c1.x')
        root.connect('c1.y2', 'sub.cc1.x1')
        root.connect('sub.cc1.y', 'sub.cc2.x')
        root.connect('sub.cc2.y', 'sub.cc1.x2')

        root.nl_solver = NewtonSolver()
        root.nl_solver.options['maxiter'] = 1
        root.ln_solver = PetscKSP()
        root.ln_solver.options['maxiter'] = 1

        sub.nl_solver = NewtonSolver()
        #sub.nl_solver.options['maxiter'] = 7
        sub.ln_solver = PetscKSP()

        prob.driver.add_desvar('p1.xx')
        prob.driver.add_objective('sub.cc2.y')

        prob.setup(check=False)
        prob.print_all_convergence()

        prob.run()

        # GMRES doesn't cause a successive build-up in the value of an out-of
        # scope param, but the linear solver doesn't converge. We can test to
        # make sure it does.
        iter_count = sub.ln_solver.iter_count
        self.assertTrue(iter_count < 20)
        self.assertTrue(not np.isnan(prob['sub.cc2.y']))


class TestPetscKSPPreconditioner(unittest.TestCase):

    def setUp(self):
        if impl is None:
            raise unittest.SkipTest("Can't run this test (even in serial) without mpi4py and petsc4py")

    # def test_sellar_derivs_grouped_precon(self):
    #     prob = Problem(impl=impl)
    #     prob.root = SellarDerivativesGrouped()

    #     prob.root.mda.nl_solver.options['atol'] = 1e-12
    #     prob.root.ln_solver = PetscKSP()
    #     prob.root.ln_solver.preconditioner = LinearGaussSeidel()
    #     prob.root.mda.ln_solver = LinearSolver()
    #     prob.setup(check=False)
    #     prob.run()

    #     # Just make sure we are at the right answer
    #     assert_rel_error(self, prob['y1'], 25.58830273, .00001)
    #     assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    #     indep_list = ['x', 'z']
    #     unknown_list = ['obj', 'con1', 'con2']

    #     Jbase = {}
    #     Jbase['con1'] = {}
    #     Jbase['con1']['x'] = -0.98061433
    #     Jbase['con1']['z'] = np.array([-9.61002285, -0.78449158])
    #     Jbase['con2'] = {}
    #     Jbase['con2']['x'] = 0.09692762
    #     Jbase['con2']['z'] = np.array([1.94989079, 1.0775421 ])
    #     Jbase['obj'] = {}
    #     Jbase['obj']['x'] = 2.98061392
    #     Jbase['obj']['z'] = np.array([9.61001155, 1.78448534])

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
    #     for key1, val1 in Jbase.items():
    #         for key2, val2 in val1.items():
    #             assert_rel_error(self, J[key1][key2], val2, .00001)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
    #     for key1, val1 in Jbase.items():
    #         for key2, val2 in val1.items():
    #             assert_rel_error(self, J[key1][key2], val2, .00001)

    # def test_converge_diverge_groups(self):
    #     prob = Problem(impl=impl)
    #     prob.root = ConvergeDivergeGroups()
    #     prob.root.ln_solver = PetscKSP()
    #     prob.root.ln_solver.preconditioner = LinearGaussSeidel()

    #     prob.root.sub1.ln_solver = LinearSolver()
    #     prob.root.sub3.ln_solver = LinearSolver()

    #     prob.setup(check=False)
    #     prob.run()

    #     # Make sure value is fine.
    #     assert_rel_error(self, prob['comp7.y1'], -102.7, 1e-6)

    #     indep_list = ['p.x']
    #     unknown_list = ['comp7.y1']

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
    #     assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
    #     assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
    #     assert_rel_error(self, J['comp7.y1']['p.x'][0][0], -40.75, 1e-6)

    # def test_fan_out_all_grouped(self):
    #     prob = Problem(impl=impl)
    #     prob.root = FanOutAllGrouped()
    #     prob.root.ln_solver = PetscKSP()

    #     prob.root.ln_solver.preconditioner = LinearGaussSeidel()
    #     prob.root.sub1.ln_solver = LinearSolver()
    #     prob.root.sub2.ln_solver = LinearSolver()
    #     prob.root.sub3.ln_solver = LinearSolver()

    #     prob.setup(check=False)
    #     prob.run()

    #     indep_list = ['p.x']
    #     unknown_list = ['sub2.comp2.y', "sub3.comp3.y"]

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
    #     assert_rel_error(self, J['sub2.comp2.y']['p.x'][0][0], -6.0, 1e-6)
    #     assert_rel_error(self, J['sub3.comp3.y']['p.x'][0][0], 15.0, 1e-6)

    #     J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
    #     assert_rel_error(self, J['sub2.comp2.y']['p.x'][0][0], -6.0, 1e-6)
    #     assert_rel_error(self, J['sub3.comp3.y']['p.x'][0][0], 15.0, 1e-6)


if __name__ == "__main__":
    unittest.main()
