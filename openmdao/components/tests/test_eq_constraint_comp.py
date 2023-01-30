import unittest
import warnings

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from openmdao.test_suite.components.sellar_feature import SellarIDF
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials


class TestEQConstraintComp(unittest.TestCase):

    def test_sellar_idf(self):
        prob = om.Problem(SellarIDF())
        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False)
        prob.setup()

        # check derivatives
        prob['y1'] = 100
        prob['equal.rhs:y1'] = 1

        prob.run_model()

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

        # check results
        prob.run_driver()

        assert_near_equal(prob['x'], 0., 1e-5)
        assert_near_equal(prob['z'], [1.977639, 0.], 1e-5)

        assert_near_equal(prob['obj_cmp.obj'], 3.18339395045, 1e-5)

        assert_almost_equal(prob['y1'], 3.16)
        assert_almost_equal(prob['d1.y1'], 3.16)

        assert_almost_equal(prob['y2'], 3.7552778)
        assert_almost_equal(prob['d2.y2'], 3.7552778)

        assert_almost_equal(prob['equal.y1'], 0.0)
        assert_almost_equal(prob['equal.y2'], 0.0)

    def test_create_on_init(self):
        prob = om.Problem()
        model = prob.model

        # find intersection of two non-parallel lines
        model.add_subsystem('indep', om.IndepVarComp('x', val=0.))
        model.add_subsystem('f', om.ExecComp('y=3*x-3', x=0.))
        model.add_subsystem('g', om.ExecComp('y=2.3*x+4', x=0.))
        model.add_subsystem('equal', om.EQConstraintComp('y', val=11.))

        model.connect('indep.x', 'f.x')
        model.connect('indep.x', 'g.x')

        model.connect('f.y', 'equal.lhs:y')
        model.connect('g.y', 'equal.rhs:y')

        model.add_design_var('indep.x', lower=0., upper=20.)
        model.add_objective('f.y')

        prob.setup(mode='fwd')

        # verify that the output variable has been initialized
        self.assertEqual(prob['equal.y'], 11.)

        # verify that the constraint has not been added
        self.assertFalse('equal.y' in model.get_constraints())

        # manually add the constraint
        model.add_constraint('equal.y', equals=0.)
        prob.setup(mode='fwd')

        prob.driver = om.ScipyOptimizeDriver(disp=False)

        prob.run_driver()

        assert_almost_equal(prob['equal.y'], 0.)
        assert_almost_equal(prob['indep.x'], 10.)
        assert_almost_equal(prob['f.y'], 27.)
        assert_almost_equal(prob['g.y'], 27.)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_create_on_init_add_constraint(self):
        prob = om.Problem()
        model = prob.model

        # find intersection of two non-parallel lines
        model.add_subsystem('indep', om.IndepVarComp('x', val=0.))
        model.add_subsystem('f', om.ExecComp('y=3*x-3', x=0.))
        model.add_subsystem('g', om.ExecComp('y=2.3*x+4', x=0.))
        model.add_subsystem('equal', om.EQConstraintComp('y', add_constraint=True))

        model.connect('indep.x', 'f.x')
        model.connect('indep.x', 'g.x')

        model.connect('f.y', 'equal.lhs:y')
        model.connect('g.y', 'equal.rhs:y')

        model.add_design_var('indep.x', lower=0., upper=20.)
        model.add_objective('f.y')

        prob.setup(mode='fwd')

        # verify that the constraint has been added as requested
        self.assertTrue('equal.y' in model.get_constraints())

        prob.driver = om.ScipyOptimizeDriver(disp=False)

        prob.run_driver()

        assert_almost_equal(prob['equal.y'], 0.)
        assert_almost_equal(prob['indep.x'], 10.)
        assert_almost_equal(prob['f.y'], 27.)
        assert_almost_equal(prob['g.y'], 27.)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_create_on_init_add_constraint_no_normalization(self):
        prob = om.Problem()
        model = prob.model

        # find intersection of two non-parallel lines
        model.add_subsystem('indep', om.IndepVarComp('x', val=-2.0))
        model.add_subsystem('f', om.ExecComp('y=3*x-3', x=0.))
        model.add_subsystem('g', om.ExecComp('y=2.3*x+4', x=0.))
        model.add_subsystem('equal', om.EQConstraintComp('y', add_constraint=True, normalize=False,
                                                         ref0=0, ref=100.0, linear=True, cache_linear_solution=True,
                                                         alias='eq_constraint'))

        model.connect('indep.x', 'f.x')
        model.connect('indep.x', 'g.x')

        model.connect('f.y', 'equal.lhs:y')
        model.connect('g.y', 'equal.rhs:y')

        model.add_design_var('indep.x', lower=0., upper=20.)
        model.add_objective('f.y')

        prob.setup(mode='fwd')

        # verify that the constraint has been added as requested
        self.assertTrue('eq_constraint' in model.get_constraints())

        # verify that the output is not being normalized
        prob.run_model()
        lhs = prob['f.y']
        rhs = prob['g.y']
        diff = lhs - rhs
        assert_near_equal(prob['equal.y'], diff)

        prob.driver = om.ScipyOptimizeDriver(disp=False)

        prob.run_driver()

        # verify that options are added
        self.assertEqual(prob.driver._cons['eq_constraint']['linear'], True)
        self.assertEqual(prob.driver._cons['eq_constraint']['cache_linear_solution'], True)
        self.assertEqual(prob.driver._cons['eq_constraint']['alias'], 'eq_constraint')

        assert_almost_equal(prob['equal.y'], 0.)
        assert_almost_equal(prob['indep.x'], 10.)
        assert_almost_equal(prob['f.y'], 27.)
        assert_almost_equal(prob['g.y'], 27.)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_vectorized(self):
        prob = om.Problem()
        model = prob.model

        n = 100

        # find intersection of two non-parallel lines, vectorized
        model.add_subsystem('indep', om.IndepVarComp('x', val=np.ones(n)))
        model.add_subsystem('f', om.ExecComp('y=3*x-3', x=np.ones(n), y=np.ones(n)))
        model.add_subsystem('g', om.ExecComp('y=2.3*x+4', x=np.ones(n), y=np.ones(n)))
        model.add_subsystem('equal', om.EQConstraintComp('y', val=np.ones(n), add_constraint=True))
        model.add_subsystem('obj_cmp', om.ExecComp('obj=sum(y)', y=np.zeros(n)))

        model.connect('indep.x', 'f.x')
        model.connect('indep.x', 'g.x')
        model.connect('f.y', 'equal.lhs:y')
        model.connect('g.y', 'equal.rhs:y')
        model.connect('f.y', 'obj_cmp.y')

        model.add_design_var('indep.x', lower=np.zeros(n), upper=20.*np.ones(n))
        model.add_objective('obj_cmp.obj')

        prob.setup(mode='fwd')

        prob.driver = om.ScipyOptimizeDriver(disp=False)

        prob.run_driver()

        assert_almost_equal(prob['equal.y'], np.zeros(n))
        assert_almost_equal(prob['indep.x'], np.ones(n)*10.)
        assert_almost_equal(prob['f.y'], np.ones(n)*27.)
        assert_almost_equal(prob['g.y'], np.ones(n)*27.)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_set_shape(self):
        prob = om.Problem()
        model = prob.model

        n = 100

        # find intersection of two non-parallel lines, vectorized
        model.add_subsystem('indep', om.IndepVarComp('x', val=np.ones(n)))
        model.add_subsystem('f', om.ExecComp('y=3*x-3', x=np.ones(n), y=np.ones(n)))
        model.add_subsystem('g', om.ExecComp('y=2.3*x+4', x=np.ones(n), y=np.ones(n)))
        model.add_subsystem('equal', om.EQConstraintComp('y', shape=(n,), add_constraint=True))
        model.add_subsystem('obj_cmp', om.ExecComp('obj=sum(y)', y=np.zeros(n)))

        model.connect('indep.x', 'f.x')
        model.connect('indep.x', 'g.x')
        model.connect('f.y', 'equal.lhs:y')
        model.connect('g.y', 'equal.rhs:y')
        model.connect('f.y', 'obj_cmp.y')

        model.add_design_var('indep.x', lower=np.zeros(n), upper=20.*np.ones(n))
        model.add_objective('obj_cmp.obj')

        prob.setup(mode='fwd')

        prob.driver = om.ScipyOptimizeDriver(disp=False)

        prob.run_driver()

        assert_almost_equal(prob['equal.y'], np.zeros(n))
        assert_almost_equal(prob['indep.x'], np.ones(n)*10.)
        assert_almost_equal(prob['f.y'], np.ones(n)*27.)
        assert_almost_equal(prob['g.y'], np.ones(n)*27.)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_vectorized_no_normalization(self):
        prob = om.Problem()
        model = prob.model

        n = 100

        # find intersection of two non-parallel lines, vectorized
        model.add_subsystem('indep', om.IndepVarComp('x', val=-2.0*np.ones(n)))
        model.add_subsystem('f', om.ExecComp('y=3*x-3', x=np.ones(n), y=np.ones(n)))
        model.add_subsystem('g', om.ExecComp('y=2.3*x+4', x=np.ones(n), y=np.ones(n)))
        model.add_subsystem('equal', om.EQConstraintComp('y', val=np.ones(n), add_constraint=True,
                                                         normalize=False))
        model.add_subsystem('obj_cmp', om.ExecComp('obj=sum(y)', y=np.zeros(n)))

        model.connect('indep.x', 'f.x')
        model.connect('indep.x', 'g.x')
        model.connect('f.y', 'equal.lhs:y')
        model.connect('g.y', 'equal.rhs:y')
        model.connect('f.y', 'obj_cmp.y')

        model.add_design_var('indep.x', lower=np.zeros(n), upper=20.*np.ones(n))
        model.add_objective('obj_cmp.obj')

        prob.setup(mode='fwd')

        prob.driver = om.ScipyOptimizeDriver(disp=False)

        # verify that the output is not being normalized
        prob.run_model()
        lhs = prob['f.y']
        rhs = prob['g.y']
        diff = lhs - rhs
        assert_near_equal(prob['equal.y'], diff)

        prob.run_driver()

        assert_almost_equal(prob['equal.y'], np.zeros(n))
        assert_almost_equal(prob['indep.x'], np.ones(n)*10.)
        assert_almost_equal(prob['f.y'], np.ones(n)*27.)
        assert_almost_equal(prob['g.y'], np.ones(n)*27.)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_scalar_with_mult(self):
        prob = om.Problem()
        model = prob.model

        # find where 2*x == x^2
        model.add_subsystem('indep', om.IndepVarComp('x', val=1.))
        model.add_subsystem('multx', om.IndepVarComp('m', val=2.))
        model.add_subsystem('f', om.ExecComp('y=x**2', x=1.))
        model.add_subsystem('equal', om.EQConstraintComp('y', use_mult=True))

        model.connect('indep.x', 'f.x')

        model.connect('indep.x', 'equal.lhs:y')
        model.connect('multx.m', 'equal.mult:y')
        model.connect('f.y', 'equal.rhs:y')

        model.add_design_var('indep.x', lower=0., upper=10.)
        model.add_constraint('equal.y', equals=0.)
        model.add_objective('f.y')

        prob.setup(mode='fwd')
        prob.driver = om.ScipyOptimizeDriver(disp=False)
        prob.run_driver()

        assert_near_equal(prob['equal.y'], 0., 1e-6)
        assert_near_equal(prob['indep.x'], 2., 1e-6)
        assert_near_equal(prob['f.y'], 4., 1e-6)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_complex_step(self):
        prob = om.Problem()
        model = prob.model

        # find where 2*x == x^2
        model.add_subsystem('indep', om.IndepVarComp('x', val=1.))
        model.add_subsystem('multx', om.IndepVarComp('m', val=2.))
        model.add_subsystem('f', om.ExecComp('y=x**2', x=1.))
        model.add_subsystem('equal', om.EQConstraintComp('y', use_mult=True))

        model.connect('indep.x', 'f.x')

        model.connect('indep.x', 'equal.lhs:y')
        model.connect('multx.m', 'equal.mult:y')
        model.connect('f.y', 'equal.rhs:y')

        model.add_design_var('indep.x', lower=0., upper=10.)
        model.add_constraint('equal.y', equals=0.)
        model.add_objective('f.y')

        prob.setup(mode='fwd', force_alloc_complex=True)
        prob.driver = om.ScipyOptimizeDriver(disp=False)
        prob.run_driver()

        with warnings.catch_warnings():
            warnings.filterwarnings(action="error", category=np.ComplexWarning)
            cpd = prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(cpd, atol=1e-10, rtol=1e-10)

    def test_vectorized_with_mult(self):
        prob = om.Problem()
        model = prob.model

        n = 100

        # find where 2*x == x^2, vectorized
        model.add_subsystem('indep', om.IndepVarComp('x', val=np.ones(n)))
        model.add_subsystem('multx', om.IndepVarComp('m', val=np.ones(n)*2.))
        model.add_subsystem('f', om.ExecComp('y=x**2', x=np.ones(n), y=np.ones(n)))
        model.add_subsystem('equal', om.EQConstraintComp('y', val=np.ones(n),
                            use_mult=True, add_constraint=True))
        model.add_subsystem('obj_cmp', om.ExecComp('obj=sum(y)', y=np.zeros(n)))

        model.connect('indep.x', 'f.x')

        model.connect('indep.x', 'equal.lhs:y')
        model.connect('multx.m', 'equal.mult:y')
        model.connect('f.y', 'equal.rhs:y')
        model.connect('f.y', 'obj_cmp.y')

        model.add_design_var('indep.x', lower=np.zeros(n), upper=np.ones(n)*10.)
        model.add_objective('obj_cmp.obj')

        prob.setup(mode='fwd')
        prob.driver = om.ScipyOptimizeDriver(disp=False)
        prob.run_driver()

        assert_near_equal(prob['equal.y'], np.zeros(n), 1e-6)
        assert_near_equal(prob['indep.x'], np.ones(n)*2., 1e-6)
        assert_near_equal(prob['f.y'], np.ones(n)*4., 1e-6)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_vectorized_with_default_mult(self):
        prob = om.Problem()
        model = prob.model

        n = 100

        # find where 2*x == x^2, vectorized
        model.add_subsystem('indep', om.IndepVarComp('x', val=np.ones(n)))
        model.add_subsystem('f', om.ExecComp('y=x**2', x=np.ones(n), y=np.ones(n)))
        model.add_subsystem('equal', om.EQConstraintComp('y', val=np.ones(n),
                            use_mult=True, mult_val=2., add_constraint=True))
        model.add_subsystem('obj_cmp', om.ExecComp('obj=sum(y)', y=np.zeros(n)))

        model.connect('indep.x', 'f.x')

        model.connect('indep.x', 'equal.lhs:y')
        model.connect('f.y', 'equal.rhs:y')
        model.connect('f.y', 'obj_cmp.y')

        model.add_design_var('indep.x', lower=np.zeros(n), upper=np.ones(n)*10.)
        model.add_objective('obj_cmp.obj')

        prob.setup(mode='fwd')
        prob.driver = om.ScipyOptimizeDriver(disp=False)
        prob.run_driver()

        assert_near_equal(prob['equal.y'], np.zeros(n), 1e-6)
        assert_near_equal(prob['indep.x'], np.ones(n)*2., 1e-6)
        assert_near_equal(prob['f.y'], np.ones(n)*4., 1e-6)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_rhs_val(self):
        prob = om.Problem()
        model = prob.model

        # find where x^2 == 4
        model.add_subsystem('indep', om.IndepVarComp('x', val=1.))
        model.add_subsystem('f', om.ExecComp('y=x**2', x=1.))
        model.add_subsystem('equal', om.EQConstraintComp('y', rhs_val=4.))

        model.connect('indep.x', 'f.x')
        model.connect('f.y', 'equal.lhs:y')

        model.add_design_var('indep.x', lower=0., upper=10.)
        model.add_constraint('equal.y', equals=0.)
        model.add_objective('f.y')

        prob.setup(mode='fwd')
        prob.driver = om.ScipyOptimizeDriver(disp=False)
        prob.run_driver()

        assert_near_equal(prob['equal.y'], 0., 1e-6)
        assert_near_equal(prob['indep.x'], 2., 1e-6)
        assert_near_equal(prob['f.y'], 4., 1e-6)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_vectorized_rhs_val(self):
        prob = om.Problem()
        model = prob.model

        n = 100

        # find where x^2 == 4, vectorized
        model.add_subsystem('indep', om.IndepVarComp('x', val=np.ones(n)))
        model.add_subsystem('f', om.ExecComp('y=x**2', x=np.ones(n), y=np.ones(n)))
        model.add_subsystem('equal', om.EQConstraintComp('y', val=np.ones(n),
                            rhs_val=np.ones(n)*4., use_mult=True, mult_val=2.))
        model.add_subsystem('obj_cmp', om.ExecComp('obj=sum(y)', y=np.zeros(n)))

        model.connect('indep.x', 'f.x')

        model.connect('indep.x', 'equal.lhs:y')
        model.connect('f.y', 'obj_cmp.y')

        model.add_design_var('indep.x', lower=np.zeros(n), upper=np.ones(n)*10.)
        model.add_constraint('equal.y', equals=0.)
        model.add_objective('obj_cmp.obj')

        prob.setup(mode='fwd')
        prob.driver = om.ScipyOptimizeDriver(disp=False)
        prob.run_driver()

        assert_near_equal(prob['equal.y'], np.zeros(n), 1e-6)
        assert_near_equal(prob['indep.x'], np.ones(n)*2., 1e-6)
        assert_near_equal(prob['f.y'], np.ones(n)*4., 1e-6)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=2e-5, rtol=2e-5)

    def test_specified_shape_rhs_val(self):
        prob = om.Problem()
        model = prob.model

        shape = (3, 2, 4)

        rhs = np.zeros(shape)

        model.add_subsystem('indep', om.IndepVarComp('x', val=np.ones(shape)))
        model.add_subsystem('equal', om.EQConstraintComp('y', val=np.ones(shape),
                                                         rhs_val=rhs))

        model.connect('indep.x', 'equal.lhs:y')

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['equal.y'], np.ones(shape) - rhs, 1e-6)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_renamed_vars(self):
        prob = om.Problem()
        model = prob.model

        # find intersection of two non-parallel lines, fx_y and gx_y
        equal = om.EQConstraintComp('y', lhs_name='fx_y', rhs_name='gx_y',
                                    add_constraint=True)

        model.add_subsystem('indep', om.IndepVarComp('x', val=0.))
        model.add_subsystem('f', om.ExecComp('y=3*x-3', x=0.))
        model.add_subsystem('g', om.ExecComp('y=2.3*x+4', x=0.))
        model.add_subsystem('equal', equal)

        model.connect('indep.x', 'f.x')
        model.connect('indep.x', 'g.x')

        model.connect('f.y', 'equal.fx_y')
        model.connect('g.y', 'equal.gx_y')

        model.add_design_var('indep.x', lower=0., upper=20.)
        model.add_objective('f.y')

        prob.setup(mode='fwd')
        prob.driver = om.ScipyOptimizeDriver(disp=False)
        prob.run_driver()

        assert_almost_equal(prob['equal.y'], 0.)
        assert_almost_equal(prob['indep.x'], 10.)
        assert_almost_equal(prob['f.y'], 27.)
        assert_almost_equal(prob['g.y'], 27.)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_multidimentional_shape_normalize(self):
        prob = om.Problem()
        model = prob.model

        n = 15

        defect_comp = model.add_subsystem('eq_comp', om.EQConstraintComp())

        defect_comp.add_eq_output(name='diff_r',
                                  shape=(n, 3),
                                  eq_units='km',
                                  lhs_name='r1', rhs_name='r2',
                                  use_mult=False,
                                  normalize=True,
                                  add_constraint=True,
                                  ref=100.0, ref0=None, adder=None, scaler=None)

        defect_comp.add_eq_output(name='diff_k',
                                  shape=(n,),
                                  eq_units='km',
                                  lhs_name='k1', rhs_name='k2',
                                  use_mult=False,
                                  normalize=True,
                                  add_constraint=True,
                                  ref=100.0, ref0=None, adder=None, scaler=None)


        prob.setup()

        prob.set_val('eq_comp.r1', np.random.rand(n, 3) * 10)
        prob.set_val('eq_comp.r2', np.random.rand(n, 3) * 10)

        prob.set_val('eq_comp.k1', np.random.rand(n) * 10)
        prob.set_val('eq_comp.k2', np.random.rand(n) * 10)

        prob.run_model()

    def test_multidimentional_shape(self):
        prob = om.Problem()
        model = prob.model

        n = 15

        defect_comp = model.add_subsystem('eq_comp', om.EQConstraintComp())

        defect_comp.add_eq_output(name='diff_r',
                                  shape=(n, 3),
                                  eq_units='km',
                                  lhs_name='r1', rhs_name='r2',
                                  use_mult=False,
                                  normalize=False,
                                  add_constraint=True,
                                  ref=100.0, ref0=None, adder=None, scaler=None)

        defect_comp.add_eq_output(name='diff_k',
                                  shape=(n,),
                                  eq_units='km',
                                  lhs_name='k1', rhs_name='k2',
                                  use_mult=False,
                                  normalize=False,
                                  add_constraint=True,
                                  ref=100.0, ref0=None, adder=None, scaler=None)


        prob.setup()

        prob.set_val('eq_comp.r1', np.random.rand(n, 3) * 10)
        prob.set_val('eq_comp.r2', np.random.rand(n, 3) * 10)

        prob.set_val('eq_comp.k1', np.random.rand(n) * 10)
        prob.set_val('eq_comp.k2', np.random.rand(n) * 10)

        prob.run_model()
    
    def test_indices(self):
        prob = om.Problem()
        model = prob.model

        n = 20

        model.add_subsystem('indep', om.IndepVarComp('x', np.ones(n)))
        model.add_subsystem('f', om.ExecComp('y=x**2', x=np.ones(n), y=np.ones(n)))
        model.add_subsystem('eq_comp', om.EQConstraintComp('y', val=np.ones(n), 
                            indices=[n-2,n-1],
                            flat_indices=True, add_constraint=True))

        model.connect('indep.x', 'f.x')
        model.add_design_var('indep.x')
        model.add_objective('f.y', index=0)

        prob.setup()

        prob.run_driver()

        assert_near_equal(prob.driver._cons['eq_comp.y']['indices']._arr, np.asarray([n-2,n-1], dtype=int))
        self.assertEqual(prob.driver._cons['eq_comp.y']['flat_indices'], True)




if __name__ == '__main__':  # pragma: no cover
    unittest.main()
