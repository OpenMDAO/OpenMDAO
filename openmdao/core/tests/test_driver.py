""" Unit tests for the Driver base class."""

from __future__ import print_function

from distutils.version import LooseVersion
from six import StringIO
import sys
import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp, Group, ExecComp, ScipyOptimizeDriver
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.general_utils import printoptions
from openmdao.test_suite.components.sellar import SellarDerivatives
from openmdao.test_suite.components.simple_comps import DoubleArrayComp, NonSquareArrayComp

class TestDriver(unittest.TestCase):

    def test_basic_get(self):

        prob = Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj')
        model.add_constraint('con1', lower=0)
        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_driver()

        designvars = prob.driver.get_design_var_values()
        self.assertEqual(designvars['pz.z'][0], 5.0 )

        designvars = prob.driver.get_objective_values()
        self.assertEqual(designvars['obj_cmp.obj'], prob['obj'] )

        designvars = prob.driver.get_constraint_values()
        self.assertEqual(designvars['con_cmp1.con1'], prob['con1']  )

    def test_scaled_design_vars(self):

        prob = Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z', ref=5.0, ref0=3.0)
        model.add_objective('obj')
        model.add_constraint('con1', lower=0)
        prob.set_solver_print(level=0)

        prob.setup(check=False)

        # Conclude setup but don't run model.
        prob.final_setup()

        dv = prob.driver.get_design_var_values()
        self.assertEqual(dv['pz.z'][0], 1.0)
        self.assertEqual(dv['pz.z'][1], -0.5)

        prob.driver.set_design_var('pz.z', np.array((2.0, -2.0)))
        self.assertEqual(prob['z'][0], 7.0)
        self.assertEqual(prob['z'][1], -1.0)

    def test_scaled_constraints(self):

        prob = Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj')
        model.add_constraint('con1', lower=0, ref=2.0, ref0=3.0)
        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        cv = prob.driver.get_constraint_values()['con_cmp1.con1'][0]
        base = prob['con1']
        self.assertEqual((base-3.0)/(2.0-3.0), cv)

    def test_scaled_objectves(self):

        prob = Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj', ref=2.0, ref0=3.0)
        model.add_constraint('con1', lower=0)
        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        cv = prob.driver.get_objective_values()['obj_cmp.obj'][0]
        base = prob['obj']
        self.assertEqual((base-3.0)/(2.0-3.0), cv)

    def test_scaled_derivs(self):

        prob = Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z', ref=2.0, ref0=0.0)
        model.add_objective('obj', ref=1.0, ref0=0.0)
        model.add_constraint('con1', lower=0, ref=2.0, ref0=0.0)
        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        base = prob.compute_totals(of=['obj', 'con1'], wrt=['z'])
        derivs = prob.driver._compute_totals(of=['obj_cmp.obj', 'con_cmp1.con1'], wrt=['pz.z'],
                                             return_format='dict')
        assert_rel_error(self, base[('con1', 'z')][0], derivs['con_cmp1.con1']['pz.z'][0], 1e-5)
        assert_rel_error(self, base[('obj', 'z')][0]*2.0, derivs['obj_cmp.obj']['pz.z'][0], 1e-5)

    def test_vector_scaled_derivs(self):

        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('px', IndepVarComp(name="x", val=np.ones((2, ))))
        comp = model.add_subsystem('comp', DoubleArrayComp())
        model.connect('px.x', 'comp.x1')

        model.add_design_var('px.x', ref=np.array([2.0, 3.0]), ref0=np.array([0.5, 1.5]))
        model.add_objective('comp.y1', ref=np.array([[7.0, 11.0]]), ref0=np.array([5.2, 6.3]))
        model.add_constraint('comp.y2', lower=0.0, upper=1.0, ref=np.array([[2.0, 4.0]]), ref0=np.array([1.2, 2.3]))

        prob.setup(check=False)
        prob.run_driver()

        derivs = prob.driver._compute_totals(of=['comp.y1'], wrt=['px.x'],
                                             return_format='dict')

        oscale = np.array([1.0/(7.0-5.2), 1.0/(11.0-6.3)])
        iscale = np.array([2.0-0.5, 3.0-1.5])
        J = comp.JJ[0:2, 0:2]

        # doing this manually so that I don't inadvertantly make an error in the vector math in both the code and test.
        J[0, 0] *= oscale[0]*iscale[0]
        J[0, 1] *= oscale[0]*iscale[1]
        J[1, 0] *= oscale[1]*iscale[0]
        J[1, 1] *= oscale[1]*iscale[1]
        assert_rel_error(self, J, derivs['comp.y1']['px.x'], 1.0e-3)

        obj = prob.driver.get_objective_values()
        obj_base = np.array([ (prob['comp.y1'][0]-5.2)/(7.0-5.2), (prob['comp.y1'][1]-6.3)/(11.0-6.3) ])
        assert_rel_error(self, obj['comp.y1'], obj_base, 1.0e-3)

        con = prob.driver.get_constraint_values()
        con_base = np.array([ (prob['comp.y2'][0]-1.2)/(2.0-1.2), (prob['comp.y2'][1]-2.3)/(4.0-2.3) ])
        assert_rel_error(self, con['comp.y2'], con_base, 1.0e-3)

    def test_vector_scaled_derivs_diff_sizes(self):

        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('px', IndepVarComp(name="x", val=np.ones((2, ))))
        comp = model.add_subsystem('comp', NonSquareArrayComp())
        model.connect('px.x', 'comp.x1')

        model.add_design_var('px.x', ref=np.array([2.0, 3.0]), ref0=np.array([0.5, 1.5]))
        model.add_objective('comp.y1', ref=np.array([[7.0, 11.0, 2.0]]), ref0=np.array([5.2, 6.3, 1.2]))
        model.add_constraint('comp.y2', lower=0.0, upper=1.0, ref=np.array([[2.0]]), ref0=np.array([1.2]))

        prob.setup(check=False)
        prob.run_driver()

        derivs = prob.driver._compute_totals(of=['comp.y1'], wrt=['px.x'],
                                                   return_format='dict')

        oscale = np.array([1.0/(7.0-5.2), 1.0/(11.0-6.3), 1.0/(2.0-1.2)])
        iscale = np.array([2.0-0.5, 3.0-1.5])
        J = comp.JJ[0:3, 0:2]

        # doing this manually so that I don't inadvertantly make an error in the vector math in both the code and test.
        J[0, 0] *= oscale[0]*iscale[0]
        J[0, 1] *= oscale[0]*iscale[1]
        J[1, 0] *= oscale[1]*iscale[0]
        J[1, 1] *= oscale[1]*iscale[1]
        J[2, 0] *= oscale[2]*iscale[0]
        J[2, 1] *= oscale[2]*iscale[1]
        assert_rel_error(self, J, derivs['comp.y1']['px.x'], 1.0e-3)

        obj = prob.driver.get_objective_values()
        obj_base = np.array([ (prob['comp.y1'][0]-5.2)/(7.0-5.2), (prob['comp.y1'][1]-6.3)/(11.0-6.3), (prob['comp.y1'][2]-1.2)/(2.0-1.2) ])
        assert_rel_error(self, obj['comp.y1'], obj_base, 1.0e-3)

        con = prob.driver.get_constraint_values()
        con_base = np.array([ (prob['comp.y2'][0]-1.2)/(2.0-1.2)])
        assert_rel_error(self, con['comp.y2'], con_base, 1.0e-3)

    def test_debug_print_option(self):

        prob = Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj')
        model.add_constraint('con1', lower=0)
        model.add_constraint('con2', lower=0, linear=True)
        prob.set_solver_print(level=0)

        prob.setup(check=False)

        # Make sure nothing prints if debug_print is the default of empty list
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            prob.run_driver()
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')
        self.assertEqual(output, [''])

        # Make sure everything prints when all options are on
        prob.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','objs']
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            prob.run_driver()
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')
        self.assertEqual(output.count("Driver debug print for iter coord: rank0:Driver|1"), 1)
        self.assertEqual(output.count("Design Vars"), 1)
        self.assertEqual(output.count("Nonlinear constraints"), 1)
        self.assertEqual(output.count("Linear constraints"), 1)
        self.assertEqual(output.count("Objectives"), 1)

        # Check to make sure an invalid debug_print option raises an exception
        with self.assertRaises(ValueError) as context:
            prob.driver.options['debug_print'] = ['bad_option']
        self.assertEqual(str(context.exception),
                         "Function is_valid returns False for debug_print.")

    def test_debug_print_desvar_physical_with_indices(self):
        prob = Problem()
        model = prob.model = Group()

        size = 3
        model.add_subsystem('p1', IndepVarComp('x', np.array([50.0] * size)))
        model.add_subsystem('p2', IndepVarComp('y', np.array([50.0] * size)))
        model.add_subsystem('comp', ExecComp('f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0',
                                             x=np.zeros(size), y=np.zeros(size),
                                             f_xy=np.zeros(size)))
        model.add_subsystem('con', ExecComp('c = - x + y',
                                            c=np.zeros(size), x=np.zeros(size),
                                            y=np.zeros(size)))

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')
        model.connect('p1.x', 'con.x')
        model.connect('p2.y', 'con.y')

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('p1.x', indices=[1], lower=-50.0, upper=50.0, ref=[5.0,])
        model.add_design_var('p2.y', indices=[1], lower=-50.0, upper=50.0)
        model.add_objective('comp.f_xy', index=1)
        model.add_constraint('con.c', indices=[1], upper=-15.0)

        prob.setup(check=False)

        prob.driver.options['debug_print'] = ['desvars',]
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout

        try:
            # formatting has changed in numpy 1.14 and beyond.
            if LooseVersion(np.__version__) >= LooseVersion("1.14"):
                with printoptions(precision=2, legacy="1.13"):
                    prob.run_driver()
            else:
                with printoptions(precision=2):
                    prob.run_driver()
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')
        # should see unscaled (physical) and the full arrays, not just what is indicated by indices
        self.assertEqual(output[3], "p1.x: array([ 50.,  50.,  50.])")
        self.assertEqual(output[4], "p2.y: array([ 50.,  50.,  50.])")

if __name__ == "__main__":
    unittest.main()
