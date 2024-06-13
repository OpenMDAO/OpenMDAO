""" Unit tests for the Driver base class."""


from packaging.version import Version
from io import StringIO
import sys
import unittest

import numpy as np

import openmdao.api as om
from openmdao.core.driver import Driver
from openmdao.utils.units import convert_units
from openmdao.utils.assert_utils import assert_near_equal, assert_warnings, assert_check_totals, assert_no_warning
from openmdao.utils.general_utils import printoptions, set_pyoptsparse_opt
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivatives
from openmdao.test_suite.components.simple_comps import DoubleArrayComp, NonSquareArrayComp
from openmdao.utils.om_warnings import OpenMDAOWarning
from openmdao.utils.mpi import MPI
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver, pyoptsparse

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


@use_tempdirs
class TestDriver(unittest.TestCase):

    def test_basic_get(self):

        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj')
        model.add_constraint('con1', lower=0)
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_driver()

        designvars = prob.driver.get_design_var_values()
        self.assertEqual(designvars['z'][0], 5.0 )

        designvars = prob.driver.get_objective_values()
        self.assertEqual(designvars['obj'], prob['obj'] )

        designvars = prob.driver.get_constraint_values()
        self.assertEqual(designvars['con1'], prob['con1']  )

    def test_scaled_design_vars(self):

        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z', ref=5.0, ref0=3.0)
        model.add_objective('obj')
        model.add_constraint('con1', lower=0)
        prob.set_solver_print(level=0)

        prob.setup()

        # Conclude setup but don't run model.
        prob.final_setup()

        dv = prob.driver.get_design_var_values()
        self.assertEqual(dv['z'][0], 1.0)
        self.assertEqual(dv['z'][1], -0.5)

        prob.driver.set_design_var('z', np.array((2.0, -2.0)))
        self.assertEqual(prob['z'][0], 7.0)
        self.assertEqual(prob['z'][1], -1.0)

    def test_scaled_constraints(self):

        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj')
        model.add_constraint('con1', lower=0, ref=2.0, ref0=3.0)
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        cv = prob.driver.get_constraint_values()['con1'][0]
        base = prob['con1']
        self.assertEqual((base-3.0)/(2.0-3.0), cv)

    def test_scaled_objectves(self):

        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj', ref=2.0, ref0=3.0)
        model.add_constraint('con1', lower=0)
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        cv = prob.driver.get_objective_values()['obj'][0]
        base = prob['obj']
        self.assertEqual((base-3.0)/(2.0-3.0), cv)

    def test_scaled_derivs(self):

        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj')
        model.add_constraint('con1')
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        base = prob.compute_totals(of=['obj', 'con1'], wrt=['z'])

        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z', ref=2.0, ref0=0.0)
        model.add_objective('obj', ref=1.0, ref0=0.0)
        model.add_constraint('con1', lower=0, ref=2.0, ref0=0.0)
        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        derivs = prob.driver._compute_totals(of=['obj', 'con1'], wrt=['z'],
                                             return_format='dict')
        assert_near_equal(base[('con1', 'z')][0], derivs['con1']['z'][0], 1e-5)
        assert_near_equal(base[('obj', 'z')][0]*2.0, derivs['obj']['z'][0], 1e-5)

    def test_vector_scaled_derivs(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp(name="x", val=np.ones((2, ))))
        comp = model.add_subsystem('comp', DoubleArrayComp())
        model.connect('px.x', 'comp.x1')

        model.add_design_var('px.x', ref=np.array([2.0, 3.0]), ref0=np.array([0.5, 1.5]))
        model.add_objective('comp.y1', ref=np.array([[7.0, 11.0]]), ref0=np.array([5.2, 6.3]))
        model.add_constraint('comp.y2', lower=0.0, upper=1.0, ref=np.array([[2.0, 4.0]]), ref0=np.array([1.2, 2.3]))

        prob.setup()
        prob.run_driver()

        derivs = prob.driver._compute_totals(of=['comp.y1'], wrt=['px.x'], return_format='dict')

        oscale = np.array([1.0/(7.0-5.2), 1.0/(11.0-6.3)])
        iscale = np.array([2.0-0.5, 3.0-1.5])
        J = comp.JJ[0:2, 0:2]

        # doing this manually so that I don't inadvertantly make an error in the vector math in both the code and test.
        J[0, 0] *= oscale[0]*iscale[0]
        J[0, 1] *= oscale[0]*iscale[1]
        J[1, 0] *= oscale[1]*iscale[0]
        J[1, 1] *= oscale[1]*iscale[1]
        assert_near_equal(J, derivs['comp.y1']['px.x'], 1.0e-3)

        obj = prob.driver.get_objective_values()
        obj_base = np.array([ (prob['comp.y1'][0]-5.2)/(7.0-5.2), (prob['comp.y1'][1]-6.3)/(11.0-6.3) ])
        assert_near_equal(obj['comp.y1'], obj_base, 1.0e-3)

        con = prob.driver.get_constraint_values()
        con_base = np.array([ (prob['comp.y2'][0]-1.2)/(2.0-1.2), (prob['comp.y2'][1]-2.3)/(4.0-2.3) ])
        assert_near_equal(con['comp.y2'], con_base, 1.0e-3)

    def test_vector_bounds_inf(self):

        # make sure no overflow when there is no specified upper/lower bound and significatn scaling
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp(name="x", val=np.ones((2, ))))
        comp = model.add_subsystem('comp', DoubleArrayComp())
        model.connect('px.x', 'comp.x1')

        model.add_design_var('px.x', ref=np.array([.1, 1e-6]))
        model.add_constraint('comp.y2', ref=np.array([.2, 2e-6]))

        prob.setup()

        desvars = model.get_design_vars()

        self.assertFalse(np.any(np.isinf(desvars['px.x']['upper'])))
        self.assertFalse(np.any(np.isinf(-desvars['px.x']['lower'])))

        responses = prob.model.get_responses()

        self.assertFalse(np.any(np.isinf(responses['comp.y2']['upper'])))
        self.assertFalse(np.any(np.isinf(-responses['comp.y2']['lower'])))

    def test_vector_scaled_derivs_diff_sizes(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp(name="x", val=np.ones((2, ))))
        comp = model.add_subsystem('comp', NonSquareArrayComp())
        model.connect('px.x', 'comp.x1')

        model.add_design_var('px.x', ref=np.array([2.0, 3.0]), ref0=np.array([0.5, 1.5]))
        model.add_objective('comp.y1', ref=np.array([[7.0, 11.0, 2.0]]), ref0=np.array([5.2, 6.3, 1.2]))
        model.add_constraint('comp.y2', lower=0.0, upper=1.0, ref=np.array([[2.0]]), ref0=np.array([1.2]))

        prob.setup()
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
        assert_near_equal(J, derivs['comp.y1']['px.x'], 1.0e-3)

        obj = prob.driver.get_objective_values()
        obj_base = np.array([ (prob['comp.y1'][0]-5.2)/(7.0-5.2), (prob['comp.y1'][1]-6.3)/(11.0-6.3), (prob['comp.y1'][2]-1.2)/(2.0-1.2) ])
        assert_near_equal(obj['comp.y1'], obj_base, 1.0e-3)

        con = prob.driver.get_constraint_values()
        con_base = np.array([ (prob['comp.y2'][0]-1.2)/(2.0-1.2)])
        assert_near_equal(con['comp.y2'], con_base, 1.0e-3)

    def test_debug_print_option(self):

        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj')
        model.add_constraint('con1', lower=0)
        model.add_constraint('con2', lower=0, linear=True)
        prob.set_solver_print(level=0)

        prob.setup()

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
            prob.run_driver(reset_iter_counts=False)
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
                         "Driver: Value (['bad_option']) of option 'debug_print' is not one of "
                         "['desvars', 'nl_cons', 'ln_cons', 'objs', 'totals'].")

    def test_debug_print_approx(self):

        prob = om.Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj')
        model.add_constraint('con1', lower=0)
        model.add_constraint('con2', lower=0, linear=True)
        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False
        prob.driver.options['maxiter'] = 0

        prob.model.approx_totals()
        prob.driver.options['debug_print'] = ['totals']

        prob.setup()

        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            prob.run_driver()
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')
        self.assertIn("{('obj', 'z'):", output[12])
        self.assertIn("{('con1', 'z'):", output[13])

    def test_debug_print_desvar_physical_with_indices(self):
        prob = om.Problem()
        model = prob.model

        size = 3
        model.add_subsystem('p1', om.IndepVarComp('x', np.array([50.0] * size)))
        model.add_subsystem('p2', om.IndepVarComp('y', np.array([50.0] * size)))
        model.add_subsystem('comp', om.ExecComp('f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0',
                                                x=np.zeros(size), y=np.zeros(size),
                                                f_xy=np.zeros(size)))
        model.add_subsystem('con', om.ExecComp('c = - x + y',
                                               c=np.zeros(size), x=np.zeros(size),
                                               y=np.zeros(size)))

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')
        model.connect('p1.x', 'con.x')
        model.connect('p2.y', 'con.y')

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('p1.x', indices=[1], lower=-50.0, upper=50.0, ref=[5.0,])
        model.add_design_var('p2.y', indices=[1], lower=-50.0, upper=50.0)
        model.add_objective('comp.f_xy', index=1)
        model.add_constraint('con.c', indices=[1], upper=-15.0)

        prob.setup()

        prob.driver.options['debug_print'] = ['desvars',]
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout

        try:
            # formatting has changed in numpy 1.14 and beyond.
            if Version(np.__version__) >= Version("1.14"):
                with printoptions(precision=2, legacy="1.13"):
                    prob.run_driver()
            else:
                with printoptions(precision=2):
                    prob.run_driver()
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')
        # should see unscaled (physical) and the full arrays, not just what is indicated by indices
        self.assertEqual(output[3], "{'p1.x': array([ 50.,  50.,  50.]), 'p2.y': array([ 50.,  50.,  50.])}")

    def test_debug_print_response_physical(self):
        prob = om.Problem()
        model = prob.model

        size = 3
        model.add_subsystem('p1', om.IndepVarComp('x', np.array([50.0] * size)))
        model.add_subsystem('p2', om.IndepVarComp('y', np.array([50.0] * size)))
        model.add_subsystem('comp', om.ExecComp('f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0',
                                                x=np.zeros(size), y=np.zeros(size),
                                                f_xy=np.zeros(size)))
        model.add_subsystem('con', om.ExecComp('c = - x + y + 1',
                                               c=np.zeros(size), x=np.zeros(size),
                                               y=np.zeros(size)))

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')
        model.connect('p1.x', 'con.x')
        model.connect('p2.y', 'con.y')

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('p1.x', indices=[1], lower=-50.0, upper=50.0)
        model.add_design_var('p2.y', indices=[1], lower=-50.0, upper=50.0)
        model.add_objective('comp.f_xy', index=1, ref=1.5)
        model.add_constraint('con.c', indices=[1], upper=-15.0, ref=1.02)

        prob.setup()

        prob.driver.options['debug_print'] = ['objs', 'nl_cons']
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout

        try:
            # formatting has changed in numpy 1.14 and beyond.
            if Version(np.__version__) >= Version("1.14"):
                with printoptions(precision=2, legacy="1.13"):
                    prob.run_driver()
            else:
                with printoptions(precision=2):
                    prob.run_driver()
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')
        # should see unscaled (physical) and the full arrays, not just what is indicated by indices
        self.assertEqual(output[3], "{'con.c': array([ 1.])}")
        self.assertEqual(output[6], "{'comp.f_xy': array([ 7622.])}")

    def test_debug_desvar_shape(self):
        # Desvar should always be printed un-flattened.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', om.IndepVarComp('x', val=np.array([[1.0, 3, 4], [7, 2, 5]])))

        model.add_design_var('p.x', np.array([[1.0, 3, 4], [7, 2, 5]]))
        prob.driver.options['debug_print'] = ['desvars']

        prob.setup()

        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout

        try:
            # formatting has changed in numpy 1.14 and beyond.
            if Version(np.__version__) >= Version("1.14"):
                with printoptions(precision=2, legacy="1.13"):
                    prob.run_driver()
            else:
                with printoptions(precision=2):
                    prob.run_driver()
        finally:
            sys.stdout = stdout

        output = strout.getvalue().split('\n')

        self.assertEqual(output[3], "{'p.x': array([[ 1.,  3.,  4.],")
        self.assertEqual(output[4], '       [ 7.,  2.,  5.]])}')

    def test_unsupported_discrete_desvar(self):
        prob = om.Problem()

        indep = om.IndepVarComp()
        indep.add_discrete_output('xI', val=0)
        prob.model.add_subsystem('p', indep)

        prob.model.add_design_var('p.xI')

        prob.driver = om.ScipyOptimizeDriver()

        prob.setup()

        with self.assertRaises(RuntimeError) as context:
            prob.final_setup()

        msg = "Discrete design variables are not supported by this driver: p.xI"
        self.assertEqual(str(context.exception), msg)

    def test_units_basic(self):
        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', 35.0, units='degF')

        model.add_subsystem('p', ivc, promotes=['x'])
        model.add_subsystem('comp1', om.ExecComp('y1 = 2.0*x',
                                                 x={'val': 2.0, 'units': 'degF'},
                                                 y1={'val': 2.0, 'units': 'degF'}),
                            promotes=['x', 'y1'])

        model.add_subsystem('comp2', om.ExecComp('y2 = 3.0*x',
                                                 x={'val': 2.0, 'units': 'degF'},
                                                 y2={'val': 2.0, 'units': 'degF'}),
                            promotes=['x', 'y2'])

        model.add_design_var('x', units='degC', lower=0.0, upper=100.0)
        model.add_constraint('y1', units='degC', lower=0.0, upper=100.0)
        model.add_objective('y2', units='degC')

        prob.setup()

        prob.run_driver()

        dv = prob.driver.get_design_var_values()
        assert_near_equal(dv['x'][0], 3.0 * 5 / 9, 1e-8)

        obj = prob.driver.get_objective_values(driver_scaling=True)
        assert_near_equal(obj['y2'][0], 73.0 * 5 / 9, 1e-8)

        con = prob.driver.get_constraint_values(driver_scaling=True)
        assert_near_equal(con['y1'][0], 38.0 * 5 / 9, 1e-8)

        meta = model.get_design_vars()
        assert_near_equal(meta['x']['lower'], 0.0, 1e-7)
        assert_near_equal(meta['x']['upper'], 100.0, 1e-7)

        meta = model.get_constraints()
        assert_near_equal(meta['y1']['lower'], 0.0, 1e-7)
        assert_near_equal(meta['y1']['upper'], 100.0, 1e-7)

        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            prob.list_driver_vars(desvar_opts=['units'], objs_opts=['units'], cons_opts=['units'])
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')

        self.assertTrue('1.666' in output[5])
        self.assertTrue('21.111' in output[12])
        self.assertTrue('40.555' in output[19])
        self.assertTrue('degC' in output[5])
        self.assertTrue('degC' in output[12])
        self.assertTrue('degC' in output[19])

    def test_units_equal(self):
        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', 35.0, units='degF')

        model.add_subsystem('p', ivc, promotes=['x'])
        model.add_subsystem('comp1', om.ExecComp('y1 = 2.0*x',
                                                 x={'val': 2.0, 'units': 'degF'},
                                                 y1={'val': 2.0, 'units': 'degF'}),
                            promotes=['x', 'y1'])

        model.add_subsystem('comp2', om.ExecComp('y2 = 3.0*x',
                                                 x={'val': 2.0, 'units': 'degF'},
                                                 y2={'val': 2.0, 'units': 'degF'}),
                            promotes=['x', 'y2'])

        model.add_design_var('x', units='degF', lower=32.0, upper=212.0)
        model.add_constraint('y1', units='degF', lower=32.0, upper=212.0)
        model.add_objective('y2', units='degF')

        prob.setup()

        prob.run_driver()

        dv = prob.driver.get_design_var_values()
        assert_near_equal(dv['x'][0], 35.0, 1e-8)

        obj = prob.driver.get_objective_values(driver_scaling=True)
        assert_near_equal(obj['y2'][0], 105.0, 1e-8)

        con = prob.driver.get_constraint_values(driver_scaling=True)
        assert_near_equal(con['y1'][0], 70.0, 1e-8)

        meta = model.get_design_vars()
        self.assertEqual(meta['x']['scaler'], None)
        self.assertEqual(meta['x']['adder'], None)

        meta = model.get_constraints()
        self.assertEqual(meta['y1']['scaler'], None)
        self.assertEqual(meta['y1']['adder'], None)

    def test_units_with_scaling(self):
        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', 35.0, units='degF')

        model.add_subsystem('p', ivc, promotes=['x'])
        model.add_subsystem('comp1', om.ExecComp('y1 = 2.0*x',
                                                 x={'val': 2.0, 'units': 'degF'},
                                                 y1={'val': 2.0, 'units': 'degF'}),
                            promotes=['x', 'y1'])

        model.add_subsystem('comp2', om.ExecComp('y2 = 3.0*x',
                                                 x={'val': 2.0, 'units': 'degF'},
                                                 y2={'val': 2.0, 'units': 'degF'}),
                            promotes=['x', 'y2'])

        model.add_design_var('x', units='degC', lower=0.0, upper=100.0, scaler=3.5, adder=77.0)
        model.add_constraint('y1', units='degC', lower=0.0, upper=100.0, scaler=3.5, adder=77.0)
        model.add_objective('y2', units='degC', scaler=3.5, adder=77.0)

        recorder = om.SqliteRecorder('cases.sql')
        prob.driver.add_recorder(recorder)

        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['record_desvars'] = True

        prob.setup()

        prob.run_driver()

        dv = prob.driver.get_design_var_values()
        assert_near_equal(dv['x'][0], ((3.0 * 5 / 9) + 77.0) * 3.5, 1e-8)

        obj = prob.driver.get_objective_values(driver_scaling=True)
        assert_near_equal(obj['y2'][0], ((73.0 * 5 / 9) + 77.0) * 3.5, 1e-8)

        con = prob.driver.get_constraint_values(driver_scaling=True)
        assert_near_equal(con['y1'][0], ((38.0 * 5 / 9) + 77.0) * 3.5, 1e-8)

        meta = model.get_design_vars()
        assert_near_equal(meta['x']['lower'], ((0.0) + 77.0) * 3.5, 1e-7)
        assert_near_equal(meta['x']['upper'], ((100.0) + 77.0) * 3.5, 1e-7)

        meta = model.get_constraints()
        assert_near_equal(meta['y1']['lower'], ((0.0) + 77.0) * 3.5, 1e-7)
        assert_near_equal(meta['y1']['upper'], ((100.0) + 77.0) * 3.5, 1e-7)

        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            prob.list_driver_vars(desvar_opts=['units'], objs_opts=['units'], cons_opts=['units'])
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')

        self.assertTrue('275.33' in output[5])
        self.assertTrue('343.3888' in output[12])
        self.assertTrue('411.444' in output[19])
        self.assertTrue('degC' in output[5])
        self.assertTrue('degC' in output[12])
        self.assertTrue('degC' in output[19])

        assert_check_totals(prob.check_totals(out_stream=None, driver_scaling=True))

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)
        case = cr.get_case(cases[0])

        dv = case.get_design_vars()
        assert_near_equal(dv['x'][0], ((3.0 * 5 / 9) + 77.0) * 3.5, 1e-8)

        obj = case.get_objectives()
        assert_near_equal(obj['y2'][0], ((73.0 * 5 / 9) + 77.0) * 3.5, 1e-8)

        con = case.get_constraints()
        assert_near_equal(con['y1'][0], ((38.0 * 5 / 9) + 77.0) * 3.5, 1e-8)

    def test_units_compute_totals(self):
        p = om.Problem()

        p.model.add_subsystem('stuff', om.ExecComp(['y = x', 'cy = x'],
                                                   x={'units': 'inch'},
                                                   y={'units': 'kg'},
                                                   cy={'units': 'kg'}),
                              promotes=['*'])

        p.model.add_design_var('x', units='ft')
        p.model.add_objective('y', units='lbm')
        p.model.add_constraint('cy', units='lbm', lower=0)

        p.setup()

        p['x'] = 1.0
        p.run_model()

        J_driver = p.driver._compute_totals()

        fact = convert_units(1.0, 'kg/inch', 'lbm/ft')
        assert_near_equal(J_driver['y', 'x'][0,0], fact, 1e-5)
        assert_near_equal(J_driver['cy', 'x'][0,0], fact, 1e-5)

    def test_units_error_messages(self):
        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', 35.0, units='degF')

        model.add_subsystem('p', ivc, promotes=['x'])
        model.add_subsystem('comp1', om.ExecComp('y1 = 2.0*x',
                                                 x={'val': 2.0, 'units': 'degF'},
                                                 y1={'val': 2.0, 'units': 'degF'}),
                            promotes=['x', 'y1'])

        model.add_design_var('x', units='ft', lower=0.0, upper=100.0, scaler=3.5, adder=77.0)
        prob.setup()

        with self.assertRaises(RuntimeError) as context:
            prob.final_setup()

        msg = "<model> <class Group>: Target for design variable x has 'degF' units, but 'ft' units were specified."
        self.assertEqual(str(context.exception), msg)

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', 35.0, units='degF')

        model.add_subsystem('p', ivc, promotes=['x'])
        model.add_subsystem('comp1', om.ExecComp('y1 = 2.0*x',
                                                 x={'val': 2.0, 'units': 'degF'},
                                                 y1={'val': 2.0, 'units': 'degF'}),
                            promotes=['x', 'y1'])

        model.add_constraint('x', units='ft', lower=0.0, upper=100.0)
        prob.setup()

        with self.assertRaises(RuntimeError) as context:
            prob.final_setup()

        msg = "<model> <class Group>: Target for constraint x has 'degF' units, but 'ft' units were specified."
        self.assertEqual(str(context.exception), msg)

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', 35.0, units=None)

        model.add_subsystem('p', ivc, promotes=['x'])
        model.add_subsystem('comp1', om.ExecComp('y1 = 2.0*x',
                                                 x={'val': 2.0},
                                                 y1={'val': 2.0, 'units': 'degF'}),
                            promotes=['x', 'y1'])

        model.add_design_var('x', units='ft', lower=0.0, upper=100.0, scaler=3.5, adder=77.0)
        prob.setup()

        with self.assertRaises(RuntimeError) as context:
            prob.final_setup()

        msg = "<model> <class Group>: Target for design variable x has no units, but 'ft' units were specified."
        self.assertEqual(str(context.exception), msg)

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', 35.0, units=None)

        model.add_subsystem('p', ivc, promotes=['x'])
        model.add_subsystem('comp1', om.ExecComp('y1 = 2.0*x',
                                                 x={'val': 2.0},
                                                 y1={'val': 2.0, 'units': 'degF'}),
                            promotes=['x', 'y1'])

        model.add_constraint('x', units='ft', lower=0.0, upper=100.0)
        prob.setup()

        with self.assertRaises(RuntimeError) as context:
            prob.final_setup()

        msg = "<model> <class Group>: Target for constraint x has no units, but 'ft' units were specified."
        self.assertEqual(str(context.exception), msg)

    def test_get_desvar_subsystem(self):
        # Test for a bug where design variables in a subsystem were not fully set up.
        prob = om.Problem()
        model = prob.model

        sub = model.add_subsystem('sub', om.Group())
        sub.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        sub.add_design_var('x', lower=-50.0, upper=50.0)
        sub.add_design_var('y', lower=-50.0, upper=50.0)
        sub.add_objective('f_xy')
        sub.add_constraint('y', lower=-40.0)

        prob.setup()

        prob.set_val('sub.x', 50.)
        prob.set_val('sub.y', 50.)

        result = prob.run_driver()

        assert_near_equal(prob['sub.x'], 6.66666667, 1e-6)
        assert_near_equal(prob['sub.y'], -7.3333333, 1e-6)

        prob.set_val('sub.x', 50.)
        prob.set_val('sub.y', 50.)

        prob.run_model()

        totals=prob.check_totals(out_stream=None)

        assert_near_equal(totals['sub.f_xy', 'sub.x']['J_fwd'], [[1.44e2]], 1e-5)
        assert_near_equal(totals['sub.f_xy', 'sub.y']['J_fwd'], [[1.58e2]], 1e-5)
        assert_near_equal(totals['sub.f_xy', 'sub.x']['J_fd'], [[1.44e2]], 1e-5)
        assert_near_equal(totals['sub.f_xy', 'sub.y']['J_fd'], [[1.58e2]], 1e-5)

    def test_get_vars_to_record(self):
        recorder = om.SqliteRecorder("cases.sql")

        prob = om.Problem(name='ABCD')
        prob.add_recorder(recorder)

        prob.model.add_subsystem('mag', om.ExecComp('y=x**2'),
                                 promotes_inputs=['*'], promotes_outputs=['*'])

        prob.model.add_subsystem('sum', om.ExecComp('z=sum(y)'),
                                 promotes_inputs=['*'], promotes_outputs=['*'])


        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.add_recorder(recorder)


        prob.recording_options['record_inputs'] = True
        prob.recording_options['excludes'] = ['*x*', '*aa*']
        prob.recording_options['includes'] = ['*z*', '*bb*']

        prob.driver.recording_options['record_inputs'] = True
        prob.driver.recording_options['excludes'] = ['*y*', '*cc*']
        prob.driver.recording_options['includes'] = ['*x*', '*dd*']

        prob.setup()

        expected_warnings = (
            (OpenMDAOWarning, "Problem ABCD: No matches for pattern '*aa*' in recording_options['excludes']."),
            (OpenMDAOWarning, "Problem ABCD: No matches for pattern '*bb*' in recording_options['includes']."),
            (OpenMDAOWarning, "ScipyOptimizeDriver: No matches for pattern '*cc*' in recording_options['excludes']."),
            (OpenMDAOWarning, "ScipyOptimizeDriver: No matches for pattern '*dd*' in recording_options['includes'].")
        )

        with assert_warnings(expected_warnings):
            prob.final_setup()

    def test_record_residuals_includes_excludes(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarProblem

        prob = SellarProblem()

        recorder = om.SqliteRecorder("rec_resids.sql")
        prob.driver.add_recorder(recorder)

        # just want record residuals
        prob.driver.recording_options['record_inputs'] = False
        prob.driver.recording_options['record_outputs'] = False
        prob.driver.recording_options['record_residuals'] = True
        prob.driver.recording_options['excludes'] = ['*y1*', '*x']   # x is an input, which we are not recording
        prob.driver.recording_options['includes'] = ['*con*', '*z']  # z is an input, which we are not recording

        prob.setup()

        expected_warnings = (
            (om.OpenMDAOWarning, "Driver: No matches for pattern '*x' in recording_options['excludes']."),
            (om.OpenMDAOWarning, "Driver: No matches for pattern '*z' in recording_options['includes']."),
        )

        with assert_warnings(expected_warnings):
            prob.final_setup()

@use_tempdirs
class TestCheckRelevance(unittest.TestCase):
    def setup_problem(self, driver=None):
        prob = om.Problem(driver=driver)
        model = prob.model

        # the customary Paraboloid model
        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_constraint('c', equals=-15.0)

        # add extra comps for testing constraints/objectives
        model.add_subsystem('p3', om.IndepVarComp('z', 50.0), promotes=['*'])
        model.add_subsystem('bad', om.ExecComp('bad = - z'), promotes=['*'])

        return prob

    def test_simple_paraboloid_irrelevant_constraint_run_once(self):
        # setup problem with default driver
        prob = self.setup_problem()

        prob.model.add_objective('f_xy')

        # add constraint that does not depend on a design var
        prob.model.add_constraint('bad', equals=-15.0)

        prob.setup()

        # the default driver does not support optimization, so a constraint that
        # does not depend on a design variable should not trigger a RuntimeError
        prob.run_driver()

    def test_simple_paraboloid_irrelevant_constraint(self):
        # define model with optimizing driver
        prob = self.setup_problem(driver=om.ScipyOptimizeDriver())

        prob.model.add_objective('f_xy')

        # add constraint that does not depend on a design var
        prob.model.add_constraint('bad', equals=-15.0)

        prob.driver.options['singular_jac_behavior'] = 'error'

        prob.setup()

        # since driver is an optimizer, a constraint that does not depend
        # on a design variable should trigger a RuntimeError
        with self.assertRaises(RuntimeError) as err:
            prob.run_driver()

        self.assertTrue("constraint(s) ['bad'] do not depend on any design variables."
                        in str(err.exception))


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestDriverMPI(unittest.TestCase):

    N_PROCS = 2

    def test_distrib_desvar_unsupported(self):

        class MyDriver(Driver):

            def __init__(self, generator=None, **kwargs):
                super().__init__(**kwargs)
                self.supports['distributed_design_vars'] = False
                self.supports._read_only = True

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp(distributed=True)
        ivc.add_output('invec')
        model.add_subsystem('p', ivc)
        model.add_design_var('p.invec', lower=0.0, upper=1.0)

        prob.driver = MyDriver()

        prob.setup()

        with self.assertRaises(RuntimeError) as context:
            prob.final_setup()

        self.assertEqual(str(context.exception),
                         "Distributed design variables are not supported by this driver, "
                         "but the following variables are distributed: [p.invec]")

    def test_distrib_desvar_get_set(self):

        comm = MPI.COMM_WORLD
        size = 3 if comm.rank == 0 else 2

        class DistribComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('w', val=1., distributed=True) # this will connect to a non-distributed IVC
                self.add_input('x', shape=size, distributed=True) # this will connect to a distributed IVC

                self.add_output('y', shape=1, distributed=True) # all-gathered output, duplicated on all procs
                self.add_output('z', shape=size, distributed=True) # distributed output

                self.declare_partials('y', 'x')
                self.declare_partials('y', 'w')
                self.declare_partials('z', 'x')

            def compute(self, inputs, outputs):
                x = inputs['x']
                local_y = np.sum((x-5)**2)
                y_g = np.zeros(self.comm.size)
                self.comm.Allgather(local_y, y_g)

                outputs['y'] = np.sum(y_g) + (inputs['w']-10)**2
                outputs['z'] = x**2

            def compute_partials(self, inputs, J):
                x = inputs['x']
                J['y', 'x'] = 2*(x-5)
                J['y', 'w'] = 2*(inputs['w']-10)
                J['z', 'x'] = np.diag(2*x)

        p = om.Problem()
        model = p.model
        driver = p.driver

        # distributed indep var, 'x'
        d_ivc = model.add_subsystem('d_ivc', om.IndepVarComp(distributed=True),
                                    promotes=['*'])
        d_ivc.add_output('x', 2*np.ones(size))

        # non-distributed indep var, 'w'
        ivc = model.add_subsystem('ivc', om.IndepVarComp(distributed=False),
                                  promotes=['*'])
        # previous version of this test set w to 2 different values depending on rank, which
        # is not valid for a non-distributed variable.  Changed to be the same on all procs.
        ivc.add_output('w', 3.0)

        # distributed component, 'dc'
        model.add_subsystem('dc', DistribComp(), promotes=['*'])

        # distributed design var, 'x'
        model.add_design_var('x', lower=-100, upper=100)
        model.add_objective('y')

        # driver that supports distributed design vars
        driver.supports._read_only = False
        driver.supports['distributed_design_vars'] = True

        p.setup()
        p.run_model()

        # get distributed design var
        driver.get_design_var_values(get_remote=None)

        assert_near_equal(driver.get_design_var_values(get_remote=True)['x'],
                          [2, 2, 2, 2, 2])

        assert_near_equal(driver.get_design_var_values(get_remote=False)['x'],
                          2*np.ones(size))

        # set distributed design var, set_remote=True
        driver.set_design_var('x', [3, 3, 3, 3, 3], set_remote=True)

        assert_near_equal(driver.get_design_var_values(get_remote=True)['x'],
                          [3, 3, 3, 3, 3])

        # set distributed design var, set_remote=False
        if comm.rank == 0:
            driver.set_design_var('x', 5.0*np.ones(size), set_remote=False)
        else:
            driver.set_design_var('x', 9.0*np.ones(size), set_remote=False)

        assert_near_equal(driver.get_design_var_values(get_remote=True)['x'],
                          [5, 5, 5, 9, 9])

        p.run_driver()

        assert_near_equal(p.get_val('dc.y', get_remote=True), [81, 81])
        assert_near_equal(p.get_val('dc.z', get_remote=True), [25, 25, 25, 81, 81])

    def test_distrib_desvar_bug(self):
        class MiniModel(om.Group):
            def setup(self):
                self.add_subsystem('dv', om.IndepVarComp('x', 3.0))
                self.add_subsystem('comp', om.ExecComp('y = (x-2)**2'))
                self.connect('dv.x', 'comp.x')

                self.add_design_var('dv.x', lower=-10, upper=10)

        class ParModel(om.ParallelGroup):
            def setup(self):
                self.add_subsystem('g0', MiniModel())
                self.add_subsystem('g1', MiniModel())

        p = om.Problem()

        pg = p.model.add_subsystem('par_group', ParModel())
        p.model.add_subsystem('obj', om.ExecComp('f = y0 + y1'))

        p.model.connect('par_group.g0.comp.y', 'obj.y0')
        p.model.connect('par_group.g1.comp.y', 'obj.y1')

        p.model.add_objective('obj.f')

        p.driver = om.ScipyOptimizeDriver()

        p.setup()
        p.run_driver()

        p.model.list_outputs()

        dvs = p.driver.get_design_var_values(get_remote=True)

        assert_near_equal(dvs['par_group.g0.dv.x'], 2)
        assert_near_equal(dvs['par_group.g1.dv.x'], 2)

    def test_scalar_scaler_and_adder(self):
        # build the model
        prob = om.Problem()
        dvs = prob.model.add_subsystem('dv', om.IndepVarComp())
        dvs.add_output("x_vec", [3.0, -4.0])
        prob.model.add_subsystem('paraboloid', om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

        # setup the optimization
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.model.connect("dv.x_vec", 'paraboloid.x', src_indices=0)
        prob.model.connect("dv.x_vec", 'paraboloid.y', src_indices=1)

        # Passing scaler as a vector here causes the problem
        prob.model.dv.add_design_var('x_vec', lower=-50, upper=50,
                                     scaler=[0.5, 0.5], adder=[0.0, 0.0])
        prob.model.add_objective('paraboloid.f')

        prob.setup()

        # run the optimization
        prob.run_driver()

        # location of the minimum
        x_star = prob.get_val('paraboloid.x')
        y_star = prob.get_val('paraboloid.y')

        assert_near_equal(x_star, 6.6666, tolerance=1.0E-3)
        assert_near_equal(y_star, -7.3333, tolerance=1.0E-3)


class TestLinearOnlyDVs(unittest.TestCase):
    def build_model(self, driver):

        prob = om.Problem()
        model = prob.model
        prob.driver = driver

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y + k'), promotes=['*'])

        prob.set_solver_print(level=0)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_design_var('k', lower=-1, upper=1)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=15.0, linear=True)

        prob.setup()

        return prob

    @unittest.skipIf(pyoptsparse is None, "pyoptsparse is required.")
    def test_pyoptsparse(self):
        OPTIMIZER = set_pyoptsparse_opt('SLSQP')[1]

        driver = pyOptSparseDriver(optimizer=OPTIMIZER, print_results=False)
        driver.opt_settings['ACC'] = 1e-9

        prob = self.build_model(driver)
        with assert_no_warning(category=om.DerivativesWarning):
            prob.run_driver()

    def test_scipy_opt(self):
        driver = om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False)
        prob = self.build_model(driver)
        with assert_no_warning(category=om.DerivativesWarning):
            prob.run_driver()


if __name__ == "__main__":
    unittest.main()
