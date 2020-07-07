""" Unit tests for the design_variable and response interface to system."""
import unittest

import numpy as np

from openmdao.api import Problem, NonlinearBlockGS, Group, IndepVarComp, ExecComp, ScipyKrylov,  \
    IndepVarComp, ScipyOptimizeDriver
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.mpi import MPI

from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDis1withDerivatives, \
     SellarDis2withDerivatives

from openmdao.core.tests.test_distribcomp import DistribInputDistribOutputComp

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class TestDesVarsResponses(unittest.TestCase):

    def test_api_on_model(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1')
        prob.model.add_constraint('con2')

        prob.setup()

        des_vars = prob.model.get_design_vars()
        obj = prob.model.get_objectives()
        constraints = prob.model.get_constraints()

        self.assertEqual(set(des_vars.keys()), {'x', 'z'})
        self.assertEqual(set(obj.keys()), {'obj_cmp.obj'})
        self.assertEqual(set(constraints.keys()), {'con_cmp1.con1', 'con_cmp2.con2'})

    def test_api_response_on_model(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_response('obj', type_="obj")
        prob.model.add_response('con1', type_="con")
        prob.model.add_response('con2', type_="con")

        prob.setup()

        des_vars = prob.model.get_design_vars()
        responses = prob.model.get_responses()
        obj = prob.model.get_objectives()
        constraints = prob.model.get_constraints()

        self.assertEqual(set(des_vars.keys()), {'x', 'z'})
        self.assertEqual(set(obj.keys()), {'obj_cmp.obj'})
        self.assertEqual(set(constraints.keys()), {'con_cmp1.con1', 'con_cmp2.con2'})
        self.assertEqual(set(responses.keys()), {'obj_cmp.obj', 'con_cmp1.con1', 'con_cmp2.con2'})

    def test_api_list_on_model(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=[-100, -20], upper=[100, 20])
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1')
        prob.model.add_constraint('con2')

        prob.setup()

        des_vars = prob.model.get_design_vars()
        obj = prob.model.get_objectives()
        constraints = prob.model.get_constraints()

        self.assertEqual(set(des_vars.keys()), {'x', 'z'})
        self.assertEqual(set(obj.keys()), {'obj_cmp.obj',})
        self.assertEqual(set(constraints.keys()), {'con_cmp1.con1', 'con_cmp2.con2'})

    def test_api_array_on_model(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z',
                                  lower=np.array([-100, -20]),
                                  upper=np.array([100, 20]))
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1')
        prob.model.add_constraint('con2')

        prob.setup()

        des_vars = prob.model.get_design_vars()
        obj = prob.model.get_objectives()
        constraints = prob.model.get_constraints()

        self.assertEqual(set(des_vars.keys()), {'x', 'z'})
        self.assertEqual(set(obj.keys()), {'obj_cmp.obj',})
        self.assertEqual(set(constraints.keys()), {'con_cmp1.con1', 'con_cmp2.con2'})

    def test_api_iter_on_model(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=range(-101, -99),
                                       upper=range(99, 101),
                                       indices=range(2))

        prob.model.add_objective('obj')
        prob.model.add_constraint('con1')
        prob.model.add_constraint('con2')

        prob.setup()

        des_vars = prob.model.get_design_vars()
        obj = prob.model.get_objectives()
        constraints = prob.model.get_constraints()

        self.assertEqual(set(des_vars.keys()), {'x', 'z'})
        self.assertEqual(set(obj.keys()), {'obj_cmp.obj',})
        self.assertEqual(set(constraints.keys()), {'con_cmp1.con1', 'con_cmp2.con2'})

    def test_api_on_subsystems(self):

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp('x', 1.0))
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        model.add_subsystem('d1', SellarDis1withDerivatives())
        model.add_subsystem('d2', SellarDis2withDerivatives())

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                               z=np.array([0.0, 0.0]), x=0.0))

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'))
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'))

        model.connect('px.x', ['d1.x', 'obj_cmp.x'])
        model.connect('pz.z', ['d1.z', 'd2.z', 'obj_cmp.z'])
        model.connect('d1.y1', ['d2.y1', 'obj_cmp.y1', 'con_cmp1.y1'])
        model.connect('d2.y2', ['d1.y2', 'obj_cmp.y2', 'con_cmp2.y2'])

        model.nonlinear_solver = NonlinearBlockGS()
        model.linear_solver = ScipyKrylov()

        px = prob.model.px
        px.add_design_var('x', lower=-100, upper=100)

        pz = prob.model.pz
        pz.add_design_var('z', lower=-100, upper=100)

        obj = prob.model.obj_cmp
        obj.add_objective('obj')

        con_comp1 = prob.model.con_cmp1
        con_comp1.add_constraint('con1')

        con_comp2 = prob.model.con_cmp2
        con_comp2.add_constraint('con2')

        prob.setup()

        des_vars = prob.model.get_design_vars()
        obj = prob.model.get_objectives()
        constraints = prob.model.get_constraints()

        self.assertEqual(set(des_vars.keys()), {'px.x', 'pz.z'})
        self.assertEqual(set(obj.keys()), {'obj_cmp.obj',})
        self.assertEqual(set(constraints.keys()), {'con_cmp1.con1', 'con_cmp2.con2'})


class TestDesvarOnModel(unittest.TestCase):

    def test_design_var_not_exist(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('junk')

        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception), "SellarDerivatives (<model>): Output not found for design variable 'junk'.")

    def test_desvar_affine_and_scaleradder(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        with self.assertRaises(ValueError) as context:
            prob.model.add_design_var('x', lower=-100, upper=100, ref=1.0,
                                      scaler=0.5)

        self.assertEqual(str(context.exception), 'Inputs ref/ref0 are mutually'
                                                 ' exclusive with'
                                                 ' scaler/adder')

        with self.assertRaises(ValueError) as context:
            prob.model.add_design_var('x', lower=-100, upper=100, ref=0.0,
                                      adder=0.5)

        self.assertEqual(str(context.exception), 'Inputs ref/ref0 are mutually'
                                                 ' exclusive with'
                                                 ' scaler/adder')

        with self.assertRaises(ValueError) as context:
            prob.model.add_design_var('x', lower=-100, upper=100, ref0=0.0,
                                      adder=0.5)

        self.assertEqual(str(context.exception), 'Inputs ref/ref0 are mutually'
                                                 ' exclusive with'
                                                 ' scaler/adder')

        with self.assertRaises(ValueError) as context:
            prob.model.add_design_var('x', lower=-100, upper=100, ref0=0.0,
                                      scaler=0.5)

        self.assertEqual(str(context.exception), 'Inputs ref/ref0 are mutually'
                                                 ' exclusive with'
                                                 ' scaler/adder')

    def test_desvar_affine_mapping(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100, ref0=-100.0,
                                  ref=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1')
        prob.model.add_constraint('con2')

        prob.setup()

        des_vars = prob.model.get_design_vars()

        x_ref0 = des_vars['x']['ref0']
        x_ref = des_vars['x']['ref']
        x_scaler = des_vars['x']['scaler']
        x_adder = des_vars['x']['adder']

        self.assertAlmostEqual( x_scaler*(x_ref0 + x_adder), 0.0, places=12)
        self.assertAlmostEqual( x_scaler*(x_ref + x_adder), 1.0, places=12)

    def test_desvar_inf_bounds(self):

        # make sure no overflow when there is no specified upper/lower bound and significatn scaling

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', scaler=1e6)
        prob.model.add_objective('obj', scaler=1e6)
        prob.model.add_constraint('con1', scaler=1e6)
        prob.model.add_constraint('con2', scaler=1e6)

        prob.setup()

        des_vars = prob.model.get_design_vars()

        self.assertFalse(np.isinf(des_vars['x']['upper']))
        self.assertFalse(np.isinf(-des_vars['x']['lower']))

        responses = prob.model.get_responses()

        self.assertFalse(np.isinf(responses['con_cmp1.con1']['upper']))
        self.assertFalse(np.isinf(responses['con_cmp2.con2']['upper']))
        self.assertFalse(np.isinf(-responses['con_cmp1.con1']['lower']))
        self.assertFalse(np.isinf(-responses['con_cmp2.con2']['lower']))

    def test_desvar_invalid_name(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        with self.assertRaises(TypeError) as context:
            prob.model.add_design_var(42, lower=-100, upper=100, ref0=-100.0,
                                      ref=100)

        self.assertEqual(str(context.exception), 'SellarDerivatives: The name argument should '
                                                 'be a string, got 42')

    def test_desvar_invalid_bounds(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        with self.assertRaises(TypeError) as context:
            prob.model.add_design_var('x', lower='foo', upper=[0, 100],
                                      ref0=-100.0, ref=100)

        self.assertEqual(str(context.exception), 'Expected values of lower to be an '
                                                 'Iterable of numeric values, '
                                                 'or a scalar numeric value. '
                                                 'Got foo instead.')

        with self.assertRaises(ValueError) as context:
            prob.model.add_design_var('x', lower=0.0, upper=['a', 'b'],
                                      ref0=-100.0, ref=100)

class TestConstraintOnModel(unittest.TestCase):

    def test_constraint_not_exist(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_constraint('junk')

        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception), "SellarDerivatives (<model>): Output not found for response 'junk'.")

    def test_constraint_affine_and_scaleradder(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        with self.assertRaises(ValueError) as context:
            prob.model.add_constraint('con1', lower=-100, upper=100, ref=1.0,
                                      scaler=0.5)

        self.assertEqual(str(context.exception), 'Inputs ref/ref0 are mutually'
                                                 ' exclusive with'
                                                 ' scaler/adder')

        with self.assertRaises(ValueError) as context:
            prob.model.add_constraint('con1', lower=-100, upper=100, ref=0.0,
                                      adder=0.5)

        self.assertEqual(str(context.exception), 'Inputs ref/ref0 are mutually'
                                                 ' exclusive with'
                                                 ' scaler/adder')

        with self.assertRaises(ValueError) as context:
            prob.model.add_constraint('x', lower=-100, upper=100, ref0=0.0,
                                      adder=0.5)

        self.assertEqual(str(context.exception), 'Inputs ref/ref0 are mutually'
                                                 ' exclusive with'
                                                 ' scaler/adder')

        with self.assertRaises(ValueError) as context:
            prob.model.add_constraint('con1', lower=-100, upper=100, ref0=0.0,
                                      scaler=0.5)

        self.assertEqual(str(context.exception), 'Inputs ref/ref0 are mutually'
                                                 ' exclusive with'
                                                 ' scaler/adder')

    def test_constraint_affine_mapping(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', lower=-100, upper=100, ref0=-100.0,
                                  ref=100)
        prob.model.add_constraint('con2')

        prob.setup()

        constraints = prob.model.get_constraints()

        con1_ref0 = constraints['con_cmp1.con1']['ref0']
        con1_ref = constraints['con_cmp1.con1']['ref']
        con1_scaler = constraints['con_cmp1.con1']['scaler']
        con1_adder = constraints['con_cmp1.con1']['adder']

        self.assertAlmostEqual( con1_scaler*(con1_ref0 + con1_adder), 0.0,
                                places=12)
        self.assertAlmostEqual( con1_scaler*(con1_ref + con1_adder), 1.0,
                                places=12)

    def test_constraint_invalid_name(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        with self.assertRaises(TypeError) as context:
            prob.model.add_design_var(42, lower=-100, upper=100, ref0=-100.0,
                                      ref=100)

        self.assertEqual(str(context.exception), 'SellarDerivatives: The name argument should '
                                                 'be a string, got 42')

    def test_constraint_invalid_bounds(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        with self.assertRaises(TypeError) as context:
            prob.model.add_design_var('x', lower='foo', upper=[0, 100],
                                      ref0=-100.0, ref=100)

        self.assertEqual(str(context.exception), 'Expected values of lower to'
                                                 ' be an Iterable of numeric'
                                                 ' values, or a scalar numeric'
                                                 ' value. Got foo instead.')

        with self.assertRaises(ValueError) as context:
            prob.model.add_design_var('x', lower=0.0, upper=['a', 'b'],
                                      ref0=-100.0, ref=100)

    def test_constraint_invalid_name(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        with self.assertRaises(TypeError) as context:
            prob.model.add_constraint(42, lower=-100, upper=100, ref0=-100.0,
                                      ref=100)

        self.assertEqual(str(context.exception), 'SellarDerivatives: The name argument should '
                                                 'be a string, got 42')

    def test_constraint_invalid_lower(self):

        prob = Problem()

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        with self.assertRaises(TypeError) as context:
            prob.model.add_constraint('con1', lower='foo', upper=[0, 100],
                                      ref0=-100.0, ref=100)

        with self.assertRaises(TypeError) as context2:
            prob.model.add_constraint('con1', lower=['zero', 5], upper=[0, 100],
                                      ref0=-100.0, ref=100)

        msg = ("Argument 'lower' can not be a string ('foo' given). You can not "
        "specify a variable as lower bound. You can only provide constant "
        "float values")
        self.assertEqual(str(context.exception), msg)

        msg2 = ("Argument 'lower' can not be a string ('['zero', 5]' given). You can not "
        "specify a variable as lower bound. You can only provide constant "
        "float values")
        self.assertEqual(str(context2.exception), msg2)

    def test_constraint_invalid_upper(self):

        prob = Problem()

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        with self.assertRaises(TypeError) as context:
            prob.model.add_constraint('con1', lower=0, upper='foo',
                                      ref0=-100.0, ref=100)

        with self.assertRaises(TypeError) as context2:
            prob.model.add_constraint('con1', lower=0, upper=[1, 'foo'],
                                      ref0=-100.0, ref=100)

        msg = ("Argument 'upper' can not be a string ('foo' given). You can not "
        "specify a variable as upper bound. You can only provide constant "
        "float values")
        self.assertEqual(str(context.exception), msg)

        msg2 = ("Argument 'upper' can not be a string ('[1, 'foo']' given). You can not "
        "specify a variable as upper bound. You can only provide constant "
        "float values")
        self.assertEqual(str(context2.exception), msg2)

    def test_constraint_invalid_equals(self):
        prob = Problem()

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        with self.assertRaises(TypeError) as context:
            prob.model.add_constraint('con1', equals='foo')

        with self.assertRaises(TypeError) as context2:
            prob.model.add_constraint('con1', equals=[1, 'two'])

        msg = ("Argument 'equals' can not be a string ('foo' given). You can "
               "not specify a variable as equals bound. You can only provide "
               "constant float values")
        self.assertEqual(str(context.exception), msg)

        msg2 = ("Argument 'equals' can not be a string ('[1, 'two']' given). You can "
               "not specify a variable as equals bound. You can only provide "
               "constant float values")
        self.assertEqual(str(context2.exception), msg2)

    def test_constraint_invalid_indices(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        with self.assertRaises(ValueError) as context:
            prob.model.add_constraint('con1', lower=0.0, upper=5.0,
                                      indices='foo')

        self.assertEqual(str(context.exception), 'SellarDerivatives: If specified, response indices must '
                                                 'be a sequence of integers.')

        with self.assertRaises(ValueError) as context:
            prob.model.add_constraint('con1', lower=0.0, upper=5.0,
                                      indices=1)

        self.assertEqual(str(context.exception), 'SellarDerivatives: If specified, response indices must '
                                                 'be a sequence of integers.')

        with self.assertRaises(ValueError) as context:
            prob.model.add_constraint('con1', lower=0.0, upper=5.0,
                                      indices=[1, 'k'])

        self.assertEqual(str(context.exception), 'SellarDerivatives: If specified, response indices must '
                                                 'be a sequence of integers.')

        # passing an iterator for indices should be valid
        prob.model.add_constraint('con1', lower=0.0, upper=5.0,
                                          indices=range(2))

    def test_error_eq_ineq_con(self):
        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        with self.assertRaises(ValueError) as context:
            prob.model.add_constraint('con1', lower=0.0, upper=5.0, equals=3.0,
                                      indices='foo')

        msg = "SellarDerivatives: Constraint 'con1' cannot be both equality and inequality."
        self.assertEqual(str(context.exception), msg)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestAddConstraintMPI(unittest.TestCase):

    N_PROCS = 2

    def test_add_bad_con(self):
        # From a bug, this message didn't work in mpi.
        prob = Problem()
        model = prob.model

        sub = model.add_subsystem('sub', SellarDerivatives())
        sub.nonlinear_solver = NonlinearBlockGS()

        sub.add_constraint('d1.junk', equals=0.0, cache_linear_solution=True)

        with self.assertRaises(RuntimeError) as context:
            prob.setup(mode='rev')

        msg = "SellarDerivatives (sub): Output not found for response 'd1.junk'."
        self.assertEqual(str(context.exception), msg)


class TestObjectiveOnModel(unittest.TestCase):

    def test_obective_not_exist(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_objective('junk')

        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
                         "SellarDerivatives (<model>): Output not found for response 'junk'.")

    def test_objective_affine_and_scaleradder(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        with self.assertRaises(TypeError) as context:
            prob.model.add_objective('con1', lower=-100, upper=100, ref=1.0,
                                      scaler=0.5)

        self.assertEqual(str(context.exception),
                         "add_objective() got an unexpected keyword argument 'lower'")

        with self.assertRaises(ValueError) as context:
            prob.model.add_objective('con1', ref=0.0, scaler=0.5)

        self.assertEqual(str(context.exception), 'Inputs ref/ref0 are mutually'
                                                 ' exclusive with'
                                                 ' scaler/adder')

        with self.assertRaises(ValueError) as context:
            prob.model.add_objective('con1', ref=0.0, adder=0.5)

        self.assertEqual(str(context.exception), 'Inputs ref/ref0 are mutually'
                                                 ' exclusive with'
                                                 ' scaler/adder')

        with self.assertRaises(ValueError) as context:
            prob.model.add_objective('x', ref0=0.0, adder=0.5)

        self.assertEqual(str(context.exception), 'Inputs ref/ref0 are mutually'
                                                 ' exclusive with'
                                                 ' scaler/adder')

        with self.assertRaises(ValueError) as context:
            prob.model.add_objective('con1', ref0=0.0, scaler=0.5)

        self.assertEqual(str(context.exception), 'Inputs ref/ref0 are mutually'
                                                 ' exclusive with'
                                                 ' scaler/adder')

    def test_objective_affine_mapping(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj', ref0=1000, ref=1010)
        prob.model.add_objective('con2')

        prob.setup()

        objectives = prob.model.get_objectives()

        obj_ref0 = objectives['obj_cmp.obj']['ref0']
        obj_ref = objectives['obj_cmp.obj']['ref']
        obj_scaler = objectives['obj_cmp.obj']['scaler']
        obj_adder = objectives['obj_cmp.obj']['adder']

        self.assertAlmostEqual( obj_scaler*(obj_ref0 + obj_adder), 0.0,
                                places=12)
        self.assertAlmostEqual( obj_scaler*(obj_ref + obj_adder), 1.0,
                                places=12)

    def test_desvar_size_err(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        for name in ['lower', 'upper', 'adder', 'scaler', 'ref', 'ref0']:
            args = {name: -np.ones(2)*100}
            with self.assertRaises(Exception) as context:
                prob.model.add_design_var('z', indices=[1], **args)
            self.assertEqual(str(context.exception),
                             "SellarDerivatives: When adding design var 'z', %s should have size 1 but instead has size 2." % name)

    def test_constraint_size_err(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        for name in ['lower', 'upper', 'equals', 'adder', 'scaler', 'ref', 'ref0']:
            args = {name: -np.ones(2)*100}
            with self.assertRaises(Exception) as context:
                prob.model.add_constraint('z', indices=[1], **args)
            self.assertEqual(str(context.exception),
                             "SellarDerivatives: When adding constraint 'z', %s should have size 1 but instead has size 2." % name)

    def test_objective_size_err(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        for name in ['adder', 'scaler', 'ref', 'ref0']:
            args = {name: -np.ones(2)*100}
            with self.assertRaises(Exception) as context:
                prob.model.add_objective('z', index=1, **args)
            self.assertEqual(str(context.exception),
                             "SellarDerivatives: When adding objective 'z', %s should have size 1 but instead has size 2." % name)

    def test_objective_invalid_name(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        with self.assertRaises(TypeError) as context:
            prob.model.add_objective(42, ref0=-100.0, ref=100)

        self.assertEqual(str(context.exception), 'SellarDerivatives: The name argument should '
                                                 'be a string, got 42')

    def test_objective_invalid_index(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        with self.assertRaises(TypeError) as context:
            prob.model.add_objective('obj', index='foo')

        self.assertEqual(str(context.exception), 'SellarDerivatives: If specified, objective index must be an int.')

        prob.model.add_objective('obj', index=1)


if __name__ == '__main__':
    unittest.main()
