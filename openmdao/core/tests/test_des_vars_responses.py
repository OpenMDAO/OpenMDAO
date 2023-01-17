""" Unit tests for the design_variable and response interface to system."""
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.mpi import MPI
from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDis1withDerivatives, \
     SellarDis2withDerivatives

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized


class TestDesVarsResponses(unittest.TestCase):

    def test_api_on_model(self):

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0))
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])))

        model.add_subsystem('d1', SellarDis1withDerivatives())
        model.add_subsystem('d2', SellarDis2withDerivatives())

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0))

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'))
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'))

        model.connect('px.x', ['d1.x', 'obj_cmp.x'])
        model.connect('pz.z', ['d1.z', 'd2.z', 'obj_cmp.z'])
        model.connect('d1.y1', ['d2.y1', 'obj_cmp.y1', 'con_cmp1.y1'])
        model.connect('d2.y2', ['d1.y2', 'obj_cmp.y2', 'con_cmp2.y2'])

        model.nonlinear_solver = om.NonlinearBlockGS()
        model.linear_solver = om.ScipyKrylov()

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

    def test_units_checking(self):
        p = om.Problem()
        model = p.model
        G1 = model.add_subsystem('G1', om.Group())
        G1.add_subsystem('C1', om.ExecComp('y = 3.*x', x={'units': 'm'}), promotes=['x'])
        G1.add_subsystem('C2', om.ExecComp('yy = 4.*xx', xx={'units': 'm'}), promotes=['xx'])

        # Constraints

        with self.assertRaises(ValueError) as cm:
            G1.add_constraint('y', units='junk')

        msg = "'G1' <class Group>: The units 'junk' are invalid for response 'y'."
        self.assertEqual(cm.exception.args[0], msg)

        with self.assertRaises(TypeError) as cm:
            G1.add_constraint('y', units=3)

        msg = "'G1' <class Group>: The units argument should be a str or None for response 'y'."
        self.assertEqual(cm.exception.args[0], msg)

        G1.add_constraint('y', units='ft*ft/ft')
        self.assertEqual(G1._static_responses['y']['units'], 'ft')

        # Objectives

        with self.assertRaises(ValueError) as cm:
            G1.add_objective('yy', units='junk')

        msg = "'G1' <class Group>: The units 'junk' are invalid for response 'yy'."
        self.assertEqual(cm.exception.args[0], msg)

        with self.assertRaises(TypeError) as cm:
            G1.add_objective('yy', units=3)

        msg = "'G1' <class Group>: The units argument should be a str or None for response 'yy'."
        self.assertEqual(cm.exception.args[0], msg)

        # Simplification
        G1.add_objective('yy', units='ft*ft/ft')
        self.assertEqual(G1._static_responses['yy']['units'], 'ft')

        # Design Variables

        with self.assertRaises(ValueError) as cm:
            G1.add_design_var('x', units='junk')

        msg = "'G1' <class Group>: The units 'junk' are invalid for design_var 'x'."
        self.assertEqual(cm.exception.args[0], msg)

        with self.assertRaises(TypeError) as cm:
            G1.add_design_var('x', units=3)

        msg = "'G1' <class Group>: The units argument should be a str or None for design_var 'x'."
        self.assertEqual(cm.exception.args[0], msg)

        # Simplification
        G1.add_design_var('x', units='ft*ft/ft')
        self.assertEqual(G1._static_design_vars['x']['units'], 'ft')

    def test_component_desvar_units1(self):

        class Paraboloid(om.ExplicitComponent):
            """
            Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
            """
            def setup(self):
                self.add_input('x', val=0.0, units='m')
                self.add_input('y', val=0.0, units='m')

                self.add_output('f_xy', val=0.0, units='m**2')

                self.add_design_var('x', lower=-50, upper=50, units='m')
                self.add_design_var('y', lower=-50, upper=50, units='m')

            def setup_partials(self):
                self.declare_partials('*', '*', method='fd')

            def compute(self, inputs, outputs):
                """
                f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

                Unconstrained Minimum at: x = 6.6667; y = -7.3333
                Constrained Minimum (x + y <= 10) at: x = 7; y = -7
                """
                x = inputs['x']
                y = inputs['y']

                outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0

        # build the model
        prob = om.Problem()
        prob.model.add_subsystem('parab', Paraboloid(),
                                 promotes_inputs=['x', 'y'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y', units='m'),
                                 promotes_inputs=['x', 'y'])

        prob.model.add_objective('parab.f_xy')
        prob.model.add_constraint('const.g', lower=0, upper=10.)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')

        # Design variables 'x' and 'y' span components,
        # so we need to provide a common initial value for them.
        prob.model.set_input_defaults('x', 3.0, units='m')
        prob.model.set_input_defaults('y', -4.0, units='m')

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob.get_val('parab.f_xy'), -27., 1e-6)
        assert_near_equal(prob.get_val('x'), 7, 1e-4)
        assert_near_equal(prob.get_val('y'), -7, 1e-4)

        # provide a common initial values again,
        # but in imperial units this time
        ft_per_m = 3.28084
        prob.model.set_input_defaults('x', 3*ft_per_m, units='ft')
        prob.model.set_input_defaults('y', 4*ft_per_m, units='ft')

        prob.setup()
        prob.run_driver()

        # minimum value (still in meters)
        assert_near_equal(prob.get_val('parab.f_xy'), -27., 1e-6)

        # location of the minimum (in feet)
        assert_near_equal(prob.get_val('x'), 7*ft_per_m, 1e-4)
        assert_near_equal(prob.get_val('y'), -7*ft_per_m, 1e-4)

    def test_component_desvar_units2(self):

        class Double(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', val=0.0, units='degF')
                self.add_output('y', val=0.0, units='degF')

                self.add_design_var('x', lower=-50, upper=50, units='ft')

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['x'] * 2

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Double(), promotes=['x'])

        prob.setup()

        with self.assertRaises(RuntimeError) as context:
            prob.final_setup()

        msg = "'comp' <class Double>: Target for design variable x has 'degF' units, but 'ft' units were specified."
        self.assertEqual(str(context.exception), msg)


class TestDesvarOnModel(unittest.TestCase):

    def test_design_var_not_exist(self):

        prob = om.Problem(name='design_var_not_exist')

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('junk')

        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
           "\nCollected errors for problem 'design_var_not_exist':"
           "\n   <model> <class SellarDerivatives>: Output not found for design variable 'junk'.")

    def test_desvar_affine_and_scaleradder(self):

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        with self.assertRaises(TypeError) as context:
            prob.model.add_design_var(42, lower=-100, upper=100, ref0=-100.0,
                                      ref=100)

        self.assertEqual(str(context.exception), '<class SellarDerivatives>: The name argument should '
                                                 'be a string, got 42')

    def test_desvar_invalid_bounds(self):

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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

        prob = om.Problem(name='constraint_not_exist')

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_constraint('junk')

        with self.assertRaises(Exception) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
           "\nCollected errors for problem 'constraint_not_exist':"
           "\n   <model> <class SellarDerivatives>: Output not found for response 'junk'.")

    def test_constraint_affine_and_scaleradder(self):

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        with self.assertRaises(TypeError) as context:
            prob.model.add_design_var(42, lower=-100, upper=100, ref0=-100.0,
                                      ref=100)

        self.assertEqual(str(context.exception), '<class SellarDerivatives>: The name argument should '
                                                 'be a string, got 42')

    def test_constraint_invalid_bounds(self):

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        with self.assertRaises(TypeError) as context:
            prob.model.add_constraint(42, lower=-100, upper=100, ref0=-100.0,
                                      ref=100)

        self.assertEqual(str(context.exception), '<class SellarDerivatives>: The name argument should '
                                                 'be a string, got 42')

    def test_constraint_invalid_lower(self):

        prob = om.Problem()

        prob.driver = om.ScipyOptimizeDriver()
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

        prob = om.Problem()

        prob.driver = om.ScipyOptimizeDriver()
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
        prob = om.Problem()

        prob.driver = om.ScipyOptimizeDriver()
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

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        with self.assertRaises(Exception) as context:
            prob.model.add_constraint('con1', lower=0.0, upper=5.0, indices='foo')

        self.assertEqual(str(context.exception), "<class SellarDerivatives>: Invalid indices foo for constraint 'con1'.")

        with self.assertRaises(Exception) as context:
            prob.model.add_constraint('con3', lower=0.0, upper=5.0, indices=[1, 'k'])

        self.assertEqual(str(context.exception), "<class SellarDerivatives>: Invalid indices [1, 'k'] for constraint 'con3'.")

        # passing an iterator for indices should be valid
        prob.model.add_constraint('con1', lower=0.0, upper=5.0, indices=range(2))

    def test_error_eq_ineq_con(self):
        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        with self.assertRaises(ValueError) as context:
            prob.model.add_constraint('con1', lower=0.0, upper=5.0, equals=3.0, indices='foo')

        msg = "<class SellarDerivatives>: Constraint 'con1' cannot be both equality and inequality."
        self.assertEqual(str(context.exception), msg)


class exampleComp(om.ExplicitComponent):
    """
    This is an example OpenMDAO explicit component which taks in inputs x, y
    and outputs a scalar f(x,y) and a square matrix (2x2) c(x,y) according to:

        f = (x-3)^2 + xy + (y+4)^2 - 3

        c[1,1] = x + y
        c[2,1] = x^2 + y^2
        c[1,2] = 2*x + y^2
        c[2,2] = x^2 + 2*y

    Partial derivatives are computed using finite-difference.
    """

    def setup(self):
        # Declare Inputs
        self.add_input("x", shape=(1), val=0.0)
        self.add_input("y", shape=(1), val=0.0)

        # Declare Outputs
        self.add_output("f_xy", shape=(1))
        self.add_output("c_xy", shape=(2, 2))

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # Retrieve Inputs
        x = inputs["x"]
        y = inputs["y"]

        # Compute Values
        f_xy = (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0

        c_xy = np.zeros((2, 2))
        c_xy[0, 0] = x + y
        c_xy[1, 0] = x ** 2 + y ** 2
        c_xy[0, 1] = 2.0 * x + y ** 2
        c_xy[1, 1] = x ** 2 + 2.0 * y

        # Set Outputs
        outputs["f_xy"] = f_xy
        outputs["c_xy"] = c_xy


def constraintExample(inds):
    """
    This function is an example for trying to implement constraints on unique
    elements of multidimensional arrays. The system consists of:
        Design Variables: x, y
        Constraints: c_xy[:,1]
        Objective: f_xy
    """
    # Initialize OM Problem and Add Component
    prob = om.Problem()
    prob.model.add_subsystem("exampleComp", exampleComp())

    # Set Optimizer
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"

    # Declare Design Variables
    prob.model.add_design_var("exampleComp.x")
    prob.model.add_design_var("exampleComp.y")

    # Declaring Constraints by Using True Dimensioned Array Indices
    prob.model.add_constraint("exampleComp.c_xy", indices=inds, flat_indices=False, lower=[0.0, 75.0], upper=[25.0, 100.0])

    # Declare Objective
    prob.model.add_objective("exampleComp.f_xy")

    # Finalize Setup
    prob.setup()

    # Define Initial Conditions
    prob.set_val("exampleComp.x", 3.0)
    prob.set_val("exampleComp.y", -4.0)

    # Run Optimization
    prob.run_driver()

    return prob


class TestConstraintFancyIndex(unittest.TestCase):

    def test_fancy_index_arr(self):
        prob = constraintExample(([0,1], [1,1]))
        np.testing.assert_allclose(prob.get_val("exampleComp.f_xy"), [10.50322551])
        np.testing.assert_allclose(prob.get_val("exampleComp.c_xy")[:, 1], [25.00000002, 75.00000024])

    def test_fancy_index_slice(self):
        prob = constraintExample(om.slicer[:, 1])
        np.testing.assert_allclose(prob.get_val("exampleComp.f_xy"), [10.50322551])
        np.testing.assert_allclose(prob.get_val("exampleComp.c_xy")[:, 1], [25.00000002, 75.00000024])


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestAddConstraintMPI(unittest.TestCase):

    N_PROCS = 2

    def test_add_bad_con(self):
        # From a bug, this message didn't work in mpi.
        prob = om.Problem(name='add_bad_con')
        model = prob.model

        sub = model.add_subsystem('sub', SellarDerivatives())
        sub.nonlinear_solver = om.NonlinearBlockGS()

        sub.add_constraint('d1.junk', equals=0.0, cache_linear_solution=True)

        with self.assertRaises(RuntimeError) as context:
            prob.setup(mode='rev')

        self.assertEqual(str(context.exception),
           "\nCollected errors for problem 'add_bad_con':"
           "\n   <model> <class Group>: 'sub' <class SellarDerivatives>: Output not found for "
           "response 'd1.junk'.")


class TestObjectiveOnModel(unittest.TestCase):

    def test_obective_not_exist(self):

        prob = om.Problem(name='obective_not_exist')

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_objective('junk')

        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
                         "\nCollected errors for problem 'obective_not_exist':\n   <model> <class SellarDerivatives>: Output not found for response 'junk'.")

    def test_objective_affine_and_scaleradder(self):

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        with self.assertRaises(TypeError) as context:
            prob.model.add_objective('con1', lower=-100, upper=100, ref=1.0,
                                      scaler=0.5)

        self.assertTrue(str(context.exception).endswith(
                        "add_objective() got an unexpected keyword argument 'lower'"))

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

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

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

    @parameterized.expand(['lower', 'upper', 'adder', 'scaler', 'ref', 'ref0'],
                          name_func=lambda f, n, p: 'test_desvar_size_err_' + '_'.join(a for a in p.args))
    def test_desvar_size_err(self, name):

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        args = {name: -np.ones(2)*100}
        prob.model.add_design_var('z', indices=[1], **args)
        with self.assertRaises(Exception) as context:
            prob.setup()
            prob.run_model()

        self.assertEqual(str(context.exception),
                         f"<model> <class SellarDerivatives>: When adding design var 'z', {name} should have size 1 but instead has size 2.")

    @parameterized.expand(['lower', 'upper', 'equals', 'adder', 'scaler', 'ref', 'ref0'],
                          name_func=lambda f, n, p: 'test_constraint_size_err_' + '_'.join(a for a in p.args))
    def test_constraint_size_err(self, name):

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        args = {name: -np.ones(2)*100}
        prob.model.add_constraint('z', indices=[1], **args)
        with self.assertRaises(Exception) as context:
            prob.setup()
            prob.run_model()

        self.assertEqual(str(context.exception),
                         f"<model> <class SellarDerivatives>: When adding constraint 'z', {name} should have size 1 but instead has size 2.")

    @parameterized.expand(['adder', 'scaler', 'ref', 'ref0'],
                          name_func=lambda f, n, p: 'test_objective_size_err_' + '_'.join(a for a in p.args))
    def test_objective_size_err(self, name):

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        args = {name: -np.ones(2)*100}
        prob.model.add_objective('z', index=1, **args)
        with self.assertRaises(Exception) as context:
            prob.setup()
            prob.run_model()

        self.assertEqual(str(context.exception),
                         f"<model> <class SellarDerivatives>: When adding objective 'z', {name} should have size 1 but instead has size 2.")

    def test_objective_invalid_name(self):

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        with self.assertRaises(TypeError) as context:
            prob.model.add_objective(42, ref0=-100.0, ref=100)

        self.assertEqual(str(context.exception), '<class SellarDerivatives>: The name argument should '
                                                 'be a string, got 42')

    def test_objective_invalid_index(self):

        prob = om.Problem()

        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        with self.assertRaises(TypeError) as context:
            prob.model.add_objective('obj', index='foo')

        self.assertEqual(str(context.exception), '<class SellarDerivatives>: If specified, objective index must be an int.')

        prob.model.add_objective('obj', index=1)

    def test_constrained_indices_scalar_support(self):
        p = om.Problem()

        indeps = p.model.add_subsystem('indeps', om.IndepVarComp(), promotes_outputs=['*'])

        indeps.add_output('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                                          0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
        indeps.add_output('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                                         -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
        indeps.add_output('r', .7)

        arctan_yox = om.ExecComp('g=arctan(y/x)', has_diag_partials=True,
                                 g=np.ones(10), x=np.ones(10), y=np.ones(10))

        p.model.add_subsystem('arctan_yox', arctan_yox)

        p.model.add_subsystem('circle', om.ExecComp('area=pi*r**2'))

        p.model.add_subsystem('r_con', om.ExecComp('g=x**2 + y**2 - r', has_diag_partials=True,
                                                   g=np.ones(10), x=np.ones(10), y=np.ones(10)))

        p.model.connect('r', ('circle.r', 'r_con.r'))
        p.model.connect('x', ['r_con.x', 'arctan_yox.x'])
        p.model.connect('y', ['r_con.y', 'arctan_yox.y'])

        p.model.approx_totals(method='cs')

        p.model.add_design_var('x')
        p.model.add_design_var('y')
        p.model.add_design_var('r', lower=.5, upper=10)
        p.model.add_constraint('y', equals=0, indices=0)
        p.model.add_objective('circle.area', ref=-1)

        p.setup(derivatives=True, force_alloc_complex=True)

        p.run_model()
        # Formerly a KeyError
        derivs = p.check_totals(compact_print=True, out_stream=None)
        assert_near_equal(0.0, derivs['indeps.y', 'indeps.x']['abs error'][0])


if __name__ == '__main__':
    unittest.main()
