import unittest
import io

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.paraboloid_problem import ParaboloidProblem
from openmdao.test_suite.components.rectangle import RectangleGroup

from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal


@use_tempdirs
class ListVarsTest(unittest.TestCase):

    def test_invalid_return_format(self):
        prob = ParaboloidProblem()
        rec = om.SqliteRecorder('test_list_outputs.db')
        prob.model.add_recorder(rec)

        prob.setup()
        prob.run_model()

        case = om.CaseReader('test_list_outputs.db').get_case(-1)

        with self.assertRaises(ValueError) as cm:
            case.list_inputs(return_format=dict)

        msg = f"Invalid value (<class 'dict'>) for return_format, " \
              "must be a string value of 'list' or 'dict'"

        self.assertEqual(str(cm.exception), msg)

        with self.assertRaises(ValueError) as cm:
            case.list_outputs(return_format='dct')

        msg = f"Invalid value ('dct') for return_format, " \
              "must be a string value of 'list' or 'dict'"

        self.assertEqual(str(cm.exception), msg)

    def test_list_outputs(self):
        """
        Confirm that includes/excludes has the same result between System.list_inputs() and
        Case.list_inputs(), and between System.list_outputs() and Case.list_outputs().
        """

        prob = ParaboloidProblem()
        rec = om.SqliteRecorder('test_list_outputs.db')
        prob.model.add_recorder(rec)

        prob.setup()
        prob.run_model()

        case = om.CaseReader('test_list_outputs.db').get_case(-1)

        prob_out = io.StringIO()
        rec_out = io.StringIO()

        # Test list_inputs() with includes
        prob.model.list_inputs(val=False, includes="comp*", out_stream=prob_out)
        case.list_inputs(val=False, includes="comp*", out_stream=rec_out)

        prob_out_str = prob_out.getvalue()
        rec_out_str = rec_out.getvalue()
        self.assertEqual(prob_out_str, rec_out_str)

        prob_out.flush()
        rec_out.flush()

        # Test list_outputs() with includes
        prob.model.list_outputs(val=False, includes="p*", out_stream=prob_out)
        case.list_outputs(val=False, includes="p*", out_stream=rec_out)

        prob_out_str = prob_out.getvalue()
        rec_out_str = rec_out.getvalue()
        self.assertEqual(prob_out_str, rec_out_str)

        prob_out.flush()
        rec_out.flush()

        # Test list_inputs() with excludes
        prob.model.list_inputs(val=False, excludes="comp*", out_stream=prob_out)
        case.list_inputs(val=False, excludes="comp*", out_stream=rec_out)

        prob_out_str = prob_out.getvalue()
        rec_out_str = rec_out.getvalue()
        self.assertEqual(prob_out_str, rec_out_str)

        prob_out.flush()
        rec_out.flush()

        # Test list_outputs() with excludes
        prob.model.list_outputs(val=False, excludes="p*", out_stream=prob_out)
        case.list_outputs(val=False, excludes="p*", out_stream=rec_out)

        prob_out_str = prob_out.getvalue()
        rec_out_str = rec_out.getvalue()
        self.assertEqual(prob_out_str, rec_out_str)

    def test_discrete_missing_attributes(self):
        class Parabola(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', val=0.0, units='m')
                self.add_discrete_input('a', val=1)

                self.add_output('f', val=0.0, units='m')
                self.declare_partials('*', '*', method='fd')

            def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
                x = inputs['x']
                a = discrete_inputs['a']
                outputs['f'] = a*(x-2)**2 + 5

        p = om.Problem()

        idv = p.model.add_subsystem('idv', om.IndepVarComp(), promotes=['*'])
        idv.add_discrete_output('a', val=5)

        p.model.add_subsystem('parab', Parabola(), promotes=['*'])

        p.model.add_design_var('x')
        p.model.add_objective('f')

        p.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')
        p.driver.add_recorder(om.SqliteRecorder('driver_cases.db'))
        p.driver.recording_options['includes'] = ['*']

        p.setup()
        p.run_driver()
        p.cleanup()

        cr = om.CaseReader("driver_cases.db")
        case = cr.get_case(-1)

        prob_out = io.StringIO()
        case_out = io.StringIO()
        self.maxDiff = 2000

        p.model.list_inputs(prom_name=True, units=True, shape=True,
                            out_stream=prob_out)

        case.list_inputs(prom_name=True, units=True, shape=True,
                         out_stream=case_out)

        self.assertEqual(prob_out.getvalue(), case_out.getvalue(),
            f"Output comparison failed:\n"
            f"===================== PROB ===================\n"
            f"{prob_out.getvalue()}"
            f"===================== CASE ===================\n"
            f"{case_out.getvalue()}"
            f"==============================================\n"
        )

        prob_out.flush()
        case_out.flush()

        p.model.list_outputs(prom_name=True, units=True, shape=True, bounds=True, scaling=True,
                             out_stream=prob_out)

        case.list_outputs(prom_name=True, units=True, shape=True, bounds=True, scaling=True,
                          out_stream=case_out)

        self.assertEqual(prob_out.getvalue(), case_out.getvalue())

    def test_list_vars(self):
        prob = om.Problem(RectangleGroup())
        prob.setup()
        prob.model.add_recorder(om.SqliteRecorder('list_vars.db'))

        prob.set_val('length', 3.)
        prob.set_val('width', 2.)
        prob.run_model()

        expected = prob.model.list_vars(units=True, out_stream=None, return_format='dict')

        case = om.CaseReader('list_vars.db').get_case(0)

        io_vars = case.list_vars(units=True, out_stream=None, return_format='dict')

        self.assertEqual(io_vars, expected)

    def test_list_vars_tags(self):
        p = om.Problem()
        model = p.model

        model.add_recorder(om.SqliteRecorder('addsubtags.db'))

        nn = 1
        ivc = om.IndepVarComp()
        ivc.add_output(name='a', shape=(nn,))
        ivc.add_output(name='b', shape=(nn,))

        model.add_subsystem(name='ivc', subsys=ivc,
                            promotes_outputs=['a', 'b'])

        adder = model.add_subsystem(name='add_subtract_comp', subsys=om.AddSubtractComp())
        adder.add_equation('adder_output', ['input_a','input_b'], tags={'foo'})
        adder.add_equation('adder_output2', ['input_a','input_a'], tags={'bar'})

        model.connect('a', 'add_subtract_comp.input_a')
        model.connect('b', 'add_subtract_comp.input_b')

        p.setup()

        p['a'] = np.random.rand(nn,)
        p['b'] = np.random.rand(nn,)

        p.run_model()

        case = om.CaseReader('addsubtags.db').get_case(0)

        a = p['a']
        b = p['b']

        for obj in (model, case):
            foo_outputs = obj.list_vars(tags={'foo'}, print_tags=True,
                                        out_stream=None, return_format='dict')
            self.assertEqual(len(foo_outputs), 1,
                             msg=f"There should be one output tagged 'foo': {foo_outputs}")

            bar_outputs = obj.list_vars(tags={'bar'}, print_tags=False,
                                        out_stream=None, return_format='dict')
            self.assertEqual(len(bar_outputs), 1,
                             msg=f"There should be one output tagged 'bar': {bar_outputs}")

            assert_near_equal(foo_outputs['add_subtract_comp.adder_output']['val'], a + b)
            assert_near_equal(bar_outputs['add_subtract_comp.adder_output2']['val'], a + a)

    def test_list_vars_bool_args(self):
        """
        Iterate through the boolean arguments to the list_vars function and toggle each one to
        its non-default value.  Make sure the output from the Case method is the same as the
        output from the System method.
        """
        from openmdao.test_suite.components.sellar_feature import SellarMDAWithUnits

        model = SellarMDAWithUnits()
        model.add_subsystem('indep', om.IndepVarComp('foo', 1.0))
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob = om.Problem(model)
        prob.setup()

        model.add_recorder(om.SqliteRecorder('list_vars.db'))

        prob.set_solver_print(0)
        prob.run_model()
        prob.cleanup()

        case = om.CaseReader('list_vars.db').get_case(0)

        import inspect
        func_sig = inspect.signature(case.list_vars)
        bool_args = [(name, param.default) for name, param in func_sig.parameters.items()
                     if param.default in {True, False}]

        for (arg, dflt) in bool_args:
            with self.subTest(f"{arg}={not dflt}"):
                self.maxDiff = None

                model_out = io.StringIO()
                case_out = io.StringIO()

                kwargs = {arg: not dflt}

                model_vars = model.list_vars(out_stream=model_out, **kwargs)
                case_vars = case.list_vars(out_stream=case_out, **kwargs)

                self.assertEqual(case_out.getvalue(), model_out.getvalue(),
                    f"Output comparison failed for {kwargs=}\n"
                    f"===================== PROB ===================\n"
                    f"{model_out.getvalue()}"
                    f"===================== CASE ===================\n"
                    f"{case_out.getvalue()}"
                    f"==============================================\n"
                )

                assert_near_equal(sorted(case_vars), sorted(model_vars),
                                  f"Return value comparison failed for {kwargs=}")


if __name__ == '__main__':
    unittest.main()
