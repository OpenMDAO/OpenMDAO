import unittest
import openmdao.api as om
from openmdao.test_suite.components.paraboloid_problem import ParaboloidProblem
import io

from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class ListOutputsTest(unittest.TestCase):
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

        read_p = om.CaseReader('test_list_outputs.db').get_case(-1)

        prob_out = io.StringIO()
        rec_out = io.StringIO()

        # Test list_inputs() with includes
        prob.model.list_inputs(val=False, includes="comp*", out_stream=prob_out)
        read_p.list_inputs(val=False, includes="comp*", out_stream=rec_out)

        prob_out_str = prob_out.getvalue()
        rec_out_str = rec_out.getvalue()
        self.assertEqual(prob_out_str, rec_out_str)

        prob_out.flush()
        rec_out.flush()

        # Test list_outputs() with includes
        prob.model.list_outputs(val=False, includes="p*", out_stream=prob_out)
        read_p.list_outputs(val=False, includes="p*", out_stream=rec_out)

        prob_out_str = prob_out.getvalue()
        rec_out_str = rec_out.getvalue()
        self.assertEqual(prob_out_str, rec_out_str)

        prob_out.flush()
        rec_out.flush()

        # Test list_inputs() with excludes
        prob.model.list_inputs(val=False, excludes="comp*", out_stream=prob_out)
        read_p.list_inputs(val=False, excludes="comp*", out_stream=rec_out)

        prob_out_str = prob_out.getvalue()
        rec_out_str = rec_out.getvalue()
        self.assertEqual(prob_out_str, rec_out_str)

        prob_out.flush()
        rec_out.flush()

        # Test list_outputs() with excludes
        prob.model.list_outputs(val=False, excludes="p*", out_stream=prob_out)
        read_p.list_outputs(val=False, excludes="p*", out_stream=rec_out)

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

        self.assertEqual(prob_out.getvalue(), case_out.getvalue())

        prob_out.flush()
        case_out.flush()

        p.model.list_outputs(prom_name=True, units=True, shape=True, bounds=True, scaling=True,
                             out_stream=prob_out)

        case.list_outputs(prom_name=True, units=True, shape=True, bounds=True, scaling=True,
                          out_stream=case_out)

        self.assertEqual(prob_out.getvalue(), case_out.getvalue())


if __name__ == '__main__':
    unittest.main()