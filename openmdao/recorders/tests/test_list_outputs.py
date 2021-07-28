import unittest
import openmdao.api as om
from openmdao.test_suite.components.paraboloid_problem import ParaboloidProblem
import io


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

if __name__ == '__main__':
    unittest.main()