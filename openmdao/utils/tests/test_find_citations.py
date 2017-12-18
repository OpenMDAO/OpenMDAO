import unittest
from io import StringIO

from openmdao.api import Problem, Group, ExecComp, IndepVarComp, NonlinearRunOnce, LinearRunOnce, NewtonSolver

from openmdao.utils.find_cite import find_citations

class CiteProblem(Problem):

    def __init__(self):
        super(Problem, self).__init__()

        self.cite = "foobar prob"
        # self.driver.cite = "foobar driver"




class TestFindCite(unittest.TestCase):

    def setUp(self):

        p = Problem()

        p.model = Group()
        p.model.cite = "foobar model"
        p.model.nonlinear_solver.cite = "foobar nonlinear_solver"
        p.model.linear_solver.cite = "foobar linear_solver"

        indeps = p.model.add_subsystem('indeps', IndepVarComp('x', 10), promotes=['*'])
        indeps.linear_solver = LinearRunOnce()

        ec = p.model.add_subsystem('ec', ExecComp('y = 2+3*x'), promotes=['*'])
        # note using newton here makes no sense in reality, but its fine for this test since we never run the model
        ec.nonlinear_solver = NewtonSolver()
        ec.cite = "foobar exec comp"

        p.setup()

        self.prob = p


    def test_find_cite_no_write(self):

        p = self.prob

        citations = find_citations(p, out_stream=False)

        # these two shouldn't have citations
        self.assertFalse(IndepVarComp in citations)
        self.assertFalse(NewtonSolver in citations)

        try:
            cite = citations[Problem]
            self.assertEqual(p.cite, cite)
        except KeyError:
            self.fail('Citation for Problem class expected')

        try:
            cite = citations[Group]
            self.assertEqual('foobar model', cite)
        except KeyError:
            self.fail('Citation for Group class expected')

        try:
            cite = citations[NonlinearRunOnce]
            self.assertEqual('foobar nonlinear_solver', cite)
        except KeyError:
            self.fail('Citation for NonlinearRunOnce class expected')

        try:
            cite = citations[LinearRunOnce]
            self.assertEqual('foobar linear_solver', cite)
        except KeyError:
            self.fail('Citation for LinearRunOnce class expected')

        try:
            cite = citations[ExecComp]
            self.assertEqual('foobar exec comp', cite)
        except KeyError:
            self.fail('Citation for ExecComp class expected')


    def test_find_cite_with_write(self):

        p = self.prob

        dest = StringIO()
        find_citations(p, out_stream=dest)

        expected = """Class: <class 'openmdao.core.problem.Problem'>
    @inproceedings{2014_openmdao_derivs,
        Author = {Justin S. Gray and Tristan A. Hearn and Kenneth T. Moore and John Hwang and Joaquim Martins and Andrew Ning},
        Booktitle = {15th AIAA/ISSMO Multidisciplinary Analysis and Optimization Conference},
        Doi = {doi:10.2514/6.2014-2042},
        Month = {2014/07/08},
        Publisher = {American Institute of Aeronautics and Astronautics},
        Title = {Automatic Evaluation of Multidisciplinary Derivatives Using a Graph-Based Problem Formulation in OpenMDAO},
        Year = {2014}
    }
Class: <class 'openmdao.core.group.Group'>
    foobar model
Class: <class 'openmdao.solvers.nonlinear.nonlinear_runonce.NonlinearRunOnce'>
    foobar nonlinear_solver
Class: <class 'openmdao.solvers.linear.linear_runonce.LinearRunOnce'>
    foobar linear_solver
Class: <class 'openmdao.components.exec_comp.ExecComp'>
    foobar exec comp"""

        self.assertEqual(expected, dest.getvalue().strip())



if __name__ == "__main__":
    unittest.main()


