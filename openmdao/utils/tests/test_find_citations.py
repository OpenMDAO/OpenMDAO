import unittest
from io import StringIO

from openmdao.api import Problem, Group, ParallelGroup, ExecComp, IndepVarComp, NonlinearRunOnce, LinearRunOnce, NewtonSolver
from openmdao.utils.mpi import MPI
from openmdao.utils.find_cite import find_citations, print_citations

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class TestFindCite(unittest.TestCase):

    def setUp(self):

        p = Problem()

        p.model.cite = "foobar model"
        p.model.nonlinear_solver.cite = "foobar nonlinear_solver"
        p.model.linear_solver.cite = "foobar linear_solver"

        indeps = p.model.add_subsystem('indeps', IndepVarComp('x', 10), promotes=['*'])
        indeps.linear_solver = LinearRunOnce()

        ec = p.model.add_subsystem('ec', ExecComp('y = 2+3*x'), promotes=['*'])
        # note using newton here makes no sense in reality, but its fine for this test since we never run the model
        ec.nonlinear_solver = NewtonSolver()
        ec.cite = "foobar exec comp"

        self.prob = p

    def test_find_cite(self):

        p = self.prob
        p.setup()

        citations = find_citations(p)

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


    def test_print_citations(self):

        p = self.prob
        p.setup()

        dest = StringIO()
        print_citations(p, out_stream=dest)

        expected = """Class: <class 'openmdao.core.problem.Problem'>
    @article{openmdao_2019,
        Author={Justin S. Gray and John T. Hwang and Joaquim R. R. A.
                Martins and Kenneth T. Moore and Bret A. Naylor},
        Title="{OpenMDAO: An Open-Source Framework for Multidisciplinary
                Design, Analysis, and Optimization}",
        Journal="{Structural and Multidisciplinary Optimization}",
        Year={2019},
        Publisher={Springer},
        pdf={http://openmdao.org/pubs/openmdao_overview_2019.pdf},
        note= {In Press}
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

    def test_print_citations_class_subset(self):

        p = self.prob
        p.setup()

        dest = StringIO()
        print_citations(p, classes=['Problem', 'LinearRunOnce'], out_stream=dest)

        expected = """Class: <class 'openmdao.core.problem.Problem'>
    @article{openmdao_2019,
        Author={Justin S. Gray and John T. Hwang and Joaquim R. R. A.
                Martins and Kenneth T. Moore and Bret A. Naylor},
        Title="{OpenMDAO: An Open-Source Framework for Multidisciplinary
                Design, Analysis, and Optimization}",
        Journal="{Structural and Multidisciplinary Optimization}",
        Year={2019},
        Publisher={Springer},
        pdf={http://openmdao.org/pubs/openmdao_overview_2019.pdf},
        note= {In Press}
        }
Class: <class 'openmdao.solvers.linear.linear_runonce.LinearRunOnce'>
    foobar linear_solver"""

        self.assertEqual(expected, dest.getvalue().strip())


class TestFindCitePar(unittest.TestCase):

    N_PROCS = 2

    def setUp(self):

        p = Problem()

        p.model.cite = "foobar model"
        p.model.nonlinear_solver.cite = "foobar nonlinear_solver"
        p.model.linear_solver.cite = "foobar linear_solver"

        indeps = p.model.add_subsystem('indeps', IndepVarComp('x', 10), promotes=['*'])
        indeps.linear_solver = LinearRunOnce()

        par = p.model.add_subsystem('par', ParallelGroup(), promotes=['*'])

        ec = par.add_subsystem('ec', ExecComp('y = 2+3*x'), promotes=['*'])
        # note using newton here makes no sense in reality, but its fine for this test since we never run the model
        ec.nonlinear_solver = NewtonSolver()
        ec.cite = "foobar exec comp"
        c2 = par.add_subsystem('c2', ExecComp('y2=x'), promotes=['*'])
        c2.cite = 'foobar exec comp'

        self.prob = p

    @unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
    def test_find_cite_petsc(self):
        p = self.prob
        p.setup()

        p.model._vector_class.cite = "foobar PETScVector"

        citations = find_citations(p)

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

        try:
            cite = citations[PETScVector]
            self.assertEqual('foobar PETScVector', cite)
        except KeyError:
            self.fail('Citation for PETScVector class expected')


if __name__ == "__main__":
    unittest.main()
