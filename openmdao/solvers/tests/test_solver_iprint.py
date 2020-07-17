""" Unit test for the solver printing behavior. """

import os
from io import StringIO
import re
import sys
import unittest

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.double_sellar import SubSellar

from openmdao.utils.general_utils import run_model
from openmdao.utils.mpi import MPI


try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class TestSolverPrint(unittest.TestCase):

    def test_feature_iprint_neg1(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = om.Problem()
        prob.model = SellarDerivatives()

        prob.setup()

        newton = prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        scipy = prob.model.linear_solver = om.ScipyKrylov()

        newton.options['maxiter'] = 2

        # use a real bad initial guess
        prob.set_val('y1', 10000)
        prob.set_val('y2', -26)

        newton.options['iprint'] = -1
        scipy.options['iprint'] = -1
        prob.run_model()

    def test_iprint_neg1(self):
        old_stdout = sys.stdout
        sys.stdout = str_out = StringIO()

        try:
            self.test_feature_iprint_neg1()
        finally:
            sys.stdout = old_stdout

        # Verify output
        self.assertEqual(str_out.getvalue().strip(), "")

    def test_feature_iprint_0(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = om.Problem()
        prob.model = SellarDerivatives()

        prob.setup()

        newton = prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        scipy = prob.model.linear_solver = om.ScipyKrylov()

        newton.options['maxiter'] = 1

        prob.set_val('y1', 10000)
        prob.set_val('y2', -26)

        newton.options['iprint'] = 0
        scipy.options['iprint'] = 0

        prob.run_model()

    def test_iprint_0(self):
        old_stdout = sys.stdout
        sys.stdout = str_out = StringIO()

        try:
            self.test_feature_iprint_0()
        finally:
            sys.stdout = old_stdout

        # Verify output
        self.assertEqual(str_out.getvalue().strip(),
                         "NL: NewtonSolver 'NL: Newton' on system '' failed to converge in "
                         "1 iterations.")

    def test_feature_iprint_1(self, stdout=None):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = om.Problem()
        prob.model = SellarDerivatives()

        prob.setup()

        newton = prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        scipy = prob.model.linear_solver = om.ScipyKrylov()

        newton.options['maxiter'] = 20

        prob.set_val('y1', 10000)
        prob.set_val('y2', -26)

        newton.options['iprint'] = 1
        scipy.options['iprint'] = 0
        prob.run_model()

    def test_iprint_1(self):
        old_stdout = sys.stdout
        sys.stdout = str_out = StringIO()

        try:
            self.test_feature_iprint_1()
        finally:
            sys.stdout = old_stdout

        # Verify output
        self.assertEqual(str_out.getvalue().strip(),
                         "NL: Newton Converged in 2 iterations")

    def test_feature_iprint_2(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = om.Problem()
        prob.model = SellarDerivatives()

        prob.setup()

        newton = prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        scipy = prob.model.linear_solver = om.ScipyKrylov()

        newton.options['maxiter'] = 20

        prob.set_val('y1', 10000)
        prob.set_val('y2', -26)

        newton.options['iprint'] = 2
        scipy.options['iprint'] = 1
        prob.run_model()

    def test_iprint_2(self):
        old_stdout = sys.stdout
        sys.stdout = str_out = StringIO()

        try:
            self.test_feature_iprint_2()
        finally:
            sys.stdout = old_stdout

        # Verify output
        lines = str_out.getvalue().strip().split('\n')
        for line in lines:
            self.assertEqual(line[:3], 'NL:')

    def test_hierarchy_iprint(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', om.Group())
        sub2 = sub1.add_subsystem('sub2', om.Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.linear_solver = om.LinearBlockGS()

        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.linear_solver = om.ScipyKrylov()
        g2.linear_solver.precon = om.LinearBlockGS()
        g2.linear_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=2)

        prob.setup()

        output = run_model(prob)

        # Check that certain things show up in our outputs
        self.assertGreaterEqual(output.count('sub1.sub2.g1'), 1)
        self.assertGreaterEqual(output.count('g2'), 1)
        self.assertGreaterEqual(output.count('NL: Newton'), 2)
        self.assertGreaterEqual(output.count('LN: SCIPY'), 2)
        self.assertGreaterEqual(output.count('precon: LNBGS'), 2)

    def test_hierarchy_iprint2(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', om.Group())
        sub2 = sub1.add_subsystem('sub2', om.Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = om.NonlinearBlockGS()
        g1.nonlinear_solver = om.NonlinearBlockGS()
        g2.nonlinear_solver = om.NonlinearBlockGS()

        prob.set_solver_print(level=2)

        prob.setup()

        output = run_model(prob)

        # Check that certain things show up in our outputs
        self.assertGreaterEqual(output.count('sub1.sub2.g1'), 2)
        self.assertGreaterEqual(output.count('g2'), 2)
        self.assertGreaterEqual(output.count('NL: NLBGS'), 2)

    def test_hierarchy_iprint3(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', om.Group())
        sub2 = sub1.add_subsystem('sub2', om.Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = om.NonlinearBlockJac()
        sub1.nonlinear_solver = om.NonlinearBlockJac()
        sub2.nonlinear_solver = om.NonlinearBlockJac()
        g1.nonlinear_solver = om.NonlinearBlockJac()
        g2.nonlinear_solver = om.NonlinearBlockJac()

        prob.set_solver_print(level=2)

        prob.setup()

        output = run_model(prob)

        # Check that certain things show up in our outputs
        self.assertGreaterEqual(output.count('sub1'), 2)
        self.assertGreaterEqual(output.count('sub1.sub2'), 2)
        self.assertGreaterEqual(output.count('sub1.sub2.g1'), 2)
        self.assertGreaterEqual(output.count('g2'), 2)
        self.assertGreaterEqual(output.count('NL: NLBJ'), 2)

    def test_feature_set_solver_print1(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.double_sellar import SubSellar

        prob = om.Problem()
        model = prob.model

        sub1 = model.add_subsystem('sub1', om.Group(), promotes_inputs=['z'])
        sub2 = sub1.add_subsystem('sub2', om.Group(), promotes_inputs=['z'])
        g1 = sub2.add_subsystem('g1', SubSellar(), promotes_inputs=['z'])
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.linear_solver = om.LinearBlockGS()

        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.linear_solver = om.ScipyKrylov()
        g2.linear_solver.precon = om.LinearBlockGS()
        g2.linear_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=2)

        prob.setup()
        prob.set_val('z', np.array([5.0, 2.0]))
        prob.run_model()

    def test_set_solver_print1(self):
        old_stdout = sys.stdout
        sys.stdout = str_out = StringIO()

        try:
            self.test_feature_set_solver_print1()
        finally:
            sys.stdout = old_stdout

        # Check that certain things show up in our outputs
        output = str_out.getvalue()
        self.assertGreaterEqual(output.count('sub1.sub2.g1'), 2)
        self.assertGreaterEqual(output.count('g2'), 2)
        self.assertGreaterEqual(output.count('NL: Newton'), 2)
        self.assertGreaterEqual(output.count('LN: LNBGS'), 2)
        self.assertGreaterEqual(output.count('LN: SCIPY'), 2)
        self.assertGreaterEqual(output.count('LS: BCHK'), 2)
        self.assertGreaterEqual(output.count('precon: LNBGS'), 2)


    def test_feature_set_solver_print2(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.double_sellar import SubSellar

        prob = om.Problem()
        model = prob.model

        sub1 = model.add_subsystem('sub1', om.Group(), promotes_inputs=['z'])
        sub2 = sub1.add_subsystem('sub2', om.Group(), promotes_inputs=['z'])
        g1 = sub2.add_subsystem('g1', SubSellar(), promotes_inputs=['z'])
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.linear_solver = om.LinearBlockGS()

        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.linear_solver = om.ScipyKrylov()
        g2.linear_solver.precon = om.LinearBlockGS()
        g2.linear_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=2)
        prob.set_solver_print(level=-1, type_='LN')

        prob.setup()
        prob.set_val('z', np.array([5.0, 2.0]))
        prob.run_model()

    def test_set_solver_print2(self):
        old_stdout = sys.stdout
        sys.stdout = str_out = StringIO()

        try:
            self.test_feature_set_solver_print2()
        finally:
            sys.stdout = old_stdout

        # Check that certain things show up in our outputs
        output = str_out.getvalue()
        self.assertGreaterEqual(output.count('sub1.sub2.g1'), 1)
        self.assertGreaterEqual(output.count('g2'), 1)
        self.assertGreaterEqual(output.count('NL: Newton'), 2)
        self.assertGreaterEqual(output.count('LS: BCHK'), 2)

    def test_feature_set_solver_print3(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.double_sellar import SubSellar

        prob = om.Problem()
        model = prob.model

        sub1 = model.add_subsystem('sub1', om.Group(), promotes_inputs=['z'])
        sub2 = sub1.add_subsystem('sub2', om.Group(), promotes_inputs=['z'])
        g1 = sub2.add_subsystem('g1', SubSellar(), promotes_inputs=['z'])
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.linear_solver = om.LinearBlockGS()

        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.linear_solver = om.ScipyKrylov()
        g2.linear_solver.precon = om.LinearBlockGS()
        g2.linear_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=0)
        prob.set_solver_print(level=2, depth=2)

        prob.setup()
        prob.set_val('z', np.array([5.0, 2.0]))
        prob.run_model()

    def test_set_solver_print3(self):
        old_stdout = sys.stdout
        sys.stdout = str_out = StringIO()

        try:
            self.test_feature_set_solver_print3()
        finally:
            sys.stdout = old_stdout

        # Check that certain things show up in our outputs
        output = str_out.getvalue()
        self.assertGreaterEqual(output.count('sub1.sub2.g1'), 1)
        self.assertGreaterEqual(output.count('g2'), 1)
        self.assertGreaterEqual(output.count('NL: Newton'), 2)
        self.assertGreaterEqual(output.count('precon: LNBGS'), 2)
        self.assertGreaterEqual(output.count('LS: BCHK'), 2)
        self.assertGreaterEqual(output.count('LN: SCIPY'), 2)

    def test_feature_set_solver_print4(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.double_sellar import SubSellar

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', om.Group())
        sub2 = sub1.add_subsystem('sub2', om.Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.linear_solver = om.LinearBlockGS()

        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.linear_solver = om.ScipyKrylov()
        g2.linear_solver.precon = om.LinearBlockGS()
        g2.linear_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=-1, type_='all')
        g2.set_solver_print(level=2, type_='NL')

        prob.setup()
        prob.run_model()

    def test_set_solver_print4(self):
        old_stdout = sys.stdout
        sys.stdout = str_out = StringIO()

        try:
            self.test_feature_set_solver_print4()
        finally:
            sys.stdout = old_stdout

        # Check that certain things show up in our outputs
        output = str_out.getvalue()
        self.assertEqual(output.count('sub1.sub2.g1'), 0)
        self.assertGreaterEqual(output.count('g2'), 1)
        self.assertGreaterEqual(output.count('NL: Newton'), 2)
        self.assertEqual(output.count('precon: LNBGS'), 0)
        self.assertGreaterEqual(output.count('LS: BCHK'), 1)
        self.assertEqual(output.count('LN: SCIPY'), 0)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MPITests(unittest.TestCase):

    N_PROCS = 2

    def test_hierarchy_iprint(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', om.Group())
        sub2 = sub1.add_subsystem('sub2', om.Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.LinearBlockGS()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.linear_solver = om.LinearBlockGS()

        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.linear_solver = om.PETScKrylov()
        g2.linear_solver.precon = om.LinearBlockGS()
        g2.linear_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=2)

        prob.setup()

        # Conclude setup but don't run model.
        prob.final_setup()

        # if USE_PROC_FILES is not set, solver convergence messages
        # should only appear on proc 0
        output = run_model(prob)
        if model.comm.rank == 0 or os.environ.get('USE_PROC_FILES'):
            self.assertTrue(output.count('\nNL: Newton Converged') == 1)
        else:
            self.assertTrue(output.count('\nNL: Newton Converged') == 0)


if __name__ == "__main__":
    unittest.main()
