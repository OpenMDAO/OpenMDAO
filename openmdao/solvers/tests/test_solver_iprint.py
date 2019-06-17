""" Unit test for the solver printing behavior. """

import os
import sys
import unittest

import numpy as np

from openmdao.api import Problem, NewtonSolver, ScipyKrylov, Group, PETScVector, \
    IndepVarComp, NonlinearBlockGS, NonlinearBlockJac, LinearBlockGS
from openmdao.test_suite.components.double_sellar import SubSellar
from openmdao.test_suite.components.sellar import SellarDerivatives

from openmdao.utils.general_utils import run_model
from openmdao.utils.mpi import MPI


class TestSolverPrint(unittest.TestCase):

    def test_feature_iprint_neg1(self):
        from openmdao.api import Problem, NewtonSolver, ScipyKrylov
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        prob.model = SellarDerivatives()

        prob.setup()

        newton = prob.model.nonlinear_solver = NewtonSolver()
        scipy = prob.model.linear_solver = ScipyKrylov()

        newton.options['maxiter'] = 2

        # use a real bad initial guess
        prob['y1'] = 10000
        prob['y2'] = -26

        newton.options['iprint'] = -1
        scipy.options['iprint'] = -1
        prob.run_model()

    def test_feature_iprint_0(self):
        from openmdao.api import Problem, NewtonSolver, ScipyKrylov
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        prob.model = SellarDerivatives()

        prob.setup()

        newton = prob.model.nonlinear_solver = NewtonSolver()
        scipy = prob.model.linear_solver = ScipyKrylov()

        newton.options['maxiter'] = 1

        prob['y1'] = 10000
        prob['y2'] = -26

        newton.options['iprint'] = 0
        scipy.options['iprint'] = 0

        prob.run_model()

    def test_feature_iprint_1(self):
        from openmdao.api import Problem, NewtonSolver, ScipyKrylov
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        prob.model = SellarDerivatives()

        prob.setup()

        newton = prob.model.nonlinear_solver = NewtonSolver()
        scipy = prob.model.linear_solver = ScipyKrylov()

        newton.options['maxiter'] = 20

        prob['y1'] = 10000
        prob['y2'] = -26

        newton.options['iprint'] = 1
        scipy.options['iprint'] = 0
        prob.run_model()

    def test_feature_iprint_2(self):
        from openmdao.api import Problem, NewtonSolver, ScipyKrylov
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        prob.model = SellarDerivatives()

        prob.setup()

        newton = prob.model.nonlinear_solver = NewtonSolver()
        scipy = prob.model.linear_solver = ScipyKrylov()

        newton.options['maxiter'] = 20

        prob['y1'] = 10000
        prob['y2'] = -20

        newton.options['iprint'] = 2
        scipy.options['iprint'] = 1
        prob.run_model()

    def test_hierarchy_iprint(self):

        prob = Problem()
        model = prob.model

        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', Group())
        sub2 = sub1.add_subsystem('sub2', Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = ScipyKrylov()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        g1.nonlinear_solver = NewtonSolver()
        g1.linear_solver = LinearBlockGS()

        g2.nonlinear_solver = NewtonSolver()
        g2.linear_solver = ScipyKrylov()
        g2.linear_solver.precon = LinearBlockGS()
        g2.linear_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=2)

        prob.setup()

        output = run_model(prob)
        # TODO: check output

    def test_hierarchy_iprint2(self):

        prob = Problem()
        model = prob.model

        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', Group())
        sub2 = sub1.add_subsystem('sub2', Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = NonlinearBlockGS()
        g1.nonlinear_solver = NonlinearBlockGS()
        g2.nonlinear_solver = NonlinearBlockGS()

        prob.set_solver_print(level=2)

        prob.setup()

        output = run_model(prob)
        # TODO: check output

    def test_hierarchy_iprint3(self):

        prob = Problem()
        model = prob.model

        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', Group())
        sub2 = sub1.add_subsystem('sub2', Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = NonlinearBlockJac()
        sub1.nonlinear_solver = NonlinearBlockJac()
        sub2.nonlinear_solver = NonlinearBlockJac()
        g1.nonlinear_solver = NonlinearBlockJac()
        g2.nonlinear_solver = NonlinearBlockJac()

        prob.set_solver_print(level=2)

        prob.setup()

        output = run_model(prob)
        # TODO: check output

    def test_feature_set_solver_print1(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, LinearBlockGS
        from openmdao.test_suite.components.double_sellar import SubSellar

        prob = Problem()
        model = prob.model

        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', Group())
        sub2 = sub1.add_subsystem('sub2', Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = ScipyKrylov()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        g1.nonlinear_solver = NewtonSolver()
        g1.linear_solver = LinearBlockGS()

        g2.nonlinear_solver = NewtonSolver()
        g2.linear_solver = ScipyKrylov()
        g2.linear_solver.precon = LinearBlockGS()
        g2.linear_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=2)

        prob.setup()
        prob.run_model()

    def test_feature_set_solver_print2(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, LinearBlockGS
        from openmdao.test_suite.components.double_sellar import SubSellar

        prob = Problem()
        model = prob.model

        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', Group())
        sub2 = sub1.add_subsystem('sub2', Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = ScipyKrylov()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        g1.nonlinear_solver = NewtonSolver()
        g1.linear_solver = LinearBlockGS()

        g2.nonlinear_solver = NewtonSolver()
        g2.linear_solver = ScipyKrylov()
        g2.linear_solver.precon = LinearBlockGS()
        g2.linear_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=2)
        prob.set_solver_print(level=-1, type_='LN')

        prob.setup()
        prob.run_model()

    def test_feature_set_solver_print3(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, LinearBlockGS
        from openmdao.test_suite.components.double_sellar import SubSellar

        prob = Problem()
        model = prob.model

        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', Group())
        sub2 = sub1.add_subsystem('sub2', Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = ScipyKrylov()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        g1.nonlinear_solver = NewtonSolver()
        g1.linear_solver = LinearBlockGS()

        g2.nonlinear_solver = NewtonSolver()
        g2.linear_solver = ScipyKrylov()
        g2.linear_solver.precon = LinearBlockGS()
        g2.linear_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=0)
        prob.set_solver_print(level=2, depth=2)

        prob.setup()
        prob.run_model()


@unittest.skipUnless(PETScVector, "PETSc is required.")
class MPITests(unittest.TestCase):

    N_PROCS = 2

    @unittest.skipUnless(MPI, "MPI is not active.")
    def test_hierarchy_iprint(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', Group())
        sub2 = sub1.add_subsystem('sub2', Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = ScipyKrylov()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        g1.nonlinear_solver = NewtonSolver()
        g1.linear_solver = LinearBlockGS()

        g2.nonlinear_solver = NewtonSolver()
        g2.linear_solver = ScipyKrylov()
        g2.linear_solver.precon = LinearBlockGS()
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
