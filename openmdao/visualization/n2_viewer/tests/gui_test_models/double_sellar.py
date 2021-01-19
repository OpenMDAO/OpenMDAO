import openmdao.api as om
from openmdao.test_suite.components.double_sellar import DoubleSellar

prob = om.Problem()
model = prob.model = DoubleSellar()

# each SubSellar group converges itself
g1 = model.g1
g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
g1.linear_solver = om.DirectSolver()  # used for derivatives

g2 = model.g2
g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
g2.linear_solver = om.DirectSolver()

# Converge the outer loop with Gauss Seidel, with a looser tolerance.
model.nonlinear_solver = om.NonlinearBlockGS(rtol=1.0e-5)
model.linear_solver = om.ScipyKrylov()
model.linear_solver.precon = om.LinearBlockGS()

prob.setup()
prob.run_model()
