import openmdao.api as om
from openmdao.test_suite.components.sellar_feature import SellarMDAWithUnits
import numpy as np

def case_reader_base():
    # build the model
    prob = om.Problem(model=SellarMDAWithUnits())

    model = prob.model
    model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                            upper=np.array([10.0, 10.0]))
    model.add_design_var('x', lower=0.0, upper=10.0)
    model.add_objective('obj')
    model.add_constraint('con1', upper=0.0)
    model.add_constraint('con2', upper=0.0)

    # setup the optimization
    driver = prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

    # Here we show how to attach recorders to each of the four objects; problem, driver, solver, and system
    # Create a recorder variable
    recorder = om.SqliteRecorder('cases.sql')
    # Attach a recorder to the problem
    prob.add_recorder(recorder)
    # Attach a recorder to the driver
    driver.add_recorder(recorder)
    # Attach a recorder to the solver
    prob.model.nonlinear_solver.add_recorder(recorder)

    prob.setup()

    # To attach a recorder to the system, you need to call it after `setup` so the model hierarchy has been generated
    obj_cmp = prob.model.obj_cmp
    obj_cmp.add_recorder(recorder)

    prob.set_solver_print(0)
    prob.run_driver()
    prob.record ("final_state")

    # Instantiate your CaseReader
    cr = om.CaseReader("cases.sql")

    return cr
