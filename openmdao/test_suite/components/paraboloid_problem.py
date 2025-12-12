import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid


class ParaboloidProblem(om.Problem):
    """
    Paraboloid problem with Constraint.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model = self.model
        model.add_subsystem('p1', om.IndepVarComp('x', 50.0, shape=()), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0, shape=()), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(default_shape=()), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y', default_shape=()), promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)
