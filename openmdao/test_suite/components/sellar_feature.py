"""
Test objects for the sellar two discipline problem.

From Sellar's analytic problem.

    Sellar, R. S., Batill, S. M., and Renaud, J. E., "Response Surface Based, Concurrent Subspace
    Optimization for Multidisciplinary System Design," Proceedings References 79 of the 34th AIAA
    Aerospace Sciences Meeting and Exhibit, Reno, NV, January 1996.
"""
import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, \
                         SellarDis2withDerivatives
from openmdao.test_suite.components.double_sellar import SubSellar


class SellarDis1(om.ExplicitComponent):
    """
    Component containing Discipline 1 -- no derivatives version.
    """

    def setup(self):

        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Local Design Variable
        self.add_input('x', val=0.)

        # Coupling parameter
        self.add_input('y2', val=1.0)

        # Coupling output
        self.add_output('y1', val=1.0)

        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y1 = z1**2 + z2 + x1 - 0.2*y2
        """
        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        x1 = inputs['x']
        y2 = inputs['y2']

        outputs['y1'] = z1**2 + z2 + x1 - 0.2*y2


class SellarDis2(om.ExplicitComponent):
    """
    Component containing Discipline 2 -- no derivatives version.
    """

    def setup(self):
        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Coupling parameter
        self.add_input('y1', val=1.0)

        # Coupling output
        self.add_output('y2', val=1.0)

        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y2 = y1**(.5) + z1 + z2
        """

        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        y1 = inputs['y1']

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        if y1.real < 0.0:
            y1 *= -1

        outputs['y2'] = y1**.5 + z1 + z2


class SellarMDA(om.Group):
    """
    Group containing the Sellar MDA.
    """

    def setup(self):
        indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('x', 1.0)
        indeps.add_output('z', np.array([5.0, 2.0]))

        cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        cycle.add_subsystem('d1', SellarDis1(), promotes_inputs=['x', 'z', 'y2'],
                            promotes_outputs=['y1'])
        cycle.add_subsystem('d2', SellarDis2(), promotes_inputs=['z', 'y1'],
                            promotes_outputs=['y2'])

        # Nonlinear Block Gauss Seidel is a gradient free solver
        cycle.nonlinear_solver = NonlinearBlockGS()

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0),
                           promotes=['x', 'z', 'y1', 'y2', 'obj'])

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])


class SellarMDALinearSolver(om.Group):
    """
    Group containing the Sellar MDA.
    """

    def setup(self):
        indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('x', 1.0)
        indeps.add_output('z', np.array([5.0, 2.0]))

        cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        d1 = cycle.add_subsystem('d1', SellarDis1(), promotes_inputs=['x', 'z', 'y2'],
                                 promotes_outputs=['y1'])
        d2 = cycle.add_subsystem('d2', SellarDis2(), promotes_inputs=['z', 'y1'],
                                 promotes_outputs=['y2'])

        cycle.nonlinear_solver = om.NonlinearBlockGS()
        cycle.linear_solver = om.DirectSolver()

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                           z=np.array([0.0, 0.0]), x=0.0),
                           promotes=['x', 'z', 'y1', 'y2', 'obj'])

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])


class SellarDis1CS(om.ExplicitComponent):
    """
    Component containing Discipline 1 -- no derivatives version.
    Uses Complex Step
    """

    def setup(self):

        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Local Design Variable
        self.add_input('x', val=0.)

        # Coupling parameter
        self.add_input('y2', val=1.0)

        # Coupling output
        self.add_output('y1', val=1.0)

        # Finite difference all partials.
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y1 = z1**2 + z2 + x1 - 0.2*y2
        """
        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        x1 = inputs['x']
        y2 = inputs['y2']

        outputs['y1'] = z1**2 + z2 + x1 - 0.2*y2


class SellarDis2CS(om.ExplicitComponent):
    """
    Component containing Discipline 2 -- no derivatives version.
    Uses Complex Step
    """

    def setup(self):
        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Coupling parameter
        self.add_input('y1', val=1.0)

        # Coupling output
        self.add_output('y2', val=1.0)

        # Finite difference all partials.
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y2 = y1**(.5) + z1 + z2
        """

        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        y1 = inputs['y1']

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        if y1.real < 0.0:
            y1 *= -1

        outputs['y2'] = y1**.5 + z1 + z2


class SellarNoDerivativesCS(om.Group):
    """
    Group containing the Sellar MDA. This version uses the disciplines without derivatives.
    """

    def setup(self):
        self.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        self.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        cycle = self.add_subsystem('cycle', om.Group(), promotes=['x', 'z', 'y1', 'y2'])
        d1 = cycle.add_subsystem('d1', SellarDis1CS(), promotes=['x', 'z', 'y1', 'y2'])
        d2 = cycle.add_subsystem('d2', SellarDis2CS(), promotes=['z', 'y1', 'y2'])

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0),
                           promotes=['x', 'z', 'y1', 'y2', 'obj'])

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        self.nonlinear_solver = NonlinearBlockGS()
        self.linear_solver = ScipyKrylov()


class SellarIDF(om.Group):
    """
    Individual Design Feasible (IDF) architecture for the Sellar problem.
    """
    def setup(self):
        # construct the Sellar model with `y1` and `y2` as independent variables
        dv = om.IndepVarComp()
        dv.add_output('x', 5.)
        dv.add_output('y1', 5.)
        dv.add_output('y2', 5.)
        dv.add_output('z', np.array([2., 0.]))

        self.add_subsystem('dv', dv)
        self.add_subsystem('d1', SellarDis1withDerivatives())
        self.add_subsystem('d2', SellarDis2withDerivatives())

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                           x=0., z=np.array([0., 0.])))

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'))
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'))

        self.connect('dv.x', ['d1.x', 'obj_cmp.x'])
        self.connect('dv.y1', ['d2.y1', 'obj_cmp.y1', 'con_cmp1.y1'])
        self.connect('dv.y2', ['d1.y2', 'obj_cmp.y2', 'con_cmp2.y2'])
        self.connect('dv.z', ['d1.z', 'd2.z', 'obj_cmp.z'])

        # rather than create a cycle by connecting d1.y1 to d2.y1 and d2.y2 to d1.y2
        # we will constrain y1 and y2 to be equal for the two disciplines

        equal = om.EQConstraintComp()
        self.add_subsystem('equal', equal)

        equal.add_eq_output('y1', add_constraint=True)
        equal.add_eq_output('y2', add_constraint=True)

        self.connect('dv.y1', 'equal.lhs:y1')
        self.connect('d1.y1', 'equal.rhs:y1')

        self.connect('dv.y2', 'equal.lhs:y2')
        self.connect('d2.y2', 'equal.rhs:y2')

        # the driver will effectively solve the cycle
        # by satisfying the equality constraints

        self.add_design_var('dv.x', lower=0., upper=5.)
        self.add_design_var('dv.y1', lower=0., upper=5.)
        self.add_design_var('dv.y2', lower=0., upper=5.)
        self.add_design_var('dv.z', lower=np.array([-5., 0.]), upper=np.array([5., 5.]))
        self.add_objective('obj_cmp.obj')
        self.add_constraint('con_cmp1.con1', upper=0.)
        self.add_constraint('con_cmp2.con2', upper=0.)

