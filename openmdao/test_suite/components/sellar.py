""" Test objects for the sellar two discipline problem.
From Sellar's analytic problem.

    Sellar, R. S., Batill, S. M., and Renaud, J. E., "Response Surface Based, Concurrent Subspace
    Optimization for Multidisciplinary System Design," Proceedings References 79 of the 34th AIAA
    Aerospace Sciences Meeting and Exhibit, Reno, NV, January 1996.
"""

import numpy as np

from openmdao.components.exec_comp import ExecComp
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.group import Group
from openmdao.solvers.nl_bgs import NonlinearBlockGS
from openmdao.solvers.ln_scipy import ScipyIterativeSolver
from openmdao.solvers.nl_newton import NewtonSolver


class SellarDis1(ExplicitComponent):
    """Component containing Discipline 1 -- no derivatives version."""

    def __init__(self):
        super(SellarDis1, self).__init__()

        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Local Design Variable
        self.add_input('x', val=0.)

        # Coupling parameter
        self.add_input('y2', val=1.0)

        # Coupling output
        self.add_output('y1', val=1.0)

        self.execution_count = 0

    def compute(self, params, unknowns):
        """Evaluates the equation
        y1 = z1**2 + z2 + x1 - 0.2*y2"""

        z1 = params['z'][0]
        z2 = params['z'][1]
        x1 = params['x']
        y2 = params['y2']

        unknowns['y1'] = z1**2 + z2 + x1 - 0.2*y2

        self.execution_count += 1


class SellarDis1withDerivatives(SellarDis1):
    """Component containing Discipline 1 -- derivatives version."""

    def compute_jacobian(self, params, unknowns, J):
        """ Jacobian for Sellar discipline 1."""

        J['y1','y2'] = -0.2
        J['y1','z'] = np.array([[2.0*params['z'][0], 1.0]])
        J['y1','x'] = 1.0


class SellarDis2(ExplicitComponent):
    """Component containing Discipline 2 -- no derivatives version."""

    def __init__(self):
        super(SellarDis2, self).__init__()

        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

        # Coupling parameter
        self.add_input('y1', val=1.0)

        # Coupling output
        self.add_output('y2', val=1.0)

        self.execution_count = 0

    def compute(self, params, unknowns):
        """Evaluates the equation
        y2 = y1**(.5) + z1 + z2"""

        z1 = params['z'][0]
        z2 = params['z'][1]
        y1 = params['y1']

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        if y1.real < 0.0:
            y1 *= -1

        unknowns['y2'] = y1**.5 + z1 + z2

        self.execution_count += 1


class SellarDis2withDerivatives(SellarDis2):
    """Component containing Discipline 2 -- derivatives version."""

    def compute_jacobian(self, params, unknowns, J):
        """ Jacobian for Sellar discipline 2."""

        J['y2', 'y1'] = .5*params['y1']**-.5
        J['y2', 'z'] = np.array([[1.0, 1.0]])


class SellarNoDerivatives(Group):
    """ Group containing the Sellar MDA. This version uses the disciplines
    without derivatives."""

    def __init__(self):
        super(SellarNoDerivatives, self).__init__()

        self.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        cycle = self.add_subsystem('cycle', Group(), promotes=['x', 'z', 'y1', 'y2'])
        cycle.ln_solver = ScipyIterativeSolver()
        d1 = cycle.add_subsystem('d1', SellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
        d2 = cycle.add_subsystem('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])

        self.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                     z=np.array([0.0, 0.0]), x=0.0),
                 promotes=['x', 'z', 'y1', 'y2', 'obj'])

        self.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        self.nl_solver = NonlinearBlockGS()
        d1.deriv_options['type'] = 'fd'
        d2.deriv_options['type'] = 'fd'


class SellarDerivatives(Group):
    """ Group containing the Sellar MDA. This version uses the disciplines
    with derivatives."""

    def __init__(self):
        super(SellarDerivatives, self).__init__()

        self.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        self.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        self.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        self.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                     z=np.array([0.0, 0.0]), x=0.0),
                 promotes=['obj', 'x', 'z', 'y1', 'y2'])

        self.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        self.nl_solver = NonlinearBlockGS()
        self.ln_solver = ScipyIterativeSolver()


class SellarDerivativesGrouped(Group):
    """ Group containing the Sellar MDA. This version uses the disciplines
    without derivatives."""

    def __init__(self):
        super(SellarDerivativesGrouped, self).__init__()

        self.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        mda = self.add_subsystem('mda', Group(), promotes=['x', 'z', 'y1', 'y2'])
        mda.ln_solver = ScipyIterativeSolver()
        d1 = mda.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        d2 = mda.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        self.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                     z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                 promotes=['obj', 'x', 'z', 'y1', 'y2'])

        self.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        mda.nl_solver = NonlinearBlockGS()
        #d1.deriv_options['type'] = 'fd'
        #d2.deriv_options['type'] = 'fd'

        self.ln_solver = ScipyIterativeSolver()

class StateConnection(ImplicitComponent):
    """ Define connection with an explicit equation"""

    def __init__(self):
        super(StateConnection, self).__init__()

        # Inputs
        self.add_input('y2_actual', 1.0)

        # States
        self.add_output('y2_command', val=1.0)

    def apply_nonlinear(self, params, unknowns, resids):
        """ Don't solve; just calculate the residual."""

        y2_actual = params['y2_actual']
        y2_command = unknowns['y2_command']

        resids['y2_command'] = y2_actual - y2_command

    def compute(self, params, unknowns):
        """ This is a dummy comp that doesn't modify its state."""
        pass

    def linearize(self, params, unknowns, J):
        """Analytical derivatives."""

        # State equation
        J[('y2_command', 'y2_command')] = -1.0
        J[('y2_command', 'y2_actual')] = 1.0


class SellarStateConnection(Group):
    """ Group containing the Sellar MDA. This version uses the disciplines
    with derivatives."""

    def __init__(self):
        super(SellarStateConnection, self).__init__()

        self.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub = self.add_subsystem('sub', Group(), promotes=['x', 'z', 'y1', 'state_eq.y2_actual', 'state_eq.y2_command', 'd1.y2', 'd2.y2'])
        sub.ln_solver = ScipyIterativeSolver()

        subgrp = sub.add_subsystem('state_eq_group', Group(), promotes=['state_eq.y2_actual', 'state_eq.y2_command'])
        subgrp.ln_solver = ScipyIterativeSolver()
        subgrp.add_subsystem('state_eq', StateConnection())

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1'])

        self.connect('state_eq.y2_command', 'd1.y2')
        self.connect('d2.y2', 'state_eq.y2_actual')

        self.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                     z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                 promotes=['x', 'z', 'y1', 'obj'])
        self.connect('d2.y2', 'obj_cmp.y2')

        self.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2'])
        self.connect('d2.y2', 'con_cmp2.y2')

        self.nl_solver = NewtonSolver()
