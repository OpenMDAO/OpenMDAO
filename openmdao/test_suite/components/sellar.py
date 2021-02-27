"""
Test objects for the sellar two discipline problem.

From Sellar's analytic problem.

    Sellar, R. S., Batill, S. M., and Renaud, J. E., "Response Surface Based, Concurrent Subspace
    Optimization for Multidisciplinary System Design," Proceedings References 79 of the 34th AIAA
    Aerospace Sciences Meeting and Exhibit, Reno, NV, January 1996.
"""
import inspect

import numpy as np

import openmdao.api as om


class SellarDis1(om.ExplicitComponent):
    """
    Component containing Discipline 1 -- no derivatives version.
    """

    def __init__(self, units=None, scaling=None):
        super().__init__()
        self.execution_count = 0
        self._units = units
        self._do_scaling = scaling

    def setup(self):

        if self._units:
            units = 'ft'
        else:
            units = None

        if self._do_scaling:
            ref = .1
        else:
            ref = 1.

        # Global Design Variable
        self.add_input('z', val=np.zeros(2), units=units)

        # Local Design Variable
        self.add_input('x', val=0., units=units)

        # Coupling parameter
        self.add_input('y2', val=1.0, units=units)

        # Coupling output
        self.add_output('y1', val=1.0, lower=0.1, upper=1000., units=units, ref=ref)

    def setup_partials(self):
        # Finite difference everything
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

        self.execution_count += 1


class SellarDis1withDerivatives(SellarDis1):
    """
    Component containing Discipline 1 -- derivatives version.
    """

    def setup_partials(self):
        # Analytic Derivs
        self.declare_partials(of='*', wrt='*')

    def compute_partials(self, inputs, partials):
        """
        Jacobian for Sellar discipline 1.
        """
        partials['y1', 'y2'] = -0.2
        partials['y1', 'z'] = np.array([[2.0 * inputs['z'][0], 1.0]])
        partials['y1', 'x'] = 1.0


class SellarDis1CS(SellarDis1):
    """
    Component containing Discipline 1 -- complex step version.
    """

    def setup_partials(self):
        # Analytic Derivs
        self.declare_partials(of='*', wrt='*', method='cs')


class SellarDis2(om.ExplicitComponent):
    """
    Component containing Discipline 2 -- no derivatives version.
    """

    def __init__(self, units=None, scaling=None):
        super().__init__()
        self.execution_count = 0
        self._units = units
        self._do_scaling = scaling

    def setup(self):
        if self._units:
            units = 'inch'
        else:
            units = None

        if self._do_scaling:
            ref = .18
        else:
            ref = 1.

        # Global Design Variable
        self.add_input('z', val=np.zeros(2), units=units)

        # Coupling parameter
        self.add_input('y1', val=1.0, units=units)

        # Coupling output
        self.add_output('y2', val=1.0, lower=0.1, upper=1000., units=units, ref=ref)

    def setup_partials(self):
        # Finite difference everything
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

        self.execution_count += 1


class SellarDis2withDerivatives(SellarDis2):
    """
    Component containing Discipline 2 -- derivatives version.
    """

    def setup_partials(self):
        # Analytic Derivs
        self.declare_partials(of='*', wrt='*')

    def compute_partials(self, inputs, J):
        """
        Jacobian for Sellar discipline 2.
        """
        y1 = inputs['y1']
        if y1.real < 0.0:
            y1 *= -1
        if y1.real < 1e-8:
            y1 = 1e-8

        J['y2', 'y1'] = .5*y1**-.5
        J['y2', 'z'] = np.array([[1.0, 1.0]])


class SellarDis2CS(SellarDis2):
    """
    Component containing Discipline 2 -- complex step version.
    """

    def setup_partials(self):
        # Analytic Derivs
        self.declare_partials(of='*', wrt='*', method='cs')


class SellarNoDerivatives(om.Group):
    """
    Group containing the Sellar MDA. This version uses the disciplines without derivatives.
    """

    def initialize(self):
        self.options.declare('nonlinear_solver', default=om.NonlinearBlockGS,
                             desc='Nonlinear solver for Sellar MDA')
        self.options.declare('nl_atol', default=None,
                             desc='User-specified atol for nonlinear solver.')
        self.options.declare('nl_maxiter', default=None,
                             desc='Iteration limit for nonlinear solver.')
        self.options.declare('linear_solver', default=om.ScipyKrylov,
                             desc='Linear solver')
        self.options.declare('ln_atol', default=None,
                             desc='User-specified atol for linear solver.')
        self.options.declare('ln_maxiter', default=None,
                             desc='Iteration limit for linear solver.')

    def setup(self):
        cycle = self.add_subsystem('cycle', om.Group(), promotes=['x', 'z', 'y1', 'y2'])
        cycle.add_subsystem('d1', SellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
        cycle.add_subsystem('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0),
                           promotes=['x', 'z', 'y1', 'y2', 'obj'])

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        self.set_input_defaults('x', 1.0)
        self.set_input_defaults('z', np.array([5.0, 2.0]))

        nl = self.options['nonlinear_solver']
        self.nonlinear_solver = nl() if inspect.isclass(nl) else nl
        if self.options['nl_atol']:
            self.nonlinear_solver.options['atol'] = self.options['nl_atol']
        if self.options['nl_maxiter']:
            self.nonlinear_solver.options['maxiter'] = self.options['nl_maxiter']

    def configure(self):
        ln = self.options['linear_solver']
        self.cycle.linear_solver = ln() if inspect.isclass(ln) else ln
        if self.options['ln_atol']:
            self.cycle.linear_solver.options['atol'] = self.options['ln_atol']
        if self.options['ln_maxiter']:
            self.cycle.linear_solver.options['maxiter'] = self.options['ln_maxiter']


class SellarDerivatives(om.Group):
    """
    Group containing the Sellar MDA. This version uses the disciplines with derivatives.
    """

    def initialize(self):
        self.options.declare('nonlinear_solver', default=om.NonlinearBlockGS,
                             desc='Nonlinear solver (class or instance) for Sellar MDA')
        self.options.declare('nl_atol', default=None,
                             desc='User-specified atol for nonlinear solver.')
        self.options.declare('nl_maxiter', default=None,
                             desc='Iteration limit for nonlinear solver.')
        self.options.declare('linear_solver', default=om.ScipyKrylov,
                             desc='Linear solver (class or instance)')
        self.options.declare('ln_atol', default=None,
                             desc='User-specified atol for linear solver.')
        self.options.declare('ln_maxiter', default=None,
                             desc='Iteration limit for linear solver.')

    def setup(self):
        self.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        self.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)', obj=0.0,
                                                  x=0.0, z=np.array([0.0, 0.0]), y1=0.0, y2=0.0),
                           promotes=['obj', 'x', 'z', 'y1', 'y2'])

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1', con1=0.0, y1=0.0),
                           promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0', con2=0.0, y2=0.0),
                           promotes=['con2', 'y2'])

        self.set_input_defaults('x', 1.0)
        self.set_input_defaults('z', np.array([5.0, 2.0]))

        nl = self.options['nonlinear_solver']
        self.nonlinear_solver = nl() if inspect.isclass(nl) else nl
        if self.options['nl_atol']:
            self.nonlinear_solver.options['atol'] = self.options['nl_atol']
        if self.options['nl_maxiter']:
            self.nonlinear_solver.options['maxiter'] = self.options['nl_maxiter']

        ln = self.options['linear_solver']
        self.linear_solver = ln() if inspect.isclass(ln) else ln
        if self.options['ln_atol']:
            self.linear_solver.options['atol'] = self.options['ln_atol']
        if self.options['ln_maxiter']:
            self.linear_solver.options['maxiter'] = self.options['ln_maxiter']


class SellarDerivativesPreAutoIVC(om.Group):
    """
    Group containing the Sellar MDA. This version uses the disciplines with derivatives.
    """

    def initialize(self):
        self.options.declare('nonlinear_solver', default=om.NonlinearBlockGS,
                             desc='Nonlinear solver (class or instance) for Sellar MDA')
        self.options.declare('nl_atol', default=None,
                             desc='User-specified atol for nonlinear solver.')
        self.options.declare('nl_maxiter', default=None,
                             desc='Iteration limit for nonlinear solver.')
        self.options.declare('linear_solver', default=om.ScipyKrylov,
                             desc='Linear solver (class or instance)')
        self.options.declare('ln_atol', default=None,
                             desc='User-specified atol for linear solver.')
        self.options.declare('ln_maxiter', default=None,
                             desc='Iteration limit for linear solver.')

    def setup(self):
        self.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        self.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)', obj=0.0,
                                                  x=0.0, z=np.array([0.0, 0.0]), y1=0.0, y2=0.0),
                           promotes=['obj', 'x', 'z', 'y1', 'y2'])

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1', con1=0.0, y1=0.0),
                           promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0', con2=0.0, y2=0.0),
                           promotes=['con2', 'y2'])

        self.set_input_defaults('x', 1.0)
        self.set_input_defaults('z', np.array([5.0, 2.0]))

        nl = self.options['nonlinear_solver']
        self.nonlinear_solver = nl() if inspect.isclass(nl) else nl
        if self.options['nl_atol']:
            self.nonlinear_solver.options['atol'] = self.options['nl_atol']
        if self.options['nl_maxiter']:
            self.nonlinear_solver.options['maxiter'] = self.options['nl_maxiter']

        ln = self.options['linear_solver']
        self.linear_solver = ln() if inspect.isclass(ln) else ln
        if self.options['ln_atol']:
            self.linear_solver.options['atol'] = self.options['ln_atol']
        if self.options['ln_maxiter']:
            self.linear_solver.options['maxiter'] = self.options['ln_maxiter']


class SellarDerivativesConnected(om.Group):
    """
    Group containing the Sellar MDA. This version uses the disciplines with derivatives.
    """

    def setup(self):
        self.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z'])
        self.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z'])

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['x', 'z'])

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'))
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'))

        self.connect('d1.y1', ['d2.y1', 'obj_cmp.y1', 'con_cmp1.y1'])
        self.connect('d2.y2', ['d1.y2', 'obj_cmp.y2', 'con_cmp2.y2'])

        self.set_input_defaults('x', 1.0)
        self.set_input_defaults('z', np.array([5.0, 2.0]))

        self.nonlinear_solver = om.NonlinearBlockGS()
        self.linear_solver = om.ScipyKrylov()


class SellarDerivativesGrouped(om.Group):
    """
    Group containing the Sellar MDA. This version uses the disciplines with derivatives.
    """

    def initialize(self):
        self.options.declare('nonlinear_solver', default=om.NonlinearBlockGS, recordable=False,
                             desc='Nonlinear solver (class or instance) for Sellar MDA')
        self.options.declare('nl_atol', default=None,
                             desc='User-specified atol for nonlinear solver.')
        self.options.declare('nl_maxiter', default=None,
                             desc='Iteration limit for nonlinear solver.')
        self.options.declare('linear_solver', default=om.ScipyKrylov, recordable=False,
                             desc='Linear solver (class or instance)')
        self.options.declare('ln_atol', default=None,
                             desc='User-specified atol for linear solver.')
        self.options.declare('ln_maxiter', default=None,
                             desc='Iteration limit for linear solver.')
        self.options.declare('mda_nonlinear_solver', default=om.NonlinearBlockGS, recordable=False,
                             desc='Nonlinear solver (class or instance)')
        self.options.declare('mda_linear_solver', default=om.ScipyKrylov, recordable=False,
                             desc='Linear solver (class or instance) for Sellar MDA')

    def setup(self):
        self.mda = mda = self.add_subsystem('mda', om.Group(), promotes=['x', 'z', 'y1', 'y2'])
        mda.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        mda.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                           promotes=['obj', 'x', 'z', 'y1', 'y2'])

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        self.set_input_defaults('x', 1.0)
        self.set_input_defaults('z', np.array([5.0, 2.0]))

        nl = self.options['nonlinear_solver']
        self.nonlinear_solver = nl() if inspect.isclass(nl) else nl
        if self.options['nl_atol']:
            self.nonlinear_solver.options['atol'] = self.options['nl_atol']
        if self.options['nl_maxiter']:
            self.nonlinear_solver.options['maxiter'] = self.options['nl_maxiter']

        ln = self.options['linear_solver']
        self.linear_solver = ln() if inspect.isclass(ln) else ln
        if self.options['ln_atol']:
            self.linear_solver.options['atol'] = self.options['ln_atol']
        if self.options['ln_maxiter']:
            self.linear_solver.options['maxiter'] = self.options['ln_maxiter']

        nl = self.options['mda_nonlinear_solver']
        self.mda.nonlinear_solver = nl() if inspect.isclass(nl) else nl

        ln = self.options['mda_linear_solver']
        self.mda.linear_solver = ln() if inspect.isclass(ln) else ln


class StateConnection(om.ImplicitComponent):
    """
    Define connection with an explicit equation.
    """

    def setup(self):
        # Inputs
        self.add_input('y2_actual', 1.0)

        # States
        self.add_output('y2_command', val=1.0)

    def setup_partials(self):
        # Declare derivatives
        self.declare_partials(of='*', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Don't solve; just calculate the residual.
        """
        y2_actual = inputs['y2_actual']
        y2_command = outputs['y2_command']

        residuals['y2_command'] = y2_actual - y2_command

    def compute(self, inputs, outputs):
        """
        This is a dummy comp that doesn't modify its state.
        """
        pass

    def linearize(self, inputs, outputs, J):
        """
        Analytical derivatives.
        """

        # State equation
        J[('y2_command', 'y2_command')] = -1.0
        J[('y2_command', 'y2_actual')] = 1.0


class SellarStateConnection(om.Group):
    """
    Group containing the Sellar MDA. This version uses the disciplines with derivatives.
    """

    def initialize(self):
        self.options.declare('nonlinear_solver', default=om.NewtonSolver(solve_subsystems=False),
                             desc='Nonlinear solver (class or instance) for Sellar MDA')
        self.options.declare('nl_atol', default=None,
                             desc='User-specified atol for nonlinear solver.')
        self.options.declare('nl_maxiter', default=None,
                             desc='Iteration limit for nonlinear solver.')
        self.options.declare('linear_solver', default=om.ScipyKrylov,
                             desc='Linear solver (class or instance)')
        self.options.declare('ln_atol', default=None,
                             desc='User-specified atol for linear solver.')
        self.options.declare('ln_maxiter', default=None,
                             desc='Iteration limit for linear solver.')

    def setup(self):
        sub = self.add_subsystem('sub', om.Group(),
                                 promotes=['x', 'z', 'y1',
                                           'state_eq.y2_actual', 'state_eq.y2_command',
                                           'd1.y2', 'd2.y2'])

        subgrp = sub.add_subsystem('state_eq_group', om.Group(),
                                   promotes=['state_eq.y2_actual', 'state_eq.y2_command'])
        subgrp.add_subsystem('state_eq', StateConnection())

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1'])

        self.connect('state_eq.y2_command', 'd1.y2')
        self.connect('d2.y2', 'state_eq.y2_actual')

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                           promotes=['x', 'z', 'y1', 'obj'])
        self.connect('d2.y2', 'obj_cmp.y2')

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2'])
        self.connect('d2.y2', 'con_cmp2.y2')

        self.set_input_defaults('x', 1.0)
        self.set_input_defaults('z', np.array([5.0, 2.0]))

        nl = self.options['nonlinear_solver']
        self.nonlinear_solver = nl() if inspect.isclass(nl) else nl
        if self.options['nl_atol']:
            self.nonlinear_solver.options['atol'] = self.options['nl_atol']
        if self.options['nl_maxiter']:
            self.nonlinear_solver.options['maxiter'] = self.options['nl_maxiter']

        ln = self.options['linear_solver']
        self.linear_solver = ln() if inspect.isclass(ln) else ln
        if self.options['ln_atol']:
            self.linear_solver.options['atol'] = self.options['ln_atol']
        if self.options['ln_maxiter']:
            self.linear_solver.options['maxiter'] = self.options['ln_maxiter']

    def configure(self):
        self.sub.linear_solver = om.ScipyKrylov()
        self.sub.state_eq_group.linear_solver = om.ScipyKrylov()


class SellarImplicitDis1(om.ImplicitComponent):
    """
    Component containing Discipline 1 -- no derivatives version.
    """

    def __init__(self, units=None, scaling=None):
        super().__init__()
        self.execution_count = 0
        self._units = units
        self._do_scaling = scaling

    def setup(self):
        if self._units:
            units = 'ft'
        else:
            units = None

        if self._do_scaling is None:
            ref = 1.
        else:
            ref = .1

        # Global Design Variable
        self.add_input('z', val=np.zeros(2), units=units)

        # Local Design Variable
        self.add_input('x', val=0., units=units)

        # Coupling parameter
        self.add_input('y2', val=1.0, units=units)

        # Coupling output
        self.add_output('y1', val=1.0, lower=-0.1, upper=1000, units=units, ref=ref)

    def setup_partials(self):
        # Derivatives
        self.declare_partials('*', '*')

    def apply_nonlinear(self, inputs, outputs, resids):
        """
        Evaluates the equation
        y1 = z1**2 + z2 + x1 - 0.2*y2
        """

        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        x1 = inputs['x']
        y2 = inputs['y2']

        y1 = outputs['y1']

        resids['y1'] = -(z1**2 + z2 + x1 - 0.2*y2 - y1)

    def linearize(self, inputs, outputs, J):
        """
        Jacobian for Sellar discipline 1.
        """
        J['y1', 'y2'] = 0.2
        J['y1', 'z'] = -np.array([[2.0 * inputs['z'][0], 1.0]])
        J['y1', 'x'] = -1.0
        J['y1', 'y1'] = 1.0


class SellarImplicitDis2(om.ImplicitComponent):
    """
    Component containing Discipline 2 -- implicit version.
    """

    def __init__(self, units=None, scaling=None):
        super().__init__()
        self.execution_count = 0
        self._units = units
        self._do_scaling = scaling

    def setup(self):
        if self._units:
            units = 'inch'
        else:
            units = None

        if self._do_scaling is None:
            ref = 1.0
        else:
            ref = .18

        # Global Design Variable
        self.add_input('z', val=np.zeros(2), units=units)

        # Coupling parameter
        self.add_input('y1', val=1.0, units=units)

        # Coupling output
        self.add_output('y2', val=1.0, lower=0.1, upper=1000., units=units, ref=ref)

    def setup_partials(self):
        # Derivatives
        self.declare_partials('*', '*')

    def apply_nonlinear(self, inputs, outputs, resids):
        """
        Evaluates the equation
        y2 = y1**(.5) + z1 + z2
        """

        z1 = inputs['z'][0]
        z2 = inputs['z'][1]
        y1 = inputs['y1'].copy()

        y2 = outputs['y2']

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        if y1.real < 0.0:
            y1 *= -1

        resids['y2'] = -(y1**.5 + z1 + z2 - y2)

    def linearize(self, inputs, outputs, J):
        """
        Jacobian for Sellar discipline 2.
        """
        y1 = inputs['y1']
        if y1.real < 0.0:
            y1 *= -1
        if y1.real < 1e-8:
            y1 = 1e-8

        J['y2', 'y1'] = -.5*y1**-.5
        J['y2', 'z'] = -np.array([[1.0, 1.0]])
        J['y2', 'y2'] = 1.0


class SellarProblem(om.Problem):
    """
    The Sellar problem with configurable model class.
    """

    def __init__(self, model_class=SellarDerivatives, **kwargs):
        super().__init__(model_class(**kwargs))

        model = self.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        # default to non-verbose
        self.set_solver_print(0)


class SellarProblemWithArrays(om.Problem):
    """
    The Sellar problem with ndarray variable options
    """

    def __init__(self, model_class=SellarDerivatives, **kwargs):
        super().__init__(model_class(**kwargs))

        model = self.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                             upper=np.array([10.0, 10.0]), indices=np.arange(2, dtype=int))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', equals=np.zeros(1))
        model.add_constraint('con2', upper=0.0)

        # default to non-verbose
        self.set_solver_print(0)
