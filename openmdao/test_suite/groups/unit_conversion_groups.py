""" Some groups that are used to test unit conversion."""

from __future__ import division, print_function

from openmdao.core.component import Component
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.group import Group


class SrcComp(Component):
    """ Source provides degrees Celsius. """

    def __init__(self):
        super(SrcComp, self).__init__()

        self.add_input('x1', 100.0)
        self.add_output('x2', 100.0, units='degC')

    def compute(self, params, unknowns):
        """ Pass through."""
        unknowns['x2'] = params['x1']

    def compute_jacobian(self, params, unknowns, resids, J):
        """ Derivative is 1.0"""
        J['x2', 'x1'] = np.array([1.0])


class TgtCompF(Component):
    """ Target expressed in degrees F."""
    def __init__(self):
        super(TgtCompF, self).__init__()

        self.add_input('x2', 100.0, units='degF')
        self.add_output('x3', 100.0)

    def compute(self, params, unknowns):
        """ Pass through."""
        unknowns['x3'] = params['x2']

    def compute_jacobian(self, params, unknowns, resids, J):
        """ Derivative is 1.0"""
        J['x3', 'x2'] = np.array([1.0])


class TgtCompFMulti(Component):
    """ Contains some extra inputs that might trip things up.
    """

    def __init__(self):
        super(TgtCompFMulti, self).__init__()

        self.add_input('_x2', 100.0, units='degF')
        self.add_input('x2', 100.0, units='degF')
        self.add_input('x2_', 100.0, units='degF')
        self.add_output('_x3', 100.0)
        self.add_output('x3', 100.0)
        self.add_output('x3_', 100.0)

    def compute(self, params, unknowns):
        """ Pass through."""
        unknowns['x3'] = params['x2']

    def compute_jacobian(self, params, unknowns, resids, J):
        """ Derivative is 1.0"""
        J['_x3', 'x2'] = np.array([1.0])
        J['_x3', '_x2'] = 0.0
        J['_x3', 'x2_'] = 0.0
        J['x3', 'x2'] = np.array([1.0])
        J['x3', '_x2'] = 0.0
        J['x3', 'x2_'] = 0.0
        J['x3_', 'x2'] = np.array([1.0])
        J['x3_', '_x2'] = 0.0
        J['x3_', 'x2_'] = 0.0


class TgtCompC(Component):
    """ Target expressed in degrees Celsius."""

    def __init__(self):
        super(TgtCompC, self).__init__()

        self.add_input('x2', 100.0, units='degC')
        self.add_output('x3', 100.0)

    def compute(self, params, unknowns):
        """ Pass through."""
        unknowns['x3'] = params['x2']

    def compute_jacobian(self, params, unknowns, resids, J):
        """ Derivative is 1.0"""
        J['x3', 'x2'] = np.array([1.0])


class TgtCompK(Component):
    """ Target expressed in degrees Kelvin."""

    def __init__(self):
        super(TgtCompK, self).__init__()

        self.add_input('x2', 100.0, units='degK')
        self.add_output('x3', 100.0)

    def compute(self, params, unknowns):
        """ Pass through."""
        unknowns['x3'] = params['x2']

    def compute_jacobian(self, params, unknowns, resids, J):
        """ Derivative is 1.0"""
        J['x3', 'x2'] = np.array([1.0])


class UnitConvGroup(Group):
    """ Group containing a defF source that feeds into three targets with
    units degF, degC, and degK respectively. Good for testing unit
    conversion."""

    def __init__(self):
        super(UnitConvGroup, self).__init__()

        self.add_subsystem('src', SrcComp())
        self.add_subsystem('tgtF', TgtCompF())
        self.add_subsystem('tgtC', TgtCompC())
        self.add_subsystem('tgtK', TgtCompK())

        self.add_subsystem('px1', IndepVarComp('x1', 100.0), promotes=['x1'])

        self.connect('x1', 'src.x1')
        self.connect('src.x2', 'tgtF.x2')
        self.connect('src.x2', 'tgtC.x2')
        self.connect('src.x2', 'tgtK.x2')


class UnitConvGroupImplicitConns(Group):
    """ Group containing a defF source that feeds into three targets with
    units degF, degC, and degK respectively. Good for testing unit
    conversion.

    In this version, all connections are Implicit.
    """

    def __init__(self):
        super(UnitConvGroupImplicitConns, self).__init__()

        self.add_subsystem('src', SrcComp(), promotes=['x1', 'x2'])
        self.add_subsystem('tgtF', TgtCompF(), promotes=['x2'])
        self.add_subsystem('tgtC', TgtCompC(), promotes=['x2'])
        self.add_subsystem('tgtK', TgtCompK(), promotes=['x2'])
        self.add_subsystem('px1', IndepVarComp('x1', 100.0), promotes=['x1'])