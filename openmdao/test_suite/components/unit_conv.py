""" Some models used in unit conversion tests."""

import numpy as np

import openmdao.api as om


class SrcComp(om.ExplicitComponent):
    """Source provides degrees Celsius."""

    def setup(self):
        self.add_input('x1', 100.0)
        self.add_output('x2', 100.0, units='degC')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """ Pass through."""
        outputs['x2'] = inputs['x1']

    def compute_partials(self, inputs, partials):
        """ Derivative is 1.0"""
        partials['x2', 'x1'] = 1.0


class SrcCompFD(SrcComp):
    """Source provides degrees Celsius."""

    def setup(self):
        self.add_input('x1', 100.0)
        self.add_output('x2', 100.0, units='degC')

        self.declare_partials('*', '*', method='fd')

    def compute_partials(self, inputs, partials):
        """ Override"""
        pass


class TgtCompF(om.ExplicitComponent):
    """Target expressed in degrees F."""

    def setup(self):
        self.add_input('x2', 100.0, units='degF')
        self.add_output('x3', 100.0)

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """ Pass through."""
        outputs['x3'] = inputs['x2']

    def compute_partials(self, inputs, partials):
        """ Derivative is 1.0"""
        partials['x3', 'x2'] = 1.0


class TgtCompFFD(TgtCompF):
    """Source provides degrees Celsius."""

    def setup(self):
        self.add_input('x2', 100.0, units='degF')
        self.add_output('x3', 100.0)

        self.declare_partials('*', '*', method='fd')

    def compute_partials(self, inputs, partials):
        """ Override"""
        pass


class TgtCompC(om.ExplicitComponent):
    """Target expressed in degrees Celsius."""

    def setup(self):
        self.add_input('x2', 100.0, units='degC')
        self.add_output('x3', 100.0)

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """ Pass through."""
        outputs['x3'] = inputs['x2']

    def compute_partials(self, inputs, partials):
        """ Derivative is 1.0"""
        partials['x3', 'x2'] = 1.0


class TgtCompCFD(TgtCompC):
    """Source provides degrees Celsius."""

    def setup(self):
        self.add_input('x2', 100.0, units='degC')
        self.add_output('x3', 100.0)

        self.declare_partials('*', '*', method='fd')

    def compute_partials(self, inputs, partials):
        """ Override"""
        pass


class TgtCompK(om.ExplicitComponent):
    """Target expressed in degrees Kelvin."""

    def setup(self):
        self.add_input('x2', 100.0, units='degK')
        self.add_output('x3', 100.0)

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """ Pass through."""
        outputs['x3'] = inputs['x2']

    def compute_partials(self, inputs, partials):
        """ Derivative is 1.0"""
        partials['x3', 'x2'] = 1.0


class TgtCompKFD(TgtCompK):
    """Source provides degrees Celsius."""

    def setup(self):
        self.add_input('x2', 100.0, units='degK')
        self.add_output('x3', 100.0)

        self.declare_partials('*', '*', method='fd')

    def compute_partials(self, inputs, partials):
        """ Override"""
        pass


class TgtCompFMulti(om.ExplicitComponent):
    """Contains some extra inputs that might trip things up."""

    def setup(self):
        self.add_input('_x2', 100.0, units='degF')
        self.add_input('x2', 100.0, units='degF')
        self.add_input('x2_', 100.0, units='degF')
        self.add_output('_x3', 100.0)
        self.add_output('x3', 100.0)
        self.add_output('x3_', 100.0)

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """ Pass through."""
        outputs['x3'] = inputs['x2']

    def compute_partials(self, inputs, J):
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


class UnitConvGroup(om.Group):
    """Group containing a degC source that feeds into three targets with
    units degF, degC, and degK respectively. Good for testing unit
    conversion."""

    def __init__(self, **kwargs):
        super(UnitConvGroup, self).__init__(**kwargs)

        self.add_subsystem('px1', om.IndepVarComp('x1', 100.0))
        self.add_subsystem('src', SrcComp())
        self.add_subsystem('tgtF', TgtCompF())
        self.add_subsystem('tgtC', TgtCompC())
        self.add_subsystem('tgtK', TgtCompK())

        self.connect('px1.x1', 'src.x1')
        self.connect('src.x2', 'tgtF.x2')
        self.connect('src.x2', 'tgtC.x2')
        self.connect('src.x2', 'tgtK.x2')


class UnitConvGroupImplicitConns(om.Group):
    """ Group containing a defF source that feeds into three targets with
    units degF, degC, and degK respectively. Good for testing unit
    conversion.

    In this version, all connections are Implicit.
    """

    def __init__(self):
        super(UnitConvGroupImplicitConns, self).__init__()

        self.add_subsystem('px1', om.IndepVarComp('x1', 100.0), promotes_outputs=['x1'])
        self.add_subsystem('src', SrcComp(), promotes_inputs=['x1'], promotes_outputs=['x2'])
        self.add_subsystem('tgtF', TgtCompF(), promotes_inputs=['x2'])
        self.add_subsystem('tgtC', TgtCompC(), promotes_inputs=['x2'])
        self.add_subsystem('tgtK', TgtCompK(), promotes_inputs=['x2'])
