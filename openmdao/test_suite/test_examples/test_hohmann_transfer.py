"""
Find the minimum delta-V for an impulsive Hohmann Transer from
Low Earth Orbit (LEO) to Geostationary Orbit (GEO)
"""
import unittest

import numpy as np

import openmdao.api as om


class VCircComp(om.ExplicitComponent):
    """
    Computes the circular orbit velocity given a radius and gravitational
    parameter.
    """
    def initialize(self):
        pass

    def setup(self):
        self.add_input('r',
                       val=1.0,
                       desc='Radius from central body',
                       units='km')

        self.add_input('mu',
                       val=1.0,
                       desc='Gravitational parameter of central body',
                       units='km**3/s**2')

        self.add_output('vcirc',
                        val=1.0,
                        desc='Circular orbit velocity at given radius '
                             'and gravitational parameter',
                        units='km/s')

        self.declare_partials(of='vcirc', wrt='r')
        self.declare_partials(of='vcirc', wrt='mu')

    def compute(self, inputs, outputs):
        r = inputs['r']
        mu = inputs['mu']

        outputs['vcirc'] = np.sqrt(mu / r)

    def compute_partials(self, inputs, partials):
        r = inputs['r']
        mu = inputs['mu']
        vcirc = np.sqrt(mu / r)

        partials['vcirc', 'mu'] = 0.5 / (r * vcirc)
        partials['vcirc', 'r'] = -0.5 * mu / (vcirc * r ** 2)


class DeltaVComp(om.ExplicitComponent):
    """
    Compute the delta-V performed given the magnitude of two velocities
    and the angle between them.
    """
    def initialize(self):
        pass

    def setup(self):
        self.add_input('v1', val=1.0, desc='Initial velocity', units='km/s')
        self.add_input('v2', val=1.0, desc='Final velocity', units='km/s')
        self.add_input('dinc', val=1.0, desc='Plane change', units='rad')

        # Note:  We're going to use trigonometric functions on dinc.  The
        # automatic unit conversion in OpenMDAO comes in handy here.

        self.add_output('delta_v', val=0.0, desc='Delta-V', units='km/s')

        self.declare_partials(of='delta_v', wrt='v1')
        self.declare_partials(of='delta_v', wrt='v2')
        self.declare_partials(of='delta_v', wrt='dinc')

    def compute(self, inputs, outputs):
        v1 = inputs['v1']
        v2 = inputs['v2']
        dinc = inputs['dinc']

        outputs['delta_v'] = np.sqrt(v1 ** 2 + v2 ** 2 - 2.0 * v1 * v2 * np.cos(dinc))

    def compute_partials(self, inputs, partials):
        v1 = inputs['v1']
        v2 = inputs['v2']
        dinc = inputs['dinc']

        delta_v = np.sqrt(v1 ** 2 + v2 ** 2 - 2.0 * v1 * v2 * np.cos(dinc))

        partials['delta_v', 'v1'] = 0.5 / delta_v * (2 * v1 - 2 * v2 * np.cos(dinc))
        partials['delta_v', 'v2'] = 0.5 / delta_v * (2 * v2 - 2 * v1 * np.cos(dinc))
        partials['delta_v', 'dinc'] = 0.5 / delta_v * (2 * v1 * v2 * np.sin(dinc))


class TransferOrbitComp(om.ExplicitComponent):
    def initialize(self):
        pass

    def setup(self):
        self.add_input('mu',
                       val=398600.4418,
                       desc='Gravitational parameter of central body',
                       units='km**3/s**2')
        self.add_input('rp', val=7000.0, desc='periapsis radius', units='km')
        self.add_input('ra', val=42164.0, desc='apoapsis radius', units='km')

        self.add_output('vp', val=0.0, desc='periapsis velocity', units='km/s')
        self.add_output('va', val=0.0, desc='apoapsis velocity', units='km/s')

        # We're going to be lazy and ask OpenMDAO to approximate our
        # partials with finite differencing here.
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        mu = inputs['mu']
        rp = inputs['rp']
        ra = inputs['ra']

        a = (ra + rp) / 2.0
        e = (a - rp) / a
        p = a * (1.0 - e ** 2)
        h = np.sqrt(mu * p)

        outputs['vp'] = h / rp
        outputs['va'] = h / ra


class TestHohmannTransfer(unittest.TestCase):

    def test_dv_at_apogee(self):
        import openmdao.api as om
        from openmdao.test_suite.test_examples.test_hohmann_transfer import  VCircComp, TransferOrbitComp, DeltaVComp

        prob = om.Problem()

        model = prob.model

        model.add_subsystem('leo', subsys=VCircComp(), promotes_inputs=[('r', 'r1'), 'mu'])
        model.add_subsystem('geo', subsys=VCircComp(), promotes_inputs=[('r', 'r2'), 'mu'])

        model.add_subsystem('transfer', subsys=TransferOrbitComp(),
                            promotes_inputs=[('rp', 'r1'), ('ra', 'r2'), 'mu'])

        model.add_subsystem('dv1', subsys=DeltaVComp(), promotes_inputs=[('dinc', 'dinc1')])

        model.connect('leo.vcirc', 'dv1.v1')
        model.connect('transfer.vp', 'dv1.v2')

        model.add_subsystem('dv2', subsys=DeltaVComp(), promotes_inputs=[('dinc', 'dinc2')])

        model.connect('transfer.va', 'dv2.v1')
        model.connect('geo.vcirc', 'dv2.v2')

        model.add_subsystem('dv_total',
                            subsys=om.ExecComp('delta_v=dv1+dv2',
                                               delta_v={'units': 'km/s'},
                                               dv1={'units': 'km/s'},
                                               dv2={'units': 'km/s'}),
                            promotes=['delta_v'])

        model.connect('dv1.delta_v', 'dv_total.dv1')
        model.connect('dv2.delta_v', 'dv_total.dv2')

        model.add_subsystem('dinc_total',
                            subsys=om.ExecComp('dinc=dinc1+dinc2',
                                               dinc={'units': 'deg'},
                                               dinc1={'units': 'deg'},
                                               dinc2={'units': 'deg'}),
                            promotes=['dinc', 'dinc1', 'dinc2'])

        prob.driver = om.ScipyOptimizeDriver()

        model.add_design_var('dinc1', lower=0, upper=28.5)
        model.add_design_var('dinc2', lower=0, upper=28.5)
        model.add_constraint('dinc', lower=28.5, upper=28.5, scaler=1.0)
        model.add_objective('delta_v', scaler=1.0)

        # set defaults for our promoted variables to remove ambiguities in value and/or units
        model.set_input_defaults('r1', val=42164.0)
        model.set_input_defaults('r2', val=398600.4418)
        model.set_input_defaults('mu', val=398600.4418)
        model.set_input_defaults('dinc1', val=0., units='deg')
        model.set_input_defaults('dinc2', val=28.5, units='deg')

        # Setup the problem
        prob.setup()

        # Execute the model with the given inputs
        prob.run_model()

        print('Delta-V (km/s):', prob['delta_v'][0])
        print('Inclination change split (deg):', prob['dinc1'][0], prob['dinc2'][0])

        prob.run_driver()

        print('Optimized Delta-V (km/s):', prob['delta_v'][0])
        print('Inclination change split (deg):', prob['dinc1'][0], prob['dinc2'][0])
