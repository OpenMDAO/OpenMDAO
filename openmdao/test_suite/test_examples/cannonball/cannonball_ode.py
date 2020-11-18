import numpy as np

import openmdao.api as om


class DynamicPressureComp(om.ExplicitComponent):

    def setup(self):
        self.add_input(name='rho', val=1.0, units='kg/m**3',
                       desc='atmospheric density')
        self.add_input(name='v', val=1.0, units='m/s',
                       desc='air-relative velocity')

        self.add_output(name='q', val=1.0, units='N/m**2',
                        desc='dynamic pressure')

        self.declare_partials(of='q', wrt='rho')
        self.declare_partials(of='q', wrt='v')

    def compute(self, inputs, outputs):
        outputs['q'] = 0.5 * inputs['rho'] * inputs['v'] ** 2


class LiftDragForceComp(om.ExplicitComponent):
    """
    Compute the aerodynamic forces on the vehicle in the wind axis frame
    (lift, drag, cross) force.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        self.add_input(name='CL', val=0.0,
                       desc='lift coefficient')
        self.add_input(name='CD', val=0.0,
                       desc='drag coefficient')
        self.add_input(name='q', val=0.0, units='N/m**2',
                       desc='dynamic pressure')
        self.add_input(name='S', val=0.0, units='m**2',
                       desc='aerodynamic reference area')

        self.add_output(name='f_lift', shape=(1, ), units='N',
                        desc='aerodynamic lift force')
        self.add_output(name='f_drag', shape=(1, ), units='N',
                        desc='aerodynamic drag force')

        self.declare_partials(of='f_lift', wrt=['q', 'S', 'CL'])
        self.declare_partials(of='f_drag', wrt=['q', 'S', 'CD'])

    def compute(self, inputs, outputs):
        q = inputs['q']
        S = inputs['S']
        CL = inputs['CL']
        CD = inputs['CD']

        qS = q * S

        outputs['f_lift'] = qS * CL
        outputs['f_drag'] = qS * CD


class FlightPathEOM2D(om.ExplicitComponent):
    """
    Computes the position and velocity equations of motion using a 2D flight path
    parameterization of states per equations 4.42 - 4.46 of _[1].

    References
    ----------
    .. [1] Bryson, Arthur Earl. Dynamic optimization. Vol. 1. Prentice Hall, p.172, 1999.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        self.add_input(name='m', val=1.0, units='kg',
                       desc='aircraft mass')
        self.add_input(name='v', val=1.0, units='m/s',
                       desc='aircraft velocity magnitude')
        self.add_input(name='T', val=0.0, units='N',
                       desc='thrust')
        self.add_input(name='alpha', val=0.0, units='rad',
                       desc='angle of attack')
        self.add_input(name='L', val=0.0, units='N',
                       desc='lift force')
        self.add_input(name='D', val=0.0, units='N',
                       desc='drag force')
        self.add_input(name='gam', val=0.0, units='rad',
                       desc='flight path angle')

        self.add_output(name='v_dot', val=0.0, units='m/s**2',
                        desc='rate of change of velocity magnitude')
        self.add_output(name='gam_dot', val=0.0, units='rad/s',
                        desc='rate of change of flight path angle')
        self.add_output(name='h_dot', val=0.0, units='m/s',
                        desc='rate of change of altitude')
        self.add_output(name='r_dot', val=0.0, units='m/s',
                        desc='rate of change of range')

    def setup_partials(self):
        self.declare_partials('v_dot', ['T', 'D', 'm', 'gam', 'alpha'])
        self.declare_partials('gam_dot', ['T', 'L', 'm', 'gam', 'alpha', 'v'])
        self.declare_partials(['h_dot', 'r_dot'], ['gam', 'v'])

    def compute(self, inputs, outputs):
        g = 9.80665
        m = inputs['m']
        v = inputs['v']
        T = inputs['T']
        L = inputs['L']
        D = inputs['D']
        gam = inputs['gam']
        alpha = inputs['alpha']

        calpha = np.cos(alpha)
        salpha = np.sin(alpha)

        cgam = np.cos(gam)
        sgam = np.sin(gam)

        mv = m * v

        outputs['v_dot'] = (T * calpha - D) / m - g * sgam
        outputs['gam_dot'] = (T * salpha + L) / mv - (g / v) * cgam
        outputs['h_dot'] = v * sgam
        outputs['r_dot'] = v * cgam


class CannonballODE(om.Group):

    def setup(self):
        self.add_subsystem(name='dynamic_pressure',
                           subsys=DynamicPressureComp(),
                           promotes=['*'])

        self.add_subsystem(name='aero',
                           subsys=LiftDragForceComp(),
                           promotes_inputs=['*'])

        self.add_subsystem(name='eom',
                           subsys=FlightPathEOM2D(),
                           promotes=['*'])

        self.connect('aero.f_drag', 'D')
        self.connect('aero.f_lift', 'L')
