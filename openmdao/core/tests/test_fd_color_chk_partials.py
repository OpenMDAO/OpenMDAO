import unittest

from io import StringIO

import openmdao.api as om
import numpy as np


class FlightDynamics2DComp(om.ExplicitComponent):
    """
    Compute the 2D flight dynamics equations of motion per equations 1-4 from
    "Energy-State approximation in performance optimization of supersonic aircraft", by
    Bryson, Desai, and Hoffman.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))
        self.options.declare('g', types=(float,), default=9.80665,
                             desc='Gravitational acceleration (m/s**2)')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('W', shape=(nn,), units='N', desc='gross vehicle weight')
        self.add_input('T', shape=(nn,), units='N', desc='vehicle thrust')
        self.add_input('D', shape=(nn,), units='N', desc='vehicle drag')
        self.add_input('L', shape=(nn,), units='N', desc='vehicle lift')
        self.add_input('TAS', shape=(nn,), units='m/s', desc='true airspeed')
        self.add_input('gamma', shape=(nn,), units='rad', desc='flight path angle')
        self.add_input('alpha', shape=(nn,), units='rad', desc='angle of attack')

        self.add_output('dXdt:TAS', shape=(nn,), units='m/s**2', desc='rate of change of true airspeed')
        self.add_output('dXdt:gamma', shape=(nn,), units='m/s**2', desc='rate of change of flight path angle')
        self.add_output('dXdt:alt', shape=(nn,), units='m/s**2', desc='rate of change of altitude')
        self.add_output('dXdt:r', shape=(nn,), units='m/s**2', desc='rate of change of range')

        self.declare_partials(of='*', wrt='*', method='cs')
        self.declare_coloring(wrt='*', method='cs', perturb_size=1e-6, num_full_jacs=2, tol=1e-20,
                              orders=10, show_summary=True, show_sparsity=False)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        g = self.options['g']

        W = inputs['W']
        T = inputs['T']
        L = inputs['L']
        D = inputs['D']
        v = inputs['TAS']
        gamma = inputs['gamma']
        alpha = inputs['alpha']

        m = W / self.options['g']

        mv = m * v

        c_gamma = np.cos(gamma)
        s_gamma = np.sin(gamma)

        outputs['dXdt:TAS'] = T * np.cos(alpha) / m - D / m - g * s_gamma
        outputs['dXdt:gamma'] = T * np.sin(alpha) / mv + L / mv - g * c_gamma / v
        outputs['dXdt:alt'] = v * s_gamma
        outputs['dXdt:r'] = v * c_gamma


class TestColoringChkPartials(unittest.TestCase):

    def test_check_partials(self):
        nn = 5

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('T', val=np.ones(nn), units='lbf')
        ivc.add_output('D', val=np.ones(nn), units='lbf')
        ivc.add_output('L', val=np.ones(nn), units='lbf')
        ivc.add_output('W', val=np.ones(nn), units='lbf')
        ivc.add_output('TAS', val=np.ones(nn), units='kn')
        ivc.add_output('alpha', val=np.ones(nn), units='deg')
        ivc.add_output('gamma', val=np.ones(nn), units='deg')

        p.model.add_subsystem('flight_dynamics_comp', subsys=FlightDynamics2DComp(num_nodes=nn),
                            promotes_inputs=['*'], promotes_outputs=['*'])

        p.setup(check=True, force_alloc_complex=True)

        p.set_val('T', 1000*np.random.rand(nn))
        p.set_val('D', 100*np.random.rand(nn))
        p.set_val('L', 10000*np.random.rand(nn))
        p.set_val('W', 10000*np.random.rand(nn))
        p.set_val('TAS', 100*np.random.rand(nn))
        p.set_val('alpha', 0.1 * np.random.rand(nn))
        p.set_val('gamma', 1.0 * np.random.rand(nn))

        p.run_model()

        s = StringIO()
        p.check_partials(out_stream=s, method='cs', show_only_incorrect=True)
        out = s.getvalue().strip()
        self.assertEqual(out, '** Only writing information about components with incorrect Jacobians **')
