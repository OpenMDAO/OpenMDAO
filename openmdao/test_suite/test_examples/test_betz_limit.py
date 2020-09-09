from distutils.version import LooseVersion
import unittest

import scipy

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


# duplicate definition here so it can be included in docs by itself
class ActuatorDisc(om.ExplicitComponent):
    """Simple wind turbine model based on actuator disc theory"""

    def setup(self):
        # Inputs
        self.add_input('a', 0.5, desc="Induced Velocity Factor")
        self.add_input('Area', 10.0, units="m**2", desc="Rotor disc area")
        self.add_input('rho', 1.225, units="kg/m**3", desc="Air density")
        self.add_input('Vu', 10.0, units="m/s", desc="Freestream air velocity, upstream of rotor")

        # Outputs
        self.add_output('Vr', 0.0, units="m/s",
                        desc="Air velocity at rotor exit plane")
        self.add_output('Vd', 0.0, units="m/s",
                        desc="Slipstream air velocity, downstream of rotor")
        self.add_output('Ct', 0.0, desc="Thrust Coefficient")
        self.add_output('thrust', 0.0, units="N",
                        desc="Thrust produced by the rotor")
        self.add_output('Cp', 0.0, desc="Power Coefficient")
        self.add_output('power', 0.0, units="W", desc="Power produced by the rotor")

    def setup_partials(self):
        self.declare_partials('Vr', ['a', 'Vu'])
        self.declare_partials('Vd', 'a')
        self.declare_partials('Ct', 'a')
        self.declare_partials('thrust', ['a', 'Area', 'rho', 'Vu'])
        self.declare_partials('Cp', 'a')
        self.declare_partials('power', ['a', 'Area', 'rho', 'Vu'])

    def compute(self, inputs, outputs):
        """ Considering the entire rotor as a single disc that extracts
        velocity uniformly from the incoming flow and converts it to
        power."""

        a = inputs['a']
        Vu = inputs['Vu']

        qA = .5 * inputs['rho'] * inputs['Area'] * Vu**2

        outputs['Vd'] = Vd = Vu * (1 - 2 * a)
        outputs['Vr'] = .5 * (Vu + Vd)

        outputs['Ct'] = Ct = 4 * a * (1 - a)
        outputs['thrust'] = Ct * qA

        outputs['Cp'] = Cp = Ct * (1 - a)
        outputs['power'] = Cp * qA * Vu

    def compute_partials(self, inputs, J):
        """ Jacobian of partial derivatives."""

        a = inputs['a']
        Vu = inputs['Vu']
        Area = inputs['Area']
        rho = inputs['rho']

        # pre-compute commonly needed quantities
        a_times_area = a * Area
        one_minus_a = 1.0 - a
        a_area_rho_vu = a_times_area * rho * Vu

        J['Vr', 'a'] = -Vu
        J['Vr', 'Vu'] = one_minus_a

        J['Vd', 'a'] = -2.0 * Vu

        J['Ct', 'a'] = 4.0 - 8.0 * a

        J['thrust', 'a'] = .5 * rho * Vu**2 * Area * J['Ct', 'a']
        J['thrust', 'Area'] = 2.0 * Vu**2 * a * rho * one_minus_a
        J['thrust', 'rho'] = 2.0 * a_times_area * Vu ** 2 * (one_minus_a)
        J['thrust', 'Vu'] = 4.0 * a_area_rho_vu * (one_minus_a)

        J['Cp', 'a'] = 4.0 * a * (2.0 * a - 2.0) + 4.0 * (one_minus_a)**2

        J['power', 'a'] = 2.0 * Area * Vu**3 * a * rho * (
            2.0 * a - 2.0) + 2.0 * Area * Vu**3 * rho * one_minus_a**2
        J['power', 'Area'] = 2.0 * Vu**3 * a * rho * one_minus_a**2
        J['power', 'rho'] = 2.0 * a_times_area * Vu ** 3 * (one_minus_a)**2
        J['power', 'Vu'] = 6.0 * Area * Vu**2 * a * rho * one_minus_a**2


class TestBetzLimit(unittest.TestCase):

    def test_betz(self):
        from distutils.version import LooseVersion
        import scipy
        import openmdao.api as om

        class ActuatorDisc(om.ExplicitComponent):
            """Simple wind turbine model based on actuator disc theory"""

            def setup(self):

                # Inputs
                self.add_input('a', 0.5, desc="Induced Velocity Factor")
                self.add_input('Area', 10.0, units="m**2", desc="Rotor disc area")
                self.add_input('rho', 1.225, units="kg/m**3", desc="air density")
                self.add_input('Vu', 10.0, units="m/s", desc="Freestream air velocity, upstream of rotor")

                # Outputs
                self.add_output('Vr', 0.0, units="m/s",
                                desc="Air velocity at rotor exit plane")
                self.add_output('Vd', 0.0, units="m/s",
                                desc="Slipstream air velocity, downstream of rotor")
                self.add_output('Ct', 0.0, desc="Thrust Coefficient")
                self.add_output('thrust', 0.0, units="N",
                                desc="Thrust produced by the rotor")
                self.add_output('Cp', 0.0, desc="Power Coefficient")
                self.add_output('power', 0.0, units="W", desc="Power produced by the rotor")

            def setup_partials(self):
                self.declare_partials('Vr', ['a', 'Vu'])
                self.declare_partials('Vd', 'a')
                self.declare_partials('Ct', 'a')
                self.declare_partials('thrust', ['a', 'Area', 'rho', 'Vu'])
                self.declare_partials('Cp', 'a')
                self.declare_partials('power', ['a', 'Area', 'rho', 'Vu'])

            def compute(self, inputs, outputs):
                """ Considering the entire rotor as a single disc that extracts
                velocity uniformly from the incoming flow and converts it to
                power."""

                a = inputs['a']
                Vu = inputs['Vu']

                qA = .5 * inputs['rho'] * inputs['Area'] * Vu ** 2

                outputs['Vd'] = Vd = Vu * (1 - 2 * a)
                outputs['Vr'] = .5 * (Vu + Vd)

                outputs['Ct'] = Ct = 4 * a * (1 - a)
                outputs['thrust'] = Ct * qA

                outputs['Cp'] = Cp = Ct * (1 - a)
                outputs['power'] = Cp * qA * Vu

            def compute_partials(self, inputs, J):
                """ Jacobian of partial derivatives."""

                a = inputs['a']
                Vu = inputs['Vu']
                Area = inputs['Area']
                rho = inputs['rho']

                # pre-compute commonly needed quantities
                a_times_area = a * Area
                one_minus_a = 1.0 - a
                a_area_rho_vu = a_times_area * rho * Vu

                J['Vr', 'a'] = -Vu
                J['Vr', 'Vu'] = one_minus_a

                J['Vd', 'a'] = -2.0 * Vu

                J['Ct', 'a'] = 4.0 - 8.0 * a

                J['thrust', 'a'] = .5 * rho * Vu**2 * Area * J['Ct', 'a']
                J['thrust', 'Area'] = 2.0 * Vu**2 * a * rho * one_minus_a
                J['thrust', 'rho'] = 2.0 * a_times_area * Vu ** 2 * (one_minus_a)
                J['thrust', 'Vu'] = 4.0 * a_area_rho_vu * (one_minus_a)

                J['Cp', 'a'] = 4.0 * a * (2.0 * a - 2.0) + 4.0 * (one_minus_a)**2

                J['power', 'a'] = 2.0 * Area * Vu**3 * a * rho * (
                2.0 * a - 2.0) + 2.0 * Area * Vu**3 * rho * one_minus_a ** 2
                J['power', 'Area'] = 2.0 * Vu**3 * a * rho * one_minus_a ** 2
                J['power', 'rho'] = 2.0 * a_times_area * Vu ** 3 * (one_minus_a)**2
                J['power', 'Vu'] = 6.0 * Area * Vu**2 * a * rho * one_minus_a**2


        # build the model
        prob = om.Problem()
        prob.model.add_subsystem('a_disk', ActuatorDisc(),
                                 promotes_inputs=['a', 'Area', 'rho', 'Vu'])

        # setup the optimization
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('a', lower=0., upper=1.)
        prob.model.add_design_var('Area', lower=0., upper=1.)

        # negative one so we maximize the objective
        prob.model.add_objective('a_disk.Cp', scaler=-1)

        prob.setup()

        prob.set_val('a', .5)
        prob.set_val('Area', 10.0, units='m**2')
        prob.set_val('rho', 1.225, units='kg/m**3')
        prob.set_val('Vu', 10.0, units='m/s')

        prob.run_driver()

        prob.model.list_inputs(values = False, hierarchical=False)
        prob.model.list_outputs(values = False, hierarchical=False)

        # minimum value
        assert_near_equal(prob['a_disk.Cp'], 16./27., 1e-4)
        assert_near_equal(prob['a'], 0.33333, 1e-4)

        # There is a bug in scipy version < 1.0 that causes this value to be wrong.
        if LooseVersion(scipy.__version__) >= LooseVersion("1.0"):
            assert_near_equal(prob['Area'], 1.0, 1e-4)

    def test_betz_with_var_tags(self):
        import openmdao.api as om

        class ActuatorDiscWithTags(om.ExplicitComponent):
            """Simple wind turbine model based on actuator disc theory"""

            def setup(self):

                # Inputs
                self.add_input('a', 0.5, desc="Induced Velocity Factor", tags="advanced")
                self.add_input('Area', 10.0, units="m**2", desc="Rotor disc area", tags="basic")
                self.add_input('rho', 1.225, units="kg/m**3", desc="air density", tags="advanced")
                self.add_input('Vu', 10.0, units="m/s",
                               desc="Freestream air velocity, upstream of rotor", tags="basic")

                # Outputs
                self.add_output('Vr', 0.0, units="m/s",
                                desc="Air velocity at rotor exit plane")
                self.add_output('Vd', 0.0, units="m/s",
                                desc="Slipstream air velocity, downstream of rotor")
                self.add_output('Ct', 0.0, desc="Thrust Coefficient")
                self.add_output('thrust', 0.0, units="N",
                                desc="Thrust produced by the rotor")
                self.add_output('Cp', 0.0, desc="Power Coefficient")
                self.add_output('power', 0.0, units="W", desc="Power produced by the rotor")

            def setup_partials(self):
                self.declare_partials('Vr', ['a', 'Vu'])
                self.declare_partials('Vd', 'a')
                self.declare_partials('Ct', 'a')
                self.declare_partials('thrust', ['a', 'Area', 'rho', 'Vu'])
                self.declare_partials('Cp', 'a')
                self.declare_partials('power', ['a', 'Area', 'rho', 'Vu'])

            def compute(self, inputs, outputs):
                """ Considering the entire rotor as a single disc that extracts
                velocity uniformly from the incoming flow and converts it to
                power."""

                a = inputs['a']
                Vu = inputs['Vu']

                qA = .5 * inputs['rho'] * inputs['Area'] * Vu ** 2

                outputs['Vd'] = Vd = Vu * (1 - 2 * a)
                outputs['Vr'] = .5 * (Vu + Vd)

                outputs['Ct'] = Ct = 4 * a * (1 - a)
                outputs['thrust'] = Ct * qA

                outputs['Cp'] = Cp = Ct * (1 - a)
                outputs['power'] = Cp * qA * Vu

            def compute_partials(self, inputs, J):
                """ Jacobian of partial derivatives."""

                a = inputs['a']
                Vu = inputs['Vu']
                Area = inputs['Area']
                rho = inputs['rho']

                # pre-compute commonly needed quantities
                a_times_area = a * Area
                one_minus_a = 1.0 - a
                a_area_rho_vu = a_times_area * rho * Vu

                J['Vr', 'a'] = -Vu
                J['Vr', 'Vu'] = one_minus_a

                J['Vd', 'a'] = -2.0 * Vu

                J['Ct', 'a'] = 4.0 - 8.0 * a

                J['thrust', 'a'] = .5 * rho * Vu**2 * Area * J['Ct', 'a']
                J['thrust', 'Area'] = 2.0 * Vu**2 * a * rho * one_minus_a
                J['thrust', 'rho'] = 2.0 * a_times_area * Vu ** 2 * (one_minus_a)
                J['thrust', 'Vu'] = 4.0 * a_area_rho_vu * (one_minus_a)

                J['Cp', 'a'] = 4.0 * a * (2.0 * a - 2.0) + 4.0 * (one_minus_a)**2

                J['power', 'a'] = 2.0 * Area * Vu**3 * a * rho * (
                2.0 * a - 2.0) + 2.0 * Area * Vu**3 * rho * one_minus_a ** 2
                J['power', 'Area'] = 2.0 * Vu**3 * a * rho * one_minus_a ** 2
                J['power', 'rho'] = 2.0 * a_times_area * Vu ** 3 * (one_minus_a)**2
                J['power', 'Vu'] = 6.0 * Area * Vu**2 * a * rho * one_minus_a**2


        # build the model
        prob = om.Problem()
        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('a', .5, tags="advanced")
        indeps.add_output('Area', 10.0, units='m**2', tags="basic")
        indeps.add_output('rho', 1.225, units='kg/m**3', tags="advanced")
        indeps.add_output('Vu', 10.0, units='m/s', tags="basic")

        prob.model.add_subsystem('a_disk', ActuatorDiscWithTags(),
                                promotes_inputs=['a', 'Area', 'rho', 'Vu'])

        # setup the optimization
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('a', lower=0., upper=1.)
        prob.model.add_design_var('Area', lower=0., upper=1.)
        # negative one so we maximize the objective
        prob.model.add_objective('a_disk.Cp', scaler=-1)

        prob.setup()
        prob.run_driver()

        prob.model.list_inputs(tags='basic', units=True, shape=True)
        prob.model.list_inputs(tags=['basic','advanced'], units=True, shape=True)

        prob.model.list_outputs(tags='basic', units=False, shape=False)
        prob.model.list_outputs(tags=['basic','advanced'], units=False, shape=False)

    def test_betz_derivatives(self):
        import openmdao.api as om

        from openmdao.test_suite.test_examples.test_betz_limit import ActuatorDisc

        prob = om.Problem()

        prob.model.add_subsystem('a_disk', ActuatorDisc())

        prob.setup()
        prob.check_partials(compact_print=True)


if __name__ == "__main__":

    unittest.main()
