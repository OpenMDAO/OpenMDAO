import unittest

from openmdao.utils.assert_utils import assert_near_equal


class TestEulerExample(unittest.TestCase):

    def test_feature_example(self):
        import numpy as np
        from scipy.optimize import minimize

        import openmdao.api as om
        from openmdao.test_suite.test_examples.cannonball.cannonball_ode import CannonballODE

        def eval_cannonball_range(gam_init, prob, complex_step=False):
            """
            Compute distance given initial speed and angle of cannonball.

            Parameters
            ----------
            gam_init : float
                Initial cannonball firing angle in degrees.
            prob : <Problem>
                OpenMDAO problem that contains the equations of motion.
            complex_step : bool
                Set to True to perform complex step.

            Returns
            -------
            float
                Negative of range in m.
            """
            dt = 0.1        # Time step
            h_init = 1.0    # Height of cannon.
            v_init = 100.0  # Initial cannonball velocity.
            h_target = 0.0  #

            v = v_init
            gam = gam_init
            h = h_init
            r = 0.0
            t = 0.0

            if complex_step:
                prob.set_complex_step_mode(True)

            while h > h_target:

                # Set values
                prob.set_val('v', v)
                prob.set_val('gam', gam, units='deg')

                # Run the model
                prob.run_model()

                # Extract rates
                v_dot = prob.get_val('v_dot')
                gam_dot = prob.get_val('gam_dot', units='deg/s')
                h_dot = prob.get_val('h_dot')
                r_dot = prob.get_val('r_dot')

                h_last = h
                r_last = r

                # Euler Integration
                v = v + dt * v_dot
                gam = gam + dt * gam_dot
                h = h + dt * h_dot
                r = r + dt * r_dot
                t += dt
                # print(v, gam, h, r)

            # Linear interpolation between last two points to get the landing point accurate.
            r_final = r_last + (r - r_last) * h_last / (h_last - h)

            if complex_step:
                prob.set_complex_step_mode(False)

            #print(f"Distance: {r_final}, Time: {t}, Angle: {gam_init}")
            return -r_final


        def gradient_cannonball_range(gam_init, prob):
            """
            Uses complex step to compute gradient of range wrt initial angle.

            Parameters
            ----------
            gam_init : float
                Initial cannonball firing angle in degrees.
            prob : <Problem>
                OpenMDAO problem that contains the equations of motion.

            Returns
            -------
            float
                Derivative of range wrt initial angle in m/deg.
            """
            step = 1.0e-14
            dr_dgam = eval_cannonball_range(gam_init + step * 1j, prob, complex_step=True)
            return dr_dgam.imag / step


        prob = om.Problem(model=CannonballODE())
        prob.setup(force_alloc_complex=True)

        # Set constants
        prob.set_val('CL', 0.0)                          # Lift Coefficient
        prob.set_val('CD', 0.05)                         # Drag Coefficient
        prob.set_val('S', 0.25 * np.pi, units='ft**2')   # Wetted Area (1 ft diameter ball)
        prob.set_val('rho', 1.225)                       # Atmospheric Density
        prob.set_val('m', 5.5)                           # Cannonball Mass

        prob.set_val('alpha', 0.0)                       # Angle of Attack (Not Applicable)
        prob.set_val('T', 0.0)                           # Thrust (Not Applicable)

        result = minimize(eval_cannonball_range, 27.0,
                          method='SLSQP',
                          jac=gradient_cannonball_range,
                          args=(prob))

        assert_near_equal(result['x'], 42.3810579, 1e-3)


if __name__ == "__main__":
    unittest.main()
