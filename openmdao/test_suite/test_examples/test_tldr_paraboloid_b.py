import unittest

from openmdao.utils.assert_utils import assert_near_equal


class TestParaboloidTLDR(unittest.TestCase):

    def test_tldr(self):
        import openmdao.api as om

        # build the model
        prob = om.Problem()
        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', 3.0)
        indeps.add_output('y', -4.0)

        prob.model.add_subsystem('paraboloid', om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

        prob.model.connect('indeps.x', 'paraboloid.x')
        prob.model.connect('indeps.y', 'paraboloid.y')

        # setup the optimization
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('indeps.x', lower=-50, upper=50)
        prob.model.add_design_var('indeps.y', lower=-50, upper=50)
        prob.model.add_objective('paraboloid.f')

        prob.setup()
        prob.run_driver()

        #
        try:
            # minimum value
            assert_near_equal(prob.get_val('paraboloid.f'), -27.33333, 1e-6)

            # location of the minimum
            assert_near_equal(prob.get_val('indeps.x'), 6.6667, 1e-4)
            assert_near_equal(prob.get_val('indeps.y'), -7.33333, 1e-4)
        except AssertionError as err:
            msg = str(err) + "\n If the code in this test must be changed, please go change sister example, " \
                             "tldr_paraboloid.py, and also the front page of the OpenMDAO website!"
            raise AssertionError(msg)

if __name__ == "__main__":

    unittest.main()
