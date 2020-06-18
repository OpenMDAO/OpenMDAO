import unittest

import numpy as np

from openmdao.utils.assert_utils import assert_near_equal


class TestSellarMDAPromoteConnect(unittest.TestCase):

    def test_sellar_mda_promote(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1, SellarDis2

        class SellarMDA(om.Group):
            """
            Group containing the Sellar MDA.
            """

            def setup(self):
                cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
                cycle.add_subsystem('d1', SellarDis1(),
                                    promotes_inputs=['x', 'z', 'y2'],
                                    promotes_outputs=['y1'])
                cycle.add_subsystem('d2', SellarDis2(),
                                    promotes_inputs=['z', 'y1'],
                                    promotes_outputs=['y2'])

                cycle.set_input_defaults('x', 1.0)
                cycle.set_input_defaults('z', np.array([5.0, 2.0]))

                # Nonlinear Block Gauss Seidel is a gradient free solver
                cycle.nonlinear_solver = om. NonlinearBlockGS()

                self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                          z=np.array([0.0, 0.0]), x=0.0),
                                   promotes=['x', 'z', 'y1', 'y2', 'obj'])

                self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'),
                                   promotes=['con1', 'y1'])
                self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'),
                                   promotes=['con2', 'y2'])


        prob = om.Problem()
        prob.model = SellarMDA()

        prob.setup()

        prob.set_val('x', 2.0)
        prob.set_val('z', [-1., -1.])

        prob.run_model()

        assert_near_equal((prob.get_val('y1')[0], prob.get_val('y2')[0], prob.get_val('obj')[0], prob.get_val('con1')[0], prob.get_val('con2')[0]),
                         (2.10951651, -0.54758253,  6.8385845,  1.05048349, -24.54758253), 1e-5)

    def test_sellar_mda_promote_in_configure(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1, SellarDis2

        class SellarMDA(om.Group):
            """
            Group containing the Sellar MDA.
            """

            def setup(self):
                # set up model hierarchy
                cycle = self.add_subsystem('cycle', om.Group())
                cycle.add_subsystem('d1', SellarDis1())
                cycle.add_subsystem('d2', SellarDis2())

                cycle.nonlinear_solver = om. NonlinearBlockGS()

                self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                          z=np.array([0.0, 0.0]), x=0.0))

                self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'))
                self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'))

            def configure(self):
                # connect everything via promotes
                self.cycle.promotes('d1', inputs=['x', 'z', 'y2'], outputs=['y1'])
                self.cycle.promotes('d2', inputs=['z', 'y1'], outputs=['y2'])

                self.promotes('cycle', any=['*'])

                self.promotes('obj_cmp', any=['x', 'z', 'y1', 'y2', 'obj'])
                self.promotes('con_cmp1', any=['con1', 'y1'])
                self.promotes('con_cmp2', any=['con2', 'y2'])


        prob = om.Problem()
        prob.model = SellarMDA()

        prob.setup()

        prob.set_val('x', 2.0)
        prob.set_val('z', [-1., -1.])

        prob.run_model()

        assert_near_equal((prob.get_val('y1')[0], prob.get_val('y2')[0], prob.get_val('obj')[0], prob.get_val('con1')[0], prob.get_val('con2')[0]),
                         (2.10951651, -0.54758253,  6.8385845,  1.05048349, -24.54758253), 1e-5)

    def test_sellar_mda_connect(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1, SellarDis2

        class SellarMDAConnect(om.Group):
            """
            Group containing the Sellar MDA. This version uses the disciplines without derivatives.
            """

            def setup(self):
                cycle = self.add_subsystem('cycle', om.Group(),
                                           promotes_inputs=['x', 'z'])
                cycle.add_subsystem('d1', SellarDis1(),
                                    promotes_inputs=['x', 'z'])
                cycle.add_subsystem('d2', SellarDis2(),
                                    promotes_inputs=['z'])
                cycle.connect('d1.y1', 'd2.y1')
                cycle.connect('d2.y2', 'd1.y2')

                cycle.set_input_defaults('x', 1.0)
                cycle.set_input_defaults('z', np.array([5.0, 2.0]))

                # Nonlinear Block Gauss Seidel is a gradient free solver
                cycle.nonlinear_solver = om.NonlinearBlockGS()

                self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                          z=np.array([0.0, 0.0]), x=0.0),
                                   promotes_inputs=['x', 'z'])

                self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'))
                self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'))

                self.connect('cycle.d1.y1', ['obj_cmp.y1', 'con_cmp1.y1'])
                self.connect('cycle.d2.y2', ['obj_cmp.y2', 'con_cmp2.y2'])

        prob = om.Problem()
        prob.model = SellarMDAConnect()

        prob.setup()

        prob.set_val('x', 2.0)
        prob.set_val('z', [-1., -1.])

        prob.run_model()

        assert_near_equal((prob.get_val('cycle.d1.y1')[0], prob.get_val('cycle.d2.y2')[0], prob.get_val('obj_cmp.obj')[0], prob.get_val('con_cmp1.con1')[0], prob.get_val('con_cmp2.con2')[0]),
                         (2.10951651, -0.54758253, 6.8385845, 1.05048349, -24.54758253), 1e-5)


    def test_sellar_mda_promote_connect(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1, SellarDis2

        class SellarMDAPromoteConnect(om.Group):
            """
            Group containing the Sellar MDA. This version uses the disciplines without derivatives.
            """

            def setup(self):
                cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
                cycle.add_subsystem('d1', SellarDis1(),
                                    promotes_inputs=['x', 'z'])
                cycle.add_subsystem('d2', SellarDis2(),
                                    promotes_inputs=['z'])
                cycle.connect('d1.y1', 'd2.y1')
                cycle.connect('d2.y2', 'd1.y2')

                cycle.set_input_defaults('x', 1.0)
                cycle.set_input_defaults('z', np.array([5.0, 2.0]))

                # Nonlinear Block Gauss Seidel is a gradient free solver
                cycle.nonlinear_solver = om.NonlinearBlockGS()

                self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                          z=np.array([0.0, 0.0]), x=0.0),
                                   promotes_inputs=['x', 'z'])

                self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'))
                self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'))

                self.connect('d1.y1', ['con_cmp1.y1', 'obj_cmp.y1'])
                self.connect('d2.y2', ['con_cmp2.y2', 'obj_cmp.y2'])


        prob = om.Problem()
        prob.model = SellarMDAPromoteConnect()

        prob.setup()

        prob.set_val('x', 2.0)
        prob.set_val('z', [-1., -1.])

        prob.run_model()

        assert_near_equal((prob.get_val('cycle.d1.y1')[0], prob.get_val('cycle.d2.y2')[0], prob.get_val('obj_cmp.obj')[0], prob.get_val('con_cmp1.con1')[0], prob.get_val('con_cmp2.con2')[0]),
                         (2.10951651, -0.54758253, 6.8385845, 1.05048349, -24.54758253), 1e-5)


if __name__ == "__main__":
    unittest.main()
