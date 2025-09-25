
import openmdao.api as om
import numpy as np

# note: size must be an even number
SIZE = 10


class CircleOpt(om.Group):

    def setup(self):
        indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes_outputs=['*'])

        # the following were randomly generated using np.random.random(10)*2-1 to randomly
        # disperse them within a unit circle centered at the origin.
        indeps.add_output('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                                          0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
        indeps.add_output('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                                          -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
        indeps.add_output('r', .7)

        self.add_subsystem('arctan_yox', om.ExecComp('g=arctan(y/x)', has_diag_partials=True,
                                                     g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)))

        self.add_subsystem('circle', om.ExecComp('area=pi*r**2'))

        self.add_subsystem('r_con', om.ExecComp('g=x**2 + y**2 - r', has_diag_partials=True,
                                                g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)))

        thetas = np.linspace(0, np.pi/4, SIZE)
        self.add_subsystem('theta_con', om.ExecComp('g = x - theta', has_diag_partials=True,
                                                    g=np.ones(SIZE), x=np.ones(SIZE),
                                                    theta=thetas))
        self.add_subsystem('delta_theta_con', om.ExecComp('g = even - odd', has_diag_partials=True,
                                                          g=np.ones(SIZE//2), even=np.ones(SIZE//2),
                                                          odd=np.ones(SIZE//2)))

        self.add_subsystem('l_conx', om.ExecComp('g=x-1', has_diag_partials=True, g=np.ones(SIZE), x=np.ones(SIZE)))

        IND = np.arange(SIZE, dtype=int)
        ODD_IND = IND[1::2]  # all odd indices
        EVEN_IND = IND[0::2]  # all even indices

        self.connect('r', ('circle.r', 'r_con.r'))
        self.connect('x', ['r_con.x', 'arctan_yox.x', 'l_conx.x'])
        self.connect('y', ['r_con.y', 'arctan_yox.y'])
        self.connect('arctan_yox.g', 'theta_con.x')
        self.connect('arctan_yox.g', 'delta_theta_con.even', src_indices=EVEN_IND)
        self.connect('arctan_yox.g', 'delta_theta_con.odd', src_indices=ODD_IND)

        self.add_design_var('x')
        self.add_design_var('y')
        self.add_design_var('r', lower=.5, upper=10)

        # nonlinear constraints
        self.add_constraint('r_con.g', equals=0)

        self.add_constraint('theta_con.g', lower=-1e-5, upper=1e-5, indices=EVEN_IND)
        self.add_constraint('delta_theta_con.g', lower=-1e-5, upper=1e-5)

        # this constrains x[0] to be 1 (see definition of l_conx)
        self.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,])

        # linear constraint
        self.add_constraint('y', equals=0, indices=[0,], linear=True)

        self.add_objective('circle.area', ref=-1)


if __name__ == '__main__':
    p = om.Problem(model=CircleOpt(), driver=om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False))
    p.setup(mode='fwd')
    p.run_driver()

    print(p['circle.area'], np.pi)
