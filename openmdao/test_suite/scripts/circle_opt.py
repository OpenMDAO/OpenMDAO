
from openmdao.api import Problem, IndepVarComp, ExecComp, ScipyOptimizeDriver
import numpy as np

# note: size must be an even number
SIZE = 10
p = Problem()

indeps = p.model.add_subsystem('indeps', IndepVarComp(), promotes_outputs=['*'])

# the following were randomly generated using np.random.random(10)*2-1 to randomly
# disperse them within a unit circle centered at the origin.
indeps.add_output('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                                  0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
indeps.add_output('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                                  -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
indeps.add_output('r', .7)

p.model.add_subsystem('arctan_yox', ExecComp('g=arctan(y/x)', vectorize=True,
                                                g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)))

p.model.add_subsystem('circle', ExecComp('area=pi*r**2'))

p.model.add_subsystem('r_con', ExecComp('g=x**2 + y**2 - r', vectorize=True,
                                        g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)))

thetas = np.linspace(0, np.pi/4, SIZE)
p.model.add_subsystem('theta_con', ExecComp('g = x - theta', vectorize=True,
                                            g=np.ones(SIZE), x=np.ones(SIZE),
                                            theta=thetas))
p.model.add_subsystem('delta_theta_con', ExecComp('g = even - odd', vectorize=True,
                                                    g=np.ones(SIZE//2), even=np.ones(SIZE//2),
                                                    odd=np.ones(SIZE//2)))

p.model.add_subsystem('l_conx', ExecComp('g=x-1', vectorize=True, g=np.ones(SIZE), x=np.ones(SIZE)))

IND = np.arange(SIZE, dtype=int)
ODD_IND = IND[1::2]  # all odd indices
EVEN_IND = IND[0::2]  # all even indices

p.model.connect('r', ('circle.r', 'r_con.r'))
p.model.connect('x', ['r_con.x', 'arctan_yox.x', 'l_conx.x'])
p.model.connect('y', ['r_con.y', 'arctan_yox.y'])
p.model.connect('arctan_yox.g', 'theta_con.x')
p.model.connect('arctan_yox.g', 'delta_theta_con.even', src_indices=EVEN_IND)
p.model.connect('arctan_yox.g', 'delta_theta_con.odd', src_indices=ODD_IND)

p.driver = ScipyOptimizeDriver()
p.driver.options['optimizer'] = 'SLSQP'
p.driver.options['disp'] = False

p.model.add_design_var('x')
p.model.add_design_var('y')
p.model.add_design_var('r', lower=.5, upper=10)

# nonlinear constraints
p.model.add_constraint('r_con.g', equals=0)

p.model.add_constraint('theta_con.g', lower=-1e-5, upper=1e-5, indices=EVEN_IND)
p.model.add_constraint('delta_theta_con.g', lower=-1e-5, upper=1e-5)

# this constrains x[0] to be 1 (see definition of l_conx)
p.model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,])

# linear constraint
p.model.add_constraint('y', equals=0, indices=[0,], linear=True)

p.model.add_objective('circle.area', ref=-1)

p.setup(mode='fwd')
p.run_driver()

print(p['circle.area'], np.pi)
