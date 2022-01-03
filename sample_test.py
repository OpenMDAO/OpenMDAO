import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.visualization.scaling_viewer.scaling_report import view_driver_scaling


p = om.Problem()

exec = om.ExecComp(['y = x**2',
                    'z = a + x**2'],
                    a={'shape': (1,)},
                    y={'shape': (101,)},
                    x={'shape': (101,)},
                    z={'shape': (101,)})

p.model.add_subsystem('exec', exec)

p.model.add_design_var('exec.a', lower=-1000, upper=1000)
p.model.add_objective('exec.y', index=50)
p.model.add_constraint('exec.z', indices=[0], equals=25)

p.model.add_constraint('exec.z', indices=[1], lower=20, alias="ALIAS_TEST")

p.driver = om.pyOptSparseDriver(optimizer='SNOPT')
p.driver.opt_settings['iSumm'] = 6

p.setup()

p.set_val('exec.x', np.linspace(-10, 10, 101))

p.run_driver()

assert_near_equal(p.get_val('exec.z')[0], 25)
assert_near_equal(p.get_val('exec.z')[50], -75)

view_driver_scaling(p.driver)