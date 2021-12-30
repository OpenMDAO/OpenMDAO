import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


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
p.model.add_constraint('exec.z', indices=[0, 1], equals=25)

p.model.add_constraint('exec.z', indices=om.slicer[1:10], lower=20, alias="ALIAS_TEST")