import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

def test_obj_and_con_same_var_different_indices():

    import openmdao.api as om

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
    p.model.add_constraint('exec.z', indices=[-1], lower=0)
    p.model.add_constraint('exec.z', indices=[50], equals=-70, alias="ALIAS_TEST")

    p.driver = om.SimpleGADriver()

    p.setup()

    p.set_val('exec.x', np.linspace(-10, 10, 101))

    p.run_driver()

    print(p.get_val('exec.z'))
    assert_near_equal(p.get_val('exec.z')[-1], 30)
    assert_near_equal(p.get_val('exec.z')[50], -75)

# test_obj_and_con_same_var_different_indices()


def test_multiple_con_and_obj_simple_ga():

    p = om.Problem()

    exec = om.ExecComp(['y = x**2',
                        'z = a + x**2'],
                        a={'shape': (1,)},
                        y={'shape': (101,)},
                        x={'shape': (101,)},
                        z={'shape': (101,)})

    p.model.add_subsystem('exec', exec)

    p.model.add_design_var('exec.a', lower=-1000, upper=1000)
    p.model.add_objective('exec.z', index=50)
    p.model.add_constraint('exec.z', indices=[0], lower=-200, alias="ALIAS_TEST")

    p.driver = om.SimpleGADriver()

    p.setup()

    p.set_val('exec.x', np.linspace(-10, 10, 101))

    p.run_driver()

    print(p.get_val('exec.z'))
    assert_near_equal(p.get_val('exec.z')[0], -200)
    assert_near_equal(p.get_val('exec.z')[50], -300)

test_multiple_con_and_obj_simple_ga()