import numpy as np
import openmdao.api as om

def test_two_constraints_different_indices():

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
    p.model.add_constraint('exec.z', indices=[-1], upper=25)
    p.model.add_constraint('exec.z', indices=[1], equals=20, alias="EXEC_COPY")


    p.driver = om.pyOptSparseDriver(optimizer='SNOPT')
    p.driver.opt_settings['iSumm'] = 6

    p.setup()

    p.set_val('exec.x', np.linspace(-10, 10, 101))

    p.run_driver()

    print(p.get_val('exec.z'))
    assert_near_equal(p.get_val('exec.z')[0], 25)
    # assert_near_equal(p.get_val('exec.z')[50], -75)

test_two_constraints_different_indices()