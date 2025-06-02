
import unittest

import openmdao.api as om



class MPISetvalBug(unittest.TestCase):
    N_PROCS = 2

    def test_set_val_mpi_bug(self):
        p = om.Problem()
        par_group = om.ParallelGroup()

        c1 = om.ExecComp('y1 = x1 ** 2', x1={'shape': (1,)}, y1={'copy_shape': 'x1'})
        g1 = om.Group()
        g1.add_subsystem('c1', c1, promotes=['*'])

        c2 = om.ExecComp('g2 = x2', x2={'shape': (1,)}, g2={'copy_shape': 'x2'})
        g2 = om.Group()
        g2.add_subsystem('c2', c2, promotes=['*'])

        #c3 = om.ExecComp('g3 = x3', x3={'shape': (1,)}, g3={'copy_shape': 'x3'})
        #g3 = om.Group()
        #g3.add_subsystem('c3', c3, promotes=['*'])

        par_group.add_subsystem('g1', g1, promotes=['*'])
        par_group.add_subsystem('g2', g2, promotes=['*'])
        #par_group.add_subsystem('g3', g3, promotes=['*'])

        p.model.add_objective('y1')
        p.model.add_design_var('x1', lower=2, upper=5)
        p.model.add_design_var('x2', lower=2, upper=5)
        #p.model.add_design_var('x3', lower=2, upper=5)
        p.model.add_constraint('g2', lower=3.)
        #p.model.add_constraint('g3', lower=3.)

        p.model.add_subsystem('par_group', par_group, promotes=['*'])

        p.driver = om.pyOptSparseDriver(optimizer='SLSQP')

        p.setup()

        # if g1 in p.model.par_group._subsystems_myproc:
        #     g1.set_val('x1', 2.5)

        # if g2 in p.model.par_group._subsystems_myproc:
        #     g2.set_val('x2', 2.5)

        # if g3 in p.model.par_group._subsystems_myproc:
        #     g3.set_val('x3', 2.5)

        g1.set_val('x1', 2.5)
        g2.set_val('x2', 2.6)
        #g3.set_val('x3', 2.5)

        p.final_setup()
        p.list_driver_vars()

