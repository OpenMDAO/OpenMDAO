"""
This test was added in response to a bug, but the bug ended up to be a mistake in our docs. We
left the test because it checks the format of the printed values from check partials.
"""
import unittest
from io import StringIO
import re

import numpy as np

import openmdao.api as om
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.mpi import MPI


class MixedDistrib2(om.ExplicitComponent):

    def setup(self):
        # Distributed Input
        self.add_input('in_dist', shape_by_conn=True, distributed=True)
        # Non-distributed Input
        self.add_input('in_nd', val=1)
        # Distributed Output
        self.add_output('out_dist', copy_shape='in_dist', distributed=True)
        # Non-distributed Output
        self.add_output('out_nd', copy_shape='in_nd')

    def compute(self, inputs, outputs):
        x = inputs['in_dist']
        y = inputs['in_nd']
        # "Computationally Intensive" operation that we wish to parallelize.
        f_x = x**2 - 2.0*x + 4.0
        # These operations are repeated on all procs.
        f_y = y ** 0.5
        g_y = y**2 + 3.0*y - 5.0
        # Compute square root of our portion of the distributed input.
        g_x = x ** 0.5
        # Distributed output
        outputs['out_dist'] = f_x + f_y
        # We need to gather the summed values to compute the total sum over all procs.
        local_sum = np.array(np.sum(g_x))
        total_sum = local_sum.copy()
        MPI.COMM_WORLD.Allreduce(local_sum, total_sum, op=MPI.SUM)
        outputs['out_nd'] = g_y * total_sum

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        x = inputs['in_dist']
        y = inputs['in_nd']

        df_dx = 2.0 * x - 2.0
        df_dy = 0.5 / y ** 0.5
        dg_dx = 0.5 / x ** 0.5
        dg_dy = 2.0 * y + 3.0

        nx = len(x)
        ny = len(y)

        if mode == 'fwd':
            if 'out_dist' in d_outputs:
                if 'in_dist' in d_inputs:
                    d_outputs['out_dist'] += df_dx * d_inputs['in_dist']
                if 'in_nd' in d_inputs:
                    d_outputs['out_dist'] += np.tile(df_dy, nx).reshape((nx, ny)).dot(d_inputs['in_nd'])

            if 'out_nd' in d_outputs:
                if 'in_dist' in d_inputs:
                    deriv = np.tile(dg_dx, ny).reshape((ny, nx)).dot(d_inputs['in_dist'])
                    deriv_sum = np.zeros(deriv.size)
                    self.comm.Allreduce(deriv, deriv_sum, op=MPI.SUM)
                    d_outputs['out_nd'] += deriv_sum
                if 'in_nd' in d_inputs:
                    d_outputs['out_nd'] += dg_dy * d_inputs['in_nd']

        else:
            if 'out_dist' in d_outputs:
                if 'in_dist' in d_inputs:
                    d_inputs['in_dist'] += df_dx * d_outputs['out_dist']
                if 'in_nd' in d_inputs:
                    d_inputs['in_nd'] += np.tile(df_dy, nx).reshape((nx, ny)).T.dot(d_outputs['out_dist'])

            if 'out_nd' in d_outputs:
                if 'out_nd' in d_outputs:
                    if 'in_dist' in d_inputs:
                        full = np.zeros(d_outputs['out_nd'].size)
                        self.comm.Allreduce(d_outputs['out_nd'], full, op=MPI.SUM)
                        d_inputs['in_dist'] += np.tile(dg_dx, ny).reshape((ny, nx)).T.dot(full)
                if 'in_nd' in d_inputs:
                    d_inputs['in_nd'] += dg_dy * d_outputs['out_nd']


@unittest.skipUnless(MPI, "MPI is required.")
class CheckPartialsRev(unittest.TestCase):

    N_PROCS = 2

    def test_cp_rev_mode(self):
        '''
        -----------------------------------------------------
        The erroneous output contained these values:

            Raw Forward Derivative (Jfor)
        [[62.5 62.5 62.5]]

            Raw Reverse Derivative (Jrev)
        [[31.25 31.25 31.25]]

            Raw FD Derivative (Jfd)
        [[62.49998444 62.49998444 62.49998444]]
        ...
            Raw Forward Derivative (Jfor)
        [[62.5 62.5 62.5 62.5]]

            Raw Reverse Derivative (Jrev)
        [[31.25 31.25 31.25 31.25]]

            Raw FD Derivative (Jfd)
        [[62.49998444 62.49998444 62.49998444 62.49998444]]
        -----------------------------------------------------
        The corrected output contains these values:

            Raw Forward Derivative (Jfor)
        [[62.5 62.5 62.5]]

            Raw Reverse Derivative (Jrev)
        [[62.5 62.5 62.5]]

            Raw FD Derivative (Jfd)
        [[62.49998444 62.49998444 62.49998444]]
        ...
            Raw Forward Derivative (Jfor)
        [[62.5 62.5 62.5 62.5]]

            Raw Reverse Derivative (Jrev)
        [[62.5 62.5 62.5 62.5]]

            Raw FD Derivative (Jfd)
        [[62.49998444 62.49998444 62.49998444 62.49998444]]
        -----------------------------------------------------
        '''
        size = 7
        comm = MPI.COMM_WORLD
        rank = comm.rank
        sizes, offsets = evenly_distrib_idxs(comm.size, size)

        prob = om.Problem()
        model = prob.model

        # Create a distributed source for the distributed input.
        ivc = om.IndepVarComp()
        ivc.add_output('x_dist', np.zeros(sizes[rank]), distributed=True)
        ivc.add_output('x_nd', val=1)

        model.add_subsystem("indep", ivc)
        model.add_subsystem("D1", MixedDistrib2())
        model.add_subsystem('con_cmp1', om.ExecComp('con1 = y**2'), promotes=['con1', 'y'])

        model.connect('indep.x_dist', 'D1.in_dist')
        model.connect('indep.x_nd', ['D1.in_nd','y'])

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        model.add_design_var('indep.x_nd', lower=5, upper=10)
        model.add_constraint('con1', upper=90)

        model.add_objective('D1.out_nd')

        prob.setup(force_alloc_complex=True)

        # Set initial values of distributed variable.
        x_dist_init = np.ones(sizes[rank])
        prob.set_val('indep.x_dist', x_dist_init)

        # Set initial values of non-distributed variable.
        prob.set_val('indep.x_nd', 10)

        prob.run_model()

        stream = StringIO()
        prob.check_partials(out_stream=stream)
        out_str = stream.getvalue()
        msg = "Problem.check_partials() output contains a reverse partial derivative divided by comm size."
        self.assertNotRegex(out_str, ".*31.25 31.25 31.25.*", msg)


