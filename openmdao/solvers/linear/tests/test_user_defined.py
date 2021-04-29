"""Test the LinearUserDefined linear solver class."""

import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.array_utils import evenly_distrib_idxs

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class DistribStateImplicit(om.ImplicitComponent):

    def setup(self):

        self.add_input('a', val=10., units='m')

        rank = self.comm.rank
        GLOBAL_SIZE = 15
        sizes, offsets = evenly_distrib_idxs(self.comm.size, GLOBAL_SIZE)

        self.add_output('states', shape=int(sizes[rank]))

        self.add_output('out_var', shape=1)
        self.local_size = sizes[rank]

        self.linear_solver = om.PETScKrylov()
        self.linear_solver.precon = om.LinearUserDefined(self.mysolve)

    def solve_nonlinear(self, i, o):
        o['states'] = i['a']

        local_sum = np.zeros(1)
        local_sum[0] = np.sum(o['states'])
        tmp = np.zeros(1)

        o['out_var'] = tmp[0]

    def apply_nonlinear(self, i, o, r):
        r['states'] = o['states'] - i['a']

        local_sum = np.zeros(1)
        local_sum[0] = np.sum(o['states'])
        global_sum = np.zeros(1)

        r['out_var'] = o['out_var'] - tmp[0]

    def apply_linear(self, i, o, d_i, d_o, d_r, mode):
        if mode == 'fwd':
            if 'states' in d_o:
                d_r['states'] += d_o['states']

                local_sum = np.array([np.sum(d_o['states'])])
                global_sum = np.zeros(1)
                self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
                d_r['out_var'] -= global_sum

            if 'out_var' in d_o:
                    d_r['out_var'] += d_o['out_var']

            if 'a' in d_i:
                    d_r['states'] -= d_i['a']

        elif mode == 'rev':
            if 'states' in d_o:
                d_o['states'] += d_r['states']

                tmp = np.zeros(1)
                if self.comm.rank == 0:
                    tmp[0] = d_r['out_var'].copy()
                self.comm.Bcast(tmp, root=0)

                d_o['states'] -= tmp

            if 'out_var' in d_o:
                d_o['out_var'] += d_r['out_var']

            if 'a' in d_i:
                    d_i['a'] -= np.sum(d_r['states'])

    def mysolve(self, d_outputs, d_residuals, mode):
        r"""
        Apply inverse jac product. The model is assumed to be in an unscaled state.

        If mode is:
            'fwd': d_residuals \|-> d_outputs

            'rev': d_outputs \|-> d_residuals

        Note: this is not the linear solution for the implicit component. We use identity so
        that simple implicit components can function in a preconditioner under linear gauss-seidel.
        To correctly solve this component, you should slot a solver in linear_solver or override
        this method.

        Parameters
        ----------
        d_outputs: Vector
            unscaled, dimensional quantities read via d_outputs[key]
        d_residuals: Vector
            unscaled, dimensional quantities read via d_residuals[key]
        mode: str
            either 'fwd' or 'rev'
        """
        # Note: we are just preconditioning with Identity as a proof of concept.
        if mode == 'fwd':
            d_outputs.set_vec(d_residuals)
        elif mode == 'rev':
            d_residuals.set_vec(d_outputs)


@unittest.skipUnless(PETScVector is not None, "PETSc is required.")
class TestUserDefinedSolver(unittest.TestCase):

    def test_method(self):
        p = om.Problem()

        p.model.add_subsystem('des_vars', om.IndepVarComp('a', val=10., units='m'), promotes=['*'])

        p.model.add_subsystem('icomp', DistribStateImplicit(), promotes=['*'])

        model = p.model

        model.linear_solver = om.PETScKrylov()
        model.linear_solver.precon = om.LinearRunOnce()

        p.setup(mode='rev', check=False)
        p.run_model()
        jac = p.compute_totals(of=['out_var'], wrt=['a'], return_format='dict')

        assert_near_equal(15.0, jac['out_var']['a'][0][0])

    def test_scaling(self):
        # Make sure values are unscaled/dimensional.

        def custom_method(d_outputs, d_residuals, mode):
            # This should be -1 because the jac setter pokes a -1.0 in phys state.
            if d_outputs['out_var'][0] != -1.0:
                raise ValueError('This value should be unscaled.')


        class ScaledComp(om.ImplicitComponent):

            def setup(self):

                self.add_input('a', val=10., units='m')

                self.add_output('states', val=20.0, ref=3333.0)
                self.add_output('out_var', val=20.0, ref=12.0)


        p = om.Problem()
        p.model.add_subsystem('des_vars', om.IndepVarComp('a', val=10., units='m'), promotes=['*'])
        p.model.add_subsystem('icomp', ScaledComp(), promotes=['*'])
        model = p.model

        model.linear_solver = om.LinearUserDefined(custom_method)

        p.setup(mode='rev', check=False)
        p.run_model()
        jac = p.compute_totals(of=['out_var'], wrt=['a'], return_format='dict')

    def test_method_default(self):
        # Uses `solve_linear` by default
        p = om.Problem()

        p.model.add_subsystem('des_vars', om.IndepVarComp('a', val=10., units='m'), promotes=['*'])

        p.model.add_subsystem('icomp', DistribStateImplicit(), promotes=['*'])

        model = p.model

        model.linear_solver = om.PETScKrylov()
        model.linear_solver.precon = om.LinearRunOnce()

        p.setup(mode='rev', check=False)

        model.icomp.linear_solver.precon = om.LinearUserDefined()

        p.run_model()
        jac = p.compute_totals(of=['out_var'], wrt=['a'], return_format='dict')

        assert_near_equal(15.0, jac['out_var']['a'][0][0])

    def test_feature(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.utils.array_utils import evenly_distrib_idxs

        class CustomSolveImplicit(om.ImplicitComponent):

            def setup(self):

                self.add_input('a', val=10., units='m')

                rank = self.comm.rank
                GLOBAL_SIZE = 15
                sizes, offsets = evenly_distrib_idxs(self.comm.size, GLOBAL_SIZE)

                self.add_output('states', shape=int(sizes[rank]))

                self.add_output('out_var', shape=1)
                self.local_size = sizes[rank]

                self.linear_solver = om.PETScKrylov()
                self.linear_solver.precon = om.LinearUserDefined(solve_function=self.mysolve)

            def solve_nonlinear(self, i, o):
                o['states'] = i['a']

                local_sum = np.zeros(1)
                local_sum[0] = np.sum(o['states'])
                tmp = np.zeros(1)

                o['out_var'] = tmp[0]

            def apply_nonlinear(self, i, o, r):
                r['states'] = o['states'] - i['a']

                local_sum = np.zeros(1)
                local_sum[0] = np.sum(o['states'])
                global_sum = np.zeros(1)

                r['out_var'] = o['out_var'] - tmp[0]

            def apply_linear(self, i, o, d_i, d_o, d_r, mode):
                if mode == 'fwd':
                    if 'states' in d_o:
                        d_r['states'] += d_o['states']

                        local_sum = np.array([np.sum(d_o['states'])])
                        global_sum = np.zeros(1)
                        self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
                        d_r['out_var'] -= global_sum

                    if 'out_var' in d_o:
                            d_r['out_var'] += d_o['out_var']

                    if 'a' in d_i:
                            d_r['states'] -= d_i['a']

                elif mode == 'rev':
                    if 'states' in d_o:
                        d_o['states'] += d_r['states']

                        tmp = np.zeros(1)
                        if self.comm.rank == 0:
                            tmp[0] = d_r['out_var'].copy()
                        self.comm.Bcast(tmp, root=0)

                        d_o['states'] -= tmp

                    if 'out_var' in d_o:
                        d_o['out_var'] += d_r['out_var']

                    if 'a' in d_i:
                            d_i['a'] -= np.sum(d_r['states'])

            def mysolve(self, d_outputs, d_residuals, mode):
                r"""
                Apply inverse jac product. The model is assumed to be in an unscaled state.

                If mode is:
                    'fwd': d_residuals \|-> d_outputs

                    'rev': d_outputs \|-> d_residuals

                Parameters
                ----------
                d_outputs: Vector
                    unscaled, dimensional quantities read via d_outputs[key]
                d_residuals: Vector
                    unscaled, dimensional quantities read via d_residuals[key]
                mode: str
                    either 'fwd' or 'rev'
                """
                # Note: we are just preconditioning with Identity as a proof of concept.
                if mode == 'fwd':
                    d_outputs.set_vec(d_residuals)
                elif mode == 'rev':
                    d_residuals.set_vec(d_outputs)

        prob = om.Problem()


        prob.model.add_subsystem('icomp', CustomSolveImplicit(), promotes=['*'])
        prob.model.set_input_defaults('a', 10., units='m')

        model = prob.model

        model.linear_solver = om.PETScKrylov()
        model.linear_solver.precon = om.LinearRunOnce()

        prob.setup(mode='rev', check=False)
        prob.run_model()
        jac = prob.compute_totals(of=['out_var'], wrt=['a'], return_format='dict')

        assert_near_equal(15.0, jac['out_var']['a'][0][0])

if __name__ == "__main__":
    unittest.main()
