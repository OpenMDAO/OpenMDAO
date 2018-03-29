from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent, \
                         ScipyOptimizeDriver, DefaultVector, DenseJacobian, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error

def lgl(n, tol=np.finfo(float).eps):
    """
    Returns the Legendre-Gauss-Lobatto nodes and weights for a
    Jacobi Polynomial with n abscissae.

    The nodes are on the range [-1, 1].

    Based on the routine written by Greg von Winckel (BSD License follows)

    Copyright (c) 2009, Greg von Winckel
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the distribution

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR aux_outputs
    PARTICULAR PURPOSE  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
    OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Parameters
    ----------
    n : int
        The number of LGL nodes requested.  The order of the polynomial is n-1.

    Returns
    -------
    x : numpy.array
        An array of the LGL nodes for a polynomial of the given order.

    w : numpy.array
        An array of the corresponding LGL weights at the nodes in x.

    """
    n = n - 1
    n1 = n + 1
    n2 = n + 2
    # Get the initial guesses from the Chebyshev nodes
    x = np.cos(np.pi * (2 * np.arange(1, n2) - 1) / (2 * n1))
    P = np.zeros([n1, n1])
    # Compute P_(n) using the recursion relation
    # Compute its first and second derivatives and
    # update x using the Newton-Raphson method.
    xold = 2

    for i in range(100):
        if np.all(np.abs(x - xold) <= tol):
            break
        xold = x
        P[:, 0] = 1.0
        P[:, 1] = x

        for k in range(2, n1):
            P[:, k] = ((2*k-1)*x*P[:, k-1]-(k-1)*P[:, k-2])/k

        x = xold - (x*P[:, n]-P[:, n-1])/(n1*P[:, n])
    else:
        raise RuntimeError('Failed to converge LGL nodes '
                           'for order {0}'.format(n))

    x.sort()

    w = 2 / (n*n1*P[:, n]**2)

    return x, w


def lagrange_matrices(x_disc, x_interp):
    """
    Compute the lagrange matrices which, given 'cardinal' nodes at which
    values are specified and 'interior' nodes at which values will be desired,
    returns interpolation and differentiation matrices which provide polynomial
    values and derivatives

    Parameters
    ----------
    x_disc : np.array
        The cardinal nodes at which values of the variable are specified.
    x_interp : np.array
        The interior nodes at which interpolated values of the variable or its derivative
        are desired.

    Returns
    -------
    Li : np.array
        A num_i x num_c matrix which, when post-multiplied by values specified
        at the cardinal nodes, yields the intepolated values at the interior
        nodes.

    Di : np.array
        A num_i x num_c matrix which, when post-multiplied by values specified
        at the cardinal nodes, yields the intepolated derivatives at the interior
        nodes.

    """

    nd = len(x_disc)

    ni = len(x_interp)
    wb = np.ones(nd)

    Li = np.zeros((ni, nd))
    Di = np.zeros((ni, nd))

    # Barycentric Weights
    for j in range(nd):
        for k in range(nd):
            if k != j:
                wb[j] /= (x_disc[j] - x_disc[k])

    # Compute Li
    for i in range(ni):
        for j in range(nd):
            Li[i, j] = wb[j]
            for k in range(nd):
                if k != j:
                    Li[i, j] *= (x_interp[i] - x_disc[k])

    # Compute Di
    for i in range(ni):
        for j in range(nd):
            for k in range(nd):
                prod = 1.0
                if k != j:
                    for m in range(nd):
                        if m != j and m != k:
                            prod *= (x_interp[i] - x_disc[m])
                    Di[i, j] += (wb[j] * prod)

    return Li, Di

class LGLFit(ExplicitComponent):
    """
    Given values at discretization nodes, provide interpolated values at midpoint nodes and
    an approximation of arclength.
    """
    def initialize(self):
        self.metadata.declare(name='num_nodes', types=int)

    def setup(self):
        n = self.metadata['num_nodes']

        self.x_lgl, self.w_lgl = lgl(n)

        self.x_mid = (self.x_lgl[1:] + self.x_lgl[:-1])/2.0

        self.L_mid, _ = lagrange_matrices(self.x_lgl, self.x_mid)
        _, self.D_lgl = lagrange_matrices(self.x_lgl, self.x_lgl)

        self.add_input('y_lgl', val=np.zeros(n), desc='given values at LGL nodes')

        self.add_output('y_mid', val=np.zeros(n-1), desc='interpolated values at midpoint nodes')
        self.add_output('yp_lgl', val=np.zeros(n), desc='approximated derivative at LGL nodes')

        self.declare_partials(of='y_mid', wrt='y_lgl', val=self.L_mid)
        self.declare_partials(of='yp_lgl', wrt='y_lgl', val=self.D_lgl/np.pi)

    def compute(self, inputs, outputs):

        outputs['y_mid'] = np.dot(self.L_mid, inputs['y_lgl'])
        outputs['yp_lgl'] = np.dot(self.D_lgl, inputs['y_lgl'])/np.pi


class DefectComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare(name='num_nodes', types=int)

    def setup(self):
        n = self.metadata['num_nodes']

        self.add_input('y_truth', val=np.zeros(n-1), desc='actual values at midpoint nodes')
        self.add_input('y_approx', val=np.zeros(n-1), desc='interpolated values at midpoint nodes')
        self.add_output('defect', val=np.zeros(n-1), desc='error values at midpoint nodes')

        arange = np.arange(n-1)
        self.declare_partials(of='defect', wrt='y_truth', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='defect', wrt='y_approx', rows=arange, cols=arange, val=-1.0)

    def compute(self, inputs, outputs):
        outputs['defect'] = inputs['y_truth'] - inputs['y_approx']


class ArcLengthFunction(ExplicitComponent):

    def initialize(self):
        self.metadata.declare(name='num_nodes', types=int)

    def setup(self):
        n = self.metadata['num_nodes']

        self.add_input('yp_lgl', val=np.zeros(n), desc='approximated derivative at LGL nodes')
        self.add_output('f_arclength', val=np.zeros(n), desc='The integrand of the arclength function')

        arange = np.arange(n)
        self.declare_partials(of='f_arclength', wrt='yp_lgl', rows=arange, cols=arange, val=1.0)

    def compute(self, inputs, outputs):
        outputs['f_arclength'] = np.sqrt(1 + inputs['yp_lgl']**2)


    def compute_partials(self, inputs, partials):
        partials['f_arclength', 'yp_lgl'] = inputs['yp_lgl'] / np.sqrt(1 + inputs['yp_lgl']**2)


class ArcLengthQuadrature(ExplicitComponent):
    """
    Computes the arclength of a polynomial segment whose values are given at the LGL nodes.
    """
    def initialize(self):
        self.metadata.declare(name='num_nodes', types=int)

    def setup(self):
        n = self.metadata['num_nodes']

        self.add_input('f_arclength', val=np.zeros(n), desc='The integrand of the arclength function')
        self.add_output('arclength', val=0.0, desc='The integrated arclength')

        _, self.w_lgl = lgl(n)

        self._mask = np.ones(n)
        self._mask[0] = 2 # / (n * (n - 1))
        self._mask[-1] = 2 # / (n * (n - 1))
        #self._mask = self._mask * np.pi

        da_df = np.atleast_2d(self.w_lgl*np.pi*self._mask)

        self.declare_partials(of='arclength', wrt='f_arclength', dependent=True, val=da_df)

    def compute(self, inputs, outputs):
        n = self.metadata['num_nodes']

        f = inputs['f_arclength']

        outputs['arclength'] = 2.0 * (f[-1] + f[0]) / (n * (n - 1)) + np.dot(self.w_lgl, f)
        outputs['arclength'] = outputs['arclength']*np.pi


class Phase(Group):

    def initialize(self):
        self.metadata.declare('order', types=int, default=10)

    def setup(self):
        order = self.metadata['order']
        n = order + 1

        # Step 1:  Make an indep var comp that provides the approximated values at the LGL nodes.
        self.add_subsystem('y_lgl_ivc', IndepVarComp('y_lgl', val=np.zeros(n), desc='values at LGL nodes'),
                              promotes_outputs=['y_lgl'])

        # Step 2:  Make an indep var comp that provides the 'truth' values at the midpoint nodes.
        x_lgl, _ = lgl(n)
        x_lgl = x_lgl * np.pi # put x_lgl on [-pi, pi]
        x_mid = (x_lgl[1:] + x_lgl[:-1]) * 0.5 # midpoints on [-pi, pi]
        self.add_subsystem('truth', IndepVarComp('y_mid',
                                                 val=np.sin(x_mid),
                                                 desc='truth values at midpoint nodes'))


        # Step 3: Make a polynomial fitting component
        self.add_subsystem('lgl_fit', LGLFit(num_nodes=n))

        # Step 4: Add the defect component
        self.add_subsystem('defect', DefectComp(num_nodes=n))

        # Step 5: Compute the integrand of the arclength function then quadrature it
        self.add_subsystem('arclength_func', ArcLengthFunction(num_nodes=n))
        self.add_subsystem('arclength_quad', ArcLengthQuadrature(num_nodes=n))


        self.connect('y_lgl', 'lgl_fit.y_lgl')
        self.connect('truth.y_mid', 'defect.y_truth')
        self.connect('lgl_fit.y_mid', 'defect.y_approx')
        self.connect('lgl_fit.yp_lgl', 'arclength_func.yp_lgl')
        self.connect('arclength_func.f_arclength', 'arclength_quad.f_arclength')


class Summer(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('n_phases', types=int)

    def setup(self):
        self.add_output('total_arc_length')

        for i in range(self.metadata['n_phases']):
            i_name = 'arc_length:p%d' % i
            self.add_input(i_name)
            self.declare_partials('total_arc_length', i_name, val=1.)

    def compute(self, inputs, outputs):
        outputs['total_arc_length'] = 0
        for i in range(self.metadata['n_phases']):
            outputs['total_arc_length'] += inputs['arc_length:p%d' % i]

def simple_model(order, dvgroup='pardv', congroup='parc', vectorize=False):
    n = order + 1

    p = Problem(model=Group())

    # Step 1:  Make an indep var comp that provides the approximated values at the LGL nodes.
    p.model.add_subsystem('y_lgl_ivc', IndepVarComp('y_lgl', val=np.zeros(n),
                          desc='values at LGL nodes'),
                          promotes_outputs=['y_lgl'])

    # Step 2:  Make an indep var comp that provides the 'truth' values at the midpoint nodes.
    x_lgl, _ = lgl(n)
    x_lgl = x_lgl * np.pi # put x_lgl on [-pi, pi]
    x_mid = (x_lgl[1:] + x_lgl[:-1])/2.0 # midpoints on [-pi, pi]
    p.model.add_subsystem('truth', IndepVarComp('y_mid',
                                                val=np.sin(x_mid),
                                                desc='truth values at midpoint nodes'))

    # Step 3: Make a polynomial fitting component
    p.model.add_subsystem('lgl_fit', LGLFit(num_nodes=n))

    # Step 4: Add the defect component
    p.model.add_subsystem('defect', DefectComp(num_nodes=n))

    # Step 5: Compute the integrand of the arclength function then quadrature it
    p.model.add_subsystem('arclength_func', ArcLengthFunction(num_nodes=n))
    p.model.add_subsystem('arclength_quad', ArcLengthQuadrature(num_nodes=n))

    p.model.connect('y_lgl', 'lgl_fit.y_lgl')
    p.model.connect('truth.y_mid', 'defect.y_truth')
    p.model.connect('lgl_fit.y_mid', 'defect.y_approx')
    p.model.connect('lgl_fit.yp_lgl', 'arclength_func.yp_lgl')
    p.model.connect('arclength_func.f_arclength', 'arclength_quad.f_arclength')

    p.model.add_design_var('y_lgl', lower=-1000.0, upper=1000.0, parallel_deriv_color=dvgroup, vectorize_derivs=vectorize)
    p.model.add_constraint('defect.defect', lower=-1e-6, upper=1e-6, parallel_deriv_color=congroup, vectorize_derivs=vectorize)
    p.model.add_objective('arclength_quad.arclength')
    p.driver = ScipyOptimizeDriver()
    return p, np.sin(x_lgl)

def phase_model(order, nphases, dvgroup='pardv', congroup='parc', vectorize=False):
    N_PHASES = nphases
    PHASE_ORDER = order

    n = PHASE_ORDER + 1

    p = Problem()

    # Step 1:  Make an indep var comp that provides the approximated values at the LGL nodes.
    for i in range(N_PHASES):
        p_name = 'p%d' % i
        p.model.add_subsystem(p_name, Phase(order=PHASE_ORDER))
        p.model.connect('%s.arclength_quad.arclength' % p_name, 'sum.arc_length:%s' % p_name)

        p.model.add_design_var('%s.y_lgl' % p_name, lower=-1000.0, upper=1000.0, parallel_deriv_color=dvgroup, vectorize_derivs=vectorize)
        p.model.add_constraint('%s.defect.defect' % p_name, lower=-1e-6, upper=1e-6, parallel_deriv_color=congroup, vectorize_derivs=vectorize)

    p.model.add_subsystem('sum', Summer(n_phases=N_PHASES))

    p.model.add_objective('sum.total_arc_length')
    p.driver = ScipyOptimizeDriver()

    x_lgl, _ = lgl(n)
    x_lgl = x_lgl * np.pi # put x_lgl on [-pi, pi]
    x_mid = (x_lgl[1:] + x_lgl[:-1])/2.0 # midpoints on [-pi, pi]

    return p, np.sin(x_lgl)



class MatMatTestCase(unittest.TestCase):


    def test_feature_vectorized_derivs(self):
        import numpy as np
        from openmdao.api import ExplicitComponent, ExecComp, IndepVarComp, Problem, ScipyOptimizeDriver

        SIZE = 5

        class ExpensiveAnalysis(ExplicitComponent):

            def setup(self):

                self.add_input('x', val=np.ones(SIZE))
                self.add_input('y', val=np.ones(SIZE))

                self.add_output('f', shape=1)

                self.declare_partials('f', 'x')
                self.declare_partials('f', 'y')

            def compute(self, inputs, outputs):

                outputs['f'] = np.sum(inputs['x']**inputs['y'])

            def compute_partials(self, inputs, J):

                J['f', 'x'] = inputs['y']*inputs['x']**(inputs['y']-1)
                J['f', 'y'] = (inputs['x']**inputs['y'])*np.log(inputs['x'])

        class CheapConstraint(ExplicitComponent):

            def setup(self):

                self.add_input('y', val=np.ones(SIZE))
                self.add_output('g', shape=SIZE)

                row_col = np.arange(SIZE, dtype=int)
                self.declare_partials('g', 'y', rows=row_col, cols=row_col)

                self.limit = 2*np.arange(SIZE)

            def compute(self, inputs, outputs):

                outputs['g'] = inputs['y']**2 - self.limit

            def compute_partials(self, inputs, J):

                J['g', 'y'] = 2*inputs['y']


        p = Problem()

        dvs = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        dvs.add_output('x', 2*np.ones(SIZE))
        dvs.add_output('y', 2*np.ones(SIZE))

        p.model.add_subsystem('obj', ExpensiveAnalysis(), promotes=['x', 'y', 'f'])
        p.model.add_subsystem('constraint', CheapConstraint(), promotes=['y', 'g'])

        p.model.add_design_var('x', lower=.1, upper=10000)
        p.model.add_design_var('y', lower=-1000, upper=10000)
        p.model.add_constraint('g', upper=0, vectorize_derivs=True)
        p.model.add_objective('f')


        p.setup(mode='rev')

        p.run_model()

        p.driver = ScipyOptimizeDriver()
        p.run_driver()

        assert_rel_error(self, p['x'], [0.10000691, 0.1, 0.1, 0.1, 0.1], 1e-5)
        assert_rel_error(self, p['y'], [0, 1.41421, 2.0, 2.44948, 2.82842], 1e-5)

    def test_simple_multi_fwd(self):
        p, expected = simple_model(order=20, vectorize=True)

        p.setup(check=False, mode='fwd')

        p.run_driver()

        #import matplotlib.pyplot as plt

        #plt.plot(p['y_lgl'], 'bo')
        #plt.plot(expected, 'go')
        #plt.show()

        y_lgl = p['y_lgl']
        assert_rel_error(self, expected, y_lgl, 1.e-5)

    def test_simple_multi_rev(self):
        p, expected = simple_model(order=20, vectorize=True)

        p.setup(check=False, mode='rev')

        p.run_driver()

        y_lgl = p['y_lgl']
        assert_rel_error(self, expected, y_lgl, 1.e-5)

    def test_phases_multi_fwd(self):
        N_PHASES = 4
        p, expected = phase_model(order=20, nphases=N_PHASES, vectorize=True)

        p.setup(check=False, mode='fwd')

        p.run_driver()

        # import matplotlib.pyplot as plt
        # for i in range(N_PHASES):
        #     offset = i*2*np.pi
        #     p_name = 'p%d' % i
        #     plt.plot(offset+x_mid, p['%s.truth.y_mid' % p_name], 'ro')
        #     plt.plot(offset+x_lgl, p['%s.y_lgl' % p_name], 'bo')
        #
        # plt.show()

        for i in range(N_PHASES):
            assert_rel_error(self, expected, p['p%d.y_lgl' % i], 1.e-5)

    def test_phases_multi_rev(self):
        N_PHASES = 4
        p, expected = phase_model(order=20, nphases=N_PHASES, vectorize=True)

        p.setup(check=False, mode='rev')

        p.run_driver()

        for i in range(N_PHASES):
            assert_rel_error(self, expected, p['p%d.y_lgl' % i], 1.e-5)


class JacVec(ExplicitComponent):

    def __init__(self, size):
        super(JacVec, self).__init__()
        self.size = size

    def setup(self):
        size = self.size
        self.add_input('x', val=np.zeros(size))
        self.add_input('y', val=np.zeros(size))
        self.add_output('f_xy', val=np.zeros(size))

    def compute(self, inputs, outputs):
        outputs['f_xy'] = inputs['x'] * inputs['y']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'x' in d_inputs:
                d_outputs['f_xy'] += d_inputs['x'] * inputs['y']
            if 'y' in d_inputs:
                d_outputs['f_xy'] += d_inputs['y'] * inputs['x']
        else:
            d_fxy = d_outputs['f_xy']
            if 'x' in d_inputs:
                d_inputs['x'] += d_fxy * inputs['y']
            if 'y' in d_inputs:
                d_inputs['y'] += d_fxy * inputs['x']

class MultiJacVec(JacVec):
    def compute_multi_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        # same as compute_jacvec_product in this case
        self.compute_jacvec_product(inputs, d_inputs, d_outputs, mode)


class ComputeMultiJacVecTestCase(unittest.TestCase):
    def setup_model(self, size, multi, vectorize, mode):
        comp_class = MultiJacVec if multi else JacVec
        p = Problem()
        model = p.model
        model.add_subsystem('px', IndepVarComp('x', val=(np.arange(5, dtype=float) + 1.) * 3.0))
        model.add_subsystem('py', IndepVarComp('y', val=(np.arange(5, dtype=float) + 1.) * 2.0))
        model.add_subsystem('comp', comp_class(size))

        model.connect('px.x', 'comp.x')
        model.connect('py.y', 'comp.y')

        model.add_design_var('px.x', vectorize_derivs=vectorize)
        model.add_design_var('py.y', vectorize_derivs=vectorize)
        model.add_constraint('comp.f_xy', vectorize_derivs=vectorize)

        p.setup(check=False, mode=mode)
        p.run_model()
        return p

    def test_compute_multi_jacvec_prod_fwd(self):
        p = self.setup_model(size=5, multi=False, vectorize=False, mode='fwd')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_rel_error(self, J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_rel_error(self, J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_multi_jacvec_prod_rev(self):
        p = self.setup_model(size=5, multi=False, vectorize=False, mode='rev')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_rel_error(self, J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_rel_error(self, J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_multi_jacvec_prod_fwd_vectorize(self):
        p = self.setup_model(size=5, multi=False, vectorize=True, mode='fwd')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_rel_error(self, J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_rel_error(self, J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_multi_jacvec_prod_rev_vectorize(self):
        p = self.setup_model(size=5, multi=False, vectorize=True, mode='rev')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_rel_error(self, J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_rel_error(self, J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_multi_jacvec_prod_fwd_multi(self):
        p = self.setup_model(size=5, multi=True, vectorize=False, mode='fwd')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_rel_error(self, J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_rel_error(self, J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_multi_jacvec_prod_rev_multi(self):
        p = self.setup_model(size=5, multi=True, vectorize=False, mode='rev')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_rel_error(self, J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_rel_error(self, J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_multi_jacvec_prod_fwd_vectorize_multi(self):
        p = self.setup_model(size=5, multi=True, vectorize=True, mode='fwd')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_rel_error(self, J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_rel_error(self, J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_multi_jacvec_prod_rev_vectorize_multi(self):
        p = self.setup_model(size=5, multi=True, vectorize=True, mode='rev')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_rel_error(self, J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_rel_error(self, J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)


if __name__ == '__main__':
    unittest.main()
