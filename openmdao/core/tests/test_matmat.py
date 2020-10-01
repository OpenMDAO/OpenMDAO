import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class QuadraticCompVectorized(om.ImplicitComponent):
    """
    A Simple Implicit Component representing a Quadratic Equation.

    R(a, b, c, x) = ax^2 + bx + c

    Solution via Quadratic Formula:
    x = (-b + sqrt(b^2 - 4ac)) / 2a
    """

    def setup(self):
        self.add_input('a', val=np.array([1.0, 2.0, 3.0]))
        self.add_input('b', val=np.array([2.0, 3.0, 4.0]))
        self.add_input('c', val=np.array([-1.0, -2.0, -3.0]))
        self.add_output('x', val=np.array([.5, .5, .5]))

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        residuals['x'] = a * x ** 2 + b * x + c

    def solve_nonlinear(self, inputs, outputs):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        outputs['x'] = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)

    def linearize(self, inputs, outputs, partials):
        a = inputs['a']
        b = inputs['b']
        x = outputs['x']
        self.inv_jac = 1.0 / (2 * a * x + b)

    def apply_multi_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        if mode == 'fwd':
            if 'x' in d_residuals:
                if 'x' in d_outputs:
                    d_residuals['x'] += (2 * a * x + b) * d_outputs['x']
                if 'a' in d_inputs:
                    d_residuals['x'] += x ** 2 * d_inputs['a']
                if 'b' in d_inputs:
                    d_residuals['x'] += x * d_inputs['b']
                if 'c' in d_inputs:
                    d_residuals['x'] += d_inputs['c']
        elif mode == 'rev':
            if 'x' in d_residuals:
                if 'x' in d_outputs:
                    d_outputs['x'] += (2 * a * x + b) * d_residuals['x']
                if 'a' in d_inputs:
                    d_inputs['a'] += x ** 2 * d_residuals['x']
                if 'b' in d_inputs:
                    d_inputs['b'] += x * d_residuals['x']
                if 'c' in d_inputs:
                    d_inputs['c'] += d_residuals['x']

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['x'] = self.inv_jac * d_residuals['x']
        elif mode == 'rev':
            d_residuals['x'] = self.inv_jac * d_outputs['x']


class QCVProblem(om.Problem):
    """
    A QuadraticCompVectorized problem with configurable component class.
    """

    def __init__(self, comp_class=QuadraticCompVectorized):
        super().__init__()

        model = self.model

        comp1 = model.add_subsystem('p', om.IndepVarComp())
        comp1.add_output('a', np.array([1.0, 2.0, 3.0]))
        comp1.add_output('b', np.array([2.0, 3.0, 4.0]))
        comp1.add_output('c', np.array([-1.0, -2.0, -3.0]))
        model.add_subsystem('comp', comp_class())

        model.connect('p.a', 'comp.a')
        model.connect('p.b', 'comp.b')
        model.connect('p.c', 'comp.c')

        model.add_design_var('p.a', vectorize_derivs=True)
        model.add_design_var('p.b', vectorize_derivs=True)
        model.add_design_var('p.c', vectorize_derivs=True)
        model.add_constraint('comp.x', vectorize_derivs=True)

        model.linear_solver = om.LinearBlockGS()


class RectangleCompVectorized(om.ExplicitComponent):
    """
    A simple Explicit Component that computes the area of a rectangle.
    """

    def setup(self):
        self.add_input('length', val=np.array([3.0, 4.0, 5.0]))
        self.add_input('width', val=np.array([1.0, 2.0, 3.0]))
        self.add_output('area', shape=(3, ))

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['area'] = inputs['length'] * inputs['width']

    def compute_multi_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'area' in d_outputs:
                if 'length' in d_inputs:
                    d_outputs['area'] += inputs['width'] * d_inputs['length']
                if 'width' in d_inputs:
                    d_outputs['area'] += inputs['length'] * d_inputs['width']
        elif mode == 'rev':
            if 'area' in d_outputs:
                if 'length' in d_inputs:
                    d_inputs['length'] += inputs['width'] * d_outputs['area']
                if 'width' in d_inputs:
                    d_inputs['width'] += inputs['length'] * d_outputs['area']


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
        at the cardinal nodes, yields the interpolated values at the interior
        nodes.

    Di : np.array
        A num_i x num_c matrix which, when post-multiplied by values specified
        at the cardinal nodes, yields the interpolated derivatives at the interior
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


class LGLFit(om.ExplicitComponent):
    """
    Given values at discretization nodes, provide interpolated values at midpoint nodes and
    an approximation of arclength.
    """

    def initialize(self):
        self.options.declare(name='num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

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


class DefectComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(name='num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

        self.add_input('y_truth', val=np.zeros(n-1), desc='actual values at midpoint nodes')
        self.add_input('y_approx', val=np.zeros(n-1), desc='interpolated values at midpoint nodes')
        self.add_output('defect', val=np.zeros(n-1), desc='error values at midpoint nodes')

        arange = np.arange(n-1)
        self.declare_partials(of='defect', wrt='y_truth', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='defect', wrt='y_approx', rows=arange, cols=arange, val=-1.0)

    def compute(self, inputs, outputs):
        outputs['defect'] = inputs['y_truth'] - inputs['y_approx']


class ArcLengthFunction(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(name='num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

        self.add_input('yp_lgl', val=np.zeros(n), desc='approximated derivative at LGL nodes')
        self.add_output('f_arclength', val=np.zeros(n), desc='The integrand of the arclength function')

        arange = np.arange(n)
        self.declare_partials(of='f_arclength', wrt='yp_lgl', rows=arange, cols=arange, val=1.0)

    def compute(self, inputs, outputs):
        outputs['f_arclength'] = np.sqrt(1 + inputs['yp_lgl']**2)

    def compute_partials(self, inputs, partials):
        partials['f_arclength', 'yp_lgl'] = inputs['yp_lgl'] / np.sqrt(1 + inputs['yp_lgl']**2)


class ArcLengthQuadrature(om.ExplicitComponent):
    """
    Computes the arclength of a polynomial segment whose values are given at the LGL nodes.
    """

    def initialize(self):
        self.options.declare(name='num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

        self.add_input('f_arclength', val=np.zeros(n), desc='The integrand of the arclength function')
        self.add_output('arclength', val=0.0, desc='The integrated arclength')

        _, self.w_lgl = lgl(n)

        self._mask = np.ones(n)
        self._mask[0] = 2  # / (n * (n - 1))
        self._mask[-1] = 2  # / (n * (n - 1))
        # self._mask = self._mask * np.pi

        da_df = np.atleast_2d(self.w_lgl*np.pi*self._mask)

        self.declare_partials(of='arclength', wrt='f_arclength', val=da_df)

    def compute(self, inputs, outputs):
        n = self.options['num_nodes']

        f = inputs['f_arclength']

        outputs['arclength'] = 2.0 * (f[-1] + f[0]) / (n * (n - 1)) + np.dot(self.w_lgl, f)
        outputs['arclength'] = outputs['arclength']*np.pi


class Phase(om.Group):

    def initialize(self):
        self.options.declare('order', types=int, default=10)

    def setup(self):
        order = self.options['order']
        n = order + 1

        # Step 1:  Make an indep var comp that provides the approximated values at the LGL nodes.
        self.add_subsystem('y_lgl_ivc', om.IndepVarComp('y_lgl', val=np.zeros(n), desc='values at LGL nodes'),
                           promotes_outputs=['y_lgl'])

        # Step 2:  Make an indep var comp that provides the 'truth' values at the midpoint nodes.
        x_lgl, _ = lgl(n)
        x_lgl = x_lgl * np.pi  # put x_lgl on [-pi, pi]
        x_mid = (x_lgl[1:] + x_lgl[:-1]) * 0.5  # midpoints on [-pi, pi]
        self.add_subsystem('truth', om.IndepVarComp('y_mid',
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


class Summer(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n_phases', types=int)

    def setup(self):
        self.add_output('total_arc_length')

        for i in range(self.options['n_phases']):
            i_name = 'arc_length:p%d' % i
            self.add_input(i_name)
            self.declare_partials('total_arc_length', i_name, val=1.)

    def compute(self, inputs, outputs):
        outputs['total_arc_length'] = 0
        for i in range(self.options['n_phases']):
            outputs['total_arc_length'] += inputs['arc_length:p%d' % i]


def simple_model(order, dvgroup='pardv', congroup='parc', vectorize=False):
    n = order + 1

    p = om.Problem()

    # Step 1:  Make an indep var comp that provides the approximated values at the LGL nodes.
    p.model.add_subsystem('y_lgl_ivc', om.IndepVarComp('y_lgl', val=np.zeros(n),
                                                       desc='values at LGL nodes'),
                          promotes_outputs=['y_lgl'])

    # Step 2:  Make an indep var comp that provides the 'truth' values at the midpoint nodes.
    x_lgl, _ = lgl(n)
    x_lgl = x_lgl * np.pi  # put x_lgl on [-pi, pi]
    x_mid = (x_lgl[1:] + x_lgl[:-1])/2.0  # midpoints on [-pi, pi]
    p.model.add_subsystem('truth', om.IndepVarComp('y_mid',
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

    p.model.add_design_var('y_lgl', lower=-1000.0, upper=1000.0,
                           parallel_deriv_color=dvgroup, vectorize_derivs=vectorize)
    p.model.add_constraint('defect.defect', lower=-1e-6, upper=1e-6,
                           parallel_deriv_color=congroup, vectorize_derivs=vectorize)
    p.model.add_objective('arclength_quad.arclength')
    p.driver = om.ScipyOptimizeDriver()
    return p, np.sin(x_lgl)


def phase_model(order, nphases, dvgroup='pardv', congroup='parc', vectorize=False):
    N_PHASES = nphases
    PHASE_ORDER = order

    n = PHASE_ORDER + 1

    p = om.Problem()

    # Step 1:  Make an indep var comp that provides the approximated values at the LGL nodes.
    for i in range(N_PHASES):
        p_name = 'p%d' % i
        p.model.add_subsystem(p_name, Phase(order=PHASE_ORDER))
        p.model.connect('%s.arclength_quad.arclength' % p_name, 'sum.arc_length:%s' % p_name)

        p.model.add_design_var('%s.y_lgl' % p_name, lower=-1000.0, upper=1000.0,
                               parallel_deriv_color=dvgroup, vectorize_derivs=vectorize)
        p.model.add_constraint('%s.defect.defect' % p_name, lower=-1e-6, upper=1e-6,
                               parallel_deriv_color=congroup, vectorize_derivs=vectorize)

    p.model.add_subsystem('sum', Summer(n_phases=N_PHASES))

    p.model.add_objective('sum.total_arc_length')
    p.driver = om.ScipyOptimizeDriver()

    x_lgl, _ = lgl(n)
    x_lgl = x_lgl * np.pi  # put x_lgl on [-pi, pi]
    # x_mid = (x_lgl[1:] + x_lgl[:-1])/2.0  # midpoints on [-pi, pi]

    return p, np.sin(x_lgl)


class MatMatTestCase(unittest.TestCase):

    def test_feature_vectorized_derivs(self):
        import numpy as np
        import openmdao.api as om

        SIZE = 5

        class ExpensiveAnalysis(om.ExplicitComponent):

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

        class CheapConstraint(om.ExplicitComponent):

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

        p = om.Problem()


        p.model.set_input_defaults('x', val=2*np.ones(SIZE))
        p.model.set_input_defaults('y', val=2*np.ones(SIZE))
        p.model.add_subsystem('obj', ExpensiveAnalysis(), promotes=['x', 'y', 'f'])
        p.model.add_subsystem('constraint', CheapConstraint(), promotes=['y', 'g'])

        p.model.add_design_var('x', lower=.1, upper=10000)
        p.model.add_design_var('y', lower=-1000, upper=10000)
        p.model.add_constraint('g', upper=0, vectorize_derivs=True)
        p.model.add_objective('f')

        p.setup(mode='rev')

        p.run_model()

        p.driver = om.ScipyOptimizeDriver()
        p.run_driver()

        assert_near_equal(p['x'], [0.10000691, 0.1, 0.1, 0.1, 0.1], 1e-5)
        assert_near_equal(p['y'], [0, 1.41421, 2.0, 2.44948, 2.82842], 1e-5)

    def test_simple_multi_fwd(self):
        p, expected = simple_model(order=20, vectorize=True)

        p.setup(mode='fwd')

        p.run_driver()

        # import matplotlib.pyplot as plt

        # plt.plot(p['y_lgl'], 'bo')
        # plt.plot(expected, 'go')
        # plt.show()

        y_lgl = p['y_lgl']
        assert_near_equal(expected, y_lgl, 1.e-5)

    def test_simple_multi_rev(self):
        p, expected = simple_model(order=20, vectorize=True)

        p.setup(mode='rev')

        p.run_driver()

        y_lgl = p['y_lgl']
        assert_near_equal(expected, y_lgl, 1.e-5)

    def test_phases_multi_fwd(self):
        N_PHASES = 4
        p, expected = phase_model(order=20, nphases=N_PHASES, vectorize=True)

        p.setup(mode='fwd')

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
            assert_near_equal(expected, p['p%d.y_lgl' % i], 1.e-5)

    def test_phases_multi_rev(self):
        N_PHASES = 4
        p, expected = phase_model(order=20, nphases=N_PHASES, vectorize=True)

        p.setup(mode='rev')

        p.run_driver()

        for i in range(N_PHASES):
            assert_near_equal(expected, p['p%d.y_lgl' % i], 1.e-5)

    def test_feature_declaration(self):
        # Tests the code that shows the signature for compute_multi_jacvec
        prob = om.Problem()
        model = prob.model

        comp1 = model.add_subsystem('p', om.IndepVarComp())
        comp1.add_output('length', np.array([3.0, 4.0, 5.0]))
        comp1.add_output('width', np.array([1.0, 2.0, 3.0]))

        model.add_subsystem('comp', RectangleCompVectorized())

        model.connect('p.length', 'comp.length')
        model.connect('p.width', 'comp.width')

        model.add_design_var('p.length', vectorize_derivs=True)
        model.add_design_var('p.width', vectorize_derivs=True)
        model.add_constraint('comp.area', vectorize_derivs=True)

        prob.setup(mode='rev')
        prob.run_model()

        J = prob.compute_totals(of=['comp.area'], wrt=['p.length', 'p.width'])
        assert_near_equal(J['comp.area', 'p.length'], np.diag(np.array([1.0, 2.0, 3.0])))
        assert_near_equal(J['comp.area', 'p.width'], np.diag(np.array([3.0, 4.0, 5.0])))

    def test_implicit(self):
        prob = QCVProblem()

        prob.setup(mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.compute_totals(of=['comp.x'], wrt=['p.a', 'p.b', 'p.c'])
        assert_near_equal(J['comp.x', 'p.a'], np.diag(np.array([-0.06066017, -0.05, -0.03971954])), 1e-4)
        assert_near_equal(J['comp.x', 'p.b'], np.diag(np.array([-0.14644661, -0.1, -0.07421663])), 1e-4)
        assert_near_equal(J['comp.x', 'p.c'], np.diag(np.array([-0.35355339, -0.2, -0.13867505])), 1e-4)

    def test_apply_multi_linear_inputs_read_only(self):
        class BadComp(QuadraticCompVectorized):
            def apply_multi_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                # inputs is read_only, should raise exception
                inputs['a'] = np.zeros(inputs['a'].shape)

        prob = QCVProblem(comp_class=BadComp)

        prob.setup(mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        with self.assertRaises(ValueError) as cm:
            prob.compute_totals()

        self.assertEqual(str(cm.exception),
                         "'comp' <class BadComp>: Attempt to set value of 'a' in input vector "
                         "when it is read only.")

    def test_apply_multi_linear_outputs_read_only(self):
        class BadComp(QuadraticCompVectorized):
            def apply_multi_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                # outputs is read_only, should raise exception
                outputs['x'] = np.zeros(outputs['x'].shape)

        prob = QCVProblem(comp_class=BadComp)

        prob.setup(mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        with self.assertRaises(ValueError) as cm:
            prob.compute_totals()

        self.assertEqual(str(cm.exception),
                         "'comp' <class BadComp>: Attempt to set value of 'x' in output vector "
                         "when it is read only.")

    def test_apply_multi_linear_dinputs_read_only(self):
        class BadComp(QuadraticCompVectorized):
            def apply_multi_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                # d_inputs is read_only, should raise exception
                d_inputs['a'] = np.zeros(inputs['a'].shape)

        prob = QCVProblem(comp_class=BadComp)

        prob.setup(mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        with self.assertRaises(ValueError) as cm:
            prob.compute_totals()

        self.assertEqual(str(cm.exception),
                         "'comp' <class BadComp>: Attempt to set value of 'a' in input vector "
                         "when it is read only.")

    def test_apply_multi_linear_doutputs_read_only(self):
        class BadComp(QuadraticCompVectorized):
            def apply_multi_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                # d_outputs is read_only, should raise exception
                d_outputs['x'] = np.zeros(outputs['x'].shape)

        prob = QCVProblem(comp_class=BadComp)

        prob.setup(mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        with self.assertRaises(ValueError) as cm:
            prob.compute_totals()

        self.assertEqual(str(cm.exception),
                         "'comp' <class BadComp>: Attempt to set value of 'x' in output vector "
                         "when it is read only.")

    def test_apply_multi_linear_dresids_read_only(self):
        class BadComp(QuadraticCompVectorized):
            def apply_multi_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                # d_residuals is read_only, should raise exception
                d_residuals['x'] = np.zeros(outputs['x'].shape)

        prob = QCVProblem(comp_class=BadComp)

        prob.setup(mode='rev')
        prob.set_solver_print(level=0)
        prob.run_model()

        with self.assertRaises(ValueError) as cm:
            prob.compute_totals()

        self.assertEqual(str(cm.exception),
                         "'comp' <class BadComp>: Attempt to set value of 'x' in residual vector "
                         "when it is read only.")


class JacVec(om.ExplicitComponent):

    def __init__(self, size):
        super().__init__()
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
    def setup_model(self, size, comp_class, vectorize, mode):
        p = om.Problem()
        model = p.model
        model.add_subsystem('px', om.IndepVarComp('x', val=(np.arange(5, dtype=float) + 1.) * 3.0))
        model.add_subsystem('py', om.IndepVarComp('y', val=(np.arange(5, dtype=float) + 1.) * 2.0))
        model.add_subsystem('comp', comp_class(size))

        model.connect('px.x', 'comp.x')
        model.connect('py.y', 'comp.y')

        model.add_design_var('px.x', vectorize_derivs=vectorize)
        model.add_design_var('py.y', vectorize_derivs=vectorize)
        model.add_constraint('comp.f_xy', vectorize_derivs=vectorize)

        p.setup(mode=mode)
        p.run_model()
        return p

    def test_compute_multi_jacvec_prod_fwd(self):
        p = self.setup_model(size=5, comp_class=JacVec, vectorize=False, mode='fwd')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_near_equal(J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_near_equal(J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_multi_jacvec_prod_rev(self):
        p = self.setup_model(size=5, comp_class=JacVec, vectorize=False, mode='rev')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_near_equal(J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_near_equal(J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_multi_jacvec_prod_fwd_vectorize(self):
        p = self.setup_model(size=5, comp_class=JacVec, vectorize=True, mode='fwd')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_near_equal(J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_near_equal(J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_multi_jacvec_prod_rev_vectorize(self):
        p = self.setup_model(size=5, comp_class=JacVec, vectorize=True, mode='rev')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_near_equal(J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_near_equal(J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_multi_jacvec_prod_fwd_multi(self):
        p = self.setup_model(size=5, comp_class=MultiJacVec, vectorize=False, mode='fwd')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_near_equal(J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_near_equal(J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_multi_jacvec_prod_rev_multi(self):
        p = self.setup_model(size=5, comp_class=MultiJacVec, vectorize=False, mode='rev')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_near_equal(J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_near_equal(J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_multi_jacvec_prod_fwd_vectorize_multi(self):
        p = self.setup_model(size=5, comp_class=MultiJacVec, vectorize=True, mode='fwd')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_near_equal(J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_near_equal(J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_multi_jacvec_prod_rev_vectorize_multi(self):
        p = self.setup_model(size=5, comp_class=MultiJacVec, vectorize=True, mode='rev')

        J = p.compute_totals(of=['comp.f_xy'], wrt=['px.x', 'py.y'])

        assert_near_equal(J[('comp.f_xy', 'px.x')], np.eye(5)*p['py.y'], 1e-5)
        assert_near_equal(J[('comp.f_xy', 'py.y')], np.eye(5)*p['px.x'], 1e-5)

    def test_compute_jacvec_product_mode_read_only(self):
        class BadComp(JacVec):
            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                # mode condition is reversed, should raise exception
                if mode == 'rev':
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

        p = self.setup_model(size=5, comp_class=BadComp, vectorize=True, mode='fwd')

        with self.assertRaises(ValueError) as cm:
            p.compute_totals()

        self.assertEqual(str(cm.exception),
                         "'comp' <class BadComp>: Attempt to set value of 'x' in input vector "
                         "when it is read only.")

        p = self.setup_model(size=5, comp_class=BadComp, vectorize=True, mode='rev')

        with self.assertRaises(ValueError) as cm:
            p.compute_totals()

        self.assertEqual(str(cm.exception),
                         "'comp' <class BadComp>: Attempt to set value of 'f_xy' in residual vector "
                         "when it is read only.")

    def test_compute_jacvec_product_inputs_read_only(self):
        class BadComp(JacVec):
            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                # inputs is read_only, should raise exception
                inputs['x'] = np.zeros(self.size)

        p = self.setup_model(size=5, comp_class=BadComp, vectorize=True, mode='fwd')

        with self.assertRaises(ValueError) as cm:
            p.compute_totals()

        self.assertEqual(str(cm.exception),
                         "'comp' <class BadComp>: Attempt to set value of 'x' in input vector "
                         "when it is read only.")

    def test_compute_multi_jacvec_product_mode_read_only(self):
        class BadComp(JacVec):
            def compute_multi_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                # mode condition is reversed, should raise exception
                if mode == 'rev':
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

        p = self.setup_model(size=5, comp_class=BadComp, vectorize=True, mode='fwd')

        with self.assertRaises(ValueError) as cm:
            p.compute_totals()

        self.assertEqual(str(cm.exception),
                         "'comp' <class BadComp>: Attempt to set value of 'x' in input vector "
                         "when it is read only.")

        p = self.setup_model(size=5, comp_class=BadComp, vectorize=True, mode='rev')

        with self.assertRaises(ValueError) as cm:
            p.compute_totals()

        self.assertEqual(str(cm.exception),
                         "'comp' <class BadComp>: Attempt to set value of 'f_xy' in residual vector "
                         "when it is read only.")

    def test_compute_multi_jacvec_product_inputs_read_only(self):
        class BadComp(JacVec):
            def compute_multi_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                # inputs is read_only, should raise exception
                inputs['x'] = np.zeros(self.size)

        p = self.setup_model(size=5, comp_class=BadComp, vectorize=True, mode='fwd')

        with self.assertRaises(ValueError) as cm:
            p.compute_totals()

        self.assertEqual(str(cm.exception),
                         "'comp' <class BadComp>: Attempt to set value of 'x' in input vector "
                         "when it is read only.")


if __name__ == '__main__':
    unittest.main()
