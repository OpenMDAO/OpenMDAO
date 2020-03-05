"""
    Optimize locations of Legendre-Gauss-Lobatto points to match a sine curve.
"""

import numpy as np

import openmdao.api as om


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
        self._mask[0] = 2 # / (n * (n - 1))
        self._mask[-1] = 2 # / (n * (n - 1))
        #self._mask = self._mask * np.pi

        da_df = np.atleast_2d(self.w_lgl*np.pi*self._mask)

        self.declare_partials(of='arclength', wrt='f_arclength', dependent=True, val=da_df)

    def compute(self, inputs, outputs):
        n = self.options['num_nodes']

        f = inputs['f_arclength']

        outputs['arclength'] = 2.0 * (f[-1] + f[0]) / (n * (n - 1)) + np.dot(self.w_lgl, f)
        outputs['arclength'] = outputs['arclength']*np.pi


class SineFitter(om.Group):

    def setup(self):

        # change this number for more compute points
        order = 7
        n = order + 1


        # Step 1:  Make an indep var comp that provides the approximated values at the LGL nodes.
        ivc = om.IndepVarComp('y_lgl', val=np.zeros(n), desc='values at LGL nodes')
        self.add_subsystem('y_lgl_ivc', ivc, promotes_outputs=['y_lgl'])

        # Step 2:  Make an indep var comp that provides the 'truth' values at the midpoint nodes.
        x_lgl, _ = lgl(n)
        x_lgl = x_lgl * np.pi # put x_lgl on [-pi, pi]
        x_mid = (x_lgl[1:] + x_lgl[:-1])/2.0 # midpoints on [-pi, pi]
        self.add_subsystem('truth', om.IndepVarComp('y_mid', val=np.sin(x_mid),
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

        self.add_design_var('y_lgl', lower=-1000.0, upper=1000.0)
        self.add_constraint('defect.defect', equals=0.)
        self.add_objective('arclength_quad.arclength')
