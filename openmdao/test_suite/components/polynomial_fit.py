"""
Component to demonstrate using an ImplicitComponent to perform a polynomial curve fit.
"""
import openmdao.api as om
import numpy as np

class PolynomialFit(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('N_cp', types=int)
        self.options.declare('N_predict', types=int)

    def setup(self):

        # data to fit, which can also be thought of as the "control points"
        # of the fit function
        self.add_input('x_cp', shape=self.options['N_cp'])
        self.add_input('y_cp', shape=self.options['N_cp'])

        # location of the points you want to evaluate the final fit function at
        self.add_input('x', shape=self.options['N_predict'])
        # computed value of the fitted polynomial at the x points
        self.add_output('y', shape=self.options['N_predict'])

        # these are the coefficients of the polynomial function you are fitting
        self.add_output('A', np.zeros(6))  # assuming a 5th order polynomial

        # analytic derivatives are left as an exercise
        # using CS here will give accurate partials, but will miss the sparsity pattern
        self.declare_partials('*', '*', method='cs')

    def apply_nonlinear(self, inputs, outputs, residuals):

        a0, a1, a2, a3, a4, a5 = outputs['A']
        X_cp = inputs['x_cp']
        Y_cp = inputs['y_cp']

        Y_computed = a0 + a1*X_cp + a2*X_cp**2 + a3*X_cp**3 + a4*X_cp**4 + a5*X_cp**5

        # error = np.sum((Y_computed-Y_cp)**2)

        # note that derivatives are showing up in the apply_nonlinear method because
        # this is the formulation we use to form the residual.
        # We are minimizing the sum of the square of the error: np.sum((Y_computed-Y_cp)**2) w.r.t A
        # hence we differentiate the objective w.r.t A and set the resulting system of equations to 0
        d_error__d_Y_computed = 2*(Y_computed-Y_cp)

        d_Y_computed__d_a0 = np.ones(self.options['N_cp'])
        d_Y_computed__d_a1 = X_cp
        d_Y_computed__d_a2 = X_cp**2
        d_Y_computed__d_a3 = X_cp**3
        d_Y_computed__d_a4 = X_cp**4
        d_Y_computed__d_a5 = X_cp**5

        residuals['A'][0] = np.sum(d_error__d_Y_computed * d_Y_computed__d_a0)
        residuals['A'][1] = np.sum(d_error__d_Y_computed * d_Y_computed__d_a1)
        residuals['A'][2] = np.sum(d_error__d_Y_computed * d_Y_computed__d_a2)
        residuals['A'][3] = np.sum(d_error__d_Y_computed * d_Y_computed__d_a3)
        residuals['A'][4] = np.sum(d_error__d_Y_computed * d_Y_computed__d_a4)
        residuals['A'][5] = np.sum(d_error__d_Y_computed * d_Y_computed__d_a5)

        X = inputs['x']
        Y = a0 + a1*X + a2*X**2 + a3*X**3 + a4*X**4 + a5*X**5
        residuals['y'] = Y - outputs['y']
