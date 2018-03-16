"""
KS Function Component.
"""

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class KSfunction(object):
    """
    Helper class for KS.

    Helper class that can be used inside other components to aggregate constraint vectors with a
    Kreisselmeier-Steinhauser Function.
    """

    def compute(self, g, rho=50.0):
        """
        Compute the value of the KS function for the given array of constraints.

        Parameters
        ----------
        g : ndarray
            Array of constraint values, where negative means satisfied and positive means violated.
        rho : float
            Constraint Aggregation Factor.

        Returns
        -------
        float
            Value of KS function.
        """
        self.rho = rho
        g_max = np.max(g)
        self.g_diff = g - g_max
        self.exponents = np.exp(rho * self.g_diff)
        self.summation = np.sum(self.exponents)
        KS = g_max + 1.0 / rho * np.log(self.summation)

        return KS

    def derivatives(self):
        """
        Compute elements of [dKS_gd, dKS_drho] based on previously computed values.

        Returns
        -------
        ndarray
            Derivative of KS function with respect to parameter values.
        """
        dsum_dg = self.rho * self.exponents
        dKS_dsum = 1.0 / (self.rho * self.summation)
        dKS_dg = dKS_dsum * dsum_dg

        dsum_drho = np.sum(self.g_diff * self.exponents)
        dKS_drho = dKS_dsum * dsum_drho

        return dKS_dg, dKS_drho


class KSComponent(ExplicitComponent):
    """
    KS function component.

    Component that aggregates a number of functions to a single value via the
    Kreisselmeier-Steinhauser Function.
    """

    def __init__(self, width=1):
        """
        Initialize the KS component.

        Parameters
        ----------
        width : dict of keyword arguments
            'Width of constraint vector.
        """
        super(KSComponent, self).__init__(width=width)

        self.options.declare('rho', 50.0, desc="Constraint Aggregation Factor.")

    def initialize(self):
        """
        Declare metadata.
        """
        self.metadata.declare('width', types=int, default=1, desc='Width of constraint vector.')

    def setup(self):
        """
        Declare inputs, outputs, and derivatives for the KS component.
        """
        width = self.metadata['width']

        # Inputs
        self.add_input('g', shape=(width, ), desc="Array of function values to be aggregated")

        # Outputs
        self.add_output('KS', 0.0, desc="Value of the aggregate KS function")

        self.declare_partials(of='KS', wrt='g')
        self._ks = KSfunction()

    def compute(self, inputs, outputs):
        """
        Compute the output of the KS function.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.

        outputs : `Vector`
            `Vector` containing outputs.
        """
        outputs['KS'] = self._ks.compute(inputs['g'], self.options['rho'])

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        partials['KS', 'g'] = self._ks.derivatives()[0]
