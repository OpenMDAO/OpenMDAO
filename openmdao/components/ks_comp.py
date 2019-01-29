"""
KS Function Component.
"""
from six.moves import range

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.general_utils import warn_deprecation


CITATIONS = """
@conference {Martins:2005:SOU,
        title = {On Structural Optimization Using Constraint Aggregation},
        booktitle = {Proceedings of the 6th World Congress on Structural and Multidisciplinary
                     Optimization},
        year = {2005},
        month = {May},
        address = {Rio de Janeiro, Brazil},
        author = {Joaquim R. R. A. Martins and Nicholas M. K. Poon}
}
"""


class KSfunction(object):
    """
    Helper class for KSComp.

    Helper class that can be used to aggregate constraint vectors with a
    Kreisselmeier-Steinhauser Function.
    """

    @staticmethod
    def _compute_values(g, rho):
        """
        Compute values needed by the KS function for the given array of constraints.

        Parameters
        ----------
        g : ndarray
            Array of constraint values, where negative means satisfied and positive means violated.
        rho : float
            Constraint Aggregation Factor.

        Returns
        -------
        tuple
            g_max, g_diff, exponents and summation as needed by compute and derivates functions.
        """
        g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
        g_diff = g - g_max
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents, axis=-1)[:, np.newaxis]
        return g_max, g_diff, exponents, summation

    @staticmethod
    def compute(g, rho=50.0):
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
        g_max, g_diff, exponents, summation = KSfunction._compute_values(g, rho)

        KS = g_max + 1.0 / rho * np.log(summation)

        return KS

    @staticmethod
    def derivatives(g, rho=50.0):
        """
        Compute elements of [dKS_gd, dKS_drho] for the given array of constraints.

        Parameters
        ----------
        g : ndarray
            Array of constraint values, where negative means satisfied and positive means violated.
        rho : float
            Constraint Aggregation Factor.

        Returns
        -------
        ndarray
            Derivative of KS function with respect to parameter values.
        """
        g_max, g_diff, exponents, summation = KSfunction._compute_values(g, rho)

        dsum_dg = rho * exponents
        dKS_dsum = 1.0 / (rho * summation)
        dKS_dg = dKS_dsum * dsum_dg

        dsum_drho = np.sum(g_diff * exponents, axis=-1)[:, np.newaxis]
        dKS_drho = dKS_dsum * dsum_drho

        return dKS_dg, dKS_drho


class KSComp(ExplicitComponent):
    """
    KS function component.

    Component that aggregates a number of functions to a single value via the
    Kreisselmeier-Steinhauser Function. This new constraint is satisfied when it
    is less than or equal to zero.

    Attributes
    ----------
    cite : str
        Listing of relevant citations that should be referenced when publishing
        work that uses this class.
    """

    def __init__(self, **kwargs):
        """
        Initialize the KS component.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super(KSComp, self).__init__(**kwargs)

        self.cite = CITATIONS

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('width', types=int, default=1, desc='Width of constraint vector.')
        self.options.declare('vec_size', types=int, default=1,
                             desc='The number of rows to independently aggregate.')
        self.options.declare('lower_flag', False,
                             desc="Set to True to reverse sign of input constraints.")
        self.options.declare('rho', 50.0, desc="Constraint Aggregation Factor.")
        self.options.declare('upper', 0.0, desc="Upper bound for constraint, default is zero.")

    def setup(self):
        """
        Declare inputs, outputs, and derivatives for the KS component.
        """
        opts = self.options
        width = opts['width']
        vec_size = opts['vec_size']

        # Inputs
        self.add_input('g', shape=(vec_size, width),
                       desc="Array of function values to be aggregated")

        # Outputs
        self.add_output('KS', shape=(vec_size, 1), desc="Value of the aggregate KS function")

        rows = np.zeros(width, dtype=np.int)
        cols = range(width)
        rows = np.tile(rows, vec_size) + np.repeat(np.arange(vec_size), width)
        cols = np.tile(cols, vec_size) + np.repeat(np.arange(vec_size), width) * width

        self.declare_partials(of='KS', wrt='g', rows=rows, cols=cols)

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
        opt = self.options

        con_val = inputs['g'] - opt['upper']
        if opt['lower_flag']:
            con_val = -con_val

        outputs['KS'] = KSfunction.compute(con_val, opt['rho'])

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
        opt = self.options
        vec_size = opt['vec_size']
        width = opt['width']

        derivs = np.empty((vec_size, width))

        con_val = inputs['g'] - opt['upper']
        if opt['lower_flag']:
            con_val = -con_val

        derivs = KSfunction.derivatives(con_val, opt['rho'])[0]

        if self.options['lower_flag']:
            derivs = -derivs

        partials['KS', 'g'] = derivs.flatten()


class KSComponent(KSComp):
    """
    Deprecated.
    """

    def __init__(self, *args, **kwargs):
        """
        Capture Initialize to throw warning.

        Parameters
        ----------
        *args : list
            Deprecated arguments.
        **kwargs : dict
            Deprecated arguments.
        """
        warn_deprecation("'KSComponent' has been deprecated. Use "
                         "'KSComp' instead.")
        super(KSComponent, self).__init__(*args, **kwargs)
