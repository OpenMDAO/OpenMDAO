"""
Smooth approximations to functions that do not have continuous derivatives.
"""

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

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


def act_tanh(x, mu=1.0E-2, z=0., a=-1., b=1.):
    """
    Differentiable activation function based on the hyperbolic tangent.

    act_tanh can be used to approximate a step function from `a` to `b`, occurring at x=z.
    Smaller values of parameter `mu` more accurately represent a step function but the "sharpness" of the corners in the
    response may be more difficult for gradient-based approaches to resolve.

    Parameters
    ----------
    x : float or jnp.array
        The input at which the value of the activation function
        is to be computed.
    mu : float
        A shaping parameter which impacts the "abruptness" of
        the activation function. As this value approaches zero
        the response approaches that of a step function.
    z : float
        The value of the independent variable about which the
        activation response is centered.
    a : float
        The initial value that the input asymptotically approaches
        as x approaches negative infinity.
    b : float
        The final value that the input asymptotically approaches
        as x approaches positive infinity.

    Returns
    -------
    float or jnp.array
        The value of the activation response at the given input.
    """
    dy = b - a
    tanh_term = jnp.tanh((x - z) / mu)
    return 0.5 * dy * (1 + tanh_term) + a


@jax.jit
def smooth_max(x, y, mu=1.0E-2):
    """
    Differentiable maximum between two arrays of the same shape.

    Parameters
    ----------
    x : float or jnp.array
        The first value or array of values for comparison.
    y : float or jnp.array
        The second value or array of values for comparison.
    mu : float
        A shaping parameter which impacts the "abruptness" of the activation function.
        As this value approaches zero the response approaches that of a step function.

    Returns
    -------
    float or jnp.array
        For each element in x or y, the greater of the values of x or y at that point.
        This function is smoothed, so near the point where x and y have equal values
        this will be approximate. The accuracy of this approximation can be adjusted
        by changing the mu parameter. Smaller values of mu will lead to more accuracy
        at the expense of the smoothness of the approximation.
    """
    x_greater = act_tanh(x=x, mu=mu, z=y, a=0, b=1)
    y_greater = 1 - x_greater
    return x_greater * x + y_greater * y


@jax.jit
def smooth_min(x, y, mu=1.0E-2):
    """
    Differentiable minimum between two arrays of the same shape.

    Parameters
    ----------
    x : float or jnp.array
        The first value or array of values for comparison.
    y : float or jnp.array
        The second value or array of values for comparison.
    mu : float
        A shaping parameter which impacts the "abruptness" of the activation function.
        As this value approaches zero the response approaches that of a step function.

    Returns
    -------
    float or jnp.array
        For each element in x or y, the greater of the values of x or y at that point. This
        function is smoothed, so near the point where x and y have equal values this will
        be approximate. The accuracy of this approximation can be adjusted by changing the
        mu parameter. Smaller values of mu will lead to more accuracy at the expense of the
        smoothness of the approximation.
    """
    x_greater = act_tanh(x=x, mu=mu, z=y, a=0, b=1)
    y_greater = 1 - x_greater
    return x_greater * y + y_greater * x


@jax.jit
def smooth_abs(x, mu=1.0E-2):
    """
    Differentiable approximation to the absolute value function.

    Parameters
    ----------
    x : float or jnp.array
        The argument to absolute value.
    mu : float
        A shaping parameter which impacts the tradeoff between the
        smoothness and accuracy of the function. As this value
        approaches zero the response approaches that of the true
        absolute value.

    Returns
    -------
    float or jnp.array
        An approximation of the absolute value. Near zero, the value will
        differ from the true absolute value but its derivative will be continuous.
    """
    act = act_tanh(x=x, mu=mu, z=0, a=-1, b=1)
    return x * act


@jax.jit
def ks_max(x, rho=100.0):
    """
    Given some array of values `x`, compute a _conservative_ maximum using the
    Kreisselmeier-Steinhauser function.

    Parameters
    ----------
    g : ndarray
        Array of values.
    rho : float
        Aggregation Factor. Larger values of rho more closely match the true maximum value.

    Returns
    -------
    float
        A conservative approximation to the minimum value in x.

    """
    x_max = jnp.max(x)
    x_diff = x - x_max
    exponents = jnp.exp(rho * x_diff)
    summation = jnp.sum(exponents)
    return x_max + 1.0 / rho * jnp.log(summation)


@jax.jit
def ks_min(x, rho=100.0):
    """
    Given some array of values `x`, compute a _conservative_ minimum using the
    Kreisselmeier-Steinhauser function.

    Parameters
    ----------
    g : ndarray
        Array of values.
    rho : float
        Aggregation Factor. Larger values of rho more closely match the true minimum value.

    Returns
    -------
    float
        A conservative approximation to the minimum value in x.
    """
    x_min = jnp.min(x)
    x_diff = x_min - x
    exponents = jnp.exp(rho * x_diff)
    summation = jnp.sum(exponents)
    return x_min - 1.0 / rho * jnp.log(summation)


if __name__ == '__main__':

    def act_tanh_numpy(x, mu=1.0E-2, z=0., a=-1., b=1.):
        """
        Differentiable activation function based on the hyperbolic tangent.
        Parameters
        ----------
        x : float or np.array
            The input at which the value of the activation function
            is to be computed.
        mu : float
            A shaping parameter which impacts the "abruptness" of
            the activation function. As this value approaches zero
            the response approaches that of a step function.
        z : float
            The value of the independent variable about which the
            activation response is centered.
        a : float
            The initial value that the input asymptotically approaches
            as x approaches negative infinity.
        b : float
            The final value that the input asymptotically approaches
            as x approaches positive infinity.
        Returns
        -------
        float or np.array
            The value of the activation response at the given input.
        """
        dy = b - a
        tanh_term = np.tanh((x - z) / mu)
        return 0.5 * dy * (1 + tanh_term) + a


    def d_act_tanh_numpy(x, mu=1.0E-2, z=0.0, a=-1.0, b=1.0,
                   dx=True, dmu=True, dz=True, da=True, db=True):
        """
        Differentiable activation function based on the hyperbolic tangent.
        Parameters
        ----------
        x : float or np.array
            The input at which the value of the activation function is to be computed.
        mu : float
            A shaping parameter which impacts the "abruptness" of the activation function.
            As this value approaches zero the response approaches that of a step function.
        z : float or np.array
            The value of the independent variable about which the activation response is centered.
        a : float
            The initial value that the input asymptotically approaches negative infinity.
        b : float
            The final value that the input asymptotically approaches positive infinity.
        dx : bool
            True if the Compute the derivative of act_tanh wrt x should be calculated. Setting this to
            False can save time when the derivative is not needed.
        dmu : bool
            True if the Compute the derivative of act_tanh wrt mu should be calculated. Setting this
            to False can save time when the derivative is not needed.
        dz : bool
            True if the Compute the derivative of act_tanh wrt z should be calculated. Setting this
            to False can save time when the derivative is not needed.
        da : bool
            True if the Compute the derivative of act_tanh wrt a should be calculated. Setting this
            to False can save time when the derivative is not needed.
        db : bool
            True if the Compute the derivative of act_tanh wrt b should be calculated. Setting this
            to False can save time when the derivative is not needed.
        Returns
        -------
        d_dx : float or np.ndarray or None
            Derivatives of act_tanh wrt x or None if argument dx is False.
        d_dmu : float or np.ndarray or None
            Derivatives of act_tanh wrt mu or None if argument dmu is False.
        d_dz : float or np.ndarray or None
            Derivatives of act_tanh wrt z or None if argument dz is False.
        d_da : float or np.ndarray or None
            Derivatives of act_tanh wrt a or None if argument da is False.
        d_db : float or np.ndarray or None
            Derivatives of act_tanh wrt b or None if argument db is False.
        """
        dy = b - a
        dy_d_2 = 0.5 * dy
        xmz = x - z
        xmz_d_mu = xmz / mu
        tanh_term = np.tanh(xmz_d_mu)

        # Avoid overflow warnings from cosh
        oo_mu_cosh2 = np.zeros_like(xmz_d_mu)
        idxs_small = np.where(np.abs(xmz_d_mu) < 20)
        if idxs_small:
            cosh2 = np.cosh(xmz_d_mu[idxs_small]) ** 2
            oo_mu_cosh2[idxs_small] = 1. / (mu * cosh2)

        return ((dy_d_2 * oo_mu_cosh2).ravel() if dx else None,  # d_dx
                -(dy_d_2 * xmz_d_mu) * oo_mu_cosh2 if dmu else None,  # d_dmu
                (-dy_d_2 * oo_mu_cosh2).ravel() if dz else None,  # d_dz
                0.5 * (1 - tanh_term) if da else None,  # d_da
                0.5 * (1 + tanh_term) if db else None)  # d_db


    from functools import partial
    import numpy as np
    import jax
    import jax.numpy as jnp
    import openmdao.api as om


    class ActTanhComp(om.ExplicitComponent):

        def initialize(self):
            self.options.declare('vec_size', types=int)

        def setup(self):
            N = self.options['vec_size']
            self.add_input('x', shape=(N,))
            self.add_input('z', shape=(1,))
            self.add_output('h', shape=(N,))

            ar = np.arange(N, dtype=int)

            self.declare_partials(of='h', wrt='x', rows=ar, cols=ar)
            self.declare_partials(of='h', wrt='z')

        @partial(jax.jit, static_argnums=(0,))
        def _compute_primal(self, x, z):
            """
            This is where the jax implementation belongs.
            """
            return act_tanh(x, 0.01, z, 0.0, 1.0)

        @partial(jax.jit, static_argnums=(0,))
        def _compute_partials_jacfwd(self, x, z):
            deriv_func = jax.jacfwd(self._compute_primal, argnums=[0, 1])
            dx, dz = deriv_func(x, z)
            return jnp.diagonal(dx), dz

        @partial(jax.jit, static_argnums=(0,))
        def _compute_partials_jacrev(self, x, z):
            deriv_func = jax.jacrev(self._compute_primal, argnums=[0, 1])
            dx, dz = deriv_func(x, z)
            return jnp.diagonal(dx), dz

        @partial(jax.jit, static_argnums=(0,))
        def _compute_partials_jvp(self, x, z):
            dx = jax.jvp(self._compute_primal,
                         primals=(x, z),
                         tangents=(jnp.ones_like(x), jnp.zeros_like(z)))[1]

            dz = jax.jvp(self._compute_primal,
                         primals=(x, z),
                         tangents=(jnp.zeros_like(x), jnp.ones_like(z)))[1]

            return dx, dz

        def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
            outputs['h'] = self._compute_primal(*inputs.values())

        def compute_partials(self, inputs, partials, discrete_inputs=None):
            # dx, dz = self._compute_partials_jvp(*inputs.values())

            x, z = inputs.values()
            dx, dmu, dz, da, db = d_act_tanh_numpy(x, 0.01, z, 0.0, 1.0, dx=True, dmu=False, dz=True, da=False, db=False)

            partials['h', 'x'] = dx
            partials['h', 'z'] = dz

    # for N in [10, 20, 50, 100, 150, 200, 300, 400, 500]:
    #     p = om.Problem()
    #     h_comp = p.model.add_subsystem('h_comp', ActTanhComp(vec_size=N))
    #     p.setup(force_alloc_complex=True)
    #
    #     p.set_val('h_comp.x', np.linspace(0, 1, h_comp.options['vec_size']))
    #     p.set_val('h_comp.z', 0.5)
    #
    #     p.run_model()
    #
    #     # call compute totals once to do the jit compilation before we run timings
    #     p.compute_totals(of=['h_comp.h'], wrt=['h_comp.x', 'h_comp.z'])
    #
    #     import timeit
    #
    #     t = timeit.timeit("p.compute_totals(of=['h_comp.h'], wrt=['h_comp.x', 'h_comp.z'])",
    #                       globals=globals(),
    #                       number=100)
    #     print(N, t/100)


    jacrev_results = [
    [10, 0.00042014292033854873],
    [20, 0.0006729008402908221],
    [50, 0.0015268770803231746],
    [100, 0.003058862089528702],
    [150, 0.004660837919800542],
    [200, 0.0064501974999438974],
    [300, 0.010076659999904224],
    [400, 0.014131232920335605],
    [500, 0.01854348958004266]]

    jacfwd_results=[
    [10, 0.00042297500011045485],
    [20, 0.0006978666596114635],
    [50, 0.0015738654200686143],
    [100, 0.0031675220897886902],
    [150, 0.00487279833003413],
    [200, 0.006663855410297401],
    [300, 0.010265454589971341],
    [400, 0.014313820829847827],
    [500, 0.018807842910173348]]

    jvp_results=[
    [10, 0.0004213391698431224],
    [20, 0.0006893787498120219],
    [50, 0.0015704424999421463],
    [100, 0.0030842158402083443],
    [150, 0.004731737080146558],
    [200, 0.006383888749987818],
    [300, 0.010029160829726607],
    [400, 0.014004908750066533],
    [500, 0.018421835839981214]]

    analytic_results= [
    [10, 0.00040093291027005763],
    [20, 0.0006627712497720494],
    [50, 0.0015353383298497647],
    [100, 0.0030298387497896327],
    [150, 0.004578538749483414],
    [200, 0.006163107499596663],
    [300, 0.00986850583984051],
    [400, 0.014005204580025748],
    [500, 0.018298543749842792]]

    import matplotlib.pyplot as plt

    anr = np.asarray(analytic_results)
    fwdr = np.asarray(jacfwd_results)
    revr = np.asarray(jacrev_results)
    jvpr = np.asarray(jvp_results)

    plt.plot(anr[:, 0], anr[:, 1] / anr[:, 1], label='analytic')
    plt.plot(anr[:, 0], fwdr[:, 1] / anr[:, 1], label='jacfwd')
    plt.plot(anr[:, 0], revr[:, 1] / anr[:, 1], label='jacrev')
    plt.plot(anr[:, 0], jvpr[:, 1] / anr[:, 1], label='jvp')

    plt.legend()
    plt.grid()
    
    plt.xlabel('Vector size of act_tanh')
    plt.ylabel('compute_totals time vs. analytic (s)')

    plt.show()