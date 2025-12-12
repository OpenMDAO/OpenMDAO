import numpy as np
import jax.numpy as jnp
import openmdao.api as om

class CoupledImplicitComp(om.ImplicitComponent):
    def setup(self):
        n = 3  # array size, adjust as needed
        self.n = n
        self.add_output('x', shape=(n,), val=np.ones(n))  # state 1
        self.add_output('y', shape=(n,), val=np.ones(n))  # state 2
        self.declare_partials('*', '*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        x = outputs['x']
        y = outputs['y']
        n = self.n

        # Compute the coupling term Ay using array operations without assignments
        # Base second difference: y[i-1] - 2y[i] + y[i+1]
        Ay_base = np.roll(y, 1) - 2 * y + np.roll(y, -1)

        # Adjust boundaries in a single expression:
        # - At i=0: -2*y[0] + y[1] (since y[-1] = 0)
        # - At i=n-1: y[n-2] - 2*y[n-1] (since y[n] = 0)
        mask_left = np.arange(n) == 0
        mask_right = np.arange(n) == n-1
        Ay = (Ay_base * (1 - mask_left - mask_right) +  # Interior
              (-2 * y + np.roll(y, -1)) * mask_left +    # Left boundary
              (np.roll(y, 1) - 2 * y) * mask_right)     # Right boundary

        # Compute residuals in one expression
        residuals['x'] = x**2 + y - 4.0 + 0.5 * Ay
        residuals['y'] = x - 2 * y + 1

    def linearize(self, inputs, outputs, partials):
        x = outputs['x']
        y = outputs['y']
        n = self.n

        # Jacobian for R_1 (dR_1/dx and dR_1/dy)
        partials['x', 'x'] = np.diag(2 * x)  # dR_1/dx = 2x (diagonal)
        J_y = np.zeros((n, n))  # dR_1/dy
        for i in range(n):
            if i == 0:
                J_y[i, i] = 1 - 2*0.5  # 1 - 1 = 0 (center)
                J_y[i, i+1] = 0.5      # right
            elif i == n-1:
                J_y[i, i-1] = 0.5      # left
                J_y[i, i] = 1 - 2*0.5  # center
            else:
                J_y[i, i-1] = 0.5      # left
                J_y[i, i] = 1 - 2*0.5  # center
                J_y[i, i+1] = 0.5      # right
        partials['x', 'y'] = J_y

        # Jacobian for R_2 (dR_2/dx and dR_2/dy)
        partials['y', 'x'] = np.eye(n)       # dR_2/dx = 1 (diagonal)
        partials['y', 'y'] = -2 * np.eye(n)  # dR_2/dy = -2 (diagonal)


class JaxCoupledImplicitComp(om.JaxImplicitComponent):
    def setup(self):
        self.n = n = 3  # array size, adjust as needed
        self.add_output('x', shape=(n,), val=np.ones(n))  # state 1
        self.add_output('y', shape=(n,), val=np.ones(n))  # state 2

    def compute_primal(self, x, y):
        n = self.n

        # Compute the coupling term Ay using array operations without assignments
        # Base second difference: y[i-1] - 2y[i] + y[i+1]
        Ay_base = jnp.roll(y, 1) - 2 * y + jnp.roll(y, -1)

        # Adjust boundaries in a single expression:
        # - At i=0: -2*y[0] + y[1] (since y[-1] = 0)
        # - At i=n-1: y[n-2] - 2*y[n-1] (since y[n] = 0)
        mask_left = jnp.arange(n) == 0
        mask_right = jnp.arange(n) == n-1
        Ay = (Ay_base * (1 - mask_left - mask_right) +  # Interior
              (-2 * y + jnp.roll(y, -1)) * mask_left +    # Left boundary
              (jnp.roll(y, 1) - 2 * y) * mask_right)     # Right boundary

        # Compute residuals in one expression
        x = x**2 + y - 4.0 + 0.5 * Ay
        y = x - 2 * y + 1
        return x, y


if __name__ == "__main__":
    import sys
    if 'jax' in sys.argv:
        klass = JaxCoupledImplicitComp
    else:
        klass = CoupledImplicitComp

    # Set up and run the problem
    prob = om.Problem()
    prob.model.add_subsystem('test', klass(), promotes=['*'])
    if 'newton' in sys.argv:
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, iprint=2)
        prob.model.linear_solver = om.DirectSolver()
    else:
        prob.model.nonlinear_solver = om.NonlinearBlockGS()
        prob.model.linear_solver = om.LinearBlockGS()

    prob.setup(force_alloc_complex=True)

    prob['x'] = np.array([1.0, 2.0, 3.0])
    prob['y'] = np.array([1.1, 2.1, 3.1])
    prob.run_model()

    print("x:", prob['x'])
    print("y:", prob['y'])

    # Check the Jacobian
    jac = prob.check_partials(method='cs', compact_print=True, show_only_incorrect=True, includes=['test'])
    # import pprint
    # pprint.pprint(jac)
