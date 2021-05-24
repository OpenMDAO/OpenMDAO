from openmdao.test_suite.components.paraboloid import Paraboloid


class ParaboloidMatVec(Paraboloid):
    """ Use matrix-vector product."""

    def setup_partials(self):
        pass

    def compute_partials(self, inputs, partials):
        """Analytical derivatives."""
        pass

    def compute_jacvec_product(self, inputs, dinputs, doutputs, mode):
        """Returns the product of the incoming vector with the Jacobian."""

        x = inputs['x'][0]
        y = inputs['y'][0]

        if mode == 'fwd':
            if 'x' in dinputs:
                doutputs['f_xy'] += (2.0*x - 6.0 + y)*dinputs['x']
            if 'y' in dinputs:
                doutputs['f_xy'] += (2.0*y + 8.0 + x)*dinputs['y']

        elif mode == 'rev':
            if 'x' in dinputs:
                dinputs['x'] += (2.0*x - 6.0 + y)*doutputs['f_xy']
            if 'y' in dinputs:
                dinputs['y'] += (2.0*y + 8.0 + x)*doutputs['f_xy']
