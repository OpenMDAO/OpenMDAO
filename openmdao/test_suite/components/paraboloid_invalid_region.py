
import openmdao.api as om


class Paraboloid(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

    This version of Paraboloid optionally raises an analysis error when the
    design variables x and y are in an invalid region defined by the specified
    "invalid_x" and "invalid_y" ranges.

    The path of evaluated points to the optmized solution is recorded as
    well as the number of analysis errors raised.

    Parameters
    ----------
    invalid_x : tuple of float or None
        The range of values for x which will trigger an AnalysisError
    invalid_y : tuple of float or None
        The range of values for y which will trigger an AnalysisError
    func : str, 'compute' or 'compute_partials'
        The function that will raise the AnalysisError (compute or compute_partials).

    Attributes
    ----------
    invalid_x : tuple of float or None
        The range of values for x which will trigger an AnalysisError
    invalid_y : tuple of float or None
        The range of values for y which will trigger an AnalysisError
    func : str, 'compute' or 'compute_partials'
        The function that will raise the AnalysisError (compute or compute_partials).
    """

    def __init__(self, invalid_x=None, invalid_y=None, func='compute'):
        super().__init__()
        self.invalid_x = invalid_x
        self.invalid_y = invalid_y
        self.func = func

        self.eval_count = -1
        self.eval_history = []
        self.raised_eval_errors = []

        self.grad_count = -1
        self.grad_history = []
        self.raised_grad_errors = []

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
        """
        self.eval_count += 1

        x = inputs['x']
        y = inputs['y']

        f_xy = outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

        self.eval_history.append((x.item(), y.item(), f_xy.item()))

        if self.invalid_x and self.func == 'compute':
            beg, end =  self.invalid_x
            if x > beg and x < end:
                self.raised_eval_errors.append(self.eval_count)
                raise om.AnalysisError(f'Invalid x: {beg} < {x.item():8.4f} < {end}).')

        if self.invalid_y and self.func == 'compute':
            beg, end =  self.invalid_y
            if y > beg and y < end:
                self.raised_eval_errors.append(self.eval_count)
                raise om.AnalysisError(f'Invalid y: {beg} < {y.item():8.4f} < {end}).')

    def compute_partials(self, inputs, partials):
        """
        Partial derivatives.
        """
        self.grad_count += 1

        x = inputs['x']
        y = inputs['y']

        partials['f_xy', 'x'] = 2.0*x - 6.0 + y
        partials['f_xy', 'y'] = 2.0*y + 8.0 + x

        self.grad_history.append((x.item(), y.item()))

        if self.invalid_x and self.func == 'compute_partials':
            beg, end =  self.invalid_x
            if x > beg and x < end:
                self.raised_grad_errors.append(self.grad_count)
                raise om.AnalysisError(f'Invalid x: {beg} < {x.item():8.4f} < {end}).')

        if self.invalid_y and self.func == 'compute_partials':
            beg, end =  self.invalid_y
            if y > beg and y < end:
                self.raised_grad_errors.append(self.grad_count)
                raise om.AnalysisError(f'Invalid y: {beg} < {y.item():8.4f} < {end}).')
