"""
Support functions for the 'openmdao scaffold' command.
"""

_explicit_template = '''
import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class {class_name}(ExplicitComponent):
    """
    <class description here>

    Attributes
    ----------
    ...
    """

    def __init__(self, **kwargs):
        """
        Intialize this component.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super({class_name}, self).__init__(**kwargs)
        # set attributes here...

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('myopt', default=1, types=int, desc='My option.')

    def setup(self):
        """
        Declare inputs, outputs, and partial derivatives.
        """
        pass

        self.add_input(name='foo', shape=None, units=None)
        self.add_output(name='bar', shape=None, units=None)

        self.declare_partials(of='bar', wrt='foo', rows=None, cols=None, val=None)

    def compute(self, inputs, outputs):
        """
        <Describe compute here>

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        outputs['bar'] = inputs['foo']

    def compute_partials(self, inputs, partials):
        """
        Compute the partial derivatives.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        partials['foo', 'bar'] = 1
'''

_implicit_template = '''
import numpy as np

from openmdao.core.implicitcomponent import ImplicitComponent


class {class_name}(ImplicitComponent):
    """
    <class description here>

    Attributes
    ----------
    ...
    """

    def __init__(self, **kwargs):
        """
        Intialize this component.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super({class_name}, self).__init__(**kwargs)
        # set attributes here...

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('myopt', default=1, types=int, desc='My option.')

    def setup(self):
        """
        Add inputs and outputs and declare partials.
        """
        self.add_input(name='foo', shape=None, units=None)
        self.add_output(name='bar', shape=None, units=None)

        self.declare_partials(of='bar', wrt='foo', rows=None, cols=None, val=None)

        # Set up the derivatives.
        self.declare_partials(of='bar', wrt='foo', rows=None, cols=None, val=None)

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        <describe apply_nonlinear here>

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        residuals['bar'] = 1

    def solve_nonlinear(self, inputs, outputs):
        """
        <describe solve_nonlinear here>

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        outputs['bar'] = inputs['foo']

    def linearize(self, inputs, outputs, partials):
        """
        Compute the non-constant partial derivatives.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        partials : partial Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        partials['bar', 'foo'] = inputs['foo']

    def solve_linear(self, d_outputs, d_residuals, mode):
        """
        <describe solve_linear here>

        Parameters
        ----------
        d_outputs : Vector
            unscaled, dimensional quantities read via d_outputs[key]
        d_residuals : Vector
            unscaled, dimensional quantities read via d_residuals[key]
        mode : str
            either 'fwd' or 'rev'
        """
        if mode == 'fwd':
            d_outputs['bar'] = d_residuals['bar']

        else:  # rev
            d_residuals['bar'] = d_outputs['bar']
'''



def _scaffold_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao scaffold' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='output file.')
    parser.add_argument('-c', '--class', action='store', dest='class_name', default='MyComp',
                        help='Name of the component class.')
    parser.add_argument('-e', '--explicit', action='store_true', dest='explicit',
                        help="Generate an ExplicitComponent.")
    parser.add_argument('-i', '--implicit', action='store_true', dest='implicit',
                        help="Generate an ImplicitComponent.")


def _scaffold_exec(options):
    """
    Execute the 'openmdao scaffold' command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    """
    outfile = options.file[0]
    if options.explicit and options.implicit:
        raise RuntimeError("Component cannot be both implicit and explicit.")

    if options.explicit:
        template = _explicit_template
    elif options.implicit:
        template = _implicit_template
    else:
        raise RuntimeError("Component must be either implicit or explicit.")

    with open(outfile, 'w') as f:
        f.write(template.format(class_name=options.class_name))
