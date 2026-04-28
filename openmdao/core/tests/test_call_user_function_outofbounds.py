"""
Test that OutOfBoundsError raised inside a user compute() function is reported
clearly rather than being masked by a secondary TypeError.

Background
----------
`System._call_user_function` (openmdao/core/system.py) wraps user-defined
callbacks (compute, apply_nonlinear, etc.) and re-raises any exception as::

    raise err_type(f"{msginfo}: Error calling {fname}(), {err}")

For most exception types this is fine because they accept a single string
argument.  `OutOfBoundsError`, however, requires five positional arguments::

    OutOfBoundsError(message, idx, value, lower, upper)

Attempting to construct it with only a formatted message string raises a
secondary ``TypeError: OutOfBoundsError.__init__() missing 4 required
positional arguments``.  This TypeError then masks the original
OutOfBoundsError and provides no useful diagnostic information to the user.
"""

import unittest

import openmdao.api as om
from openmdao.components.interp_util.outofbounds_error import OutOfBoundsError
from openmdao.utils.assert_utils import assert_warning
from openmdao.utils.testing_utils import use_tempdirs


class RaisesOutOfBoundsComp(om.ExplicitComponent):
    """
    Minimal component that intentionally raises OutOfBoundsError in compute().

    This simulates what an interpolation component does when an input value
    falls outside the tabulated domain.
    """

    def setup(self):
        self.add_input('x', val=0.5)
        self.add_output('y', val=1.0)

    def compute(self, inputs, outputs):
        x = float(inputs['x'].flat[0])
        lower, upper = 0.0, 1.0
        if x < lower or x > upper:
            raise OutOfBoundsError(
                f"Input x={x} is outside the interpolation range [{lower}, {upper}].",
                idx=0,
                value=x,
                lower=lower,
                upper=upper,
            )
        outputs['y'] = x ** 2


@use_tempdirs
class TestCallUserFunctionOutOfBounds(unittest.TestCase):
    """
    Verify that OutOfBoundsError raised in a component compute() surfaces as a
    clear RuntimeError (or OutOfBoundsError) rather than a secondary TypeError.

    In OpenMDAO <= 3.43.0, this was raising the following:

        TypeError: OutOfBoundsError.__init__() missing 4 required positional
        arguments: 'idx', 'value', 'lower', and 'upper'
    """

    def test_outofbounds_not_masked_by_typeerror(self):
        """OutOfBoundsError in compute() must not be hidden by a secondary TypeError."""
        p = om.Problem()
        p.model.add_subsystem('comp', RaisesOutOfBoundsComp())
        p.setup()
        p.set_val('comp.x', 2.0)  # outside [0, 1] — will trigger OutOfBoundsError

        # Before the fix this raises:
        #   TypeError: OutOfBoundsError.__init__() missing 4 required positional arguments
        #
        # After the fix it should raise RuntimeError (or OutOfBoundsError) with
        # a message that mentions both the component and the original error.
        with self.assertRaises((RuntimeError, OutOfBoundsError)) as ctx:
            p.run_model()

        # The error message must contain the component path so the user can
        # identify which component is out of bounds.
        self.assertIn('comp', str(ctx.exception))

        # The error message must also carry the original out-of-bounds description.
        self.assertIn('outside the interpolation range', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
