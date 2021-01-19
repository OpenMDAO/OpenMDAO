"""Define the test group classes."""
from openmdao.core.group import Group


class ParametericTestGroup(Group):
    """
    Test Group expected by `ParametricInstance`. Groups inheriting from this should extend
    `default_params` to include valid parametric options for that model.

    Attributes
    ----------
    expected_totals : dict or None
        Dictionary mapping (out, in) pairs to the associated total derivative. Optional
    total_of : iterable
        Iterable containing which outputs to take the derivative of.
    total_wrt : iterable
        Iterable containing which variables with which to take the derivative of the above.
    expected_values : dict or None
        Dictionary mapping variable names to expected values. Optional.
    default_params : dict
        Dictionary containing the available options and default values for parametric sweeps.
    """
    def __init__(self, **kwargs):

        self.expected_totals = None
        self.total_of = None
        self.total_wrt = None
        self.expected_values = None
        self.default_params = {
            'local_vector_class': ['default', 'petsc'],
            'assembled_jac': [True, False],
            'jacobian_type': ['matvec', 'dense', 'sparse-csc'],
        }

        super().__init__()

        self.options.declare('local_vector_class', default='default',
                             values=['default', 'petsc'],
                             desc='Which local vector implementation to use.')
        self.options.declare('assembled_jac', default=True,
                             types=bool,
                             desc='If an assemebled Jacobian should be used.')
        self.options.declare('jacobian_type', default='matvec',
                             values=['dense', 'matvec', 'sparse-csc'],
                             desc='Controls the type of the assembled jacobian.')

        self.options.update(kwargs)
