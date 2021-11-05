""" Unit tests for the notebook_utils."""

import unittest
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

try:
    import IPython
except ImportError:
    IPython = False

import openmdao.api as om
from openmdao.utils.assert_utils import assert_warning

class StateOptionsDictionary(om.OptionsDictionary):
    """
    An OptionsDictionary specific to states.

    Parameters
    ----------
    read_only : bool
        If True, setting (via __setitem__ or update) is not permitted.
    """
    def __init__(self, read_only=False):
        super(StateOptionsDictionary, self).__init__(read_only)

        self.declare(name='name', types=str,
                     desc='name of ODE state variable')


class _TestComp(om.ExplicitComponent):
    """
    A simple test component to test show_options_table.
    """

    def __init__(self, required_arg, **kwargs):
        super().__init__(**kwargs)

        assert(required_arg == 1)

    def initialize(self):
        self.options.declare('required_kwarg', types=(int,))

    def setup(self):
        foo = self.options['required_kwarg']
        assert(foo == 2)


@unittest.skipUnless(tabulate and IPython, "Tabulate and IPython are required")
class TestNotebookUtils(unittest.TestCase):

    def test_show_obj_options(self):
        from openmdao.utils import notebook_utils
        notebook_utils.ipy = True

        options = om.show_options_table("openmdao.utils.tests.test_notebook_utils.StateOptionsDictionary")

        self.assertEqual(options, None)

    def test_show_options_w_attr(self):
        from openmdao.utils import notebook_utils
        notebook_utils.ipy = True

        options = om.show_options_table("openmdao.components.balance_comp.BalanceComp")

        self.assertEqual(options, None)

    def test_show_options_with_kwargs(self):
        from openmdao.utils import notebook_utils
        notebook_utils.ipy = True

        options = om.show_options_table('openmdao.utils.tests.test_notebook_utils._TestComp',
                                        init_args=(1,), init_kwargs={'required_kwarg': 2})

        self.assertEqual(options, None)

    def test_show_options_table_warning(self):
        from openmdao.utils import notebook_utils
        notebook_utils.ipy = False

        msg = ("IPython is not installed. Run `pip install openmdao[notebooks]` or `pip install "
               "openmdao[docs]` to upgrade.")

        with assert_warning(UserWarning, msg):
            om.show_options_table("openmdao.components.balance_comp.BalanceComp")
