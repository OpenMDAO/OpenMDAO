import unittest
import io
from contextlib import redirect_stderr

from openmdao.utils.om_warnings import warn_deprecation
import openmdao.api as om


class TestOMWarnings(unittest.TestCase):

    def setUp(self):
        """
        Ensure that OpenMDAO warnings are using their default filter action.
        """
        om.reset_warnings()

    def test_warnings_filters(self):
        # OMDeprecationWarning should only generate one warning
        # because it has the 'once' filter

        # first call should generate a warning
        f = io.StringIO()
        with redirect_stderr(f):
            warn_deprecation('msg')
        err = f.getvalue()
        self.assertTrue(len(err) > 0 )

        f.truncate(0)
        f.seek(0)

        # second call should not generate a warning
        with redirect_stderr(f):
            warn_deprecation('msg')
        err = f.getvalue()
        self.assertEqual(len(err), 0 )

    def test_expired_warning(self):
        with self.assertRaises(RuntimeError) as ctx:
            warn_deprecation('msg', expires='0.0.1')
        self.assertEqual(str(ctx.exception), 'Deprecation message expired in version 0.0.1')                             


if __name__ == "__main__":
    unittest.main()
