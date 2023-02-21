import unittest
import warnings
import io
from contextlib import redirect_stderr


class TestOMWarnings(unittest.TestCase):

    def setUp(self):
        """
        Ensure that OpenMDAO warnings are using their default filter action.
        """
        import openmdao.api as om
        om.reset_warnings()

    def test_warnings_filters(self):
        # OMDeprecationWarning should only generate one warning
        # because it has the 'once' filter

        from openmdao.utils.om_warnings import OMDeprecationWarning

        # first call should generate a warning
        f = io.StringIO()
        with redirect_stderr(f):
            warnings.warn('msg', OMDeprecationWarning)
        err = f.getvalue()
        self.assertTrue(len(err) > 0 )

        f.truncate(0)
        f.seek(0)

        # second call should not generate a warning
        with redirect_stderr(f):
            warnings.warn('msg', OMDeprecationWarning)
        err = f.getvalue()
        self.assertEqual(len(err), 0 )


if __name__ == "__main__":
    unittest.main()
