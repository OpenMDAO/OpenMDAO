import unittest

from openmdao.core.system import _abs_from_tol_violation, _ErrorData


class TestAbsFromTolViolation(unittest.TestCase):

    def test_empty_lists(self):
        # Test that empty inputs return an empty list.
        result = _abs_from_tol_violation([], [], atol=5, rtol=0.1)
        self.assertEqual(result, [])

    def test_all_attributes_set(self):
        # Test when all attributes are set.
        # For each attribute, the function should compute: tv + atol + rtol * (tv_val second element)
        tv = _ErrorData(forward=100, reverse=200, fwd_rev=300)
        tv_val = _ErrorData(forward=(None, 10), reverse=(None, 20), fwd_rev=(None, 30))
        result = _abs_from_tol_violation([tv], [tv_val], atol=5, rtol=0.1)
        # Expected: forward = 100 + 5 + 0.1*10 = 106, reverse = 200 + 5 + 0.1*20 = 207, fwd_rev = 300 + 5 + 0.1*30 = 308
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].forward, 106)
        self.assertAlmostEqual(result[0].reverse, 207)
        self.assertAlmostEqual(result[0].fwd_rev, 308)

    def test_partial_attributes_none(self):
        # Test when some attributes are None. Only compute for attributes that are not None.
        tv = _ErrorData(forward=None, reverse=150, fwd_rev=None)
        tv_val = _ErrorData(forward=(None, 10), reverse=(None, 50), fwd_rev=(None, 30))
        result = _abs_from_tol_violation([tv], [tv_val], atol=2, rtol=0.05)
        # Expected: only reverse computed: 150 + 2 + 0.05*50 = 150 + 2 + 2.5 = 154.5
        self.assertEqual(len(result), 1)
        self.assertFalse(hasattr(result[0], 'forward') and result[0].forward is not None, "Forward should not be set")
        self.assertFalse(hasattr(result[0], 'fwd_rev') and result[0].fwd_rev is not None, "fwd_rev should not be set")
        self.assertAlmostEqual(result[0].reverse, 154.5)

    def test_multiple_entries(self):
        # Test with multiple tv and tv_val entries.
        tv1 = _ErrorData(forward=10, reverse=None, fwd_rev=30)
        tv_val1 = _ErrorData(forward=(None, 5), reverse=(None, 0), fwd_rev=(None, 15))

        tv2 = _ErrorData(forward=None, reverse=20, fwd_rev=None)
        tv_val2 = _ErrorData(forward=(None, 0), reverse=(None, 10), fwd_rev=(None, 0))

        result = _abs_from_tol_violation([tv1, tv2], [tv_val1, tv_val2], atol=3, rtol=0.2)

        # For tv1: forward = 10 + 3 + 0.2*5 = 10 + 3 + 1 = 14, fwd_rev = 30 + 3 + 0.2*15 = 30 + 3 + 3 = 36
        # tv1.reverse is None so it won't be set
        # For tv2: reverse = 20 + 3 + 0.2*10 = 20 + 3 + 2 = 25
        self.assertEqual(len(result), 2)

        # Check first entry
        self.assertAlmostEqual(result[0].forward, 14)
        self.assertAlmostEqual(result[0].fwd_rev, 36)
        self.assertFalse(hasattr(result[0], 'reverse') and result[0].reverse is not None, "Reverse should not be set for first entry")

        # Check second entry
        self.assertAlmostEqual(result[1].reverse, 25)
        self.assertFalse(hasattr(result[1], 'forward') and result[1].forward is not None, "Forward should not be set for second entry")
        self.assertFalse(hasattr(result[1], 'fwd_rev') and result[1].fwd_rev is not None, "fwd_rev should not be set for second entry")


if __name__ == '__main__':
    unittest.main()