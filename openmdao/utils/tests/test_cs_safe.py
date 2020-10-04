import numpy as np
import unittest 

from openmdao.utils import cs_safe

from openmdao.utils.assert_utils import assert_near_equal



class TestCSSafeFuctions(unittest.TestCase): 

    def test_abs(self): 

        test_data = np.array([1, -1, -2, 2, 5.675, -5.676], dtype='complex')

        assert_near_equal(cs_safe.abs(test_data), np.abs(test_data))

        test_data += complex(0,1e-50)
        cs_derivs = cs_safe.abs(test_data).imag/1e-50
        expected = [1, -1, -1, 1, 1, -1]

        assert_near_equal(cs_derivs, expected)

    def test_norm(self): 

        test_data = np.array([[1, 2, 3, -4],[5, 6, 7, -8]], dtype='complex')
        
        assert_near_equal(cs_safe.norm(test_data,axis=None), np.linalg.norm(test_data,axis=None))
        assert_near_equal(cs_safe.norm(test_data,axis=0), np.linalg.norm(test_data,axis=0))
        assert_near_equal(cs_safe.norm(test_data,axis=1), np.linalg.norm(test_data,axis=1))

        deriv_test_data = test_data.copy()
        deriv_test_data[0,0] += complex(0, 1e-50)
        
        cs_deriv = cs_safe.norm(deriv_test_data).imag/1e-50

        expected = 1/np.linalg.norm(test_data) * test_data[0,0].real
        assert_near_equal(cs_deriv, expected)


if __name__ == "__main__": 

    unittest.main()