import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.func_utils import get_func_info
from openmdao.utils.general_utils import shape2tuple


_shapes = [
    [(3,2), (2,5), (5,4), (3,5), (2,4)],
    [3, 3, 3, (), ()],
]

class FuncAPITestCase(unittest.TestCase):
    def test_default_in_shapes(self):
        # input default values only
        for s1, s2, s3, xshp, yshp in _shapes:
            def func(a=np.zeros(s1), b=np.zeros(s2), c=np.zeros(s3)):
                x = a.dot(b)
                y = b.dot(c)
                return x, y

            ininfo, outinfo = get_func_info(func)
            self.assertEqual(shape2tuple(s1), ininfo['a']['shape'])
            self.assertEqual(shape2tuple(s2), ininfo['b']['shape'])
            self.assertEqual(shape2tuple(s3), ininfo['c']['shape'])
            self.assertEqual(xshp, outinfo['x']['shape'])
            self.assertEqual(yshp, outinfo['y']['shape'])

    def test_default_scalar_in_shapes(self):
        # input default values (some array some scalar)
        def func(a=np.zeros((3,2)), b=5., c=np.zeros((1,4))):
            x = a * b
            y = b * c
            return x, y

        ininfo, outinfo = get_func_info(func)
        self.assertEqual((3,2), ininfo['a']['shape'])
        self.assertEqual((), ininfo['b']['shape'])
        self.assertEqual((1,4), ininfo['c']['shape'])
        self.assertEqual((3,2), outinfo['x']['shape'])
        self.assertEqual((1,4), outinfo['y']['shape'])

    def test_annotation_in_shapes(self):
        # input shape annotations only
        for s1, s2, s3, xshp, yshp in _shapes:
            def func(a:{'shape': s1}, b:{'shape': s2}, c:{'shape': s3}):
                x = a.dot(b)
                y = b.dot(c)
                return x, y

            ininfo, outinfo = get_func_info(func)
            self.assertEqual(s1, ininfo['a']['shape'])
            self.assertEqual(s2, ininfo['b']['shape'])
            self.assertEqual(s3, ininfo['c']['shape'])
            self.assertEqual(xshp, outinfo['x']['shape'])
            self.assertEqual(yshp, outinfo['y']['shape'])

    def test_mixed_annotation_default_in_shapes(self):
        # input mixes shape annotations with default value
        for s1, s2, s3, xshp, yshp in _shapes:
            def func(a:{'shape': s1}, b:{'shape': s2}, c=np.zeros(s3)):
                x = a.dot(b)
                y = b.dot(c)
                return x, y

            ininfo, outinfo = get_func_info(func)
            self.assertEqual(s1, ininfo['a']['shape'])
            self.assertEqual(s2, ininfo['b']['shape'])
            self.assertEqual(shape2tuple(s3), ininfo['c']['shape'])
            self.assertEqual(xshp, outinfo['x']['shape'])
            self.assertEqual(yshp, outinfo['y']['shape'])

    def test_annotation_all_shapes(self):
        # input and output shape annotations
        for s1, s2, s3, xshp, yshp in _shapes:
            def func(a:{'shape': s1}, b:{'shape': s2}, c:{'shape': s3}) -> [('x', {'shape':xshp}), ('y', {'shape':yshp})]:
                return a.dot(b), b.dot(c)

            ininfo, outinfo = get_func_info(func)
            self.assertEqual(s1, ininfo['a']['shape'])
            self.assertEqual(s2, ininfo['b']['shape'])
            self.assertEqual(s3, ininfo['c']['shape'])
            self.assertEqual(xshp, outinfo['x']['shape'])
            self.assertEqual(yshp, outinfo['y']['shape'])

    # def test_annotation_all_shapes(self):
    #     # input and output shape annotations
    #     for s1, s2, s3, xshp, yshp in _shapes:
    #         def func(a:{'shape': s1}, b:{'shape': s2}, c:{'shape': s3}) -> [('x', {'shape':xshp}), ('y', {'shape':yshp})]:
    #             x = a.dot(b)
    #             y = b.dot(c)
    #             return x, y

    #         ininfo, outinfo = get_func_info(func)
    #         self.assertEqual(s1, ininfo['a']['shape'])
    #         self.assertEqual(s2, ininfo['b']['shape'])
    #         self.assertEqual(s3, ininfo['c']['shape'])
    #         self.assertEqual(xshp, outinfo['x']['shape'])
    #         self.assertEqual(yshp, outinfo['y']['shape'])


if __name__ == '__main__':
    unittest.main()
