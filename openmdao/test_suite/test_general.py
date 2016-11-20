
from __future__ import division, print_function

import unittest

from openmdao.test_suite.update_test_general import CompTestCaseBase
from openmdao.test_suite.components.implicit_components     import TestImplCompNondLinear
from openmdao.test_suite.components.explicit_components     import TestExplCompNondLinear
from openmdao.api import DefaultVector, NewtonSolver, ScipyIterativeSolver
from openmdao.parallel_api import PETScVector

class CompTestCase(CompTestCaseBase):

    def test_0(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (1,))


    def test_1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (2,))


    def test_2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_3(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_4(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (1,))


    def test_5(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (2,))


    def test_6(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_7(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_8(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (1,))


    def test_9(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (2,))


    def test_10(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_11(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_12(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (1,))


    def test_13(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (2,))


    def test_14(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_15(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_16(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_17(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_18(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_19(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_20(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_21(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_22(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_23(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_24(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_25(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_26(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_27(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_28(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_29(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_30(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_31(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_32(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (1,))


    def test_33(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (2,))


    def test_34(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_35(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_36(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (1,))


    def test_37(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (2,))


    def test_38(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_39(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_40(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (1,))


    def test_41(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (2,))


    def test_42(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_43(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_44(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (1,))


    def test_45(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (2,))


    def test_46(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_47(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_48(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (1,))


    def test_49(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (2,))


    def test_50(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (2, 1))


    def test_51(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (1, 2))


    def test_52(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (1,))


    def test_53(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (2,))


    def test_54(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (2, 1))


    def test_55(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (1, 2))


    def test_56(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (1,))


    def test_57(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (2,))


    def test_58(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (2, 1))


    def test_59(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (1, 2))


    def test_60(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (1,))


    def test_61(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (2,))


    def test_62(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (2, 1))


    def test_63(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (1, 2))


    def test_64(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (1,))


    def test_65(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (2,))


    def test_66(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_67(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_68(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (1,))


    def test_69(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (2,))


    def test_70(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_71(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_72(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (1,))


    def test_73(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (2,))


    def test_74(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_75(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_76(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (1,))


    def test_77(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (2,))


    def test_78(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_79(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_80(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (1,))


    def test_81(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (2,))


    def test_82(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_83(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_84(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (1,))


    def test_85(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (2,))


    def test_86(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_87(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_88(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (1,))


    def test_89(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (2,))


    def test_90(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_91(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_92(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (1,))


    def test_93(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (2,))


    def test_94(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_95(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_96(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_97(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_98(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_99(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_100(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_101(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_102(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_103(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_104(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_105(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_106(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_107(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_108(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_109(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_110(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_111(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_112(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_113(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_114(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_115(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_116(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_117(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_118(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_119(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_120(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_121(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_122(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_123(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_124(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_125(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_126(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_127(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_128(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_129(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_130(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_131(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_132(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_133(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_134(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_135(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_136(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_137(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_138(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_139(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_140(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_141(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_142(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_143(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


    def test_144(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (1,))


    def test_145(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (2,))


    def test_146(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_147(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_148(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (1,))


    def test_149(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (2,))


    def test_150(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_151(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_152(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (1,))


    def test_153(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (2,))


    def test_154(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_155(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_156(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (1,))


    def test_157(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (2,))


    def test_158(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_159(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_160(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_161(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_162(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_163(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_164(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_165(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_166(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_167(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_168(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_169(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_170(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_171(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_172(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_173(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_174(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_175(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_176(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (1,))


    def test_177(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (2,))


    def test_178(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_179(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_180(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (1,))


    def test_181(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (2,))


    def test_182(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_183(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_184(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (1,))


    def test_185(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (2,))


    def test_186(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_187(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_188(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (1,))


    def test_189(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (2,))


    def test_190(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_191(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_192(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (1,))


    def test_193(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (2,))


    def test_194(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (2, 1))


    def test_195(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (1, 2))


    def test_196(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (1,))


    def test_197(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (2,))


    def test_198(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (2, 1))


    def test_199(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (1, 2))


    def test_200(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (1,))


    def test_201(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (2,))


    def test_202(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (2, 1))


    def test_203(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (1, 2))


    def test_204(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (1,))


    def test_205(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (2,))


    def test_206(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (2, 1))


    def test_207(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (1, 2))


    def test_208(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (1,))


    def test_209(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (2,))


    def test_210(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_211(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_212(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (1,))


    def test_213(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (2,))


    def test_214(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_215(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_216(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (1,))


    def test_217(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (2,))


    def test_218(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_219(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_220(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (1,))


    def test_221(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (2,))


    def test_222(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_223(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_224(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (1,))


    def test_225(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (2,))


    def test_226(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_227(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_228(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (1,))


    def test_229(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (2,))


    def test_230(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_231(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_232(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (1,))


    def test_233(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (2,))


    def test_234(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_235(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_236(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (1,))


    def test_237(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (2,))


    def test_238(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_239(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_240(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_241(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_242(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_243(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_244(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_245(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_246(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_247(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_248(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_249(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_250(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_251(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_252(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_253(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_254(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_255(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_256(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_257(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_258(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_259(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_260(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_261(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_262(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_263(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_264(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_265(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_266(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_267(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_268(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_269(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_270(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_271(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_272(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_273(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_274(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_275(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_276(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_277(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_278(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_279(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_280(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_281(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_282(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_283(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_284(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_285(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_286(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_287(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


    def test_288(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (1,))


    def test_289(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (2,))


    def test_290(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_291(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_292(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (1,))


    def test_293(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (2,))


    def test_294(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_295(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_296(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (1,))


    def test_297(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (2,))


    def test_298(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_299(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_300(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (1,))


    def test_301(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (2,))


    def test_302(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_303(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_304(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_305(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_306(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_307(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_308(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_309(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_310(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_311(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_312(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_313(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_314(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_315(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_316(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_317(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_318(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_319(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_320(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (1,))


    def test_321(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (2,))


    def test_322(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_323(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_324(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (1,))


    def test_325(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (2,))


    def test_326(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_327(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_328(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (1,))


    def test_329(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (2,))


    def test_330(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_331(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_332(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (1,))


    def test_333(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (2,))


    def test_334(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_335(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_336(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (1,))


    def test_337(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (2,))


    def test_338(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (2, 1))


    def test_339(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (1, 2))


    def test_340(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (1,))


    def test_341(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (2,))


    def test_342(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (2, 1))


    def test_343(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (1, 2))


    def test_344(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (1,))


    def test_345(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (2,))


    def test_346(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (2, 1))


    def test_347(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (1, 2))


    def test_348(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (1,))


    def test_349(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (2,))


    def test_350(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (2, 1))


    def test_351(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (1, 2))


    def test_352(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (1,))


    def test_353(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (2,))


    def test_354(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_355(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_356(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (1,))


    def test_357(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (2,))


    def test_358(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_359(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_360(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (1,))


    def test_361(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (2,))


    def test_362(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_363(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_364(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (1,))


    def test_365(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (2,))


    def test_366(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_367(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_368(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (1,))


    def test_369(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (2,))


    def test_370(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_371(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_372(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (1,))


    def test_373(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (2,))


    def test_374(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_375(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_376(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (1,))


    def test_377(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (2,))


    def test_378(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_379(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_380(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (1,))


    def test_381(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (2,))


    def test_382(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_383(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_384(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_385(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_386(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_387(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_388(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_389(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_390(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_391(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_392(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_393(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_394(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_395(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_396(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_397(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_398(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_399(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_400(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_401(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_402(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_403(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_404(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_405(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_406(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_407(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_408(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_409(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_410(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_411(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_412(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_413(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_414(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_415(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_416(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_417(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_418(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_419(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_420(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_421(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_422(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_423(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_424(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_425(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_426(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_427(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_428(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_429(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_430(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_431(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


    def test_432(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (1,))


    def test_433(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (2,))


    def test_434(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_435(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_436(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (1,))


    def test_437(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (2,))


    def test_438(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_439(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_440(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (1,))


    def test_441(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (2,))


    def test_442(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_443(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_444(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (1,))


    def test_445(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (2,))


    def test_446(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_447(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_448(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_449(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_450(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_451(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_452(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_453(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_454(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_455(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_456(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_457(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_458(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_459(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_460(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_461(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_462(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_463(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_464(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (1,))


    def test_465(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (2,))


    def test_466(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_467(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_468(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (1,))


    def test_469(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (2,))


    def test_470(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_471(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_472(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (1,))


    def test_473(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (2,))


    def test_474(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_475(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_476(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (1,))


    def test_477(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (2,))


    def test_478(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_479(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_480(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (1,))


    def test_481(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (2,))


    def test_482(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (2, 1))


    def test_483(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (1, 2))


    def test_484(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (1,))


    def test_485(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (2,))


    def test_486(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (2, 1))


    def test_487(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (1, 2))


    def test_488(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (1,))


    def test_489(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (2,))


    def test_490(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (2, 1))


    def test_491(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (1, 2))


    def test_492(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (1,))


    def test_493(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (2,))


    def test_494(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (2, 1))


    def test_495(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (1, 2))


    def test_496(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (1,))


    def test_497(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (2,))


    def test_498(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_499(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_500(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (1,))


    def test_501(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (2,))


    def test_502(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_503(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_504(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (1,))


    def test_505(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (2,))


    def test_506(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_507(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_508(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (1,))


    def test_509(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (2,))


    def test_510(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_511(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_512(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (1,))


    def test_513(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (2,))


    def test_514(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_515(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_516(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (1,))


    def test_517(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (2,))


    def test_518(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_519(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_520(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (1,))


    def test_521(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (2,))


    def test_522(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_523(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_524(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (1,))


    def test_525(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (2,))


    def test_526(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_527(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_528(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_529(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_530(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_531(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_532(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_533(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_534(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_535(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_536(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_537(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_538(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_539(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_540(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_541(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_542(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_543(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_544(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_545(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_546(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_547(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_548(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_549(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_550(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_551(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_552(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_553(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_554(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_555(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_556(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_557(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_558(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_559(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_560(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_561(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_562(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_563(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_564(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_565(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_566(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_567(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_568(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_569(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_570(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_571(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_572(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_573(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_574(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_575(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


    def test_576(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (1,))


    def test_577(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (2,))


    def test_578(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_579(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_580(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (1,))


    def test_581(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (2,))


    def test_582(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_583(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_584(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (1,))


    def test_585(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (2,))


    def test_586(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_587(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_588(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (1,))


    def test_589(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (2,))


    def test_590(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_591(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_592(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_593(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_594(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_595(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_596(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_597(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_598(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_599(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_600(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_601(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_602(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_603(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_604(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_605(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_606(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_607(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_608(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (1,))


    def test_609(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (2,))


    def test_610(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_611(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_612(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (1,))


    def test_613(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (2,))


    def test_614(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_615(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_616(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (1,))


    def test_617(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (2,))


    def test_618(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_619(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_620(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (1,))


    def test_621(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (2,))


    def test_622(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_623(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_624(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (1,))


    def test_625(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (2,))


    def test_626(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (2, 1))


    def test_627(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (1, 2))


    def test_628(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (1,))


    def test_629(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (2,))


    def test_630(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (2, 1))


    def test_631(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (1, 2))


    def test_632(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (1,))


    def test_633(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (2,))


    def test_634(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (2, 1))


    def test_635(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (1, 2))


    def test_636(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (1,))


    def test_637(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (2,))


    def test_638(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (2, 1))


    def test_639(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (1, 2))


    def test_640(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (1,))


    def test_641(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (2,))


    def test_642(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_643(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_644(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (1,))


    def test_645(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (2,))


    def test_646(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_647(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_648(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (1,))


    def test_649(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (2,))


    def test_650(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_651(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_652(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (1,))


    def test_653(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (2,))


    def test_654(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_655(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_656(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (1,))


    def test_657(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (2,))


    def test_658(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_659(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_660(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (1,))


    def test_661(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (2,))


    def test_662(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_663(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_664(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (1,))


    def test_665(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (2,))


    def test_666(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_667(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_668(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (1,))


    def test_669(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (2,))


    def test_670(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_671(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_672(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_673(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_674(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_675(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_676(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_677(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_678(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_679(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_680(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_681(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_682(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_683(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_684(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_685(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_686(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_687(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_688(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_689(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_690(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_691(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_692(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_693(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_694(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_695(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_696(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_697(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_698(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_699(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_700(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_701(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_702(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_703(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_704(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_705(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_706(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_707(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_708(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_709(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_710(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_711(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_712(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_713(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_714(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_715(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_716(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_717(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_718(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_719(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


    def test_720(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (1,))


    def test_721(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (2,))


    def test_722(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_723(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_724(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (1,))


    def test_725(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (2,))


    def test_726(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_727(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_728(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (1,))


    def test_729(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (2,))


    def test_730(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_731(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_732(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (1,))


    def test_733(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (2,))


    def test_734(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_735(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_736(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_737(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_738(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_739(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_740(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_741(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_742(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_743(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_744(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_745(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_746(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_747(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_748(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_749(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_750(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_751(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_752(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (1,))


    def test_753(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (2,))


    def test_754(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_755(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_756(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (1,))


    def test_757(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (2,))


    def test_758(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_759(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_760(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (1,))


    def test_761(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (2,))


    def test_762(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_763(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_764(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (1,))


    def test_765(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (2,))


    def test_766(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_767(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_768(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (1,))


    def test_769(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (2,))


    def test_770(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (2, 1))


    def test_771(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (1, 2))


    def test_772(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (1,))


    def test_773(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (2,))


    def test_774(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (2, 1))


    def test_775(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (1, 2))


    def test_776(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (1,))


    def test_777(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (2,))


    def test_778(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (2, 1))


    def test_779(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (1, 2))


    def test_780(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (1,))


    def test_781(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (2,))


    def test_782(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (2, 1))


    def test_783(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (1, 2))


    def test_784(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (1,))


    def test_785(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (2,))


    def test_786(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_787(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_788(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (1,))


    def test_789(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (2,))


    def test_790(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_791(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_792(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (1,))


    def test_793(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (2,))


    def test_794(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_795(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_796(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (1,))


    def test_797(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (2,))


    def test_798(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_799(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_800(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (1,))


    def test_801(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (2,))


    def test_802(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_803(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_804(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (1,))


    def test_805(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (2,))


    def test_806(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_807(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_808(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (1,))


    def test_809(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (2,))


    def test_810(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_811(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_812(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (1,))


    def test_813(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (2,))


    def test_814(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_815(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_816(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_817(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_818(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_819(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_820(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_821(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_822(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_823(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_824(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_825(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_826(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_827(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_828(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_829(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_830(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_831(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_832(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_833(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_834(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_835(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_836(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_837(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_838(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_839(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_840(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_841(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_842(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_843(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_844(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_845(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_846(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_847(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_848(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_849(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_850(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_851(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_852(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_853(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_854(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_855(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_856(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_857(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_858(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_859(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_860(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_861(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_862(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_863(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


    def test_864(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (1,))


    def test_865(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (2,))


    def test_866(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_867(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_868(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (1,))


    def test_869(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (2,))


    def test_870(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_871(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_872(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (1,))


    def test_873(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (2,))


    def test_874(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_875(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_876(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (1,))


    def test_877(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (2,))


    def test_878(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_879(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_880(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_881(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_882(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_883(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_884(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_885(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_886(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_887(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_888(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_889(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_890(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_891(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_892(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_893(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_894(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_895(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_896(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (1,))


    def test_897(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (2,))


    def test_898(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_899(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_900(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (1,))


    def test_901(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (2,))


    def test_902(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_903(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_904(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (1,))


    def test_905(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (2,))


    def test_906(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_907(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_908(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (1,))


    def test_909(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (2,))


    def test_910(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_911(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_912(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (1,))


    def test_913(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (2,))


    def test_914(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (2, 1))


    def test_915(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (1, 2))


    def test_916(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (1,))


    def test_917(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (2,))


    def test_918(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (2, 1))


    def test_919(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (1, 2))


    def test_920(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (1,))


    def test_921(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (2,))


    def test_922(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (2, 1))


    def test_923(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (1, 2))


    def test_924(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (1,))


    def test_925(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (2,))


    def test_926(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (2, 1))


    def test_927(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (1, 2))


    def test_928(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (1,))


    def test_929(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (2,))


    def test_930(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_931(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_932(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (1,))


    def test_933(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (2,))


    def test_934(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_935(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_936(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (1,))


    def test_937(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (2,))


    def test_938(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_939(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_940(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (1,))


    def test_941(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (2,))


    def test_942(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_943(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_944(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (1,))


    def test_945(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (2,))


    def test_946(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_947(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_948(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (1,))


    def test_949(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (2,))


    def test_950(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_951(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_952(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (1,))


    def test_953(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (2,))


    def test_954(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_955(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_956(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (1,))


    def test_957(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (2,))


    def test_958(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_959(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_960(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_961(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_962(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_963(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_964(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_965(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_966(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_967(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_968(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_969(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_970(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_971(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_972(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_973(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_974(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_975(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_976(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_977(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_978(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_979(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_980(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_981(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_982(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_983(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_984(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_985(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_986(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_987(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_988(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_989(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_990(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_991(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_992(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_993(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_994(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_995(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_996(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_997(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_998(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_999(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_1000(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_1001(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_1002(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_1003(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_1004(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_1005(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_1006(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_1007(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


    def test_1008(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (1,))


    def test_1009(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (2,))


    def test_1010(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_1011(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_1012(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (1,))


    def test_1013(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (2,))


    def test_1014(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_1015(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_1016(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (1,))


    def test_1017(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (2,))


    def test_1018(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_1019(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_1020(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (1,))


    def test_1021(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (2,))


    def test_1022(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_1023(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_1024(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_1025(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_1026(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_1027(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_1028(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_1029(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_1030(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_1031(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_1032(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_1033(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_1034(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_1035(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_1036(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_1037(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_1038(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_1039(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_1040(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (1,))


    def test_1041(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (2,))


    def test_1042(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_1043(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_1044(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (1,))


    def test_1045(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (2,))


    def test_1046(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_1047(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_1048(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (1,))


    def test_1049(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (2,))


    def test_1050(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_1051(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_1052(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (1,))


    def test_1053(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (2,))


    def test_1054(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_1055(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_1056(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (1,))


    def test_1057(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (2,))


    def test_1058(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (2, 1))


    def test_1059(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (1, 2))


    def test_1060(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (1,))


    def test_1061(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (2,))


    def test_1062(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (2, 1))


    def test_1063(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (1, 2))


    def test_1064(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (1,))


    def test_1065(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (2,))


    def test_1066(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (2, 1))


    def test_1067(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (1, 2))


    def test_1068(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (1,))


    def test_1069(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (2,))


    def test_1070(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (2, 1))


    def test_1071(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (1, 2))


    def test_1072(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (1,))


    def test_1073(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (2,))


    def test_1074(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_1075(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_1076(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (1,))


    def test_1077(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (2,))


    def test_1078(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_1079(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_1080(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (1,))


    def test_1081(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (2,))


    def test_1082(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_1083(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_1084(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (1,))


    def test_1085(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (2,))


    def test_1086(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_1087(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_1088(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (1,))


    def test_1089(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (2,))


    def test_1090(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_1091(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_1092(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (1,))


    def test_1093(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (2,))


    def test_1094(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_1095(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_1096(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (1,))


    def test_1097(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (2,))


    def test_1098(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_1099(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_1100(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (1,))


    def test_1101(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (2,))


    def test_1102(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_1103(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_1104(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_1105(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_1106(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_1107(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_1108(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_1109(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_1110(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_1111(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_1112(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_1113(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_1114(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_1115(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_1116(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_1117(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_1118(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_1119(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_1120(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_1121(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_1122(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_1123(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_1124(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_1125(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_1126(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_1127(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_1128(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_1129(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_1130(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_1131(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_1132(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_1133(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_1134(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_1135(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_1136(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_1137(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_1138(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_1139(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_1140(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_1141(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_1142(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_1143(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_1144(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_1145(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_1146(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_1147(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_1148(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_1149(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_1150(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_1151(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


if __name__ == '__main__':
    unittest.main()
