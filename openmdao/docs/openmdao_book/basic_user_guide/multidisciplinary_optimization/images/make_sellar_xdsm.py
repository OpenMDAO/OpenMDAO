
from pyxdsm.XDSM import XDSM, FUNC, IFUNC, RIGHT

def make_xdsm():
    """
    Create the XDSM diagram for the sellar problem.
    """
    x = XDSM(use_sfmath=False)

    x.add_system("dis1", FUNC, [r"\text{Discipline 1}", "y_1 = z_1^2 + z_2 + x - 0.2 y_2"])
    x.add_system("dis2", FUNC, [r"\text{Discipline 2}", r"y_2 = \sqrt{y_1} + z_1 + z_2"])
    x.add_system("obj", IFUNC, [r"\text{Objective}", r"f = x^2 + z_2 + y_1 + e^{-y_2}"])
    x.add_system("con1", IFUNC, [r"\text{Constraint 1}", r"g_1 = 3.16 - y_1"])
    x.add_system("con2", IFUNC, [r"\text{Constraint 2}", r"g_2 = y_2 - 24.0"])

    x.connect("dis1", "dis2", "y_1")
    x.connect("dis1", "obj", "y_1")
    x.connect("dis1", "con1", "y_1")

    x.connect("dis2", "dis1", "y_2")
    x.connect("dis2", "obj", "y_2")
    x.connect("dis2", "con2", "y_2")

    x.add_output("obj", "f", side=RIGHT)
    x.add_output("con1", "g_1", side=RIGHT)
    x.add_output("con2", "g_2", side=RIGHT)

    x.add_input("dis1", ["x, z_1, z_2"])
    x.add_input("dis2", ["z_1, z_2"])
    x.add_input("obj", ["x, z_2"])

    x.write("sellar_xdsm")

if __name__ == '__main__':
    make_xdsm()
