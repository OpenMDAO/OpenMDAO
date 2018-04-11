import os

from pyxdsm.matrix_eqn import MatrixEquation

IMPLICIT_COLOR="yellow"


def par_system():
    sys = MatrixEquation()

    sys.add_variable(1, size=2, text="x")
    sys.add_variable(2, size=2, text="y")
    sys.add_variable(3, size=4, text="s1", color=IMPLICIT_COLOR)
    sys.add_variable(3.5, size=1, text="z1")
    sys.add_variable(4, size=4, text="s2", color=IMPLICIT_COLOR)
    sys.add_variable(4.5, size=1, text="z2")

    sys.connect(1, 2)
    sys.connect(2, [3,4], color="blue")
    sys.connect(3, 3.5, color=IMPLICIT_COLOR)
    sys.connect(4, 4.5, color=IMPLICIT_COLOR)

    return sys

#############################################################
par = par_system()

par.jacobian()
par.write('parallel_adj_jac', cleanup=False)

#############################################################
par = par_system()

par.jacobian(transpose=True)
par.spacer()
par.vector(base_color='red', highlight=[2, 2, 2, 2, 1, 1])
par.spacer()
par.vector(base_color='red', highlight=[2, 2, 1, 1, 2, 2])
par.operator('=')
par.vector(base_color='green', highlight=[1, 1, 1, 2, 1, 1])
par.spacer()
par.vector(base_color='green', highlight=[1, 1, 1, 1, 1, 2])

par.write('parallel_adj_separate')

#############################################################
par = par_system()

par.jacobian(transpose=True)
par.spacer()
par.vector(base_color='red', highlight=[2, 2, 2, 2, 0, 0])
par.vector(base_color='red', highlight=[2, 2, 0, 0, 2, 2])
par.operator('=')
par.vector(base_color='green', highlight=[1, 1, 1, 2, 0, 0])
par.vector(base_color='green', highlight=[1, 1, 0, 0, 1, 2])

par.write('parallel_adj_combined')




# convert the pdf's to png (requires image magick library to be installed)
def convert(name):
    os.system('convert -density 300 {0}.pdf -quality 90 {0}.png'.format(name))


convert('parallel_adj_jac')
convert('parallel_adj_separate')
convert('parallel_adj_combined')
