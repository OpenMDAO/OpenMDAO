import os

from pyxdsm.matrix_eqn import MatrixEquation


def simul_system():
    simul = MatrixEquation()

    simul.add_variable(1, size=1, text="a")
    simul.add_variable(2, size=1, text="b")

    simul.add_variable(3, size=1, text="c")
    simul.add_variable(4, size=2, )
    simul.add_variable(5, size=2, )
    simul.add_variable('gc', size=1, text=r"$g_c$")

    simul.add_variable(6, size=1, text="d")
    simul.add_variable(7, size=2, )
    simul.add_variable(8, size=2, )
    simul.add_variable('gd', size=1, text=r"$g_d$")

    simul.add_variable(9, size=1, text="e")
    simul.add_variable(10, size=2, )
    simul.add_variable(11, size=2, )
    simul.add_variable('ge', size=1, text=r"$g_e$")

    simul.add_variable(12, size=1, text="f")

    simul.connect(1, [4,5,7,8,10,11,12,'gc','gd','ge'])
    simul.connect(2, [4,5,7,8,10,11,12,'gc','gd','ge'])

    simul.connect(3, 4)
    simul.connect(4, 5)
    simul.connect(5, 4)
    simul.connect(3, 'gc')
    simul.connect(4, 'gc')
    simul.connect(5, 'gc')

    simul.connect(6, 7)
    simul.connect(7, 8)
    simul.connect(8, 7)
    simul.connect(6, 'gd')
    simul.connect(7, 'gd')
    simul.connect(8, 'gd')

    simul.connect(9, 10)
    simul.connect(10, 11)
    simul.connect(11, 10)
    simul.connect(9, 'ge')
    simul.connect(10, 'ge')
    simul.connect(11, 'ge')

    simul.connect(11, 12)

    return simul

#############################################################
simul_0 = simul_system()

simul_0.jacobian()
simul_0.write('simultaneous_jac')

#############################################################
simul_1 = simul_system()

simul_1.jacobian()
simul_1.spacer()
simul_1.vector(base_color='red', highlight=[2, 1] + 13*[2,])
simul_1.spacer()
simul_1.vector(base_color='green', highlight=[1,] + 14*[2,])
simul_1.operator('=')
simul_1.vector(base_color='red', highlight=[2,]+14*[1,])
simul_1.spacer()
simul_1.vector(base_color='green', highlight=[1,2]+13*[1,])

simul_1.write('simultaneous_dense')




#############################################################
simul_2 = simul_system()

simul_2.jacobian()
simul_2.spacer()
simul_2.vector(base_color='red', highlight=[1,1,2,2,2,2,1,1,1,1,1,1,1,1,1])
simul_2.spacer()
simul_2.vector(base_color='green', highlight=[1,1,1,1,1,1,2,2,2,2,1,1,1,1,1])
simul_2.spacer()
simul_2.vector(base_color='yellow', highlight=[1,1,1,1,1,1,1,1,1,1,2,2,2,2,2])
simul_2.spacer()
simul_2.operator('=')
simul_2.vector(base_color='red', highlight=[1,1,2,1,1,1,1,1,1,1,1,1,1,1,1])
simul_2.spacer()
simul_2.vector(base_color='green', highlight=[1,1,1,1,1,1,2,1,1,1,1,1,1,1,1])
simul_2.spacer()
simul_2.vector(base_color='yellow', highlight=[1,1,1,1,1,1,1,1,1,1,2,1,1,1,1])
simul_2.spacer()
simul_2.write('simultaneous_sparse_separate')


#############################################################
simul_combined = simul_system()

simul_combined.jacobian()
simul_combined.spacer()
simul_combined.vector(base_color='red', highlight=[1,1,2,2,2,2,2,2,2,2,2,2,2,2,2])
simul_combined.operator('=')
simul_combined.vector(base_color='green', highlight=[1,1,2,1,1,1,2,1,1,1,2,1,1,1,1])
simul_combined.write('simultaneous_sparse_combined')

# convert the pdf's to png (requires image magick library to be installed)
def convert(name):
    os.system('convert -density 300 {0}.pdf -quality 90 {0}.png'.format(name))


convert('simultaneous_jac')
convert('simultaneous_dense')
convert('simultaneous_sparse_separate')
convert('simultaneous_sparse_combined')
