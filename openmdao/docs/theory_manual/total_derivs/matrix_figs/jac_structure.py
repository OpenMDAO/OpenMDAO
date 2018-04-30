import os

from pyxdsm.matrix_eqn import MatrixEquation


############################################
# Feed Forward Uncoupled System
############################################

def build_sys():

    sys = MatrixEquation()
    sys.add_variable(1, size=1)
    sys.add_variable(2, size=1)
    sys.add_variable(3, size=2)
    sys.add_variable(4, size=2)
    sys.add_variable(5, size=1)

    sys.connect(1, [2,3,4,5])
    sys.connect(2, [3,4,5])
    sys.connect(3, [4,5])
    sys.connect(4, [5])

    return sys

#-----------
# fwd
#-----------
sys = build_sys()

sys.jacobian()
sys.spacer()
sys.vector(base_color='red', highlight=[2,2,2,2,2])
sys.spacer()
sys.operator('=')
sys.vector(base_color='green', highlight=[2,1,1,1,1])

sys.write('uncoupled_fwd')

#-----------
# rev
#-----------
sys = build_sys()

sys.jacobian(transpose=True)
sys.spacer()
sys.vector(base_color='red', highlight=[2,2,2,2,2])
sys.spacer()
sys.operator('=')
sys.vector(base_color='green', highlight=[2,1,1,1,1])

sys.write('uncoupled_rev')


############################################
# Coupled Uncoupled System
############################################

sys = MatrixEquation()
sys.add_variable(1, size=1)
sys.add_variable(2, size=1)
sys.add_variable(3, size=2)
sys.add_variable(4, size=2)
sys.add_variable(5, size=1)

sys.connect(1, [2,3,4,5])
sys.connect(2, [5])
sys.connect(3, [2,4,5])
sys.connect(4, [2,5])
sys.connect(5, [2,3,4])

sys.jacobian()
sys.spacer()
sys.vector(base_color='red', highlight=[2,2,2,2,2])
sys.spacer()
sys.operator('=')
sys.vector(base_color='green', highlight=[2,1,1,1,1])

sys.write('coupled_fwd')

# convert the pdf's to png (requires image magick library to be installed)
def convert(name):
    os.system('convert -density 300 {0}.pdf -quality 90 {0}.png'.format(name))

convert('uncoupled_fwd')
convert('uncoupled_rev')
convert('coupled_fwd')




