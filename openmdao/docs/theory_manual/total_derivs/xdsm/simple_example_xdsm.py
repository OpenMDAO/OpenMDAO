from pyxdsm.XDSM import XDSM

#
opt = 'Optimization'
solver = 'MDA'
comp = 'ImplicitAnalysis'
group = 'Metamodel'
func = 'Function'

x = XDSM()

x.add_system('dv', func, (r'x=1', r'\text{Design Variable}'))
x.add_system('d1', func, (r'y_1=y_2^2', r'\text{Discipline 1}'))
x.add_system('d2', comp, (r'\exp(-y_1 y_2) - x y_2', r'\text{Discipline 2}'))
x.add_system('f', func, (r'y_1^2 - y_2 + 3', r'\text{Objective}'))


x.connect('dv', 'd2', 'x')
x.connect('d1', 'd2', 'y_1')
x.connect('d2', 'd1', 'y_1')
x.connect('d1', 'f', 'y_1')
x.connect('d2', 'f', 'y_2')

x.write('simple_example_xdsm')