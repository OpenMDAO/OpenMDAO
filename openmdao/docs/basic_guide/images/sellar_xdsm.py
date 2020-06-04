from pyxdsm.XDSM import XDSM

#
opt = 'Optimization'
solver = 'MDA'
group = 'Group'
func = 'Function'


x = XDSM()
x.add_system('d1', func, [r'Discipline 1','y_1 = z_1^2 + z_2 + x_1 - 0.2y_2'])
x.add_system('d2', func, [r'Discipline 2', 'y_2 = \sqrt{y_1} + z_1 + z_2'])
x.add_system('f', func, [r'Objective', 'f = x^2 + z_1 + y_1 + e^{-y_2}'])
x.add_system('g1', func, [r'Constraint 1','g1 = 3.16-y_1 '])
x.add_system('g2', func, [r'Constraint 2','g_2 = y_2 - 24.0'])

x.connect('d1', 'd2', r'y_1')
x.connect('d1', 'f', r'y_1')
x.connect('d1', 'g1', r'y_1')

x.connect('d2', 'd1', r'y_2')
x.connect('d2', 'f', r'y_2')
x.connect('d2', 'g2', r'y_2')

x.add_input('d1', r'x, z_1, z_2')
x.add_input('d2', r'z_1, z_2')
x.add_input('f', r'x, z_1')

x.add_output('f', r'f', side='right')
x.add_output('g1', r'g_1', side='right')
x.add_output('g2', r'g_2', side='right')

x.write('sellar_xdsm')
