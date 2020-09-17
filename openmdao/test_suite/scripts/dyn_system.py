
import numpy as np
import openmdao.api as om

# just a small dynamic shape model to test the command line dyn shape graph plotting

if __name__ == '__main__':
    p = om.Problem()
    indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', val=np.ones((2,3))))
    indep.add_output('x2', val=np.ones((4,2)))

    p.model.add_subsystem('C1', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                x1={'shape_by_conn': True, 'copy_shape': 'y1'},
                                                x2={'shape_by_conn': True, 'copy_shape': 'y2'},
                                                y1={'shape_by_conn': True, 'copy_shape': 'x1'},
                                                y2={'shape_by_conn': True, 'copy_shape': 'x2'}))

    p.model.add_subsystem('C2', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                x1={'shape_by_conn': True, 'copy_shape': 'y1'},
                                                x2={'shape_by_conn': True, 'copy_shape': 'y2'},
                                                y1={'shape_by_conn': True, 'copy_shape': 'x1'},
                                                y2={'shape_by_conn': True, 'copy_shape': 'x2'}))

    p.model.connect('indep.x1', 'C1.x1')
    p.model.connect('indep.x2', 'C1.x2')
    p.model.connect('C1.y1', 'C2.x1')
    p.model.connect('C1.y2', 'C2.x2')

    p.setup()
    p.run_model()
