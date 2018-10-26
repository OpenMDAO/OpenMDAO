"""Define `Group`s with parallel topologies for testing"""

from __future__ import division, print_function

from openmdao.core.group import Group
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.components.exec_comp import ExecComp


class FanOut(Group):
    """
    Topology where one comp broadcasts an output to two target
    components.
    """

    def __init__(self):
        super(FanOut, self).__init__()

        self.add_subsystem('p', IndepVarComp('x', 1.0))
        self.add_subsystem('comp1', ExecComp(['y=3.0*x']))
        self.add_subsystem('comp2', ExecComp(['y=-2.0*x']))
        self.add_subsystem('comp3', ExecComp(['y=5.0*x']))

        self.connect("p.x", "comp1.x")
        self.connect("comp1.y", "comp2.x")
        self.connect("comp1.y", "comp3.x")


class FanOutGrouped(Group):
    """
    Topology where one component broadcasts an output to two target
    components.
    """

    def __init__(self):
        super(FanOutGrouped, self).__init__()

        self.add_subsystem('iv', IndepVarComp('x', 1.0))
        self.add_subsystem('c1', ExecComp(['y=3.0*x']))

        self.sub = self.add_subsystem('sub', ParallelGroup())
        self.sub.add_subsystem('c2', ExecComp(['y=-2.0*x']))
        self.sub.add_subsystem('c3', ExecComp(['y=5.0*x']))

        self.add_subsystem('c2', ExecComp(['y=x']))
        self.add_subsystem('c3', ExecComp(['y=x']))

        self.connect('iv.x', 'c1.x')

        self.connect('c1.y', 'sub.c2.x')
        self.connect('c1.y', 'sub.c3.x')

        self.connect('sub.c2.y', 'c2.x')
        self.connect('sub.c3.y', 'c3.x')


class FanIn(Group):
    """
    Topology where two comps feed a single comp.
    """

    def __init__(self):
        super(FanIn, self).__init__()

        self.add_subsystem('p1', IndepVarComp('x1', 1.0))
        self.add_subsystem('p2', IndepVarComp('x2', 1.0))
        self.add_subsystem('comp1', ExecComp(['y=-2.0*x']))
        self.add_subsystem('comp2', ExecComp(['y=5.0*x']))
        self.add_subsystem('comp3', ExecComp(['y=3.0*x1+7.0*x2']))

        self.connect("comp1.y", "comp3.x1")
        self.connect("comp2.y", "comp3.x2")
        self.connect("p1.x1", "comp1.x")
        self.connect("p2.x2", "comp2.x")


class FanInGrouped(Group):
    """
    Topology where two components in a Group feed a single component
    outside of that Group.
    """

    def __init__(self):
        super(FanInGrouped, self).__init__()

        iv = self.add_subsystem('iv', IndepVarComp())
        iv.add_output('x1', 1.0)
        iv.add_output('x2', 1.0)
        iv.add_output('x3', 1.0)

        self.sub = self.add_subsystem('sub', ParallelGroup())
        self.sub.add_subsystem('c1', ExecComp(['y=-2.0*x']))
        self.sub.add_subsystem('c2', ExecComp(['y=5.0*x']))

        self.add_subsystem('c3', ExecComp(['y=3.0*x1+7.0*x2']))

        self.connect("sub.c1.y", "c3.x1")
        self.connect("sub.c2.y", "c3.x2")

        self.connect("iv.x1", "sub.c1.x")
        self.connect("iv.x2", "sub.c2.x")

class FanInGrouped2(Group):
    """
    Topology where two components in a Group feed a single component
    outside of that Group. This is slightly different than FanInGrouped
    in that it has two different IndepVarComps.  This configuration
    is used to test a reverse indexing MPI bug that does not appear
    when using FanInGrouped.
    """

    def __init__(self):
        super(FanInGrouped2, self).__init__()

        p1 = self.add_subsystem('p1', IndepVarComp('x', 1.0))
        p2 = self.add_subsystem('p2', IndepVarComp('x', 1.0))

        self.sub = self.add_subsystem('sub', ParallelGroup())
        self.sub.add_subsystem('c1', ExecComp(['y=-2.0*x']))
        self.sub.add_subsystem('c2', ExecComp(['y=5.0*x']))

        self.add_subsystem('c3', ExecComp(['y=3.0*x1+7.0*x2']))

        self.connect("sub.c1.y", "c3.x1")
        self.connect("sub.c2.y", "c3.x2")

        self.connect("p1.x", "sub.c1.x")
        self.connect("p2.x", "sub.c2.x")


class DiamondFlat(Group):
    """
    Topology: one - two - one.

    This one is flat."""

    def __init__(self):
        super(DiamondFlat, self).__init__()

        self.add_subsystem('iv', IndepVarComp('x', 2.0))

        self.add_subsystem('c1', ExecComp([
            'y1 = 2.0*x1**2',
            'y2 = 3.0*x1'
        ]))

        self.add_subsystem('c2', ExecComp('y1 = 0.5*x1'))
        self.add_subsystem('c3', ExecComp('y1 = 3.5*x1'))

        self.add_subsystem('c4', ExecComp([
            'y1 = x1 + 2.0*x2',
            'y2 = 3.0*x1 - 5.0*x2'
        ]))

        # make connections
        self.connect('iv.x', 'c1.x1')
        self.connect('c1.y1', 'c2.x1')
        self.connect('c1.y2', 'c3.x1')
        self.connect('c2.y1', 'c4.x1')
        self.connect('c3.y1', 'c4.x2')


class Diamond(Group):
    """
    Topology: one - two - one.
    """

    def __init__(self):
        super(Diamond, self).__init__()

        self.add_subsystem('iv', IndepVarComp('x', 2.0))

        self.add_subsystem('c1', ExecComp([
            'y1 = 2.0*x1**2',
            'y2 = 3.0*x1'
        ]))

        sub = self.add_subsystem('sub', ParallelGroup())
        sub.add_subsystem('c2', ExecComp('y1 = 0.5*x1'))
        sub.add_subsystem('c3', ExecComp('y1 = 3.5*x1'))

        self.add_subsystem('c4', ExecComp([
            'y1 = x1 + 2.0*x2',
            'y2 = 3.0*x1 - 5.0*x2'
        ]))

        # make connections
        self.connect('iv.x', 'c1.x1')

        self.connect('c1.y1', 'sub.c2.x1')
        self.connect('c1.y2', 'sub.c3.x1')

        self.connect('sub.c2.y1', 'c4.x1')
        self.connect('sub.c3.y1', 'c4.x2')


class ConvergeDivergeFlat(Group):
    """
    Topology one - two - one - two - one. This model was critical in
    testing parallel reverse scatters. This version is perfectly flat.
    """

    def __init__(self):
        super(ConvergeDivergeFlat, self).__init__()

        self.add_subsystem('iv', IndepVarComp('x', 2.0))

        self.add_subsystem('c1', ExecComp([
            'y1 = 2.0*x1**2',
            'y2 = 3.0*x1'
        ]))

        self.add_subsystem('c2', ExecComp('y1 = 0.5*x1'))
        self.add_subsystem('c3', ExecComp('y1 = 3.5*x1'))

        self.add_subsystem('c4', ExecComp([
            'y1 = x1 + 2.0*x2',
            'y2 = 3.0*x1 - 5.0*x2'
        ]))

        self.add_subsystem('c5', ExecComp('y1 = 0.8*x1'))
        self.add_subsystem('c6', ExecComp('y1 = 0.5*x1'))
        self.add_subsystem('c7', ExecComp('y1 = x1 + 3.0*x2'))

        # make connections
        self.connect('iv.x', 'c1.x1')

        self.connect('c1.y1', 'c2.x1')
        self.connect('c1.y2', 'c3.x1')

        self.connect('c2.y1', 'c4.x1')
        self.connect('c3.y1', 'c4.x2')

        self.connect('c4.y1', 'c5.x1')
        self.connect('c4.y2', 'c6.x1')

        self.connect('c5.y1', 'c7.x1')
        self.connect('c6.y1', 'c7.x2')


class ConvergeDiverge(Group):
    """
    Topology: one - two - one - two - one.

    Used for testing parallel reverse scatters.
    """

    def __init__(self):
        super(ConvergeDiverge, self).__init__()

        self.add_subsystem('iv', IndepVarComp('x', 2.0))

        self.add_subsystem('c1', ExecComp([
            'y1 = 2.0*x1**2',
            'y2 = 3.0*x1'
        ]))

        g1 = self.add_subsystem('g1', ParallelGroup())
        g1.add_subsystem('c2', ExecComp('y1 = 0.5*x1'))
        g1.add_subsystem('c3', ExecComp('y1 = 3.5*x1'))

        self.add_subsystem('c4', ExecComp([
            'y1 = x1 + 2.0*x2',
            'y2 = 3.0*x1 - 5.0*x2'
        ]))

        g2 = self.add_subsystem('g2', ParallelGroup())
        g2.add_subsystem('c5', ExecComp('y1 = 0.8*x1'))
        g2.add_subsystem('c6', ExecComp('y1 = 0.5*x1'))

        self.add_subsystem('c7', ExecComp('y1 = x1 + 3.0*x2'))

        # make connections
        self.connect('iv.x', 'c1.x1')

        self.connect('c1.y1', 'g1.c2.x1')
        self.connect('c1.y2', 'g1.c3.x1')

        self.connect('g1.c2.y1', 'c4.x1')
        self.connect('g1.c3.y1', 'c4.x2')

        self.connect('c4.y1', 'g2.c5.x1')
        self.connect('c4.y2', 'g2.c6.x1')

        self.connect('g2.c5.y1', 'c7.x1')
        self.connect('g2.c6.y1', 'c7.x2')


class ConvergeDivergeGroups(Group):
    """
    Topology: one - two - one - two - one.

    Used for testing parallel reverse scatters. This version contains some
    deeper nesting.
    """

    def __init__(self):
        super(ConvergeDivergeGroups, self).__init__()

        self.add_subsystem('iv', IndepVarComp('x', 2.0))

        g1 = self.add_subsystem('g1', ParallelGroup())
        g1.add_subsystem('c1', ExecComp([
            'y1 = 2.0*x1**2',
            'y2 = 3.0*x1'
        ]))

        g2 = g1.add_subsystem('g2', ParallelGroup())
        g2.add_subsystem('c2', ExecComp('y1 = 0.5*x1'))
        g2.add_subsystem('c3', ExecComp('y1 = 3.5*x1'))

        g1.add_subsystem('c4', ExecComp([
            'y1 = x1 + 2.0*x2',
            'y2 = 3.0*x1 - 5.0*x2'
        ]))

        g3 = self.add_subsystem('g3', ParallelGroup())
        g3.add_subsystem('c5', ExecComp('y1 = 0.8*x1'))
        g3.add_subsystem('c6', ExecComp('y1 = 0.5*x1'))

        self.add_subsystem('c7', ExecComp('y1 = x1 + 3.0*x2'))

        # make connections
        self.connect('iv.x', 'g1.c1.x1')

        g1.connect('c1.y1', 'g2.c2.x1')
        g1.connect('c1.y2', 'g2.c3.x1')

        self.connect('g1.g2.c2.y1', 'g1.c4.x1')
        self.connect('g1.g2.c3.y1', 'g1.c4.x2')

        self.connect('g1.c4.y1', 'g3.c5.x1')
        self.connect('g1.c4.y2', 'g3.c6.x1')

        self.connect('g3.c5.y1', 'c7.x1')
        self.connect('g3.c6.y1', 'c7.x2')


class FanInSubbedIDVC(Group):
    """
    Classic Fan In with indepvarcomps buried below the parallel group, and a summation
    component.
    """

    def setup(self):
        sub = self.add_subsystem('sub', ParallelGroup())
        sub1 = sub.add_subsystem('sub1', Group())
        sub2 = sub.add_subsystem('sub2', Group())

        sub1.add_subsystem('p1', IndepVarComp('x', 3.0))
        sub2.add_subsystem('p2', IndepVarComp('x', 5.0))
        sub1.add_subsystem('c1', ExecComp(['y = 2.0*x']))
        sub2.add_subsystem('c2', ExecComp(['y = 4.0*x']))
        sub1.connect('p1.x', 'c1.x')
        sub2.connect('p2.x', 'c2.x')

        self.add_subsystem('sum', ExecComp(['y = z1 + z2']))
        self.connect('sub.sub1.c1.y', 'sum.z1')
        self.connect('sub.sub2.c2.y', 'sum.z2')

        self.sub.sub1.add_design_var('p1.x')
        self.sub.sub2.add_design_var('p2.x')
        self.add_objective('sum.y')