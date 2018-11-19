"""
This script works in serial but fails in parallel.
Desired behavior is to have problem var setting intelligently give the value
to the proc which owns that part of the problem.
For example, proc 1 would get the value 3 and proc 2 would get 5 below, if
proc 1 owned comp1 and proc 2 owned comp2.
"""

from openmdao.api import Problem, ExecComp, Group, ParallelGroup

# build the model
prob = Problem()

group = prob.model.add_subsystem('group', ParallelGroup())

comp = ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3', y=2.0)
group.add_subsystem('comp1', comp)

comp = ExecComp('g = x*y', y=2.0)
group.add_subsystem('comp2', comp)

prob.setup()

prob['group.comp1.x'] = 4.
prob['group.comp2.x'] = 5.

prob.run_model()

print("group.comp1.f", prob['group.comp1.f'])   # 42
print("group.comp2.g", prob['group.comp2.g'])   # 10


