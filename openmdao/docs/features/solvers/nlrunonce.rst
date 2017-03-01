:orphan:

.. _nlrunonce:

Nonlinear Solver: NLRunOnce
===========================

The simplest solver in OpenMDAO is the NLRunOnce solver, which executes the
system's components or subsystems sequentially. Not iteration is performed by
this solver, so it can only be used in systems where the following conditions
are satisfied:

1. System does not contain a cycle.
2. System does not contain any implicit states.

Note that a subsystem may contain cycles or impicit states provided that it is
fitted with a solver that can handle them such as NewtonSolver.



