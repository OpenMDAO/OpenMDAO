************************************************
Determining How Many Times a System was Executed
************************************************

All OpenMDAO `Systems` have a set of local counters that keep track of how many times they have
been executed.

**iter_count**

    Counts the number of times this system has called _solve_nonlinear. This also
    corresponds to the number of times that the system's outputs are recorded if a recorder
    is present.

**iter_count_apply**

    Counts the number of times the system has called _apply_nonlinear. For ExplicitComponent,
    calls to apply_nonlinear also call compute, so number of executions can be found by adding
    this and iter_count together. Recorders do no record calls to _apply_nonlinear.

**iter_count_without_approx**

    Counts the number of times the system has iterated but excludes any that occur during
    approximation of derivatives.

When you have an `ExplicitComponent`, the number stored in iter_count may not match the total
number of times that the "compute" function has been called.  This is because compute is also
called whenever '_apply_nonlinear' is called to compute the norm of the current residual. For
an explicit equation, the residual is defined as the difference in the value of the outputs
before and after execution, and an additional execution is required to compute this.

The correct execution count for an ExplicitComponent can always be obtained by adding iter_count
and iter_count_apply.

The recorder iteration coordinate will always match the iter_count because calls to apply_nonlinear
are not recorded.
