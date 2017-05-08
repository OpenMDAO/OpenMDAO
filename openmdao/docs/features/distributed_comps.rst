
Distributed Components
----------------------

At times when you need to perform a computation using large input arrays, you may
want to perform that computation in multiple processes, where each process
operates on some subset of the input values. This may be done purely for
performance reasons, or it may be necessary because the entire input will not fit
in the memory of a single machine.  In any case, this can be accomplished in
OpenMDAO using a distributed component.

We've already seen here <add ref to src_indices discussion> that using src_indices
we can connect an input to only a subset of an output variable.
