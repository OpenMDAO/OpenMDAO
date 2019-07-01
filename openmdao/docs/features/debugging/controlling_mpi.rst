.. _controlling-mpi:

*************************
Controlling MPI Detection
*************************

By default, OpenMDAO will attempt to import the mpi4py module. If that fails,
or if :code:`MPI.COMM_WORLD.size` is :code:`1`, a warning is printed and
execution continues normally without MPI support:

  ``Unable to import mpi4py. Parallel processing unavailable.``

This can be unfortunate if MPI processing was desired, so behavior can be
modified by setting the environment variable :code:`OPENMDAO_REQUIRE_MPI`:

- A value of :code:`True` (or :code:`Yes`, :code:`1`, or
  :code:`Always`; case-insensitive) will raise an exception if mpi4py fails to
  load. If successful, MPI will be used even if :code:`MPI.COMM_WORLD.size` is
  only :code:`1`.

- Any other value will prevent loading of the mpi4py module, disabling MPI
  usage. This can be useful when:

  * MPI is disallowed (e.g. certain HPC cluster head nodes)
  * Loading mpi4py causes unacceptable overhead
  * Displaying the warning message is undesirable
