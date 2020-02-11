.. _controlling-mpi:

*************************
Controlling MPI Detection
*************************

By default, OpenMDAO will attempt to import the mpi4py module. If that fails,
or if :code:`MPI.COMM_WORLD.size` is :code:`1`, a warning is printed and
execution continues normally without MPI support:

  ``Unable to import mpi4py. Parallel processing unavailable.``

Continuing can be problematic if MPI processing was intended, so this behavior
can be modified by setting the environment variable
:code:`OPENMDAO_REQUIRE_MPI`:

- A value of :code:`True` (or :code:`Yes`, :code:`1`, or
  :code:`Always`; case-insensitive) will raise an exception if mpi4py fails to
  load. If successful, MPI will be used even if :code:`MPI.COMM_WORLD.size` is
  only :code:`1`.

- Any other value will prevent loading of the mpi4py module, disabling MPI
  usage. This can be useful when:

  * MPI is disallowed (e.g. certain HPC cluster head nodes)
  * Loading mpi4py causes unacceptable overhead
  * Displaying the warning message is undesirable


*******************
MPI Troubleshooting
*******************

This section describes how to fix certain MPI related problems.


The following errors may occur when using certain versions of Open MPI:

Fix the **There are not enough slots available in the system...** error by defining

    **OMPI_MCA_rmaps_base_oversubscribe=1**

in your environment.


Fix the **A system call failed during shared memory initialization that should not have...**
error by setting

    **OMPI_MCA_btl=self,tcp**

in your environment.


***********
MPI Testing
***********

Running OpenMDAO's MPI tests requires the use of the
`testflo <https://github.com/OpenMDAO/testflo>`_  package.  You must have a working
MPI library installed, for example *openmpi* or *MVAPICH*, and the
*mpi4py*, *petsc4py*, and *pyoptsparse* python packages must be installed in your python
environment.

