.. _instbasedmemory:

****************
Memory Profiling
****************

The :code:`openmdao mem` command can be used to obtain an estimate of the memory usage of
python function calls.

For example:

.. code-block:: none

   openmdao mem -p openmdao -p openaerostruct --tree run_CRM.py


This will generate output to the console that looks like this:

.. code-block:: none

    3486.46  (1 calls)  Problem.setup
    |3486.46  (1 calls)  System._setup:(Group)
    |   |3485.49  (1 calls)  Group._setup_procs
    |   |   |3481.97  (1 calls)  Group._setup_procs:(AeroPoint)
    |   |   |   |3479.86  (1 calls)  Group._setup_procs:(VLMStates)
    |   |   |   |   |2617.19  (2 calls)  Component._setup_procs:(EvalVelMtx)
    |   |   |   |   |   |3917.41  (2 calls)  EvalVelMtx.setup
    |   |   |   |   |   |   |  27.19  (2 calls)  ExplicitComponent.add_output:(EvalVelMtx)
    |   |   |   |   |   |   |   |  27.19  (2 calls)  Component.add_output:(EvalVelMtx)
    |   |   |   |   |   |   |   |   |  27.09  (2 calls)  </Users/banaylor/dev/blue/openmdao/utils/general_utils.py:151>.ensure_compatible
    |   |   |   |   | 454.27  (2 calls)  Component._setup_procs:(GetVectors)
    |   |   |   |   |   | 454.27  (2 calls)  GetVectors.setup
    |   |   |   |   | 207.82  (1 calls)  Component._setup_procs:(VLMMtxRHSComp)
    |   |   |   |   |   | 207.82  (1 calls)  VLMMtxRHSComp.setup
    |   |   |   |   |   |   |  27.09  (3 calls)  Component.add_input:(VLMMtxRHSComp)
    |   |   |   |   |   |   |   |  27.09  (3 calls)  </Users/banaylor/dev/blue/openmdao/utils/general_utils.py:151>.ensure_compatible
    |   |   |   |   |   |   |   9.13  (2 calls)  ExplicitComponent.add_output:(VLMMtxRHSComp)
    |   |   |   |   |   |   |   |   9.12  (2 calls)  Component.add_output:(VLMMtxRHSComp)
    |   |   |   |   |   |   |   |   |   9.03  (2 calls)  </Users/banaylor/dev/blue/openmdao/utils/general_utils.py:151>.ensure_compatible
    |   |   |   |   | 162.62  (1 calls)  Component._setup_procs:(EvalVelocities)
    |   |   |   |   |   | 162.62  (1 calls)  EvalVelocities.setup
    |   |   |   |   |  36.21  (1 calls)  Component._setup_procs:(SolveMatrix)
    |   |   |   |   |   |  36.21  (1 calls)  SolveMatrix.setup
    |   |   |   |   1.31  (1 calls)  Component._setup_procs:(VLMGeometry)
    |   |   |   |   |   1.31  (1 calls)  VLMGeometry.setup
    |   |   |   3.35  (1 calls)  Group._setup_procs:(Geometry)
    |   |   |   |   2.85  (1 calls)  Group._setup_procs:(GeometryMesh)
    |   |   |   |   |   1.15  (1 calls)  Component._setup_procs:(Rotate)
    |   |   |   |   |   |   1.15  (1 calls)  Rotate.setup
    1399.43  (1 calls)  Problem.final_setup
    |1399.41  (1 calls)  System._final_setup:(Group)
    |   | 689.68  (1 calls)  Group._setup_transfers
    |   |   | 689.68  (1 calls)  DefaultTransfer._setup_transfers
    |   |   |   | 688.50  (1 calls)  Group._setup_transfers:(AeroPoint)
    |   |   |   |   | 688.50  (1 calls)  DefaultTransfer._setup_transfers
    |   |   |   |   |   | 688.33  (1 calls)  Group._setup_transfers:(VLMStates)
    |   |   |   |   |   |   | 688.33  (1 calls)  DefaultTransfer._setup_transfers
    |   |   |   |   |   |   |   | 516.52  (54 calls)  DefaultTransfer._setup_transfers.merge
    |   | 354.49  (1 calls)  System._setup_bounds:(Group)
    |   | 354.33  (1 calls)  System.set_initial_values:(Group)
    4.27  (1 calls)  /Users/banaylor/dev/blue/openmdao/api.py:1<module>
    |   1.09  (1 calls)  /Users/banaylor/dev/blue/openmdao/drivers/pyoptsparse_driver.py:7<module>
    |   1.06  (1 calls)  /Users/banaylor/dev/blue/openmdao/components/meta_model_structured_comp.py:1<module>
    169.36  (2 calls)  ExplicitComponent._solve_nonlinear:(GetVectors)
    | 113.30  (2 calls)  DefaultVector.set_const
    |  56.69  (2 calls)  GetVectors.compute
    17.88  (1 calls)  ExplicitComponent._solve_nonlinear:(VLMMtxRHSComp)
    |   9.04  (1 calls)  DefaultVector.set_const
    |   8.84  (1 calls)  VLMMtxRHSComp.compute
    8.57  (13 calls)  Group._transfer:(VLMStates)
    |   8.57  (13 calls)  DefaultTransfer.transfer
    147.45  (2 calls)  EvalVelMtx.compute
    | 686.95  (10 calls)  </Users/banaylor/dev/OpenAeroStruct/openaerostruct/aerodynamics/eval_mtx.py:18>._compute_finite_vortex
    |   |  57.44  (10 calls)  </Users/banaylor/dev/OpenAeroStruct/openaerostruct/utils/vector_algebra.py:10>.compute_dot
    |   |  21.36  (20 calls)  </Users/banaylor/dev/OpenAeroStruct/openaerostruct/utils/vector_algebra.py:90>.compute_norm
    54.19  (2 calls)  DefaultVector.set_const

    Max mem usage: 5400.57 MB


The memory use is mapped to the call tree structure .  Note that functions are tracked based on
their full call tree path, so that the same function can appear multiple times in the tree,
called from different places, and the different memory usage for those multiple calls can be
seen in the tree.

To see a flat listing rather than a tree, don't use the `--tree` arg, and you'll get output like
this:

.. code-block:: none

    1.06  (225 calls)  Vector.__init__:(DefaultVector)
    1.06  (1 calls)  /Users/banaylor/dev/blue/openmdao/components/meta_model_structured_comp.py:1<module>
    1.09  (1 calls)  /Users/banaylor/dev/blue/openmdao/drivers/pyoptsparse_driver.py:7<module>
    1.15  (1 calls)  Component._setup_procs:(Rotate)
    1.15  (1 calls)  Rotate.setup
    1.31  (1 calls)  Component._setup_procs:(VLMGeometry)
    1.31  (1 calls)  VLMGeometry.setup
    2.85  (1 calls)  Group._setup_procs:(GeometryMesh)
    3.35  (1 calls)  Group._setup_procs:(Geometry)
    4.27  (1 calls)  /Users/banaylor/dev/blue/openmdao/api.py:1<module>
    8.57  (13 calls)  Group._transfer:(VLMStates)
    8.58  (43 calls)  DefaultTransfer.transfer
    8.84  (1 calls)  VLMMtxRHSComp.compute
    9.12  (2 calls)  Component.add_output:(VLMMtxRHSComp)
    9.13  (2 calls)  ExplicitComponent.add_output:(VLMMtxRHSComp)
    17.88  (1 calls)  ExplicitComponent._solve_nonlinear:(VLMMtxRHSComp)
    21.36  (24 calls)  </Users/banaylor/dev/OpenAeroStruct/openaerostruct/utils/vector_algebra.py:90>.compute_norm
    27.09  (3 calls)  Component.add_input:(VLMMtxRHSComp)
    27.19  (2 calls)  Component.add_output:(EvalVelMtx)
    27.19  (2 calls)  ExplicitComponent.add_output:(EvalVelMtx)
    36.21  (1 calls)  Component._setup_procs:(SolveMatrix)
    36.21  (1 calls)  SolveMatrix.setup
    56.69  (2 calls)  GetVectors.compute
    57.44  (14 calls)  </Users/banaylor/dev/OpenAeroStruct/openaerostruct/utils/vector_algebra.py:10>.compute_dot
    64.25  (145 calls)  </Users/banaylor/dev/blue/openmdao/utils/general_utils.py:151>.ensure_compatible
    147.45  (2 calls)  EvalVelMtx.compute
    162.62  (1 calls)  Component._setup_procs:(EvalVelocities)
    162.62  (1 calls)  EvalVelocities.setup
    169.36  (2 calls)  ExplicitComponent._solve_nonlinear:(GetVectors)
    177.06  (36 calls)  DefaultVector.set_const
    207.82  (1 calls)  Component._setup_procs:(VLMMtxRHSComp)
    207.82  (1 calls)  VLMMtxRHSComp.setup
    354.33  (1 calls)  System.set_initial_values:(Group)
    354.49  (1 calls)  System._setup_bounds:(Group)
    454.27  (2 calls)  Component._setup_procs:(GetVectors)
    454.27  (2 calls)  GetVectors.setup
    517.44  (186 calls)  DefaultTransfer._setup_transfers.merge
    686.95  (10 calls)  </Users/banaylor/dev/OpenAeroStruct/openaerostruct/aerodynamics/eval_mtx.py:18>._compute_finite_vortex
    688.33  (1 calls)  Group._setup_transfers:(VLMStates)
    688.50  (1 calls)  Group._setup_transfers:(AeroPoint)
    689.68  (1 calls)  Group._setup_transfers
    1399.41  (1 calls)  System._final_setup:(Group)
    1399.44  (2 calls)  Problem.final_setup
    2068.28  (7 calls)  DefaultTransfer._setup_transfers
    2617.19  (2 calls)  Component._setup_procs:(EvalVelMtx)
    3479.86  (1 calls)  Group._setup_procs:(VLMStates)
    3481.97  (1 calls)  Group._setup_procs:(AeroPoint)
    3485.49  (1 calls)  Group._setup_procs
    3486.46  (1 calls)  System._setup:(Group)
    3486.46  (1 calls)  Problem.setup
    3917.41  (2 calls)  EvalVelMtx.setup

    Max mem usage: 5400.57 MB


The `-p` argument(s) determine which package(s) will be traced. In the example above, the
`openmdao` and `openaerostruct` packages were traced.  If no `-p` args are supplied, the
`openmdao` package is assumed.  Note however that if *any* `-p` args are supplied, then the
`openmdao` package will not be traced unless explicitly specified.

The output can be filtered by minimum memory usage so that the parts with memory
usage below a certain amount will not be shown.  The default minimum memory usage is 1 MB.
If you wanted to set the minimum memory usage to 100 MB, for example, you could do it like this:


.. code-block:: none

   openmdao mem <your_python_script_here> --min=100


Running `openmdao mem` generates a raw memory dump file with a default name of `mem_trace.raw`.
To display the memory profile using a pre-existing memory dump file, you can use the
`openmdao mempost` command, for example:


.. code-block:: none

   openmdao mempost mem_trace.raw --min=100


This just allows you to take different looks at the memory profile without having to re-run
your code.


As usual, any additional options for the `openmdao mem` and `openmdao mempost` commands can
be seen by providing the `-h` argument, for example:

.. embed-shell-cmd::
    :cmd: openmdao mem -h


.. note::

   These memory usage numbers are only estimates, based on the changes in the process memory
   measured before and after each method call.  The exact memory use is difficult to determine due
   to the presence of python's own internal memory management and garbage collection.
