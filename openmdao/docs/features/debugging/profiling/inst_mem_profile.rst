.. _instbasedmemory:

****************
Memory Profiling
****************

The :code:`openmdao mem` command can be used to obtain an estimate of the memory usage of
python function calls.

For example:

.. code-block:: none

   openmdao mem <your_python_script_here>


This will generate output to the console that looks like this:

.. code-block:: none

    5312.46  (1 calls)  /Users/jubs/OpenAeroStruct/openaerostruct/examples/run_CRM.py:1:<module>
        5.68  (1 calls)  /Users/jubs/OpenMDAO/openmdao/api.py:1:<module>
        1.73  (1 calls)  /Users/jubs/OpenMDAO/openmdao/components/external_code_comp.py:1:<module>
            1.69  (1 calls)  /Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/distutils/__init__.py:1:<module>
            1.55  (1 calls)  /Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/distutils/ccompiler.py:1:<module>
        1.28  (1 calls)  /Users/jubs/OpenMDAO/openmdao/components/meta_model_structured_comp.py:1:<module>
            1.26  (1 calls)  /Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/scipy/interpolate/__init__.py:173:<module>
        1.17  (1 calls)  /Users/jubs/OpenMDAO/openmdao/drivers/pyoptsparse_driver.py:7:<module>
            1.16  (1 calls)  /Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/pyoptsparse/__init__.py:2:<module>
    3757.77  (1 calls)  Problem.setup Problem#0
        3757.76  (1 calls)  System._setup Group#0:''
        3757.12  (1 calls)  Group._setup_procs Group#0:''
            2.95  (1 calls)  Group._setup_procs Geometry#0:wing
                2.61  (1 calls)  Group._setup_procs GeometryMesh#0:mesh
                1.10  (1 calls)  Component._setup_procs Rotate#0:rotate
                    1.10  (1 calls)  Rotate.setup Rotate#0:wing.mesh.rotate
            3754.11  (1 calls)  Group._setup_procs AeroPoint#0:aero_point_0
            3753.10  (1 calls)  Group._setup_procs VLMStates#0:aero_states
                283.37  (1 calls)  Component._setup_procs GetVectors#0:get_vectors
                283.37  (1 calls)  GetVectors.setup GetVectors#0:aero_point_0.aero_states.get_vectors
                    56.66  (3 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/numeric.py:146>.ones
                    56.72  (1 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/numeric.py:1039>.outer
                1462.68  (1 calls)  Component._setup_procs EvalVelMtx#0:mtx_assy
                2112.93  (1 calls)  EvalVelMtx.setup EvalVelMtx#0:aero_point_0.aero_states.mtx_assy
                    162.56  (5 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:382>.repeat
                    162.56  (5 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:50>._wrapfunc
                    812.81  (4 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/lib/shape_base.py:844>.tile
                    650.38  (4 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/einsumfunc.py:824>.einsum
                    1372.92  (2 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/lib/function_base.py:4703>.delete
                        81.29  (2 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/numeric.py:146>.ones
                    27.09  (1 calls)  ExplicitComponent.add_output EvalVelMtx#0:aero_point_0.aero_states.mtx_assy
                        27.09  (1 calls)  Component.add_output EvalVelMtx#0:aero_point_0.aero_states.mtx_assy
                        27.09  (1 calls)  </Users/jubs/OpenMDAO/openmdao/utils/general_utils.py:63>.ensure_compatible
                            27.09  (1 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/numeric.py:146>.ones
                207.89  (1 calls)  Component._setup_procs VLMMtxRHSComp#0:mtx_rhs
                207.89  (1 calls)  VLMMtxRHSComp.setup VLMMtxRHSComp#0:aero_point_0.aero_states.mtx_rhs
                    27.18  (3 calls)  Component.add_input VLMMtxRHSComp#0:aero_point_0.aero_states.mtx_rhs
                        27.18  (3 calls)  </Users/jubs/OpenMDAO/openmdao/utils/general_utils.py:63>.ensure_compatible
                        27.09  (3 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/numeric.py:146>.ones
                    9.03  (2 calls)  ExplicitComponent.add_output VLMMtxRHSComp#0:aero_point_0.aero_states.mtx_rhs
                        9.03  (2 calls)  Component.add_output VLMMtxRHSComp#0:aero_point_0.aero_states.mtx_rhs
                        9.03  (2 calls)  </Users/jubs/OpenMDAO/openmdao/utils/general_utils.py:63>.ensure_compatible
                            9.03  (2 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/numeric.py:146>.ones
                    54.19  (4 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/einsumfunc.py:824>.einsum
                36.14  (1 calls)  Component._setup_procs SolveMatrix#0:solve_matrix
                    36.14  (1 calls)  SolveMatrix.setup SolveMatrix#0:aero_point_0.aero_states.solve_matrix
                    9.03  (3 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/numeric.py:1039>.outer
                170.12  (1 calls)  Component._setup_procs GetVectors#1:get_vectors_force
                170.12  (1 calls)  GetVectors.setup GetVectors#1:aero_point_0.aero_states.get_vectors_force
                    56.72  (1 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/numeric.py:1039>.outer
                1428.43  (1 calls)  Component._setup_procs EvalVelMtx#1:mtx_assy_forces
                2078.68  (1 calls)  EvalVelMtx.setup EvalVelMtx#1:aero_point_0.aero_states.mtx_assy_forces
                    162.56  (5 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:382>.repeat
                    162.56  (5 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:50>._wrapfunc
                    812.81  (4 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/lib/shape_base.py:844>.tile
                    650.31  (4 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/einsumfunc.py:824>.einsum
                    1371.62  (2 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/lib/function_base.py:4703>.delete
                        81.28  (2 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/numeric.py:146>.ones
                162.57  (1 calls)  Component._setup_procs EvalVelocities#0:eval_velocities
                162.57  (1 calls)  EvalVelocities.setup EvalVelocities#0:aero_point_0.aero_states.eval_velocities
                    54.19  (3 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/einsumfunc.py:824>.einsum
    1399.38  (1 calls)  Problem.final_setup Problem#0
        1399.37  (1 calls)  System._final_setup Group#0:''
        354.37  (1 calls)  System._setup_bounds Group#0:''
        689.50  (1 calls)  Group._setup_transfers Group#0:''
            689.50  (1 calls)  DefaultTransfer._setup_transfers
            688.57  (1 calls)  Group._setup_transfers AeroPoint#0:aero_point_0
                688.57  (1 calls)  DefaultTransfer._setup_transfers
                688.32  (1 calls)  Group._setup_transfers VLMStates#0:aero_point_0.aero_states
                    688.32  (1 calls)  DefaultTransfer._setup_transfers
                    516.78  (54 calls)  DefaultTransfer._setup_transfers.merge
        354.32  (1 calls)  System.set_initial_values Group#0:''
    148.66  (1 calls)  Problem.run_model Problem#0
        148.65  (1 calls)  System.run_solve_nonlinear Group#0:''
        148.65  (1 calls)  Group._solve_nonlinear Group#0:''
            148.65  (1 calls)  NonlinearRunOnce.solve NonlinearRunOnce#0
            148.35  (1 calls)  Group._solve_nonlinear AeroPoint#0:aero_point_0
                148.35  (1 calls)  NonlinearRunOnce.solve NonlinearRunOnce#2
                148.21  (1 calls)  Group._solve_nonlinear VLMStates#0:aero_point_0.aero_states
                    148.21  (1 calls)  NonlinearRunOnce.solve NonlinearRunOnce#4
                        8.58  (13 calls)  Group._transfer VLMStates#0:aero_point_0.aero_states
                        8.51  (1 calls)  DefaultTransfer.transfer DefaultTransfer#41
                    112.67  (1 calls)  ExplicitComponent._solve_nonlinear GetVectors#0:aero_point_0.aero_states.get_vectors
                        56.65  (1 calls)  DefaultVector.set_const DefaultVector#68
                        56.66  (1 calls)  GetVectors.compute GetVectors#0:aero_point_0.aero_states.get_vectors
                        17.43  (1 calls)  ExplicitComponent._solve_nonlinear VLMMtxRHSComp#0:aero_point_0.aero_states.mtx_rhs
                        9.04  (1 calls)  DefaultVector.set_const DefaultVector#77
                        8.39  (1 calls)  VLMMtxRHSComp.compute VLMMtxRHSComp#0:aero_point_0.aero_states.mtx_rhs
                            9.03  (2 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/einsumfunc.py:824>.einsum
                        4.84  (1 calls)  ImplicitComponent._solve_nonlinear SolveMatrix#0:aero_point_0.aero_states.solve_matrix
                        4.84  (1 calls)  SolveMatrix.solve_nonlinear SolveMatrix#0:aero_point_0.aero_states.solve_matrix
                            4.77  (1 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/scipy/linalg/decomp_lu.py:17>.lu_factor
                        56.73  (1 calls)  ExplicitComponent._solve_nonlinear GetVectors#1:aero_point_0.aero_states.get_vectors_force
                        56.65  (1 calls)  DefaultVector.set_const DefaultVector#86
                        29.45  (1 calls)  ExplicitComponent._solve_nonlinear EvalVelMtx#1:aero_point_0.aero_states.mtx_assy_forces
                        27.09  (1 calls)  DefaultVector.set_const DefaultVector#89
                        246.20  (1 calls)  EvalVelMtx.compute EvalVelMtx#1:aero_point_0.aero_states.mtx_assy_forces
                        429.35  (5 calls)  </Users/jubs/OpenAeroStruct/openaerostruct/aerodynamics/eval_mtx.py:18>._compute_finite_vortex
                            18.07  (10 calls)  </Users/jubs/OpenAeroStruct/openaerostruct/utils/vector_algebra.py:90>.compute_norm
                                18.06  (10 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:1778>.sum
                                18.06  (10 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/_methods.py:31>._sum
                            18.60  (5 calls)  </Users/jubs/OpenAeroStruct/openaerostruct/utils/vector_algebra.py:39>.compute_cross
                                18.60  (5 calls)  </Users/jubs/miniconda2/envs/blue3/lib/python3.6/site-packages/numpy/core/numeric.py:1591>.cross

    Max mem usage: 5825.35 MB


The memory use is mapped to the call tree structure .  Note that functions are tracked based on
their full call tree path, so that the same function can appear multiple times in the tree,
called from different places, and the different memory usage for those multiple calls can be
seen in the tree.

The tree can be filtered by minimum memory usage so that the parts of the tree with memory
usage below a certain amount will not be shown.  The default minimum memory usage is 1 MB.
If you wanted to set the minimum memory usage to 100 MB, for example, you could do it like this:


.. code-block:: none

   openmdao mem <your_python_script_here> --min=100


Running `openmdao mem` generates a raw memory dump file with a default name of `mem_trace.raw`.
To display the memory profile using a pre-existing memory dump file, you can use the
`openmdao mempost` command as follows:


.. code-block:: none

   openmdao mempost mem_trace.raw --min=100


This just allows you to take different looks at the memory profile without having to re-run
your code.


.. note::

   These memory usage numbers are only estimates, based on the changes in the process memory
   measured before and after each method call.  The exact memory use is difficult to determine due
   to the presence of python's own internal memory management and garbage collection.
