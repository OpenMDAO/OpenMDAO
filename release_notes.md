**********************************
# Release Notes for OpenMDAO 3.2.0

July 21, 2020

OpenMDAO 3.2.0 introduces significant changes in the way OpenMDAO works.
The primary change is that the manual creation of IndepVarComp outputs is no longer required because OpenMDAO will create them for you in the background.
This was done in a backwards compatible way so that old models will still run and you can still use IndepVarComps if you want to.
However, in the vast majority of cases, you don't need to manually create either the IndepVarComp component or add outputs to it anymore.

This feature aids in creating modular systems. Previously, there was always debate as to whether a system should "own" its own IndepVarComp.
Doing so, however, meant that the outputs of that IVC couldn't be passed in externally without changing the system.
This is no longer the case.
Now any inputs that remain unconnected at the end of the problem setup sequence will be connected to special automatic IndepVarComp outputs that are addressable using the pathname of the inputs.
We have updated the docs to use the new style where IndepVarComps are not manually created. We now consider it best practice not to manually create IndepVarComps in the vast majority of cases.

Another new feature is the openmdao.api.slicer which helps specify src_indices for connections or indices for design variables, objectives, and constraints.
This is intended to allow the user to easily connect portions of an output to an input (such as selecting an individual column from a matrix, for instance).
The old method of providing an explicit list of indices should still work. However, the new slicer gives greater functionality since it allows for the use of general slicing syntax such as `om.slicer[0:10:1, :]`.
Using the slicer object is now considered the new best coding practice for giving src_indices or indices arguments.

We hope these changes will reduce the development burden on our users.
As this is a significant change, some issues may have slipped through our testing, and any feedback is always welcomed via issues on the GitHub repository.

## Backwards Incompatible API Changes:

- Fix a typo in user_terminate_signal option, deprecated the old. #1469
- Remove support for factorial function in ExecComp. #1483

## Backwards Incompatible NON-API Changes:

- _all_subsystem_iter has been removed.  Users should use the systems_iter method instead.

## New Features:

- N2 viewer changes to improve performance. #1475
- Local table of contents (navigation) added to docs sidebar. #1477
- N2 now uses coloring info by default to show dependence in each component. #1478
- Added test to capture error message when running in parallel with non distributed components. #1484
- N2 adds a spinner to give the user an indication when it isn't finished rendering. #1485
- All case recorder files now always contain system options information about all systems in the model. #1486
- Model data now compressed when saved in N2 HTML file. #1490
- Added support for om.slicer as a way to pass in slices to src_indices/indices arguments in add_design_var, add_response, add_constraint, promotes, and connect. #1491
- Added flag for `under_approx` to let the user know when a system is operating under any derivative approximation scheme, and a new counter `iter_count_without_approx`. #1492
- Allow the linear algebra components (DotProductComp, CrossProductComp, MatrixVectorProductComp, VectorMagnitudeComp) to each perform multiple calculations. #1503
- Automatically add a single, user-hidden, top-level IndepVarComp to provide outputs for every unconnected input in the model. #1539

## Bug Fixes:

- Fix bug where exception was raised while printing bounds violations via "print_bound_enforce" option on linesearch if one side of an output was unbounded. #1466
- Fixed an exception when using AddSubtractComp constructor instead of its add_equation method. #1474
- Minor fixes for Appveyor CI. #1495
- IndepVarComp did not use the options mechanism like other Components, which was an issue for a user that wanted to subclass it with additional options. #1536
- Problem.record() provides a useful error message if called before final_setup. #1545
- Fixed a bug that was obliterating options passed to BalanceComp, and added tests for better coverage. #1546
- Added quotes to install command in docs to prevent problems in some shell environments. #1547

**********************************
# Release Notes for OpenMDAO 3.1.1

June 12, 2020

OpenMDAO 3.1.1 is a periodic release of OpenMDAO.

This update addresses several issues with distributed models (thanks to @bbrelje) and more improvements to the N2 diagram.

## Backwards Incompatible API Changes:

None

## Backwards Incompatible NON-API Changes:

- Renamed `_post_configure` to `_config_check`, and will no longer support adding inputs/outputs after configure. Any code which previously cached inputs/outputs to be added after configure should now do so directly. #1385

## New Features:

- An error is now raised if a component has a 0-length array #1391
- BoundsEnforceLS linesearch no longer warns if bounds are not set #1393
- Added SVG icons for the N2 toolbar. #1394
- Added method signature for add_equation to the AddSubtractComp feature doc #1395
- Added recording of the top level solver abs_err and rel_err to the case recording of Problem #1398
- Added support for group approximation across distributed components in certain cases. #1402
- The `Group.promotes` method now supports specification of `src_indices`. #1408
- OpenMDAO now warns the user about out of order parallel components and recommends the use of a NonlinearBlockJac to converge the components in a parallel group. #1412
- Updated docs for options dictionary to include example for `check_valid` argument. #1417
- Added support for response indices on distributed variables. #1426
- Raise a RuntimeError rather than excluding pyOptSparseDriver from the API when pyoptsparse is not installed or not installed properly. #1427
- Added Problem source to list_cases #1428
- Integrated IPOPT into CI testing #1431
- Added CI Support for SNOPT 7.7 and PyoptSparse 2.1.0 #1435
- Also control importing of petsc4py with the use of OPENMDAO_REQUIRE_MPI #1445
- Added `out_stream` option to CaseReader list_* methods #1446
- Added tests using DOE and GA driver with distributed driver responses . #1450
- `guess_nonlinear` now works with NonlinearBlockGS #1452
- User can specify units for KSComp inputs and outputs #1455

## Bug Fixes:

- Improved several N2 resizing issues #1386
- Fixed a bug when approximating totals on a group that caused derivatives to be incorrect if a component under the group had partials declared with the 'val' argument. #1392
- Fixed N2 search autocomplete appending '.' to results #1399
- Allow partials on distributed components with zero-length inputs/outputs on some processors #1400
- Approximated totals now work for objectives in parallel groups #1404
- Distributed outputs as driver responses are now supported #1402
- The N2 home button now restores the diagram to its original state #1409
- Redundant calls no longer made to the matrix-free API when running under parallel derivative coloring (for fan-out parallel problems). #1413
- Fix for distributed components over-allocating processors #1416
- Fixed N2 toolbar icons overlapping when resized; compacted toolbar #1423
- Fixed collapsing the zoomed element's parent, as well as other collapse/search bugs #1430
- Fixed bug during (second) setup. #1447
- Fixed an out-of-order warning that should not have raised a warning #1451

**********************************
# Release Notes for OpenMDAO 3.1.0

May 01, 2020

OpenMDAO 3.1.0 is a periodic release of OpenMDAO.
This release features more uniformity in recording options of Problem, Driver, System, and Solver.
It adds more capability to the Problem recorder for when users only wish to record the final state of the model rather than iteration-by-iteration.
There have been numerous updates relating to the N2 viewer and distributed components.

## Backwards Incompatible API Changes:

- <POEM001> Updates the definition of light year (ly) to the IAU definition and adds units for astronomical units (au) and parsec (pc).  #1204
- <POEM014> Built-in XDSM viewer removed from OpenMDAO and is now supported as an external plugin. #1240
- assert_rel_error deprecated replaced with assert_near_equal. #1260 #1264
- <POEM017> Adds ability to specify units on design variables, constraints, and objectives. #1265
- <POEM013> convert_units and unit_conversion added to openmdao.api. #1267
- <POEM019> Fixed various issues with directional derivatives. #1314
- Renamed Problem.record_iteration to Problem.record. #1355
- Case reader `system_metadata` deprecated and replaced by `system_options`. #1271 #1273

## Backwards Incompatible NON-API Changes:

- N2 viewer no longer treats colons in variable names as special separators. #1275

## New Features:

- Enabled certain command-line debugging functions to work with problems embedded in tests. #1222
- Added default opt_settings for IPOPT and added a demonstration of the sellar problem using IPOPT via pyoptsparse. #1234
- compute_total_coloring can now be called with `of` and `wrt` lists rather than just on desvars and responses. #1241
- Added ability to deprecate options in OptionsDictionary. #1253
- Improved reporting of ill-conditioned Jacobians in DirectSolver. #1192
- We can use the directional derivative for check_partials derivative calculation of a matrix-free subjacobian. #1274
- DOEDriver now supports discrete variables. #1293
- Added an option 'record_derivatives' to Problem and then provided a way to access this information via a case reader. #1281
- Added --version command to the command line interface. #1300
- Added recording of inputs, outputs, and residuals to Problem. Add recording of outputs and residuals to Driver. #1311
- Raise error if DirectSolver or BroydenSolver are above a parallel group or distributed component. #1321
- Added option to KSComp to automatically add a corresponding constraint to the optimization. #1323
- set_solver_print is now a public method on System. #1325
- Added/updated docs for Group promotes() method. #1326
- Users are now prevented from adding objective or constraints on distributed components.  This is a temporary fix until a more comprehensive solution is implemented. #1333
- Bounds arrays are now only allocated when required by the linesearch, saving signficant memory for very large models. #1345
- The N2 info panel now displays both the absolute and promoted names of variables. #1309

## Bug Fixes:

- Fixed a bug where dynamic coloring was not updated when doing subsequent setup calls. #1233
- Fixed some output formatting issues in check_totals. #1258 #1259
- Fixed a bug that occurred when a jacobian was filled with an incorrect shape. #1261
- Fixed a bug in the Akima spline that resulted when using 4 points to define the spline. #1263
- Fixed a bug where a driver's response indices were being changed during total derivative computation. #1288
- Skip decorators are now correctly handled when running doc tests. #1310
- Cleaned up the feature documentation for MatrixVectorProductComp. #1320
- Fixed an MPI hang when using DirectSolver on a serial model run on multiple procs. #1329
- Fixed FD derivatives on distributed components. #1337
- Fixed bug in options passing for BSplineComp which was causing hangs. #1338
- Fixed a bug for case where src_indices are declared for a 1D source, and are given as a flat list with flat_src_indices set to False. If any entry beyond the first were out-of-bounds, no exception was being raised. #1347
- Correctly raise connection error in mpi when src_indices are out-of-bounds. #1353
- Fix for a hang when setting up a distributed comp with flat_src_indices. #1353
- Fixed a bug where driver `supports` options could be overridden by the user. #1358
- Fixed a bug in CaseRecorder that was causing an error if a variable name contained a comma. #1380
- Fixed sizing issues for the N2 diagram.
- Fixed a bug that was preventing the N2 diagram from displaying in the event of a connection error. #1194
- Fixed a missing image in the N2 diagram and updated the legend. #1324
- Fixed a bug in rendering when a selected connection contained a variable out of the current scope. #1335
- Fixed differences in rendering an N2 from a recorded file vs an in-memory model. #1299

**********************************
# Release Notes for OpenMDAO 3.0.0

March 05, 2020

OpenMDAO 3.0.0 is the first version of OpenMDAO to drop support for Python 2.7.  Going forward,
we will support Python >= 3.6.  Functionally, 3.0.0 is identical to 2.10.1 with the exception
that long-standing deprecations have been removed and will now result in errors.

When upgrading to OpenMDAO 3.0.0, users should
1. Get their models working with version 2.10.1
2. Make necessary changes to remove any OpenMDAO-specific deprecation warnings
3. Upgrade to OpenMDAO 3.0.0 and verify that their models are working as expected.


## Backwards Incompatible API Changes:

The following are deprecated functionality since 2.0 that has been officially removed as of version 3.0.0:

- <POEM 004> AkimaSplineComp and BsplineComp have been **removed**, replaced with SplineComp.

- <POEM 008> The user is now **required** to set a value for the solver option `solve_subsystems`
    on Newton and Broyden solvers. Previously the default was `False`, but you would get
    a deprecation warning if this value was not explicitly set.

- The class for the PETSc linear solver `PetscKSP` is renamed `PetscKrylov`.

- The class for the Scipy linear solver `ScipyIterativeSolver` is renamed `ScipyKrylov`.

- The `NonLinearRunOnce` solver is now `NonlinearRunOnce`.

- The `FloatKrigingSurrogate` is now `KrigingSurrogate`.

- Optimizer `ScipyOptimizer` is now `ScipyOptimizeDriver`.

- API function and CLI command `view_model` are renamed `n2`.

- Component `ExternalCode` is renamed `ExternalCodeComp`.

- Component `KSComponent` is renamed `KSComp`.

- Component `MetaModelStructured` is renamed `MetaModelStructuredComp`.

- Component `MetaModel` is renamed `MetaModelUnStructuredComp`.

- Component `MultiFiMetaModel` is renamed `MultiFiMetaModelUnStructuredComp`.

- Case attribute `iteration_coordinate` is renamed `name`.

- Solver option `err_on_maxiter` is renamed `err_on_non_converge`.

- Solver attribute `preconditioner` is renamed `precon`.

- Nonlinear solver option `reraise_child_analysiserror` now defaults to False.

- BroydenSolver and NewtonSolver linesearch now defaults to BoundsEnforceLS.

- OptionsDictionary option argument `_type` is renamed `types`.

- Component attribute `distributed` has been changed to an option.

- The attribute `root` of Problem is renamed `model`.

- Group attribute `nl_solver` is renamed `nonlinear_solver`.

- Group attribute `ln_solver` is renamed `linear_solver`.

- System attribute `metadata` is renamed `options`.

- `units = 'unitless'` is removed.  Use `units = None` instead.

- NewtonSolver attribute `line_search` is now `linesearch`.

- Multi-variable specification via list or tuple to IndepVarComp is now invalid. Use add_output for each variable instead.

- Driver method `set_simul_deriv_color` is removed. Use `use_fixed_coloring` instead.

- Problem setup method argument `vector_class` removed. Use `distributed_vector_class` instead.

- pyOptSparseDriver method `dynamic_simul_derivs` removed. Use `declare_coloring` instead.

## Backwards Incompatible NON-API Changes:

Note: These changes are not technically API related, but
still may cause your results to differ when upgrading.

- <POEM 012> The old implementation of KrigingSurrogate was hard coded to the
  `gesdd` method for inverses. The method is now an option, but we've changed
  the default to `gesvd` because its more numerically stable.
  This might introduce slight numerical changes to the Kriging fits.

## New Features:

- None


## Bug Fixes:
- None

***********************************
# Release Notes for OpenMDAO 2.10.1

March 03, 2020

## Backwards Incompatible API Changes:

None

## Backwards Incompatible NON-API Changes:

None

## New Features:

None

## Bug Fixes:
- Fixed a bug that caused errors in jacobian values in certain models when computing derivatives in reverse mode under MPI.

***********************************
# Release Notes for OpenMDAO 2.10.0

February 27, 2020

Note: This is the last release of OpenMDAO 2.X. Updating to this release will be a 
critical step in preparing for the OpenMDAO 3.X releases


## Backwards Incompatible API Changes: 

- <POEM 004> AkimaSplineComp and BsplineComp have been deprecated, replaced with SplineComp

- <POEM 008>
  - The user is now required to set a value for the solver option `solve_subsystems` 
    on Newton and Broyden solvers. Previously the default was `False`, but you will now get 
    a deprecation warning if this value is not explicitly set. 

- Specification of `includes` and `excludes` for solver case recording options 
  has been changed to use promoted names relative to the location of that solver in the model. 

## Backwards Incompatible NON-API Changes: 

Note: These changes are not technically API related, but 
still may cause your results to differ when upgrading. 

- <POEM 012> The old implementation of KrigingSurrogate was hard coded to the 
  `gesdd` method for inverses. The method is now an option, but we've changed 
  the default to `gesvd` because its more numerically stable. 
  This might introduce slight numerical changes to the Kriging fits. 

- Refactor of how reverse mode derivative solves work for parallel components 
  that may require the removal of reduce calls in some components that were 
  needed previously 

- Change to the way SimpleGA does crossover to be more correct (was basically just mutuation before)


## New Features:

- <POEM 002> users can now manually terminate an optimization using operating system signals 

- <POEM 003> 
  - users can now add I/O, connections, and promotions in configure 
  - new API method added to groups to allow promotions after a subsystem has already been added

- <POEM 004> SplineComp was added to the standard component library, 
  and you can also import the interpolation algorithms to use in your own components

- <POEM 005> OpenMDAO plugin system has been defined, so users can now publish their own plugins on github. 
  several utilities added to openmdao command line tools related to plugins

- <POEM 007> ExternalCodeComp and ExternalCodeImplicitComp can now accept strings 
  (previously a lists of strings was required) for `command`, `command_apply`, and `command_solve`. 

- <POEM 010> User can now mark some options as `recordable=False` as an alternative to 
  using `system.recording_options['options_exclude']`

- <POEM 012> KrigingSurrogate now has a new option `lapack_driver` to determine how an inverse is computed
  The original implementation used a default `gesdd` method, but we've changed to `gesvd`.

- Slight improvements to the metamodel viewer output

- Improvement to the bidirectional Jacobian coloring so it will fall back to 
  pure forward or reverse if thats faster

- Users can now set the shape for outputs of `BalanceComp` and `EqConstraintsComp`

- Improvements to `debug_print` for driver related to reporting which derivatives 
  are being computed when coloring is active

- The N2 viewer now works for models that are run in parallel (i.e. if you're using mpi to call python)

- `list_inputs` and `list_outputs` both have arguments for including local_size and 
  global_size of distributed outputs in their reports


## Bug Fixes: 
- Fixed a hang in SimpleGA driver related to mpi calls 
- Fixed a bug in SimpleGA related to setting a vector of constraints
- Various bug-fixes to allow running with Python 3.8
- Fixed a bug in ScipyOptimizer/Cobyla related to objective and dv scaling
- Corrected bug preventing the user from passing a list to `set_val` when the variable is an array

***********************************
# Release Notes for OpenMDAO 2.9.1

October 8, 2019

2.9.1 is a quick patch release to correct a couple of issues:

## Bug Fixes:
- The previous release inadvertantly changed the default "bounds_enforce" option on the BoundsEnforce
  line search to "vector".  This release restores the default to the originally intended "scalar".
- The entry point for the `web_view` utility function was fixed. (Thank you Nicholas Bollweg)
- Fixed a bug that prevented model reconfigurability in certain situations.
- Fixed a missing `sqrt` in the documentation for the Hohmann example.
- Fixed a bug that prevented proper fallback from bidirectional coloring to unidirectional coloring
  in certain cases where the bidirectional coloring was worse than the unidirectional coloring.

## New Features:
- Small improvements were made to the MetaModel viewer, along with some additional testing.


***********************************
# Release Notes for OpenMDAO 2.9.0

October 1, 2019

## New Features:

- better detection of when you're running under MPI, and new environment variable to forcibly disable MPI
- users can now issue connections from within the `configure` method on group 
  (this is useful if you wanted to be able to interrogate your children before issuing connections)
- significant overhead reduction due to switching from 'np.add.at' to 'np.bincount' for some matrix-vector
  products and data transfers.

- `openmdao` command line tool: 
    - new `openmdao scaffold` command line tool to quickly bootstrap component and group files
    - `openmdao summary` command gives more data including number of constraints
    - `openmdao tree` command now uses different colors for implicit and explicit components and 
      includes sizes of inputs and outputs
    - the `openmdao --help` tool gives better formatted help now

- CaseRecording: 
    - Case objects now have a `name` attribute to make identifying them simpler
    - Case objects now support __getitem__ (i.e. <case>[<some_var>]) and the
      get_val/set_val methods to directly mimic the problem interface
    - list_inputs and list_outputs methods on Case objects now support optional includes and excludes patterns     

- Documentation:    
    - added documentation for the list_inputs and list_outputs method on the Case object
    - added example use of implicit component for add_discrete_input docs
    - some minor cleanup to the cantilever beam optimization example to remove some unused code

- Derivatives: 
    - slightly different method of computing the total derivative sparsity pattern has been implemented
      that should be a bit more robust (but YMMV)
    - the partials object passed into `linearize` and `compute_partials` methods on implicit
      and explicit component respectively can now be iterated on like a true dictionary

- Solvers: 
    - a true Goldstein condition was added to the ArmijoGoldstein line search 
      (you can now choose if you want Armijo or Goldstein) *** Peter Onodi ***
    - DirectSolver now works under MPI 
      (but be warned that it might be very slow to use it this way, because it does a lot of MPI communication!!!)
    - BroydenSolver now works in parallel
    - improved error message when you have a nonconverged solver 
    - improved error message when solver failure is due to Nan or Inf

- Components: 
    - list_inputs and list_outputs methods on System objects now support optional includes and excludes patterns
    - variables can now have tags added as additional meta-data when they are declared,
      which can then be used to filter the output from list_inputs and list_ouputs 
    - significant speed improvements and increased surrogate options for StructuredMetaModel
    - improved error message when required options are not provided to a component during instantiation
    - user gets a clear error msg if they try to use `np.` or `numpy.` namespaces inside ExecComp strings
    - in the declare_coloring method, a threshold can now be set such that the generated coloring must improve
      things at least as much as the given percentage or the coloring will not be used. This can happen in cases
      where multiple instances of the same component class appear in a given model and each instance can potentially
      generate a different sparsity which results in a different coloring.

- Problem
    - user can set specific slices of variables using the set_val method on problem
    - added problem.compute_jacvec_prod function for matrix-free total derivative products. 
      Useful for wrapping for loops around an OpenMDAO problem and propagating derivatives through it.

- Visualization
    - minor reformatting of the N2 menu to make better use of vertical screen space
    - new visualziation tool for examining the surrogates in StructuredMetaModelComp and UnstructuredMetaModelComp
    - improved `openmdao view_connections` functionality for better filtering and toggling between 
      promoted and absolute names
    - component names can now optionally be shown on the diagonal for XDSM diagrams *** Peter Onodi ***

## Backwards Incompatible API Changes: 
- `openmdao view_model` has been deprecated, use `openmdao n2` instead
- `vectorize` argument to ExecComp has been deprecated, use `has_diag_partials` instead
- changed NewtonSolver option from `err_on_maxiter` to `err_on_non_converge` (old option deprecated)
- in StructuredMetamodel, original methods 'slinear', 'cubic', 'quintic' renamed to 'scipy_linear', 'scipy_cubic', 'scipy_quintic';
  new methods 'linear' and 'cubic' have been added that are faster python implementations. 'quintic' is no longer used.

## Bug Fixes:
- `prom_name=True` now works with list_outputs on CaseReader (it was broken before)
- fixed a corner case where N2 diagram wasn't listing certain variables as connected, even though they were
- fixed a bug in check_totals when the FD-norm is zero 
- fixed corner case bug where assembled jacobians for implicit components were not working correctly
- fixed a problem with Aitken acceleration in parallel for NonlinearBlockGaussSeidel solver
- fixed DOE driver bug where vars were recorded in driver_scaled form.
- discrete variables are now recorded
- list_outputs/list_inputs on a Case now lists in Execution order, similarly to the System methods
- fixed a bug where the derivatives calculated with complex step around a model with a BroydenSolver were wrong.
- fixed a bug in the coloring config check

***********************************
# Release Notes for OpenMDAO 2.8.0

June 27, 2019

## Bug Fixes:
- Fixed a bug in PETScVector norm calculation, which was totally wrong for serial models combined
  with distributed ones 
- Fixed a bug with the solver debug_print option when output file already existed
- Fixed the incorrect Shockley diode equation in the Circuit example 
- Fixed a few small bugs in group level FD  

## New Features:
- Stopped reporting a warning for a corner case regarding promoted inputs that are connected inside
  their owning group, because everyone hated it! 
- Optional normalization of input variables to multifi_cokriging 
- New matplotlib based sparsity matrix viewer for coloring of partial and total derivatives 
- Preferred import style now changed to `import openmdao.api as om`
- Discrete variables now show up when calling the list_input/list_output functions 
- Discrete variables now show up in view_model 
- ScipyOptimizeDriver now raises an exception when the objective is missing
- A legend can be optionally added to the XDSM diagram. Not shown by default.
  *** contributed by Peter Onodi ***

***********************************
# Release Notes for OpenMDAO 2.7.1

May 30, 2019

2.7.1 is a quick patch release to correct/update the 2.7.0 release notes.

***********************************
# Release Notes for OpenMDAO 2.7.0

May 28, 2019

## New Features:

- You can now define guess_nonlinear method at the group level
- New documentation added about the N2 diagram usage
- Significant improvement to documentation search functionality
  (by default, only searches the feature docs and user guide now)

- Derivatives: 
    - Improved support for full-model complex-step when models have guess_nonlinear methods defined
    - **Experimental** FD and CS based coloring methods for partial derivative approximation 
      Valuable for efficiently using FD/CS on vectorized (or very sparse) components

- Solvers: 
    - `Solver failed to converge` message now includes solver path name to make it more clear what failed
    - Improved pathname information in the singular matrix error from DirectSolver
    - Directsolver has an improved error message when it detects identical rows or columns in Jacobian
    - NonlinearGaussSeidel solver now accounts for residual scaling in its convergence criterion
    - New naming scheme for solver debug print files (the old scheme was making names so long it caused OSErrors)

- Components: 
    - ExecComp now allows unit=<something> and shape=<something> arguments that apply to all variables in the expression
    - New AkimaSpline component with derivatives with respect to training data inputs

- Visualization
    - Several improvements for the N2 diagram for large models
    - N2 diagram html files have been reduced in size significantly
    - `openmdao view_model` command line utility now supports case record database files

- (Experimental) Automatic XDSM generator (using either Latex with pyXDSM or html with XDSMjs)
  *** contributed by Peter Onodi ***
  *** uses XDSMjs v0.6.0 by Rémi Lafage (https://github.com/OneraHub/XDSMjs) ***

## Backwards Incompatible API Changes: 
- New APIs for total derivative coloring that are more consistent with partial derivative coloring
  (previous APIs are deprecated and coloring files generated with the previous API will not work)
- The API for providing a guess function to the BalanceComp has changed. 
  guess_function is now passed into BalanceComp as an init argument
- Changed the N2 diagram json data formatting to make the file size smaller 
  You can't use older case record databases to generate an N2 diagram with latest version
- All component methods related to execution now include `discrete_inputs` and `discrete_outputs` arguments 
  when a component is defined with discrete i/o. (if no discrete i/o is defined, the API remains unchanged)
  (includes `solve_nonlinear`, `apply_nonlinear`, `linearize`, `apply_linear`, `compute`, `compute_jac_vec_product`, `guess_nonlinear`) 
- The internal Driver API has changed, a driver should execute the model with `run_solve_nonlinear` to ensure that proper scaling operations occur 

## Bug Fixes:
- CaseRecorder was reporting incorrect values of scaled variables (analysis was correct, only case record output was wrong)
- the problem level `record_iteration` method was not properly respecting the `includes` specification
- ExecComp problem when vectorize=True, but only shape was defined. 
- Incorrect memory allocation in parallel components when local size of output went to 0
- Multidimensional `src_indices` were not working correctly with assembled Jacobians
- Fixed problem with genetic algorithm not working with vector design variables *** contributed by jennirinker ***
- Fixed incompatibility with mpich mpi library causing "PMPI_Allgather(945).: Buffers must not be aliased" error *** fzhale and nbons ***

***********************************
# Release Notes for OpenMDAO 2.6.0

February 22, 2018

## New Features:
- MetaModelStructured will detect NaN in requested sample and print a readable message.
- ScipyOptimizeDriver now supports Hessian calculation option for optimizers that use it.
- User can specify src_indices that have duplicates or have two inputs on a single component connected to the same output, and still have CSC Jacobian work.
- User can get/set problem values in a straightforward manner: prob['var'] = 2., etc. even when running in parallel.
    - Problem 'get' access to distributed variables will raise an exception since it isn't clear what behavior is expected,
      i.e. should prob['comp.x'] return the full distributed variable or just the local part.
- Directsolver has an improved error message when it detects identical rows or columns in Jacobian.
- The NonlinearBlockGS solver has been updated with a less expensive implementation that does not call the compute method of Explicit components as many times.
- User can request a directional-derivative check (similar to SNOPTs built-in level 0 check) for check_partials.
- check_partials with compact_print will now always show all check pairs.
- The N^2 diagram (from `openmdao view_model`) now shows the solver hierarchy.
- KSComp has an improved, vectorized implementation.
- list_inputs now includes a 'shape' argument.
- User can generate an XDSM from a model (`openmdao xdsm`).
- User can set `units=<something>` for an execcomp, when all variables have the same units.

## Backwards Incompatible API Changes:
- The default bounds enforcement for BoundsEnforceLS is now 'scalar' (was 'vector')
- Direct solver will now use 'assemble_jac=True' by default
- Recording options 'includes' and 'excludes' now use promoted names for outputs (absolute path names are still used for inputs)
- FloatKrigingSurrogate has been deprecated since it does not provide any unique functionality.
- FloatMultifiKrigingSurrogate has been deleted because it was never used, was incorrectly implemented, and provides no new functionality.

##  Bug Fixes:
- Complex-step around Newton+DirectSolver now works with assembled Jacobians.
- Armijolinesearch implementation was incorrect when used with solve_subsystems. The implementation is now correct.

***********************************
# Release Notes for OpenMDAO 2.5.0

October 31, 2018

## New Features:
- list_outputs() method now includes a `prom_name` argument to include the promoted name in the printed output (Thanks John Jasa!).
- N2 viewer now includes tool tip (hover over) showing the promoted name of any variable.
- Improved error msg when building sparse partials with duplicate col/row entries.
- You can now build the docs without MPI/PETSc installed (you will get warnings, but no longer errors).
- Major internal refactor to reduce overhead on compute_totals calls (very noticeable on large design spaces but still cheap problems).
- Components now have a `under_complex_step` attribute you can check to see if complex-step is currently active.
- Components `distributed` attribute has been moved to an option (old attribute has been deprecated).
- MetaModelUnstructured will now use FD for partial derivatives of surrogates that don't provide analytic derivatives (user can override the default settings if they wish to use CS or different FD config).
- Improvements to SimpleGA to make it more stable, and added support constraints via penalty functions (Thanks madsmpedersen and onodip).
- Parallel FD and CS at the component and group level is now supported.
- Can turn off the analytic derivative sub-system if it is not needed via an argument to setup().
- Derivative coloring now works for problems that run under MPI.

- New Components in the standard library:
    - Mux and Demux components.

- New CaseRecording/CaseReading Features:
    - DesVar AND output variable bounds are both reordered now.
    - Improved error msg if you try to load a non-existent file with CaseReader.

- **Experimental Feature**: Discrete data passing is now supported as an experimental feature... we're still testing it, and may change the API!

## Backwards Incompatible API Changes:
-- `get_objectives` method on CaseReader now returns a dict-like object.
- Output vector is now locked (read-only) when inside the `apply_nonlinear` method (you shouldn't have been changing it then anyway!).
- default step size for complex-step has been changed to 1e-40.
- Moderate refactor of the CaseReader API. It is now self-consistent.

***********************************
# Release Notes for OpenMDAO 2.4.0

August 1, 2018

## New Features:
- Better error message when upper and lower have the wrong shape for add_design_var and add_constraint methods.
- pyOptSparseDriver now runs the initial condition for ALPSO and NSGA-II optimizers.
- Normalization in EQConstraintComp and BalanceComp is now optional.

- New Components in the standard library:
    - VectorMagnitudeComp

- New Solvers in the standard library:
    - BroydenSolver

- New CaseRecording/CaseReading Features:
    - <Problem>.load_case() method lets you pull values from a case back into the problem.
    - Updated the Case recording format for better performance.
    - User can call <Problem>.record_iteration() to save specific cases.
      Recording options for this method are separate from driver/solver options.
      This is so you can record one (or more) cases manually with *ALL* the variables in it if you want to.

## Backwards Incompatible API Changes:
- The input and output vectors are now put into read-only modes within certain component methods.
  NOTE: This REALLY REALLY should not break anyone's models, but it is technically a backwards-incompatible change...
        If you were changing values in these arrays when you shouldn't have been, you're going to get an error now.
        Fix it... the new error is enforcing correct behavior.
        If it was working before, you got lucky.

## Bug Fixes:
- Various small bug fixes to check_partials and check_totals.
- ArmijoGoldstein linesearch iteration counter works correctly now.
- the record_derivatives recorder_option now actually does something!

***********************************
# Release Notes for OpenMDAO 2.3.1

June 20, 2018


2.3.1 is a quick patch release to correct/update a few release-supporting documents such as the 2.3.0 release notes.
However, a few new features did manage to sneak in:


## New Features:
- Support for simultaneous derivatives in reverse mode.
- Added a '-m' option (possible values 'fwd' and 'rev') to the 'openmdao simul_coloring' command to allow coloring in either direction.
- Users are now warned during 'check_config' if no recorders are defined.

***********************************
# Release Notes for OpenMDAO 2.3.0

June 12, 2018

## New Features:
- Drivers have new `debug_print` option that writes design vars, objective, and constraints to stdout for each iteration.

- Added a DOEDriver for serial and parallel cases to the standard library.
    (For people who had adopted DOEDriver early, via github, the following things have changed since initial implementation:)
    - The option that controls the parallel behavior for DOEDriver was changed to `procs_per_model` from `parallel`.
    - User can provide a CSV file to the DOEDriver to run an arbitrary set of cases.

- Solvers have new `debug_print` option that will report the initial condition of the solver if it fails to converge.

- SimpleGADriver changes:
    - Will now compute a population size based on 4*sum(bits) + 1 if pop_size=0.
    - Can support models that need more than one processor.

- Problem changes:
    - New Problem methods `<problem>.get_val()` and `<problem>.set_val()` allow the user to specify a unit, and we'll convert to the correct unit (or throw an error if the unit is not compatible).
    - `list_problem_vars()` method will report all design vars, objectives, and constraints.

- OpenMDAO command line utility now works under MPI.
- ExternalCodeComp has new option, `allowed_return_codes` to specify valid return codes.
- Components that request complex-step for partial derivative checks will now fall back to finite-difference if `force_alloc_complex=False`.
- You can now pass a case_prefix argument to run_model or run_driver to name the cases for case recording.
- Beam example problem now runs 20x faster.

- New Analytic Derivatives Features
    - Major re-write of the Feature documentation on the Core Feature, "Working with Analytic Derivatives."
    - User can now set AssembledJacobians for linear solvers. (Big speedup for a lot of lower-order or serial problems).
    - Automatic calculation of total Jacobian sparsity for fwd derivatives for optimizers that support it (lower compute cost when using SNOPT).
    - Automatic simultaneous coloring algorithm for separable problems to speed up derivatives calculations.
    - Option for caching of linear solution vectors to speed up linear solves when using iterative solvers.
    - check_partials now allows you to use `includes` and `excludes` to limit which components get checked.

- New Components in the standard library:
    - CrossProductComp
    - DotProductComp
    - MatrixVectorProductComp
    - ExternalCodeImplicitComp
    - EQConstraintComp

- Updates to existing Components in standard library:
    - LinearSystemComp has been vectorized.
    - KSComp has been vectorized.
    - BsplineComp has been vectorized.

- New CaseRecording/CaseReading Features:
    - CaseRecorder now saves metadata/options from all groups and components.
    - CaseReader now supports `list_inputs` and `list_outputs` methods.
    - User can iterate over cases from a CaseReader in either flat or hierarchical modes.
    - User can re-load a case back into a model via the `load_case()` method.


## Backwards-Compatible API Changes:
- Changed all instances of `metadata` to `options`. `<system>.metadata` has now been deprecated.

- Class name changes to Components, so all end in "Comp" (old class names have been deprecated):
    - MetaModelStructured -> MetaModelStructuredComp
    - MetaModelUnStructured -> MetaModelUnStructuredComp
    - MultiFiMetaModelUnStructured -> MultiFiMetaModelUnStructuredComp
    - ExternalCode -> ExternalCodeComp

- Driver metadata/options can now be set via init args (can still be set as options as well).
- User no longer needs to specify PETSCvector for parallel cases (the framework figures it out now).

- check_totals and check_partials no longer support the `supress_output` argument. Use `out_stream=None` instead.


## Backwards-Incompatible API Changes:
- `comps` argument in check_partials was replaced with `includes` and `excludes` arguments.
- Re-design of the CaseReader API. Unification of all cases, and simpler iteration over case hierarchies.
- `partial_type` argument for LinearSystemComponent has been removed
- `maxiter` option from BoundsEnforceLS has been removed

## Bug Fixes:
- Fixed bug in component-level FD that caused it to call the model too many times.
- Added proper error check for situation when user creates a variable name containing illegal characters.
- Better error msg when adding objectives or constraints with variables that don't exist under MPI execution.
- Vectorization for MetaModelUnStructuredComp was super-duper broken. It's fixed now.


***********************************
# Release Notes for OpenMDAO 2.2.1

April 2, 2018

## New Features:
- check_partials() improvements to formatting and clarity. Report of potentially-bad derivatives summarized at bottom of output.
- check_partials() only compares fwd analytic to FD for any components that provide derivatives directly through the Jacobian
    argument to compute_partials or linearize. (significantly less output to view now).
- Docs for UnstructuredMetaModel improved.
- pyoptsparse wrapper only calls run_model before optimization if LinearConstraints are included.
- ScipyOptimizerDriver is now smarter about how it handles linear constraints. It caches the derivatives and doesn't recompute them anymore.
- Docs for ExternalCode improved to show how to handle derivatives.
- cache_linear_solution argument to add_design_var, add_constraint, add_objective, allows iterative linear solves to use previous solution as initial guess.
- New solver debugging tool via the `debug_print` option: writes out initial state values, so failed cases can be more easily replicated.
- Added generic KS component.
- Added generic Bspline component.
- Improved error msg when class is passed into add_subsystem.
- Automated Jacobian coloring algorithm now works across all variables (previously, it was just local within a variable).
- Major refactor of the `compute_totals` method to clean up and simplify.

## Backwards-Compatible API Changes:
N/A

## Backwards-Incompatible API changes:
N/A

## Bug Fixes:
- compute_totals works without any arguments now (just uses the default set of des_vars, objectives, and constraints)
- UnstructuredMetaModel can now be sub-classed
- Deprecated ScipyOptimizer class wasn't working correctly, but can now actually be used.

***********************************
# Release Notes for OpenMDAO 2.2.0

February 9, 2018

## New Features:
- `DirectSolver` now tells you which row or column is singular when it gets a singluar matrix error.
- `ScipyOptimizeDriver` now handles linear constraints more efficiently by only computing them one time.
- Added the `openmdao` command line script to allow for model checking, visualization, and profiling without making modifications to the run script.
- Added a `SimpleGADriver` with a basic genetic algorithm implementation.
- Added a `MetaModelStructured` component with a interpolative method.
- New option for derivative calculations: Simultaneous derivatives, useful when you have totally disjoint Jacobians (e.g. diagonal Jacobians).
- Automatic coloring algorithm added to compute the valid coloring scheme for simultaneous derivatives.
- `list_outputs` method updated with new display options, ability to filter variables by residual value, ability to change sorting scheme, and ability to display unit details.
- openmdao citation helper added to the `openmdao` command line script, making it easy to generate the correct bibtex citations based on which classes are being used.
- `NewtonSolver` modified so that maxiter=0 case now will compute residuals, but not do a linear solve (useful for debugging nonlinear errors).

## Backwards-Compatible API Changes:
- Changed `ScipyOptimizer` to `ScipyOptimizeDriver` for consistency (deprecated older class).
- Renamed `MetaModel` to `MetaModelUnstructured` to allow for new structured interpolant (deprecated old class).
- Renamed `PetscKSP` to `PETScKrylov` for consistency. (deprecated old class).
- Renamed `ScipyIterativeSolver` to `ScipyKrylov` for consistency. (deprecated old class).


## Backwards-Incompatible API changes:
- CaseRecorder now uses variables' promoted names for storing and accessing data.
- Removed `DeprecatedComp` from codebase.
- `list_residuals` method on Groups and Components removed.

## Bug Fixes:
- Fixed error check for duplicate connections to a single input from multiple levels of the hierarchy

***********************************
# Release Notes for OpenMDAO 2.1.0

December 7, 2017

## New Features:
- Configure setup hook allowing changing of solver settings after hierarchy tree is instantiated
- Component metadata system for specifying init_args with error checking
- Parallel Groups
- Units Reference added to the Docs
- Case recording now records all variables by default
- `openmdao` console script that can activate useful debugging features
  (e.g. view_model) without editing the run script
- Scipy COBYLA optimizer converts des var bounds to constraints (the algorithm doesn't natively handle bounds)
- StructuredMetaModel component offers a simple spline interpolation routine for structured data

## Backwards Compatible API Changes:
- `NonlinearRunOnce` changed `NonLinearRunOnce` for consistency (old class deprecated)
- `types_` argument to `self.metadata.declare` changed to `types`. (old argument deprecated)
- `types` and `values` arguments to `self.metadata.declare`
-  `BalanceComp` has a `use_mult` argument to control if it has a `mult` input, defaulting to false
   (the mult input isn't used most of the time)
- Renamed `MetaModel` to `UnstructuredMetaModel` and `MultiFiMetaModel` to `UnStructuredMultiFiMetaModel`


## Backwards Incompatible API Changes:
- Case Recording options API updated with `.recording_options` attribute on Driver, Solver, and System classes
- `get_subsystem` changed to a private method, removed from public API of System
- `check_partials` now has a `method` argument that controls which type of check

## Bug Fixes:
- Improved error msg on a corner case for when user doesn't declare a partial derivative
- Fixed docs embedding bug when `<>` included in the output text


***********************************
# Release Notes for OpenMDAO 2.0.2
October 19, 2017

- Fixing further packaging errors by updating packaging information.
- Added tutorials for derivatives of explicit and implicit components.


***********************************
# Release Notes for OpenMDAO 2.0.1
October 19, 2017

- Attempting to fix test errors that were happening by updating the packaging information.


***********************************
# Release Notes for OpenMDAO 2.0.0
October 19, 2017

First public release of 2.0.0.


## New Features:
(first release... so EVERYTHING!)
- Drivers:
    - ScipyOptimizer
    - PyoptSparseOptimizer
- Solvers:
    - Nonlinear:
        - NonlinearBlockGS
        - NonlinearBlockJac
        - NonlinearRunOnce
        - NewtonSolver
    - Linear:
        - LinearBlockGS
        - LinearBlockJac
        - LinearRunOnce
        - DirectSolver
        - PetscKSP
        - ScipyIterativeSolver
        - LinearUserDefined
    - LineSearch
        - AmijoGoldsetinLS
        - BoundsEnforceLS
- Components:
    - MetaModel
    - ExecComp
    - BalanceComp
    - LinearSystemComp

- General Capability:
    - Implicit and Explicit Components
    - Serial and Parallel Groups
    - Parallel distributed components (components that need an MPI comm)
    - Unit conversions
    - Variable scaling
    - Specifying component metadata
    - Analytic derivatives


##  Currently Missing Features that existed in 1.7.3:
- Pass-by-object variables (anything that is not a float)
- File Variables
- automatic ordering of groups/components based on connections
- Design of Experiments Driver (DOE)
- Parallel Finite Difference
- File-Wrapping utility library & External Code Component
- Approximate active set constraint calculation skipping
- Brent Solver
- CaseRecorders for CSV, HDF5, and Dump formats

##  Bug Fixes
N/A
