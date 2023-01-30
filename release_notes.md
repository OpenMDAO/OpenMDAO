***********************************
# Release Notes for OpenMDAO 3.25.0

January 27, 2023

OpenMDAO 3.25.0 includes only one change, which is the convention OpenMDAO uses when transferring data between distributed and non-distributed variables. The underlying principle is that serial variables and their derivatives must have consistent values across all ranks where those variables exist.

This is a backwards incompatible change that could break some matrix-free components with a mix of distributed and non-distributed variables.  Users developing components involving distributed inputs should consult [POEM 075](https://github.com/OpenMDAO/POEMs/blob/master/POEM_075.md).

## New Deprecations

- None

## Backwards Incompatible API Changes

- **POEM 75** implementation: changes the convention OpenMDAO uses when transferring data between distributed and non-distributed variables. [#2751](https://github.com/OpenMDAO/OpenMDAO/pull/2751)

## Backwards Incompatible Non-API Changes

- None

## New Features

- None

## Bug Fixes

- None

## Miscellaneous

- None


***********************************
# Release Notes for OpenMDAO 3.24.0

January 25, 2023

OpenMDAO 3.24.0 serves as a transitional release that marks a change in the way distributed I/O is handled.
Moving forward from 3.25.0 onward, users developing components involving distributed inputs should consult
[POEM 075](https://github.com/OpenMDAO/POEMs/blob/master/POEM_075.md), as OpenMDAO is changing this convention
and the switch will not be backwards compatible.

For the N2 diagram, we've added information about connections between systems when a connection node is highlighted in
NodeInfo mode.

## New Deprecations

- **POEM 75** implementation: Problems involving systems with distributed inputs will raise a deprecation regarding upcoming changes to their behavior. [#2784](https://github.com/OpenMDAO/OpenMDAO/pull/2784)

## Backwards Incompatible API Changes

- None

## Backwards Incompatible Non-API Changes

- None

## New Features

- N2: Connections displayed during mouseover of a connection node when in NodeInfo mode [#2778](https://github.com/OpenMDAO/OpenMDAO/pull/2778)
- Added set_val as a method of System. [#2785](https://github.com/OpenMDAO/OpenMDAO/pull/2785)

## Bug Fixes

- Fixed a bug in the handling of design variable bounds conditions when using COBYLA. [#2770](https://github.com/OpenMDAO/OpenMDAO/pull/2770)
- Fixed a bug in System.get_io_metadata() that caused the `discrete` property to always be shown as `True`. [#2771](https://github.com/OpenMDAO/OpenMDAO/pull/2771)
- Fixed a bug that prevented `Problem.set_solver_print` from impacting output in multi-run scenarios [#2773](https://github.com/OpenMDAO/OpenMDAO/pull/2773)
- Fixed a bug in `set_output_solver_options`, `set_design_var_options`, and `set_constraint_options` that prevented them from working when given vector values. [#2782](https://github.com/OpenMDAO/OpenMDAO/pull/2782)

## Miscellaneous

- Fixed a bug in our docstring linting so that now we can detect class attributes that aren't created in __init__. [#2780](https://github.com/OpenMDAO/OpenMDAO/pull/2780)


***********************************
# Release Notes for OpenMDAO 3.23.0

January 10, 2023

OpenMDAO 3.23.0 fixes a few bugs and and provides functionality with numpy 1.24, which removed several previously deprecated features.

POEM 79 is implemented. This will cause a warning to be issued if the initial value of a design variable is outside of the bounds it was given.
Previously, this behavior was handled differently by different optimizers, with `IPOPT` clipping the values to lay within the bounds, while other optimizers just silently proceeded starting from an invalid design point.
More often than not, setting design variables to invalid values is an oversight by the user and so they will receive a warning in the current release if such a condition is found.
This warning will be changed to an exception in OpenMDAO 3.25, but the user will continue to have the ability to choose whether OpenMDAO warns, raises, or ignores the condition.

We've also addressed several bugs that were found by users, so please continue to submit those issues.

## New Deprecations

- **POEM 79** implementation: Warning issued for design variables whose initial value exceeds their bounds. This will become an exception in OpenMDAO 3.25. [#2747](https://github.com/OpenMDAO/OpenMDAO/pull/2747)

## Backwards Incompatible API Changes

- None

## Backwards Incompatible Non-API Changes

- None

## New Features

- **POEM 79** implementation: Warning issued for design variables whose initial value exceeds their bounds. This will become an exception in OpenMDAO 3.25. [#2747](https://github.com/OpenMDAO/OpenMDAO/pull/2747)
- Updates for numpy 1.24 removed features. [#2738](https://github.com/OpenMDAO/OpenMDAO/pull/2738)

## Bug Fixes

- Fixed a bug with units check when design variable is specified in a Component. [#2742](https://github.com/OpenMDAO/OpenMDAO/pull/2742)
- Fixed a bug regarding size 0 arrays in inputs report. [#2743](https://github.com/OpenMDAO/OpenMDAO/pull/2743)
- Fixed an errant warning when regarding non-existent report hooks. [#2744](https://github.com/OpenMDAO/OpenMDAO/pull/2744)
- The optimization report should now correctly compute the min/max value for discrete desvars. [#2746](https://github.com/OpenMDAO/OpenMDAO/pull/2746)

## Miscellaneous

- Added skipUnless to a couple distributed recording tests that need pyDOE2 [#2733](https://github.com/OpenMDAO/OpenMDAO/pull/2733)
- Temporarily ignored GitPython vulnerability in audit [#2734](https://github.com/OpenMDAO/OpenMDAO/pull/2734)

***********************************
# Release Notes for OpenMDAO 3.22.0

December 14, 2022

OpenMDAO 3.22.0 contains a variety of new user-facing capabilities and bug fixes.
Here are some of the highlights:

We've removed our dependency on the tabulate package.
The dependencies on pyDOE2 and pyparsing are no longer required unless the user
attempts to use an OpenMDAO feature that requires them. In this case, that means
the DOEDriver or the external code file-wrapping capabilities, respectively.

OpenMDAO will now raise an exception if a single solver instance is attached to multiple systems.

We've implemented several [POEMs](https://github.com/OpenMDAO/POEMs) in this release.
POEM 74 is implemented, and OpenMDAO will suggest closely-matching connection targets if you happen to misspell it during the `connect` call.
POEM 70 adds a new _inputs report_ (`inputs.html`) that allows the user to quickly view the available inputs in a model, see which are ultimately connected to IndepVarComps, and to see which are design variables controlled by the Driver.
POEM 69 allows users to provide more clear names for the residuals associated with implicit outputs, rather than assigning them the same name.
Users were sometimes confused that the existing implementation seemed to associate some residuals specifically with some outputs, when in reality it often just matters that a solver be given `N` implicit outputs and `N` corresponding residuals.

## New Deprecations

- Setting `prob.model` to a component is now deprecated, as the technical burden of supporting this corner case has outweighed its usefulness. The `model` assigned to a Problem should now always be a Group.

## Backwards Incompatible API Changes

- None

## Backwards Incompatible Non-API Changes

- None

## New Features

- **POEM 70** implementation: Added the inputs report, and a generate_table function to eliminate dependency on the tabulate package. [#2655](https://github.com/OpenMDAO/OpenMDAO/pull/2655)
- Added a command line test for the scaling report. [#2657](https://github.com/OpenMDAO/OpenMDAO/pull/2657)
- Changed ImplicitComponent.apply_nonlinear() to raise NotImplementedError [#2664](https://github.com/OpenMDAO/OpenMDAO/pull/2664)
- Changed check_partials to skip components with no outputs or no inputs [#2667](https://github.com/OpenMDAO/OpenMDAO/pull/2667)
- OpenMDAO will now consider relevance information when running _linearize. As a result of this change, a component with finite differenced or complex stepped partials will not compute them for inputs that do not contribute to the model relevancy (as measured from the model's objectives and constraints to the design variables.) [#2675](https://github.com/OpenMDAO/OpenMDAO/pull/2675)
- **POEM 74** implementation: Suggest variables for failed connection [#2681](https://github.com/OpenMDAO/OpenMDAO/pull/2681)
- Made pyDOE2 an optional dependency. [#2689](https://github.com/OpenMDAO/OpenMDAO/pull/2689)
- Added support for constraint alias in the case reader. [#2698](https://github.com/OpenMDAO/OpenMDAO/pull/2698)
- **POEM 69** implementation: Allow residual names to be different from the corresponding implicit output names. [#2709](https://github.com/OpenMDAO/OpenMDAO/pull/2709)
- Changed default for Driver supports['optimization'] to False [#2715](https://github.com/OpenMDAO/OpenMDAO/pull/2715)
- Deprecated Component as a model, handle gracefully. [#2716](https://github.com/OpenMDAO/OpenMDAO/pull/2716)
- Added pyoptsparse to windows build on GitHub Actions. [#2718](https://github.com/OpenMDAO/OpenMDAO/pull/2718)
- Removed pyparsing as an explicit OpenMDAO dependency. [#2723](https://github.com/OpenMDAO/OpenMDAO/pull/2723)
- OpenMDAO will now raise an exception if a solver is assigned to more than one System [#2724](https://github.com/OpenMDAO/OpenMDAO/pull/2724)
- Cached setup errors so they can be raised at one time. [#2726](https://github.com/OpenMDAO/OpenMDAO/pull/2726)

## Bug Fixes

- Added `indep_var` tag to work with inputs report for backward compatibility. [#2683](https://github.com/OpenMDAO/OpenMDAO/pull/2683)
- Fixed an issue with embedded newlines in tables and added some '*grid' formats. [#2679](https://github.com/OpenMDAO/OpenMDAO/pull/2679)
- Fixed doc for linking [#2680](https://github.com/OpenMDAO/OpenMDAO/pull/2680)
- Fixed an issue with hang in guess_nonlinear in ParallelGroup [#2671](https://github.com/OpenMDAO/OpenMDAO/pull/2671)
- Fixed various numpy deprecation warnings. [#2708](https://github.com/OpenMDAO/OpenMDAO/pull/2708)

## Miscellaneous

- Moved filterwarnings into catch_warnings context. [#2677](https://github.com/OpenMDAO/OpenMDAO/pull/2677)
- Dropped required Python version back to 3.7. [#2662](https://github.com/OpenMDAO/OpenMDAO/pull/2662)
- Changed map test to use weighted interpolant. [#2658](https://github.com/OpenMDAO/OpenMDAO/pull/2658)
- Incremented version in issue template. [#2661](https://github.com/OpenMDAO/OpenMDAO/pull/2661)
- Added document describing how to contribute to OpenMDAO via issues, POEMS and pull requests. [#2663](https://github.com/OpenMDAO/OpenMDAO/pull/2663)
- Added "Debugging your Optimizations" document. [#2672](https://github.com/OpenMDAO/OpenMDAO/pull/2672)
- Updated CI to use conda vs mamba due to Python 3.11 release [#2687](https://github.com/OpenMDAO/OpenMDAO/pull/2687)
- Updated GitHub workflow to work properly with Python 3.11 [#2699](https://github.com/OpenMDAO/OpenMDAO/pull/2699)
- Separated vulnerabilities check from the CI build process [#2703](https://github.com/OpenMDAO/OpenMDAO/pull/2703)
- Updated GitHub test workflow [#2704](https://github.com/OpenMDAO/OpenMDAO/pull/2704)
- Updated version of JAX used in GitHub workflow [#2705](https://github.com/OpenMDAO/OpenMDAO/pull/2705)
- Removed audit step from Windows job in GitHub workflow [#2707](https://github.com/OpenMDAO/OpenMDAO/pull/2707)
- Fixed description of ExternalCodeComp 'command' option. [#2719](https://github.com/OpenMDAO/OpenMDAO/pull/2719)


***********************************
# Release Notes for OpenMDAO 3.21.0

September 28, 2022

OpenMDAO 3.21.0 implements several bug fixes and a few new capabilities. DOEDriver can now compute and record
derivatives, which can be useful for testing. AnalysisError will now correctly feed back to IPOPT when using
pyoptsparse. This feature previously worked only with SNOPT. The behavior with other optimizers varies as not all
are capable of dealing with failed analyses. Finally, `assert_no_approx_partials`, from `openmdao.utils.assert_utils`,
can now be used to separately test models and their children for the presence of partials approximated with
complex-step, finite-difference, or both.

## New Deprecations

- None

## Backwards Incompatible API Changes

- None

## Backwards Incompatible Non-API Changes

- None

## New Features

- Added --dependency_versions option to command line interface to show versions of OpenMDAO dependencies. [#2628](https://github.com/OpenMDAO/OpenMDAO/pull/2628)
- Handle AnalysisError within pyoptsparse_driver using optimizers other than SNOPT. [#2646](https://github.com/OpenMDAO/OpenMDAO/pull/2646)
- **POEM_073**: Allow DOEDriver to record derivatives to allow user to check derivatives across model. [#2648](https://github.com/OpenMDAO/OpenMDAO/pull/2648)
- Added optional arguments `method` and `excludes` to `assert_no_approx_partials` [#2652](https://github.com/OpenMDAO/OpenMDAO/pull/2652)

## Bug Fixes

- Accounted for y-intercept on bounds in linear constraints given to pyoptsparse. [#2605](https://github.com/OpenMDAO/OpenMDAO/pull/2605)
- Fixed bug where models with discrete connections were reconfigured between setup calls. [#2610](https://github.com/OpenMDAO/OpenMDAO/pull/2610)
- Added a better error message when DirectSolver fails and broadened conditions to check for rank deficiency. [#2614](https://github.com/OpenMDAO/OpenMDAO/pull/2614)
- Sorted information for N2 generation to ensure consistency in created diagrams and fixed an error when drawing cycle arrows. [#2616](https://github.com/OpenMDAO/OpenMDAO/pull/2616)
- Fixed file save dialogs for the N2 diagram. [#2619](https://github.com/OpenMDAO/OpenMDAO/pull/2619)
- Fixed a bug in check_totals when an aliased constraint is defined in a child system. [#2621](https://github.com/OpenMDAO/OpenMDAO/pull/2621)
- Fixed the error message to be more helpful when a user calls `Problem.list_problem_vars()` before `final_setup`. [#2624](https://github.com/OpenMDAO/OpenMDAO/pull/2624)
- Removed deprecation warnings under python 3.10 when entry points are used. [#2625](https://github.com/OpenMDAO/OpenMDAO/pull/2625)
- Fixed failure in load_case when variable is distributed. [#2627](https://github.com/OpenMDAO/OpenMDAO/pull/2627)
- Updated check for when coloring doesn't meet minimum improvement percentage to prevent errant messages. [#2630](https://github.com/OpenMDAO/OpenMDAO/pull/2630)
- Fixed to allow a shape mismatch in AutoIVC if all promoted inputs have different shapes but unambiguous storage orders. [#2632](https://github.com/OpenMDAO/OpenMDAO/pull/2632)
- Fixed a bug in `AddSubtractComp` when reusing an input defined in a previous equation. [#2636](https://github.com/OpenMDAO/OpenMDAO/pull/2636)
- Include myst_nb with optional [notebooks] dependencies to support Google Colab. [#2638](https://github.com/OpenMDAO/OpenMDAO/pull/2638)
- Fix for CaseViewer issue caused by matplotlib change. [#2640](https://github.com/OpenMDAO/OpenMDAO/pull/2640)
- Fix to allow "OPENMDAO_REPORTS=none" to disable reports, per the docs. [#2649](https://github.com/OpenMDAO/OpenMDAO/pull/2649)

## Miscellaneous

- Added `long_description_content_type` to setup.py for rendering on pypi. [#2604](https://github.com/OpenMDAO/OpenMDAO/pull/2604)
- Updated `Sellar` model used for testing and documentation.. [#2613](https://github.com/OpenMDAO/OpenMDAO/pull/2613)
- Added coverage improvements. [#2615](https://github.com/OpenMDAO/OpenMDAO/pull/2615)
- Due to issues with conda, switch github workflow to use mamba and pre-install some jupyter dependencies. [#2618](https://github.com/OpenMDAO/OpenMDAO/pull/2618)
- Added MacOS build to CI. [#2623](https://github.com/OpenMDAO/OpenMDAO/pull/2623)
- Replaced all calls to time.time() with time.perf_counter(). [#2643](https://github.com/OpenMDAO/OpenMDAO/pull/2643)
- Fixed some out-of-date documentation dealing with fixed-size interpolants. [#2644](https://github.com/OpenMDAO/OpenMDAO/pull/2644)

***********************************
# Release Notes for OpenMDAO 3.20.0

August 10, 2022

OpenMDAO 3.20.0 adds an optimization summary to our reports, adds a new "restart from last successful" capability to
solvers, adds the ability to only show incorrect totals in `check_totals`, and fixes several issues in the code and documentation.

## New Deprecations

- None

## Backwards Incompatible API Changes

- None

## Backwards Incompatible Non-API Changes

- None

## New Features

- Added optimization summary report. [#2561](https://github.com/OpenMDAO/OpenMDAO/pull/2561)
- Added 'restart_from_successful' option to solvers to allow restarting from the last known good state. See [POEM 068](https://github.com/OpenMDAO/POEMs/blob/master/POEM_068.md) for more information. [#2590](https://github.com/OpenMDAO/OpenMDAO/pull/2590)
- Added 'show_only_incorrect' option to `Problem.check_totals`. [#2595](https://github.com/OpenMDAO/OpenMDAO/pull/2595)

## Bug Fixes

- Specified utf-8 encoding when reading certain files. [#2563](https://github.com/OpenMDAO/OpenMDAO/pull/2563)
- Added code to close coloring plot to prevent matplotlib warnings. [#2569](https://github.com/OpenMDAO/OpenMDAO/pull/2569)
- Fixed a bug in our petsc vector get_norm method. [#2572](https://github.com/OpenMDAO/OpenMDAO/pull/2572)
- Fixed some issues with reports dirs and the index page displayed by 'openmdao view_reports' [#2578](https://github.com/OpenMDAO/OpenMDAO/pull/2578)
- Fixed N2 diagram NL-solver toolbar button. [#2579](https://github.com/OpenMDAO/OpenMDAO/pull/2579)
- Changed to wipeFilters() in Diagram and ModelData classes [#2584](https://github.com/OpenMDAO/OpenMDAO/pull/2584)
- Fixed several minor N2 bugs [#2592](https://github.com/OpenMDAO/OpenMDAO/pull/2592)
- Fixed a hang in opt_report when using MPI. [#2596](https://github.com/OpenMDAO/OpenMDAO/pull/2596)

## Miscellaneous

- Updated setup.py to reflect requirement of Python >= 3.8 plus some changes to the readme. [#2560](https://github.com/OpenMDAO/OpenMDAO/pull/2560)
- Changed CI workflow to use mamba instead of conda due to conda issues. [#2565](https://github.com/OpenMDAO/OpenMDAO/pull/2565)
- Fixed link to citing in readme.md [#2567](https://github.com/OpenMDAO/OpenMDAO/pull/2567)
- Updated issue templates to use github forms and make submitting issues a bit easier. [#2576](https://github.com/OpenMDAO/OpenMDAO/pull/2576)
- Update GitHub workflow [#2582](https://github.com/OpenMDAO/OpenMDAO/pull/2582)
- Fixed pep issues from pycodestyle 2.9.0 [#2586](https://github.com/OpenMDAO/OpenMDAO/pull/2586)
- Fixed docs for NonlinearBlockJac [#2587](https://github.com/OpenMDAO/OpenMDAO/pull/2587)
- Cleaned up some documentation. [#2588](https://github.com/OpenMDAO/OpenMDAO/pull/2588)

***********************************
# Release Notes for OpenMDAO 3.19.0

June 29, 2022

OpenMDAO 3.19.0 removes some long-standing deprecations, continues the addition of faster interpolants, fixes a few bugs, and includes various code cleanup tasks.

## New Deprecations

- None

## Backwards Incompatible API Changes

- Various two-word OptionsDictionary entries, which cannot be provided as arguments because they are not valid python names, have been removed. A number of other existing deprecations have been removed. See [#2551](https://github.com/OpenMDAO/OpenMDAO/pull/2551)

## Backwards Incompatible Non-API Changes

- None

## New Features

- Added 1D and 2D Lagrange2 and Lagrange3 methods [#2527](https://github.com/OpenMDAO/OpenMDAO/pull/2527)
- Added ability to let users specify constants in ExecComp expressions [#2547](https://github.com/OpenMDAO/OpenMDAO/pull/2547)
- Added various reporting system updates [#2548](https://github.com/OpenMDAO/OpenMDAO/pull/2548)
- Various deprecation changes [#2551](https://github.com/OpenMDAO/OpenMDAO/pull/2551)

## Bug Fixes

- Cleanup of test_scaling_report_mpi.py because importing TestScalingReport causes testflo to run both the non-mpi and the MPI versions of the TestCase when processing this file. [#2525](https://github.com/OpenMDAO/OpenMDAO/pull/2525)
- Fixed a bug in which reverse-mode derivatives were not computed correctly for models that contained both unit conversion and solver scaling. [#2526](https://github.com/OpenMDAO/OpenMDAO/pull/2526)
- Handle unavailable attributes in Case.list_outputs (allows discrete variables). [#2529](https://github.com/OpenMDAO/OpenMDAO/pull/2529)
- Update GA & DE drivers to handle INF_BOUND on constraints [#2533](https://github.com/OpenMDAO/OpenMDAO/pull/2533)
- Added exception when user attempts to add a solver to ExplicitComponent [#2538](https://github.com/OpenMDAO/OpenMDAO/pull/2538)
- Made minor fixes to generic model diagram code [#2549](https://github.com/OpenMDAO/OpenMDAO/pull/2549)
- Made dynamic sizing use info from group input defaults [#2550](https://github.com/OpenMDAO/OpenMDAO/pull/2550)
- Fixed a bug when pyoptsparse was used with coloring and scaling report [#2553](https://github.com/OpenMDAO/OpenMDAO/pull/2553)

## Miscellaneous

- Replace distutils.LooseVersion with packaging Version [#2532](https://github.com/OpenMDAO/OpenMDAO/pull/2532)
- Refactor of _apply_linear/_solve_linear and block linear solvers [#2534](https://github.com/OpenMDAO/OpenMDAO/pull/2534)
- Added pip-audit to github workflow [#2536](https://github.com/OpenMDAO/OpenMDAO/pull/2536)
- Updated dependency versions in the GitHub workflow [#2546](https://github.com/OpenMDAO/OpenMDAO/pull/2546)
- Updated Numpy and pyOptSparse versions in GitHub workflow [#2554](https://github.com/OpenMDAO/OpenMDAO/pull/2554)
- Remove use of deprecated 'asscalar' in unit test [#2556](https://github.com/OpenMDAO/OpenMDAO/pull/2556)

***********************************
# Release Notes for OpenMDAO 3.18.0

May 05, 2022

OpenMDAO 3.18.0 adds some enhanced MPI capability to OpenMDAO, and includes several bug fixes and internal cleanup.

The implementation of [POEM 065](https://github.com/OpenMDAO/POEMs/blob/master/POEM_065.md) allows users to specify
`proc_group` for subsystems in ParallelGroups. This enables the grouping of less computationally expensive subsystems onto
fewer processors to achieve better balancing.

We also recently discovered that `math.prod` in the standard Python library as of 3.8 is over an order of magnitude
faster than `numpy.prod` for reasonably small arrays. Since this function is frequently used to determine sizes given
the shape of a variable in OpenMDAO-related tools, we've updated the `shape_to_len` function to take advantage of it
and encourage users to use this implementation over `numpy.prod`.

The generalization of the N2 diagram also continues in this release, with a goal of allowing its use in other contexts.

## New Deprecations

- None

## Backwards Incompatible API Changes

- None

## Backwards Incompatible Non-API Changes

- None

## New Features

- Implemented [POEM 065](https://github.com/OpenMDAO/POEMs/blob/master/POEM_065.md), allowing user to specify `proc_group` and `proc_weight` for subsystems. [#2487](https://github.com/OpenMDAO/OpenMDAO/pull/2487)
- Allow pyoptsparse `opt_prob` to be printed before optimization with `print_opt_prob` option. [#2492](https://github.com/OpenMDAO/OpenMDAO/pull/2492)
- Updated `shape_to_len` and made it call `math.prod` if Python >= 3.8. [#2512](https://github.com/OpenMDAO/OpenMDAO/pull/2512)

## Bug Fixes

- Fixed a bug in ExecComp where the value was not saved as a Numpy array, causing issues with the N2 diagram. [#2482](https://github.com/OpenMDAO/OpenMDAO/pull/2482)
- Fixed formatting of options table in notebooks. [#2485](https://github.com/OpenMDAO/OpenMDAO/pull/2485)
- Fixed indexer.py such that sizes created using numpy.prod are always integers. [#2485](https://github.com/OpenMDAO/OpenMDAO/pull/2486)
- Fixed a few bugs related to discrete variables. [#2495](https://github.com/OpenMDAO/OpenMDAO/pull/2495)
- Aligned the population initialization of the SimpleGA and DifferentialEvolution drivers. [#2499](https://github.com/OpenMDAO/OpenMDAO/pull/2499)
- Fixed an issue where an exception was being raised when dynamic sizing was used with distributed arrays having a local size of zero. [#2503](https://github.com/OpenMDAO/OpenMDAO/pull/2503)
- Fixed an issue where reverse slicing for `src_indices` (`om.slicer[::-1]`) was not working. [#2505](https://github.com/OpenMDAO/OpenMDAO/pull/2505)
- Added a fix to handle more than just `ImportError` in pyOptSparseDriver. [#2508](https://github.com/OpenMDAO/OpenMDAO/pull/2508)
- Added a missing comma to test dependencies in setup.py. [#2513](https://github.com/OpenMDAO/OpenMDAO/pull/2513)
- Fixed infinite recursion when running `openmdao scaling` when coloring is active. [#2515](https://github.com/OpenMDAO/OpenMDAO/pull/2515)
- Added a fix for a bug where complex step around an implicit component containing a Newton solver would discard the imaginary part. [#2517](https://github.com/OpenMDAO/OpenMDAO/pull/2517)
- Fixed a bug in which calling run_model from within run_driver would cause the case_prefix to be lost. [#2520](https://github.com/OpenMDAO/OpenMDAO/pull/2520)

## Miscellaneous

- Added generalized versions of Diagram, Layout classes for the N2 and updated to D3 v7.3. [#2484](https://github.com/OpenMDAO/OpenMDAO/pull/2484)
- Moved MPI scaling report test to its own file to address CI issues. [#2489](https://github.com/OpenMDAO/OpenMDAO/pull/2489)
- Enable triggering of Dymos tests after a successful push to OpenMDAO. [#2490](https://github.com/OpenMDAO/OpenMDAO/pull/2490)
- Added generalized Style, Arrow, Window, Search, Toolbar, SymbolType and UserInterface classes for N2. [#2493](https://github.com/OpenMDAO/OpenMDAO/pull/2493)
- Added 2022 Development Roadmap [#2496](https://github.com/OpenMDAO/OpenMDAO/pull/2496)
- Added generalized Matrix and Legend classes for N2. [#2500](https://github.com/OpenMDAO/OpenMDAO/pull/2500)
- Changed a few test tolerances to avoid sporadic failures on Python 3.7 test environment. [#2501](https://github.com/OpenMDAO/OpenMDAO/pull/2501)
- Cleaned up dependencies. [#2502](https://github.com/OpenMDAO/OpenMDAO/pull/2502)
- Added scripts to create and test a generic N2 model and to build an N2 screenshot for the help screen. [#2510](https://github.com/OpenMDAO/OpenMDAO/pull/2510)
- Fixed text formatting in notebook describing parallel groups. [#2519](https://github.com/OpenMDAO/OpenMDAO/pull/2519)

***********************************
# Release Notes for OpenMDAO 3.17.0

March 21, 2022

OpenMDAO 3.17.0 adds a few significant capabilities, as well as several bugfixes and performance improvements.

The first major capability is the addition of a reports system.  OpenMDAO has provided various feedback through
standard output, either automatically or at the users request.  The new report capability will allow OpenMDAO to provide
richer feedback via html and other outputs.  In addition, reports that are useful and cheap to generate, such as the N2
diagram and the scaling report, will be generated automatically and placed in a reports directory.  The location of
this reports directory is user-configurable, and options exists for disabling reports when they are unwanted.  Finally,
the reports system is intended to be extensible, so developers of tools which use OpenMDAO will be able to generate
their own reports.

OpenMDAO will now also allow aliases for constraints.  This feature will enable users to apply multiple constraints
to the same variable - such as imposing a equality constraint on one array element and lower/upper bounds on other
elements.  In the past, users would typically create new "pass-thru" components that would accept different portions
of these array outputs.  Using the OpenMDAO model hierarchy to "namespace" constraints is also no longer necessary,
since aliases can be used to give a more human-readable description of the constraint.

We're continuing to add visualization tools. This release features OptionsWidget and CaseViewer.  OptionsWidget provides
a simple GUI to change options of a model in a Jupyter notebook.  CaseViewer provides a GUI for rapidly plotting
variables across case iterations from a case record file.  CaseViewer currently only works in a Jupyter notebook,
but we plan on allowing it to be used as a standalone tool in the future.

## New Deprecations

- Added deprecation if options names are not valid python names. [#2378](https://github.com/OpenMDAO/OpenMDAO/pull/2378)

## Backwards Incompatible API Changes

- None

## Backwards Incompatible Non-API Changes

- None

## New Features

- Added several updates to the hooks system. [#2404](https://github.com/OpenMDAO/OpenMDAO/pull/2404)
- N2 toolbar buttons added to access collapse/expand and variable filter modes [#2407](https://github.com/OpenMDAO/OpenMDAO/pull/2407)
- Represent filtered variables in the N2 as collapsed instead of hidden [#2422](https://github.com/OpenMDAO/OpenMDAO/pull/2422)
- Allow complex numbers in InterpND, more fixed-size interpolants for 1D and 2D slinear, and some slinear performance improvements. [#2436](https://github.com/OpenMDAO/OpenMDAO/pull/2436)
- Implementation of the reports system, per POEM 060. [#2448](https://github.com/OpenMDAO/OpenMDAO/pull/2448)
- Added a widget to modify options in a Jupyter notebook environment before setup. [#2456](https://github.com/OpenMDAO/OpenMDAO/pull/2456)
- Added a CaseViewer tool for quick visualization of case data in a Jupyter notebook. [#2470](https://github.com/OpenMDAO/OpenMDAO/pull/2470)
- Implementation of POEM 063, constraint aliases and allowing multiple constraints on different indices of array variables. [#2473](https://github.com/OpenMDAO/OpenMDAO/pull/2473)

## Bug Fixes

- When generating the scaling report, run the model if it hasn't been run. [#2413](https://github.com/OpenMDAO/OpenMDAO/pull/2413)
- Allow leading underscore in valid options names. [#2418](https://github.com/OpenMDAO/OpenMDAO/pull/2418)
- Removed exception when get_io_metadata called on dyn shaped variable before shape is known [#2424](https://github.com/OpenMDAO/OpenMDAO/pull/2424)
- Fixed Iterable deprecation and a couple tests for Python 3.10 [#2426](https://github.com/OpenMDAO/OpenMDAO/pull/2426)
- DOE driver no longer generates redundant recordings when not running in parallel [#2430](https://github.com/OpenMDAO/OpenMDAO/pull/2430)
- Fixed a bug when a flat slice is specified as the src_indices for a non-flat source, and added usage of slicer to connection documentation. [#2443](https://github.com/OpenMDAO/OpenMDAO/pull/2443)
- Fixed a bug and some changes to recording under MPI [#2445](https://github.com/OpenMDAO/OpenMDAO/pull/2445)
- Fixed a bug in CaseReader's parsing of iteration coordinates. [#2454](https://github.com/OpenMDAO/OpenMDAO/pull/2454)
- Fixed some issues with src indices under promotion and some MPI issues. [#2455](https://github.com/OpenMDAO/OpenMDAO/pull/2455)
- Fixed an MPI hang when running the scaling report. [#2457](https://github.com/OpenMDAO/OpenMDAO/pull/2457)
- Fixed a broken usage message for cmd line tools. [#2458](https://github.com/OpenMDAO/OpenMDAO/pull/2458)
- Finished fixing logic to parse system from iteration coordinate. [#2459](https://github.com/OpenMDAO/OpenMDAO/pull/2459)
- Added a check for discrete outputs that are not design vars in Driver._get_voi_val. [#2464](https://github.com/OpenMDAO/OpenMDAO/pull/2464)
- Added a test for solver recording when Broyden is the solver and also eliminate double recording of Broyden solver iterations. [#2465](https://github.com/OpenMDAO/OpenMDAO/pull/2465)
- Added handling of discrete variables to the scaling report. [#2466](https://github.com/OpenMDAO/OpenMDAO/pull/2466)
- Fixed some issues with indexing values for plots in the CaseViewer [#2477](https://github.com/OpenMDAO/OpenMDAO/pull/2477)

## Miscellaneous

- Removed 'remove-output' tag on metamodel notebooks with options tables [#2403](https://github.com/OpenMDAO/OpenMDAO/pull/2403)
- Updated docstrings for numpydoc 1.2 compatibility [#2421](https://github.com/OpenMDAO/OpenMDAO/pull/2421)
- Refactored the HTML preprocessor implementation for the N2. [#2452](https://github.com/OpenMDAO/OpenMDAO/pull/2452)
- Added installation of `libscotch` to the oldest workflow. [#2467](https://github.com/OpenMDAO/OpenMDAO/pull/2467)
- Continued N2 refactor by adding generic classes for ModelData, TreeNode, and others. [#2468](https://github.com/OpenMDAO/OpenMDAO/pull/2468)

***********************************
# Release Notes for OpenMDAO 3.16.0

January 06, 2022

OpenMDAO 3.16.0 fixes the definition of the "Kelvin as energy" unit, improves documentation and exception handling, adds verification that options names are valid python names, and adds improvements to the N2 visualization.

We've also adjusted our nomenclature with respect to parallel processing.
Instead of inputs and outputs either being "distributed" or "serial", we now refer to them as "distributed" or "non-distributed."

Finally, the minimum supported version of Python is now 3.7, since 3.6 has reached the end of its life.  While OpenMDAO may still work with prior versions of Python, we recommend users upgrade to supported versions.


## New Deprecations

- Added deprecation if options names are not valid python names. [#2378](https://github.com/OpenMDAO/OpenMDAO/pull/2378)

## Backwards Incompatible API Changes

- None

## Backwards Incompatible Non-API Changes

- None

## New Features

- If no explicit name given to a Problem, assign one automatically. [#2368](https://github.com/OpenMDAO/OpenMDAO/pull/2368)
- Added fwd/rev coloring support for function comps (using jax). [#2370](https://github.com/OpenMDAO/OpenMDAO/pull/2370)
- Added deprecation if options names are not valid python names. [#2378](https://github.com/OpenMDAO/OpenMDAO/pull/2378)
- Added traceback information in a few places where OpenMDAO handles then re-raises exceptions. [#2384](https://github.com/OpenMDAO/OpenMDAO/pull/2384)

## Bug Fixes

- Replaced np.bool with bool to silence numpy deprecation. [#2375](https://github.com/OpenMDAO/OpenMDAO/pull/2375)
- Fixed 'Ken' (kelvin-as-energy-unit) definition based on 2018 standard. [#2386](https://github.com/OpenMDAO/OpenMDAO/pull/2386)

## Miscellaneous

- Added a search box to the N2 variable selection dialog. [#2361](https://github.com/OpenMDAO/OpenMDAO/pull/2361)
- Removed Deprecation Warning For Serial to Distributed IO Connections. [#2363](https://github.com/OpenMDAO/OpenMDAO/pull/2363)
- Added a complex step summary notebook. [#2376](https://github.com/OpenMDAO/OpenMDAO/pull/2376)
- Removed old 'experimental docs' directory. [#2382](https://github.com/OpenMDAO/OpenMDAO/pull/2382)
- Updated ISSUE_TEMPLATE to remove redundant summary section.  [#2388](https://github.com/OpenMDAO/OpenMDAO/pull/2388)
- Overhaul documentation for matrix-free derivatives on components with distributed variables. [#2390](https://github.com/OpenMDAO/OpenMDAO/pull/2390)
- Made a few changes to improve coverage. [#2394](https://github.com/OpenMDAO/OpenMDAO/pull/2394)
- Changed minimum Python version to 3.7 since 3.6 has reached end-of-life. [#2395](https://github.com/OpenMDAO/OpenMDAO/pull/2395)
- Made a small fix to make distributed example more complex-safe. [#2398](https://github.com/OpenMDAO/OpenMDAO/pull/2398)

***********************************
# Release Notes for OpenMDAO 3.15.0

November 24, 2021

OpenMDAO 3.15.0 is released to address a derivative bug in the new 1D-akima interpolant.

Calling `set_val` on a variable that is ultimately sourced from 'AutoIVC' was causing a poorly worded error when the given value was not compatible.
This has ben fixed.

It also adds a new explicitly `'unitless'` unit.
Currently the standard practice is to use `None` as the units for dimensionless quantities.
Variables with units `None` are allowed to be connected to variables of other units, and no conversion is performed.
Using `'unitless'` will all users more rigorous checking when connecting a dimensionless unit to another variable.

The new unit `'percent'`, is also added, which is defined as `unitless/100`.

## New Deprecations

- None

## Backwards Incompatible API Changes

- None

## Backwards Incompatible Non-API Changes

- None

## New Features

- Added 'unitless' and 'percent' quantities, from [POEM 059](https://github.com/OpenMDAO/POEMs/blob/master/POEM_059.md) [#2340](https://github.com/OpenMDAO/OpenMDAO/pull/2340)

## Bug Fixes

- Fix for deriv calc on vectorized 1D-akima [#2352](https://github.com/OpenMDAO/OpenMDAO/pull/2352)
- Provide a better error message when setting an invalid value to an auto_ivc output. [#2355](https://github.com/OpenMDAO/OpenMDAO/pull/2355)

## Miscellaneous

- None

***********************************
# Release Notes for OpenMDAO 3.14.0

November 18, 2021

OpenMDAO 3.14.0 features new function wrapping capability that makes it easier to tie existing code into OpenMDAO,
and significantly increased interpolation performance when the training data is fixed.

## New Deprecations

- Renamed fixed interp methods to follow convention in [POEM 058](https://github.com/OpenMDAO/POEMs/blob/master/POEM_058.md), 'trilinear' and 'akima1D' are now deprecated and changed to '3D-slinear' and '1D-akima', respectively. [#2329](https://github.com/OpenMDAO/OpenMDAO/pull/2329) [#2332](https://github.com/OpenMDAO/OpenMDAO/pull/2332)

## Backwards Incompatible API Changes

- None

## Backwards Incompatible Non-API Changes

- None

## New Features

- Added implementation of ExplicitFuncComp, the explicit function wrapping capability described in [POEM 056](https://github.com/OpenMDAO/POEMs/blob/master/POEM_056.md) and [POEM 057](https://github.com/OpenMDAO/POEMs/blob/master/POEM_057.md). [#2309](https://github.com/OpenMDAO/OpenMDAO/pull/2309)
- Added a new method '3D-lagrange3' for interpolation on a fixed 3D grid. This method is vectorized and caches the coefficients for each cell, which results in a much more efficient execution. [#2314](https://github.com/OpenMDAO/OpenMDAO/pull/2314)
- Added implementation of ImplicitFuncComp for implicit function wrapping capability described in [POEM 056](https://github.com/OpenMDAO/POEMs/blob/master/POEM_056.md) and [POEM 057](https://github.com/OpenMDAO/POEMs/blob/master/POEM_057.md). [#2321](https://github.com/OpenMDAO/OpenMDAO/pull/2321)
- Added a fixed '3D-lagrange2' interpolant, and implemented some other performance improvements. [#2337](https://github.com/OpenMDAO/OpenMDAO/pull/2337)
- Added a variable selection dialog added to N2 diagram, use Alt+Right Click to access it. [#2339](https://github.com/OpenMDAO/OpenMDAO/pull/2339)
- Added warning when user calls list_inputs before final_setup [#2341](https://github.com/OpenMDAO/OpenMDAO/pull/2341)

## Bug Fixes

- Fixed output of reverse mode partials in check_partials if they were divided by comm size. [#2310](https://github.com/OpenMDAO/OpenMDAO/pull/2310)
- Handle recording of constraints that use input names, which is how bounds are implemented when using COBYLA. [#2323](https://github.com/OpenMDAO/OpenMDAO/pull/2323)
- Fixed documentation pages that use automethod but were missing eval-rst. [#2331](https://github.com/OpenMDAO/OpenMDAO/pull/2331)

## Miscellaneous

- Pinned playwight version to <1.15 so GUI testing still works on RHEL/CentOS. [#2311](https://github.com/OpenMDAO/OpenMDAO/pull/2311)
- Changed GitHub actions to use Python <3.9. [#2313](https://github.com/OpenMDAO/OpenMDAO/pull/2313)
- Added step to GitHub workflow to check for security vulnerabilities. [#2317](https://github.com/OpenMDAO/OpenMDAO/pull/2317)
- Fixed an issue with documentation upload. [#2318](https://github.com/OpenMDAO/OpenMDAO/pull/2318)
- Updated advanced recording documentation page. [#2322](https://github.com/OpenMDAO/OpenMDAO/pull/2322)
- Additional speed improvements for interpolation. [#2325](https://github.com/OpenMDAO/OpenMDAO/pull/2325)
- Improved performance for setting jacobian elements. [#2335](https://github.com/OpenMDAO/OpenMDAO/pull/2335)

***********************************
# Release Notes for OpenMDAO 3.13.1

October 13, 2021

OpenMDAO 3.13.1 is a patch release to fix a minor bug in the new trilinear interpolant.

## New Deprecations

- None

## Backwards Incompatible API Changes

- None

## Backwards Incompatible Non-API Changes

- None

## New Features

- The implementations for Trilinear and akima1D are now vectorized with coefficient caching. [#2300](https://github.com/OpenMDAO/OpenMDAO/pull/2300)

## Bug Fixes

- Fixed a bug where extrapolation was mistakenly not handled in the Trilinear method. [#2300](https://github.com/OpenMDAO/OpenMDAO/pull/2300)
- Prevent mpi4py & petsc4py from importing when they're not wanted [#2290](https://github.com/OpenMDAO/OpenMDAO/pull/2290)

## Miscellaneous

- Remove reference to since-deleted code regarding 2-3 compatibility [#2299](https://github.com/OpenMDAO/OpenMDAO/pull/2299)

***********************************
# Release Notes for OpenMDAO 3.13.0

October 06, 2021

OpenMDAO 3.13.0 implements a change in the way indices are handled.
See [POEM 053](https://github.com/OpenMDAO/POEMs/blob/master/POEM_053.md) and [POEM 054](https://github.com/OpenMDAO/POEMs/blob/master/POEM_054.md)
for more information.

We've also added some security to the OpenMDAO repository by requiring GPG-signed commits when accepting pull requests.
This is an extra layer of security that helps prevent nefarious pull-requests from being made to the code-base.
See the documentation [here](http://openmdao.org/twodocs/versions/latest/other_useful_docs/developer_docs/signing_commits.html) for more information.

## New Deprecations

- Behavior of `'rel'` step_calc for finite difference is changed and issues a deprecation warning if used, see [#2209](https://github.com/OpenMDAO/OpenMDAO/pull/2209) for more information.

## Backwards Incompatible API Changes

- When indexing into multi-dimensional arrays while connecting or adding design variables, constraints, or objectives, the indices will now behave like numpy indices.  Flattened behavior can be achieved using `flat_src_indices` (in connections) or `flat_indices` (in other methods).

## Backwards Incompatible Non-API Changes

- None

## New Features

- Completed the transition to numpy style indexing described in POEM_054. [#2279](https://github.com/OpenMDAO/OpenMDAO/pull/2279)
- Added information on min and max values of array variables in our list methods per [POEM 055](https://github.com/OpenMDAO/POEMs/blob/master/POEM_055.md).  Thank you @andrewellis55 for the contribution. [#2280](https://github.com/OpenMDAO/OpenMDAO/pull/2280)
- Added an implementation of the function metadata API per [POEM 056](https://github.com/OpenMDAO/POEMs/blob/master/POEM_056.md) [#2281](https://github.com/OpenMDAO/OpenMDAO/pull/2281)
- Improved performance of cubic metamodel interpolation, and added interpolants `akima1d` and `trilinear` that are limited to fixed dimensions and no derivatives wrt the training data, but are significantly faster. These interpolants are currently under development so some changes to the API may occur. [#2285](https://github.com/OpenMDAO/OpenMDAO/pull/2285)

## Bug Fixes

- None

## Miscellaneous

- Added docs for signing/verifying commits [#2273](https://github.com/OpenMDAO/OpenMDAO/pull/2273)
- remove unused travis CI files [#2274](https://github.com/OpenMDAO/OpenMDAO/pull/2274)
- Fix test that was failing due to varying precision in warning message. [#2287](https://github.com/OpenMDAO/OpenMDAO/pull/2287)

***********************************
# Release Notes for OpenMDAO 3.12.0

September 15, 2021

OpenMDAO 3.12.0 provides a transitional release as we change the way the specification of indices are handled.
In the future, specification of indices will be similar to the way it is done in Numpy, as opposed to an OpenMDAO-specific method.
See [POEM 053](https://github.com/OpenMDAO/POEMs/blob/master/POEM_053.md) and [POEM 054](https://github.com/OpenMDAO/POEMs/blob/master/POEM_054.md)
for more information.

## New Deprecations

- Behavior of `'rel'` step_calc for finite difference is changed and issues a deprecation warning if used, see [#2209](https://github.com/OpenMDAO/OpenMDAO/pull/2209) for more information.

## Backwards Incompatible API Changes

- Disabled the option to use coloring on approximated semi-total derivatives. [#2245](https://github.com/OpenMDAO/OpenMDAO/pull/2245)

## Backwards Incompatible Non-API Changes

## New Features

- Improved behavior of relative step_calc for finite difference in accordance with [POEM 051](https://github.com/OpenMDAO/POEMs/blob/master/POEM_051.md). [#2209](https://github.com/OpenMDAO/OpenMDAO/pull/2209)
- Add a check to openmdao check to report any options in the model that are not serializable. [#2225](https://github.com/OpenMDAO/OpenMDAO/pull/2225)
- Shift-click can now close all persistent Node Info windows in the N2 viewer. [#2226](https://github.com/OpenMDAO/OpenMDAO/pull/2226)
- Update to src_indices indexing to behave like numpy indexing (old indexing is still there but deprecated) [#2235](https://github.com/OpenMDAO/OpenMDAO/pull/2235)
- Added code to issue warnings in cases where apparently flat indices index into non-flat source arrays when the flat_src_indices or flat_indices flags have not been set to True. [#2248](https://github.com/OpenMDAO/OpenMDAO/pull/2248)
- Added a converter from old OpenMDAO indexing format to the new numpy-like one. [#2267](https://github.com/OpenMDAO/OpenMDAO/pull/2267)

## Bug Fixes

- Fixed the hang described in [#2191](https://github.com/OpenMDAO/OpenMDAO/pull/2191) so all procs will call multi_proc_exception_check()  [#2208](https://github.com/OpenMDAO/OpenMDAO/pull/2208)
- Added support for multidimensional shapes in EQConstraintComp and BalanceComp. [#2210](https://github.com/OpenMDAO/OpenMDAO/pull/2210)
- Fixed bug where an exception was raised if you tried to get a value for an input that was connected to a source with a src_indices slice. [#2217](https://github.com/OpenMDAO/OpenMDAO/pull/2217)
- Fixed a bug in the interpolation bracketing algorithm that caused a hysteresis in the analytic derivative when interpolating on one of the table's gridpoints. [#2230](https://github.com/OpenMDAO/OpenMDAO/pull/2230)
- Fixed a bug that caused an IndexError to be raised when using a directional derivative during check_partials on an MPI model. [#2233](https://github.com/OpenMDAO/OpenMDAO/pull/2233)
- Fixed a bug where an exception was raised if a directional derivative was used to check a sparsely declared subjacobian. [#2241](https://github.com/OpenMDAO/OpenMDAO/pull/2241)
- Fixed for IndexError raised in the lagrange3 semi-structured interpolation. [#2243](https://github.com/OpenMDAO/OpenMDAO/pull/2243)
- Disabled the option to use coloring on approximated semi-total derivatives. [#2245](https://github.com/OpenMDAO/OpenMDAO/pull/2245)
- Fixed case where openmdao check for deprecated indexing was calling len(self.shape) on an indexer with a shape that wasn't a tuple. [#2264](https://github.com/OpenMDAO/OpenMDAO/pull/2264)

## Miscellaneous

- Updates to github workflow. [#2211](https://github.com/OpenMDAO/OpenMDAO/pull/2211)
- The documentation for distributed variables has been rewritten to demonstrate how to use them after recent refactors. [#2218](https://github.com/OpenMDAO/OpenMDAO/pull/2218)
- Refactor playwright code to enable re-use with other GUI testing [#2228](https://github.com/OpenMDAO/OpenMDAO/pull/2228)
- Replaced custom linting with Numpydoc.validate linting [#2231](https://github.com/OpenMDAO/OpenMDAO/pull/2231)
- Some improvements to MetaModelStructuredComp documentation for dynamically generated training values. [#2236](https://github.com/OpenMDAO/OpenMDAO/pull/2236)
- Provide a way for users to exclude search results from source docs [#2237](https://github.com/OpenMDAO/OpenMDAO/pull/2237)
- Modified the code that uploads documentation to be able to handle extra arguments (e.g. port number). [#2244](https://github.com/OpenMDAO/OpenMDAO/pull/2244)

***********************************
# Release Notes for OpenMDAO 3.11.0

August 03, 2021

OpenMDAO 3.11.0 introduces SemiStructuredMetaModelComp as well as some bug fixes and performance improvements.
It fixes the known issue from 3.10.0 in which the derivatives of serial outputs wrt distributed inputs were broken.

## New Deprecations

- None

## Backwards Incompatible API Changes:

- None

## Backwards Incompatible NON-API Changes:

- None

## New Features:

- New MetaModelSemiStructuredComp component that can interpolate on "semi-structured" data. [#2185](https://github.com/OpenMDAO/OpenMDAO/pull/2185)

## Bug Fixes:

- Fixed deprecation of 'value' in get_io_metadata(). [#2191](https://github.com/OpenMDAO/OpenMDAO/pull/2191)
- Improved error message for global shape errors. [#2192](https://github.com/OpenMDAO/OpenMDAO/pull/2192)
- Changed handling of seeds when computing total derivatives - distributed to serial derivatives now working. [#2197](https://github.com/OpenMDAO/OpenMDAO/pull/2197)
- The "includes" and "excludes" arguments for list_outputs and list_inputs on case objects now support strings. [#2198](https://github.com/OpenMDAO/OpenMDAO/pull/2198)

## Miscellaneous:

- Fixed errors in multifi_cokriging documentation. [#2183](https://github.com/OpenMDAO/OpenMDAO/pull/2183)
- Introduced some speed improvements to final_setup when recorders are present. [#2184](https://github.com/OpenMDAO/OpenMDAO/pull/2184)
- Switched from Pyppeteer to Playwright for N2 GUI tests. [#2187](https://github.com/OpenMDAO/OpenMDAO/pull/2187)
- The MetaModelStructuredComp now uses backward difference for fd checks when the method is 'slinear' so that the finite difference step direction aligns with the bin bracketing. [#2195](https://github.com/OpenMDAO/OpenMDAO/pull/2195)
- Fixed the inputs of the Hohmann transfer example in the docs. [#2202](https://github.com/OpenMDAO/OpenMDAO/pull/2202)


***********************************
# Release Notes for OpenMDAO 3.10.0

July 14, 2021

OpenMDAO 3.10.0 features a few API changes and a migration of documentation to JupyterBook.

## Known Issues

Currently total derivatives of serial outputs with respect to distributed inputs do not work.
This corner case did not exist in prior releases, but was created by the new distributed variable API from [POEM_046](https://github.com/OpenMDAO/POEMs/blob/master/POEM_046.md). We will fix this in the next release.

## New Deprecations

- Component option `distributed` is now replaced with `distributed=True` on `add_input` and `add_output`. [#2073](https://github.com/OpenMDAO/OpenMDAO/pull/2073)
- Removed an inconsistency where various options/arguments used different forms of `val`/`value`/`values`.  All have been changed to `val`. [#2112](https://github.com/OpenMDAO/OpenMDAO/pull/2112)
- We've switched our documentation to JupyterBook, which allows our documentation examples to be run online through Google Colab.

## Backwards Incompatible API Changes:

- Option `vectorize_derivs` has been removed from optimization variables. [#2116](https://github.com/OpenMDAO/OpenMDAO/pull/2116)

## Backwards Incompatible NON-API Changes:

- None

## New Features:

- Replaced options['distributed'] usages with distributed=True in add_input/add_output [#2073](https://github.com/OpenMDAO/OpenMDAO/pull/2073)
- New documentation using Jupyter-Notebooks. [#2076](https://github.com/OpenMDAO/OpenMDAO/pull/2076)
- Added flag to System to tell when under finite difference. [#2089](https://github.com/OpenMDAO/OpenMDAO/pull/2089)
- Added code to handle the case where the sql_meta file already exists in case recording. [#2099](https://github.com/OpenMDAO/OpenMDAO/pull/2099)
- 'indices' for design vars and constraints can now be slices or 'fancy' indices. [#2102](https://github.com/OpenMDAO/OpenMDAO/pull/2102)
- Refactor of parallel derivative coloring and removal of vectorize_derivs [#2116](https://github.com/OpenMDAO/OpenMDAO/pull/2116)
- Users who add a recorder after setup is called get improved error messages [#2125](https://github.com/OpenMDAO/OpenMDAO/pull/2125)
- OpenMDAO now checks that derivative settings are different for computing derivatives and checking derivatives. [#2166](https://github.com/OpenMDAO/OpenMDAO/pull/2166)

## Bug Fixes:

- Added rank check to only print driver summary to single rank. [#2077](https://github.com/OpenMDAO/OpenMDAO/pull/2077)
- Updated the norm calculation to account for None and zero length array. [#2084](https://github.com/OpenMDAO/OpenMDAO/pull/2084)
- Fixed Return Value Type of add_output for IndepVarComp. [#2104](https://github.com/OpenMDAO/OpenMDAO/pull/2104)
- Whenever get_io_metadata is called, update metadata and promoted names to account for any new i/o that was added. [#2106](https://github.com/OpenMDAO/OpenMDAO/pull/2106)
- Fixed a couple of bugs in the scaling report viewer. [#2109](https://github.com/OpenMDAO/OpenMDAO/pull/2109)
- Fixed invalid value in QuadraticComp for docs. [#2112](https://github.com/OpenMDAO/OpenMDAO/pull/2112)
- Fixed inconsistency of val vs value in OpenMDAO [#2118](https://github.com/OpenMDAO/OpenMDAO/pull/2118)
- Fixed a bug in ExecComp when expressions were added with missing value/shape info [#2121](https://github.com/OpenMDAO/OpenMDAO/pull/2121)
- Fixed an issue where ExecComp was required to have at least one expression before configure is called. [#2123](https://github.com/OpenMDAO/OpenMDAO/pull/2123)
- Generalized distributed vector check in Problem.setup() [#2126](https://github.com/OpenMDAO/OpenMDAO/pull/2126)
- Fixed bug in scaling/unscaling the linear vector during reverse mode in model with solver scaling defined on an output. via 'ref'. [#2131](https://github.com/OpenMDAO/OpenMDAO/pull/2131)
- Fixed deprecation warning; remove extraneous 'value' from metadata [#2146](https://github.com/OpenMDAO/OpenMDAO/pull/2146)
- Fix for exception when adding src_indices with self.promotes on a previously promoted input. [#2149](https://github.com/OpenMDAO/OpenMDAO/pull/2149)
- Now all metadata for COBYLA design variable bounds is copied to constraints. [#2156](https://github.com/OpenMDAO/OpenMDAO/pull/2156)
- Fixed broken docs scaling link [#2160](https://github.com/OpenMDAO/OpenMDAO/pull/2160)
- Moved and renamed the openmdao warnings module to utils.om_warnings and fixed a few issues with it.  [#2162](https://github.com/OpenMDAO/OpenMDAO/pull/2162)
- Fixed invalid keyword arg in issue_warning call [#2167](https://github.com/OpenMDAO/OpenMDAO/pull/2167)
- Refactored DOEDriver to eliminate recorder pointer copies. [#2174](https://github.com/OpenMDAO/OpenMDAO/pull/2174)
- Fixed for multiple issues with computing analytic and approx derivatives for components with a mix of serial and distributed variables. [#2177](https://github.com/OpenMDAO/OpenMDAO/pull/2177)
- Fixed some issues so now OpenMDAO works with networkx==2.6 [#2178](https://github.com/OpenMDAO/OpenMDAO/pull/2178)

## Miscellaneous:

- Fixed doc links in README; change coveralls base_dir arg. [#2081](https://github.com/OpenMDAO/OpenMDAO/pull/2081)
- Added a doc for load_cases. [#2082](https://github.com/OpenMDAO/OpenMDAO/pull/2082)
- Cleaned up notebooks and made reset_notebook a console script. [#2086](https://github.com/OpenMDAO/OpenMDAO/pull/2086)
- Updated workflow for docs. [#2087](https://github.com/OpenMDAO/OpenMDAO/pull/2087)
- Moved asserts in a notebook into hidden cells. [#2092](https://github.com/OpenMDAO/OpenMDAO/pull/2092)
- Some minor clean up of the former doc tests. [#2093](https://github.com/OpenMDAO/OpenMDAO/pull/2093)
- Added conditional doc build reports to github workflow. [#2094](https://github.com/OpenMDAO/OpenMDAO/pull/2094)
- Updated show_options_table func for Dymos documentation. [#2095](https://github.com/OpenMDAO/OpenMDAO/pull/2095)
- Fixed a couple of lingering doc issues. [#2097](https://github.com/OpenMDAO/OpenMDAO/pull/2097)
- Sourced coveralls from pypi in actions workflow [#2101](https://github.com/OpenMDAO/OpenMDAO/pull/2101)
- Added tests for return values from add_input and add_output [#2105](https://github.com/OpenMDAO/OpenMDAO/pull/2105)
- Update readme with new instructions for building docs [#2108](https://github.com/OpenMDAO/OpenMDAO/pull/2108)
- Fixed github repo information for JupyterBook docs [#2115](https://github.com/OpenMDAO/OpenMDAO/pull/2115)
- Fixed Keplers equation example in docs to show better ordering [#2117](https://github.com/OpenMDAO/OpenMDAO/pull/2117)
- Removed deprecated usage of src_indices from docs [#2124](https://github.com/OpenMDAO/OpenMDAO/pull/2124)
- Added sphinx_sitemap to docs; tweak CI & badges [#2129](https://github.com/OpenMDAO/OpenMDAO/pull/2129)
- Replaced 'value' with 'val' in ExecComp across the documentation [#2130](https://github.com/OpenMDAO/OpenMDAO/pull/2130)
- Minor documentation tweaks. [#2135](https://github.com/OpenMDAO/OpenMDAO/pull/2135)
- Remove a bunch of warnings from the notebooks in the docs. [#2145](https://github.com/OpenMDAO/OpenMDAO/pull/2145)
- Updated distributed parab component to remove deprecation warnings from the documentation. [#2143](https://github.com/OpenMDAO/OpenMDAO/pull/2143)
- Added deprecation column to options table in documentation. [#2151](https://github.com/OpenMDAO/OpenMDAO/pull/2151)
- Added full test_suite examples with display_source to allow users to see the underlying code examples being imported from test_suite. [#2159](https://github.com/OpenMDAO/OpenMDAO/pull/2159)
- Docs now explicitly import the om namespace so they can be more easily reproduced outside of the notebook. [#2165](https://github.com/OpenMDAO/OpenMDAO/pull/2165)
- Added min/max example to ks_comp feature doc [#2170](https://github.com/OpenMDAO/OpenMDAO/pull/2170)
- Addressed some issues with CI until changes to networkx are accounted for. [#2172](https://github.com/OpenMDAO/OpenMDAO/pull/2172)
- Fixed broken links in documentation to papers at the MDO lab. [#2173](https://github.com/OpenMDAO/OpenMDAO/pull/2173)

**********************************
# Release Notes for OpenMDAO 3.9.2

May 14, 2021

OpenMDAO 3.9.2 is a patch release that fixes some bugs found in 3.9.1.

## New Deprecations

- None

## Backwards Incompatible API Changes:

- None

## Backwards Incompatible NON-API Changes:

- None

## New Features:

- Added `require_pyoptsparse` decorator for skipping tests when the necessary optimizers are not available. [#2064](https://github.com/OpenMDAO/OpenMDAO/pull/2064)

## Bug Fixes:

- Changed thread.isAlive to thread.is_alive for Python 3.9 compatibility in the profiler. [#2045](https://github.com/OpenMDAO/OpenMDAO/pull/2045)
- Updated pyOptSparseDriver to handle an API change in pyOptSparse in version 2.5.1 [#2051](https://github.com/OpenMDAO/OpenMDAO/pull/2051)
- Fixed DemuxComp issue not handling 3D arrays [#2052](https://github.com/OpenMDAO/OpenMDAO/pull/2052)
- Fixed KeyError during recording setup when running under MPI when design variable was an input [#2053](https://github.com/OpenMDAO/OpenMDAO/pull/2053)
- Require Sphinx < 4.0 for building docs for Python 3.6 compatibility. [#2055](https://github.com/OpenMDAO/OpenMDAO/pull/2055)
- Added a small fix to MetaModelUnStructuredComp for multiple nD array training inputs [#2058](https://github.com/OpenMDAO/OpenMDAO/pull/2058)
- Fixed _linearize was being called on linear solver unnecessarily during check_partials and partial sparsity computation [#2060](https://github.com/OpenMDAO/OpenMDAO/pull/2060)
- Make sure variables to record are sorted so they are in sync across multiple procs. [#2065](https://github.com/OpenMDAO/OpenMDAO/pull/2065)
- Fixed a bug in set_val when src_indices reference into a flat array but are not in a flat array themselves. [#2067](https://github.com/OpenMDAO/OpenMDAO/pull/2067)
- Improved error message when there are zero rows or cols in the total Jacobian [#2069](https://github.com/OpenMDAO/OpenMDAO/pull/2069)

## Miscellaneous:

- Updated GitHub Workflow to include repo details when submitting coverage statistics. [#2043](https://github.com/OpenMDAO/OpenMDAO/pull/2043), [#2044](https://github.com/OpenMDAO/OpenMDAO/pull/2044)
- Privatize some filewrap classes to solve problem with new doc build. [#2063](https://github.com/OpenMDAO/OpenMDAO/pull/2063)

**********************************
# Release Notes for OpenMDAO 3.9.1

May 04, 2021

OpenMDAO 3.9.1 is a patch release that fixes a bug in check_totals as well
as resolving an issue that prevents systems from being repickled without
being run.

## New Deprecations

- None

## Backwards Incompatible API Changes:

- None

## Backwards Incompatible NON-API Changes:

- None

## New Features:

- None

## Bug Fixes:

- Fixed a bug that prevented systems from being repickled without running them. [#2034](https://github.com/OpenMDAO/OpenMDAO/pull/2034)
- Fixed a bug in check_totals that was causing an IndexError. [#2038](https://github.com/OpenMDAO/OpenMDAO/pull/2038)

## Miscellaneous:

- Refactored the github workflow to make it more modular. [#2036](https://github.com/OpenMDAO/OpenMDAO/pull/2036)

**********************************
# Release Notes for OpenMDAO 3.9.0

April 28, 2021

OpenMDAO 3.9.0 features an API change to the serial/parallel implementation
in OpenMDAO, a significant performance increase in approximated partials,
and several other improvements.

Most OpenMDAO components allow inputs and outputs to be added after the
component is instantiated.  This now applies to ExecComp as well.  New
expressions can be added to ExecComp using the `add_expr` method.

OpenMDAO can sometimes produce numerous verbose warnings during setup.
We now implement our own Warning classes so that the verbosity of these
warnings can be changed through the use of Python's warning filtering
functionality.

See the following sections for a complete list of changes.

## New Deprecations

- Environment variable OPENMDAO_REQUIRE_MPI is now OPENMDAO_USE_MPI. [#1968](https://github.com/OpenMDAO/OpenMDAO/pull/1968)

## Backwards Incompatible API Changes:

- dynamically shaped connection between distributed outputs and serial inputs are no longer allowed. [#2013](https://github.com/OpenMDAO/OpenMDAO/pull/2013)
- dynamically shaped connections now by default assume local data transfer to distributed inputs, i.e., size/shape and src_indices in the input correspond to the local part of the output. [#2013](https://github.com/OpenMDAO/OpenMDAO/pull/2013)
- old behavior of connecting distributed outputs to serial inputs was to have the serial input in each proc contain the full distributed size of the output. To achieve this same behavior, user must now specify src_indices for the input of om.slicer[:] (recommended) or an index array matching the full distrib size of the output. [#2013](https://github.com/OpenMDAO/OpenMDAO/pull/2013)

## Backwards Incompatible NON-API Changes:

- None

## New Features:

- The unit 'as' for attoseconds now works. [#1963](https://github.com/OpenMDAO/OpenMDAO/pull/1963)
- Warnings for MPI/PETSc failed import are now suppressed. [#1968](https://github.com/OpenMDAO/OpenMDAO/pull/1968)
- Added OpenMDAO version to case recorder file [#1971](https://github.com/OpenMDAO/OpenMDAO/pull/1971)
- Added `_setup_check()` hook that is called immediately after setup. [#1976](https://github.com/OpenMDAO/OpenMDAO/pull/1976)
- User can now add expressions to ExecComp using "add_expr". [#1977](https://github.com/OpenMDAO/OpenMDAO/pull/1977)
- Added argument to list_problem_vars to optionally return unscaled values. [#1980](https://github.com/OpenMDAO/OpenMDAO/pull/1980)
- Added OpenMDAO-specific warnings. [#1982](https://github.com/OpenMDAO/OpenMDAO/pull/1982)
- Added notebook utils for upcoming documentation built in Jupyter Book [#1984](https://github.com/OpenMDAO/OpenMDAO/pull/1984)
- Reduced the size of the case database files by compressing JSON & BLOBs in case db, and create separate metadata db for parallel runs [#1989](https://github.com/OpenMDAO/OpenMDAO/pull/1989)
- Decreased memory required for partial coloring by changing the way the sparsity matrix is computed. [#2000](https://github.com/OpenMDAO/OpenMDAO/pull/2000)
- Added a change to get_val such that, when `from_src` is false, the data is obtained locally rather than from the source. [#2005](https://github.com/OpenMDAO/OpenMDAO/pull/2005)
- Changed default behavior when computing src_indices between variables when one side is distributed. [#2013](https://github.com/OpenMDAO/OpenMDAO/pull/2013)


## Bug Fixes:
- Fixed a bug where the global_size was incorrect for a design variable that is specified with indices on an IVC output declared as a distributed component. [#1986](https://github.com/OpenMDAO/OpenMDAO/pull/1986)
- Generate more informative error message if a src_indices containing slicing operators was incorrectly set. [#1988](https://github.com/OpenMDAO/OpenMDAO/pull/1988)
- Fix to allow showing N2 diagrams in Jupyter notebook when the file path is absolute. [#1998](https://github.com/OpenMDAO/OpenMDAO/pull/1998)
- Disable USE_PROC_FILES when MPI is used in a notebook. [#2001](https://github.com/OpenMDAO/OpenMDAO/pull/2001)
- Fixed a bug so that order doesn't matter when promoting wildcards and aliased names on add_subsystem. [#2002](https://github.com/OpenMDAO/OpenMDAO/pull/2002)
- Covered ipython imports for imports without [all] import [#2006](https://github.com/OpenMDAO/OpenMDAO/pull/2006)
- Fixed bug in handling of response from github query when looking for plugins [#2010](https://github.com/OpenMDAO/OpenMDAO/pull/2010)
- Fixed bug where the rel_err on a case was actually the abs_err. [#2012](https://github.com/OpenMDAO/OpenMDAO/pull/2012)
- Fixed a bug with multidimensional inputs in DOE driver [#2016](https://github.com/OpenMDAO/OpenMDAO/pull/2016)
- Fixed scaffold test warnings and a KrigingSurrogate test failure [#2018](https://github.com/OpenMDAO/OpenMDAO/pull/2018)

## Miscellaneous:

- Added support for sparse FD, setting of jacobian columns, and using sparse arrays in coloring algorithms. [#1967](https://github.com/OpenMDAO/OpenMDAO/pull/1967)
- Moved CI to Github Actions. [#1974](https://github.com/OpenMDAO/OpenMDAO/pull/1974)
- Updated warning logic for pull requests. [#1979](https://github.com/OpenMDAO/OpenMDAO/pull/1979)
- Github CI now checks coverage. [#1995](https://github.com/OpenMDAO/OpenMDAO/pull/1995)
- Added IPython to notebook and doc dependencies. [#2004](https://github.com/OpenMDAO/OpenMDAO/pull/2004)
- Added CI testing without optional dependencies. [#2008](https://github.com/OpenMDAO/OpenMDAO/pull/2008)
- Implemented a workaround for a github actions issue. [#2015](https://github.com/OpenMDAO/OpenMDAO/pull/2015)
- Docs are now uploaded from github action instead of travis. [#2020](https://github.com/OpenMDAO/OpenMDAO/pull/2020)
- Removed Travis CI badge from README [#2022](https://github.com/OpenMDAO/OpenMDAO/pull/2022)

**********************************
# Release Notes for OpenMDAO 3.8.0

March 16, 2021

OpenMDAO 3.8.0 includes some memory usage reductions for large problems,
and some user-configurable feedback from drivers when problems are not
well formulated.

## Backwards Incompatible API Changes:

- None

## Backwards Incompatible NON-API Changes:

- None


## New Features:

- Added debug printing for totals for approx_totals [#1902](https://github.com/OpenMDAO/OpenMDAO/pull/1902)
- New N2 capability to save and load views. [#1904](https://github.com/OpenMDAO/OpenMDAO/pull/1904)
- `user_terminate_signal` now defaults to None [#1906](https://github.com/OpenMDAO/OpenMDAO/pull/1906)
- Added check for Google Colab environment in notebook_mode [#1921](https://github.com/OpenMDAO/OpenMDAO/pull/1921)
- Cleaned up pyoptSparseDriver error handling so that unhandled exceptions are re-raised in place of the cryptic pyoptsparse one. [#1923](https://github.com/OpenMDAO/OpenMDAO/pull/1923)
- Added `to_table` method to OptionsDictionary to provide `embed_options` replacement capability in Jupyter. [#1924](https://github.com/OpenMDAO/OpenMDAO/pull/1924)
- In pyoptsparse and scipyoptimizer drivers, raise a warning during first full derivative calculation if a row or column of the total derivatives matrix is all zero. [#1938](https://github.com/OpenMDAO/OpenMDAO/pull/1938)
- Overhaul internal storage of scaling vectors. [#1947](https://github.com/OpenMDAO/OpenMDAO/pull/1947)
- Added code to show view_connections in notebook/colab/docs [#1948](https://github.com/OpenMDAO/OpenMDAO/pull/1948)
- Add N2 search terms to forward/back history [#1952](https://github.com/OpenMDAO/OpenMDAO/pull/1952)

## Bug Fixes:

- Fixed problem where `has_diag_partials` was ignored when `shape_by_conn` or `copy_shape` were used. [#1907](https://github.com/OpenMDAO/OpenMDAO/pull/1907)
- Fixed `shape_by_conn` bug for distributed input to serial output. [#1914](https://github.com/OpenMDAO/OpenMDAO/pull/1914)
- Fixed bugs in `list_outputs` `residuals_tol` filtering. [#1937](https://github.com/OpenMDAO/OpenMDAO/pull/1937)
- Fixed bug when using InterpND to interpolate on 1D data. [#1941](https://github.com/OpenMDAO/OpenMDAO/pull/1941)
- Fixed NaNs that may still show up in the N2 diagram [#1945](https://github.com/OpenMDAO/OpenMDAO/pull/1945)
- Fixed text mistake when a linesearch prints its violations. [#1949](https://github.com/OpenMDAO/OpenMDAO/pull/1949)
- Replaced unsafe chars when creating HTML element ids in N2 [#1950](https://github.com/OpenMDAO/OpenMDAO/pull/1950)
- Fixed a couple of small debug print bugs. [#1954](https://github.com/OpenMDAO/OpenMDAO/pull/1954)

## Miscellaneous:

- Updated screen grabs for scaling report [#1900](https://github.com/OpenMDAO/OpenMDAO/pull/1900)
- Updated docstrings for check argument in Problem methods setup and check_config [#1919](https://github.com/OpenMDAO/OpenMDAO/pull/1919)
- Updated various case reader and notebook/tabulate related code, tests and docs [#1927](https://github.com/OpenMDAO/OpenMDAO/pull/1927)
- Automated N2 toolbar font generation and toolbar help screen generation [#1942](https://github.com/OpenMDAO/OpenMDAO/pull/1942)
- Exclude new icon script from linting tests because it terminates the tests. [#1946](https://github.com/OpenMDAO/OpenMDAO/pull/1946)

**********************************
# Release Notes for OpenMDAO 3.7.0

February 11, 2021

OpenMDAO 3.7.0 adds the ability for users to use their own Python functions
from within ExecComp.  It also adds to OpenMDAO's utility in a Jupyter notebook environment
by formatting output of the informative methods (list_outputs, view_connections, etc)
in a notebook-friendly table format.  Other changes include performance improvements
when using partial derivative coloring, continued improvements to the N2 viewer,
and removal of the mock dependency

## Backwards Incompatible API Changes:

- None

## Backwards Incompatible NON-API Changes:

- If case recorder outputs are returned scaled, this no longer (erroneously) remains true on subsequent calls. [#1850](https://github.com/OpenMDAO/OpenMDAO/pull/1850)

## New Features:

- ExecComp expressions are visible in the N2 Node Info Panel [#1888](https://github.com/OpenMDAO/OpenMDAO/pull/1888)
- [POEM 036](https://github.com/OpenMDAO/POEMs/blob/master/POEM_039.md): User created functions can now be used with ExecComp. ExecComp now allows `shape_by_conn` and `copy_shape` options for IO. [#1852](https://github.com/OpenMDAO/OpenMDAO/pull/1852)
- Options for all solvers are always recorded [#1845](https://github.com/OpenMDAO/OpenMDAO/pull/1845)
- Added detection of a Jupyter notebook environment, table formatting and in cell HTML output [#1844](https://github.com/OpenMDAO/OpenMDAO/pull/1844)
- Added a display to the N2 so that if an error occurs the user has some idea of what happened [#1838](https://github.com/OpenMDAO/OpenMDAO/pull/1838)

## Bug Fixes:

- Fixed N2 connections toolbar show/hide buttons [#1890](https://github.com/OpenMDAO/OpenMDAO/pull/1890)
- Fixed a bug in nonlinear and linear solvers when running under MPI so that AnalysisErrors for non-convergence are raised on all processors instead of just root. [#1878](https://github.com/OpenMDAO/OpenMDAO/pull/1878)
- Fixed a bug in doe when design variables have indices defined. [#1873](https://github.com/OpenMDAO/OpenMDAO/pull/1873)
- Added a fix to ensure that IPython is an optional import in the N2 viewer and connection viewer [#1869](https://github.com/OpenMDAO/OpenMDAO/pull/1869)
- Fixed a small bug to give better error message when promoting an input with units together with an input without units. [#1867](https://github.com/OpenMDAO/OpenMDAO/pull/1867)
- Fixed a bug where declaring partials wrt '*' caused extra subjacs for outputs wrt other outputs to be added to the subjacs. This resulted in total derivative colorings that were less efficient than they should be. [#1862](https://github.com/OpenMDAO/OpenMDAO/pull/1862)
- Fixed a bug that prevented the jacobian heatmap legend from displaying in firefox [#1859](https://github.com/OpenMDAO/OpenMDAO/pull/1859)
- openmdao CLI now reports an error if there is a dashed arg other than -h or --version before the command or filename [#1858](https://github.com/OpenMDAO/OpenMDAO/pull/1858)
- Case recorder now returns a copy of data when getting variables [#1850](https://github.com/OpenMDAO/OpenMDAO/pull/1850)
- Fixed bug that made a Gatherv call fail when calling list outputs on a distributed model [#1847](https://github.com/OpenMDAO/OpenMDAO/pull/1847)


## Miscellaneous:

- Cleanup of N2 window code to remove duplication [#1871](https://github.com/OpenMDAO/OpenMDAO/pull/1871)
- Got rid of memory allocation for -identity subjacs for matrix free ExplicitComponents [#1863](https://github.com/OpenMDAO/OpenMDAO/pull/1863)
- Switch from mock to the stdlib unittest.mock [#1860](https://github.com/OpenMDAO/OpenMDAO/pull/1860)
- Switched Favicon to SVG instead of ICO for dynamic switching of website icon based on theme. [#1848](https://github.com/OpenMDAO/OpenMDAO/pull/1848)
- Added clarification to the documentation that one must be running under MPI to use DOEDriver in parallel [#1837](https://github.com/OpenMDAO/OpenMDAO/pull/1837)

**********************************
# Release Notes for OpenMDAO 3.6.0

January 14, 2021

OpenMDAO 3.6.0 provides new features, several bug fixes, as well as
documentation and visualization updates.

Thank you to users @Dakror and @cfe316 for contributing to this release.

## Backwards Incompatible API Changes:

- None

## Backwards Incompatible NON-API Changes:

- None

## New Features:

- (POEM_032) Implemented a new scaling report feature that is intended to provide the user with some information as to how well their problem is scaled. [#1820](https://github.com/OpenMDAO/OpenMDAO/pull/1820)
- (POEM_036) Implemented caching of the Kriging training weights by serializing them and saving them to a file so they can be used in later runs to save computation time. [#1830](https://github.com/OpenMDAO/OpenMDAO/pull/1830)
- Added button to n2 to show or hide desvars, constraints, and objectives [#1793](https://github.com/OpenMDAO/OpenMDAO/pull/1793)
- Units are now simplified upon creation. [#1796](https://github.com/OpenMDAO/OpenMDAO/pull/1796)
- Added info about surrogate in n2 [#1800](https://github.com/OpenMDAO/OpenMDAO/pull/1800)
- Added note to N2 NodeInfo when displaying initial value [#1805](https://github.com/OpenMDAO/OpenMDAO/pull/1805)
- Added the ability to hide the solver hierarchy in the N2 [#1807](https://github.com/OpenMDAO/OpenMDAO/pull/1807)
- Added warning when nonlinear solver stalls three times [#1818](https://github.com/OpenMDAO/OpenMDAO/pull/1818)

## Bug Fixes:

- Fixed a bug related to show_progress in check_totals [#1794](https://github.com/OpenMDAO/OpenMDAO/pull/1794)
- A few fixes for the standalone InterpND [#1797](https://github.com/OpenMDAO/OpenMDAO/pull/1797)
- Fix for an undefined variable in error msg related to shape_by_conn [#1799](https://github.com/OpenMDAO/OpenMDAO/pull/1799)
- Fixed issue where resid_tol failed with vectorized resids [#1804](https://github.com/OpenMDAO/OpenMDAO/pull/1804)
- Only raise distributed variable errors when MPI comm size > 1 [#1814](https://github.com/OpenMDAO/OpenMDAO/pull/1814)
- Fixed shape bug in DVs when indices were used [#1815](https://github.com/OpenMDAO/OpenMDAO/pull/1815)
- Make get_var_meta private, and fix a test failure introduced from the latest scipy. [#1819](https://github.com/OpenMDAO/OpenMDAO/pull/1819)

## Miscellaneous:

- Added new logos to docs [#1806](https://github.com/OpenMDAO/OpenMDAO/pull/1806)
- Fixed dead link in NonlinearBlockGS docs [#1810](https://github.com/OpenMDAO/OpenMDAO/pull/1810)
- Correct the very short "input + output" example to have both an input and an output. [#1812](https://github.com/OpenMDAO/OpenMDAO/pull/1812)
- Fixed flag for pip --upgrade in the README. [#1826](https://github.com/OpenMDAO/OpenMDAO/pull/1826)
- fix for change in sphinx-doc v3.4.0 (PR 8445) [#1827](https://github.com/OpenMDAO/OpenMDAO/pull/1827)

**********************************
# Release Notes for OpenMDAO 3.5.0

December 04, 2020

OpenMDAO 3.5.0 adds the ability for users to complex-step across an OpenMDAO model externally,
as well as a few performance tweaks.

## Backwards Incompatible API Changes:

- None

## Backwards Incompatible NON-API Changes:

- None

## New Features:

- Allow user to complex step across a Problem. [#1777](https://github.com/OpenMDAO/OpenMDAO/pull/1777)
- Internal refactor of vector to have real and complex step modes share memory. [#1778](https://github.com/OpenMDAO/OpenMDAO/pull/1778)
- Fixed deprecation warning when user passed a pandas dataframe to discrete input [#1780](https://github.com/OpenMDAO/OpenMDAO/pull/1780)
- Added a 'get_remote' arg to compute_totals [#1783](https://github.com/OpenMDAO/OpenMDAO/pull/1783)

## Bug Fixes:

- Fixed setup() memory leak [#1782](https://github.com/OpenMDAO/OpenMDAO/pull/1782)
- Fixed an indexing bug dealing with src_indices in view_connections [#1788](https://github.com/OpenMDAO/OpenMDAO/pull/1788)

## Miscellaneous:
- Rearranged an mpi test and added fwd test [#1781](https://github.com/OpenMDAO/OpenMDAO/pull/1781)

**********************************
# Release Notes for OpenMDAO 3.4.1

November 13, 2020

OpenMDAO 3.4.1 contains mostly bug fixes, and adds a complex-compatible
two-argument arctangent function.

PR #1760 allows multiple sets of src_indices to be given to connections.
Previously, if an input was promoted using some set of src_indices, one could not connect an output to it using another set of src_indices.
This difference in behavior has been fixed.

## Backwards Incompatible API Changes:

- None

## Backwards Incompatible NON-API Changes:

- None

## New Features:

- src_indices can now be applied at multiple levels. #1760
- Added complex safe arctan2 to utils.cs_safe and to the functions available in ExecComp. #1759
- Current value appears in N2 node info panel instead of initial value. #1755
- Implemented non-modal, multi-capable node info panels in the N2 for viewing multiple data values at once. #1744

## Bug Fixes:

- Fix for bug in parallel setup in version 3.4  #1765
- Added residuals_tol logic to filter out by residual tolerance specified by the user in list_outputs. #1757

## Miscellaneous:
- None

**********************************
# Release Notes for OpenMDAO 3.4.0

October 01, 2020

OpenMDAO 3.4.0 adds an experimental feature that allows inputs to be
shaped based upon their source.  We're still testing this feature but we
encourage users to try it out and see if it works for their use cases.

## Backwards Incompatible API Changes:

- list_outputs will now return the system-relative promoted path of outputs
- list_inputs will now return the system-relative absolute path of inputs

## Backwards Incompatible NON-API Changes:

- Refactor of internal data structures and some cleanups. #1693

## New Features:

- Implementation for POEM_022 - Determining variable shape at runtime based on connections #1671
- AnalysisErrors reached during optimization now produce noisy warnings in pyOptSparseDriver. #1672
- Added check to optionally hide noisy AutoIVC warnings #1680
- Errors in user-defined methods (e.g. compute, apply_linear, etc.) now report the class and pathname where the error occurred. #1697
- Auto-ivc component visible again in N2, and connections sourced from it are now highlighted #1698

## Bug Fixes:

- Added support for distributed design variables in get_design_vars #1659
- Fixed a bug in pyoptsparse where, on certain Windows setups, the signal package may not have SIGUSR1 defined and the user gets an AttributeError when instantiating the Driver. #1675
- Fix bug where the N2 was gathering options that are not recordable. #1676
- Add support for Nan values in the N2 diagram #1677
- Fixed N2 testing code to detect console errors and added regression test for nan value bug in N2 #1679
- Fixed issue with case recording on a DOEDriver with a parallel model where it did not record all remote variables. #1689
- Fixed a bug where equivalent units promoted to the same name were requiring set_input_defaults unnecessarily. #1690
- Added some improvements in the sparsity of derivatives for ExecComps with multiple expressions #1699
- Connection of unitless variables is more reliable for automatically generated units #1704

## Miscellaneous:
- Added declaration for distributed design variables to the `supports` dictionary. #1678
- Added miscellaneous speedups for models with large numbers of inputs. #1686
- Change calls to super to use Python3 syntax. #1695



**********************************
# Release Notes for OpenMDAO 3.3.0

September 04, 2020

OpenMDAO 3.3.0 features some changes to the setup/configure stack that
are intended to make it easier to implement complex models in OpenMDAO.

The new group method `get_io_metadata` is available from the `configure`
method of Groups.  During setup, a Group's configure method is run
after the setup methods of all children have been run.
The `get_io_metadata` method allows one to inquire about the inputs and outputs
within the Group and its descendents.  For instance, it can be used to find the names,
units, and shapes of inputs and outputs in the descendent components of a Group.

A new Differential Evolution driver has been added.
This driver is for use on problems with continuous design variables,
and is roughly 3x faster than the existing Simple GA Driver.

## Backwards Incompatible API Changes:

- list_outputs will now return the system-relative promoted path of outputs
- list_inputs will now return the system-relative absolute path of inputs

## Backwards Incompatible NON-API Changes:

None

## New Features:

- Added more information to the node info window in the n2. #1610
- Removed warning when recording deprecated options in viewer data. #1613
- <POEM 029> Adds get_io_metadata method to retrieve metadata of underlying inputs and outputs. #1618
- Internal definition of undefined inputs is now managed with an _UNDEFINED constant in openmdao.core.constants. #1622
- N2 code getting tree dict updated to use _UNDEFINED constant. #1623
- When encountering an AnalysisError, drivers will now indicate which component raised the error. #1650
- Differential Evolution driver added. #1662
- <POEM 031> Added Aitken Relaxation to the Linear Block Gauss-Seidel solver. #1663
- Tagging capability added to AddSubtractComp #1664

## Bug Fixes:

- Fix for a bug where design variables declared in a subsystem are not set up correctly. #1604
- Fix for Keyerror when debug printing pyoptsparse derivatives with auto_ivc design vars. #1605
- Fix for a bug in add_objective/add_constraint when adding using an input variable name #1616
- Fixed pyoptsparse sparse specification for auto_ivc #1619
- Fixed a bug where setting value with units on a simple component model would fail #1626
- Fix for problem with view_connections when model has discrete variables #1627
- Case recorders now save model options correctly if run_driver is called more than once. #1635
- Added logic to stop error being raised if promotes '*' is used and no matches are found. #1636
- Fixed mistake in error message: set_input_defaults expects keyword val not value. #1638
- Fix for unserializable object failure when running n2. #1639
- Fix for typo in handling of flat_src_indices. #1642
- Fix for problem with src_indices applied during promotes called from parent when multiple subsystem inputs are promoted to the same name. #1645
- Fix for issue with calling promotes on descendants that are not direct children. #1647
- Clarified deprecation warnings for IndepVarComp args. #1649
- Fix to avoid building ParOpt on travis without MPI #1652
- Fixed a bug in the accuracy of SimpleGA Driver that was causing issues on CI. #
- Added a new iteration counter for apply_nonlinear to address an issue with counting iterations. #1656

## Miscellaneous:

- Added some knobs to the N2 pre-collapsing functionality. #1614
- Rewrote the unconnected input check to work as originally intended in the auto_ivc environment. #1658
- Added error checks for a couple of get_val corner cases, plus some cleanups. #1660


**********************************
# Release Notes for OpenMDAO 3.2.1

August 07, 2020

OpenMDAO 3.2.1 is a minor update intended to address issues introduced in 3.2.0.

On the user-facing side, there are continuing improvements to the N2 visualization tool.
Version 3.2.0 introduced the ability to use `om.slicer` to specify indices and src_indices via slices.  This update now allows the Ellipsis (`...`) in the specification of slices, which should make connecting subsets of multidimensional outputs much simpler.
Nonlinear solvers also now have the ability to detect stalls.

## Backwards Incompatible API Changes:

- Removed unused args from IndepVarComp.add_output #1576

## Backwards Incompatible NON-API Changes:


## New Features:

- Pinned N2 arrows are now transitioned between updates #1563
- Update vector API to eliminate the need to access internal vector data structures directly. This is a good idea in general but is mandatory for nocopy transfers, where the internal '_data' array in input vectors no longer contains storage for all of the variables. #1567
- Support for ellipsis objects in om.Slicer #1564
- Added Stall Detection to the nonlinear solvers. #1574

## Bug Fixes:

- Reworked error/warning behavior for set_input_defaults so that ambiguities resolved at a higher level trigger a warning instead of an error. #1568
- The N2 code and several areas of the docs still referenced param/unknown terms instead of input/output. These were updated, and a few other areas of N2 code were cleaned up as well. #1570
- An error message is now raised if set_val is called before setup. #1578
- Calls made to set_input_defaults inside of the configure method are no longer ignored. #1583
- When multiple inputs promoted to the same name were explicitly connected to an output, a units ambiguity error was generated even if all of the inputs had the same units. #1587
- Fixed bug in directional derivatives of implicit states when checking partials on a matrix-free component. #1589
- Allow BalanceComp I/O to be sized from rhs_val if val is unavailable #1591
- Fixed bug in group promotes with input src_indices being ignored. #1595

## Miscellaneous:

- Multiple versions of the TLDR Paraboloid example were consolidated #1555
- Fix for test failure on AppVeyor. Compare against Numpy integer datatype. #1586

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
