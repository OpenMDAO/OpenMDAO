.. index:: ExternalCodeImplicitComp Example

.. _externalcodeimplicitcomp_feature:

************************
ExternalCodeImplicitComp
************************

`ExternalCodeImplicitComp` is very similar to `ExternalCodeComp` in that it runs an external program in a subprocess on your
operating system. But it treats the `Component` as an `ImplicitComponent` rather than an `ExplicitComponent`. See
:ref:`ExternalCodeComp <externalcodecomp_feature>` for basic information about how `ExternalCodeComp` works.

`ExternalCodeImplicitComp` has most of the same options as `ExternalCodeComp`, but there is one major difference.

    .. embed-options::
        openmdao.components.external_code_comp
        ExternalCodeImplicitComp
        options

When using an `ExternalCodeImplicitComp`, you have the option to define two external programs rather than one. The
first of these is "command_apply", which is the command that you want to run to evaluate the residuals. You should
always specify a value for this option. The second is "command_solve", which is the command that you want to run
to let the external program solve its own states. This is optional, but you should specify it if your code can
solve itself, and if you want it to do so (for example, while using a Newton solver with "solve_subsystems" turned
on in a higher-level `Group`.)

ExternalCodeImplicitComp Example
---------------------------------------

Here is a simple example of the use of an `ExternalCodeImplicitComp` Component. The external code in the example
is a Python script that evaluates the output and residual for the implicit relationship between the area ratio and
mach number in an isentropic flow. We use the same external code for both "command_apply" and "command_solve", but
in each case we pass it different flags.

.. embed-code::
    openmdao.components.tests.extcode_mach

The following model makes use of this external code.

.. embed-code::
    openmdao.components.tests.test_external_code_comp.TestExternalCodeImplicitCompFeature.test_simple_external_code_implicit_comp
    :layout: interleave



.. tags:: ExternalCodeImplicitComp, ExternalCodeComp, FileWrapping, Component
