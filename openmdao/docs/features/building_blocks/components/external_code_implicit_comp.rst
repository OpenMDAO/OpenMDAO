.. index:: ExternalCodeImplicitComp Example

.. _externalcodeimplicitcomp_feature:

************************
ExternalCodeImplicitComp
************************

`ExternalCodeImplicitComp` is very similar to `ExternalCodeComp` in that it runs an external program in a subprocess on your
operating system. But it treats the `Component` as an `ImplicitComponent` rather than an `ExplicitComponent`. See
:ref:`ExternalCodeComp <externalcodecomp_feature>` for basic information about how `ExternalCodeComp` works.

`ExternalCodeImplicitComp` has the same options as `ExternalCodeComp`.

    .. embed-options::
        openmdao.components.external_code_comp
        ExternalCodeImplicitComp
        options


ExternalCodeImplicitComp Example
---------------------------------------

Here is a simple example of the use of a `ExternalCodeImplicitComp` Component. The external code is a Python script
that evaluates the output and residual for the implicit relationship between the area ratio and mach number in an
isentropic flow.

.. embed-code::
    openmdao.components.tests.extcode_mach

The following model makes use of this external code.

.. embed-code::
    openmdao.components.tests.test_external_code_comp.TestExternalCodeImplicitCompFeature.test_simple_external_code_implicit_comp_with_solver
    :layout: interleave



.. tags:: ExternalCodeImplicitComp, ExternalCodeComp, FileWrapping, Component
