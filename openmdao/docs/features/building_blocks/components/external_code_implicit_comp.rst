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
--------------------------------

Here is another example where there are external codes for both an `ImplicitComponent` and `ExplicitComponent`.
We will modify the code given in
:ref:`the circuit tutorial <defining_icomps_tutorial>`. We will replace one of the nodes and one of the resistors
in the model with external codes.

In this example, our external codes will be simple Python scripts. Here they are:

.. embed-code::
    openmdao.components.tests.extcode_resistor

.. embed-code::
    openmdao.components.tests.extcode_node

Here is the modified circuit example using these external codes.

.. embed-code::
    openmdao.components.tests.test_external_code_comp.TestExternalCodeImplicitCompFeature.test_circuit_plain_newton_using_extcode
    :layout: interleave


.. tags:: ExternalCodeImplicitComp, ExternalCodeComp, FileWrapping, Component
