# OpenMDAO Development Map 2023

Author: Rob Falck
Date: 2025-05-12

## What is this roadmap for?
As in previous years, this roadmap exists to provide general guidance for our development decisions in the coming year, and to provide a retrospective for our performance in the previous year.

## Performance Assessment for 2024

While we neglected to post an official roadmap for 2024, we made significant progress on reducing the activation energy for users.

We made considerable strides into automatic differentiation by closely coupling the capability of the JAX framework with OpenMDAO's JAXExplicitComponent and JaxImplicitComponent. This is in addition to existing AD capability on the Julia side via `OpenMDAO.jl` and the work being developed on the Computational Systems Design Language (CSDL) at UCSD.

AnalysisDriver is a generalization of capability regarding model analysis and visualization. DOEDriver's behavior was governed by the optimization problem as defined upon the system. Performing a simple parameter sweep across model inputs wasn't simple to do. The work on the AnalaysisDriver and related visualizations should improve the situation.

While we made some good strides into developing instructional content on the web, changes in our team and our budget have put an end to those efforts in the short term.

## 2025 Focus Areas

1. Continued expansion of coupling with AD tools. Given what we can do with JAX, it makes sense to make similar efforts towards the portion of the community that relies upon PyTorch for similar capability. We will look at building components that wrap PyTorch models in much the same way that we can wrap JAX models today.

2. Post-Optimality Sensitivity and Suboptimization

We're finishing up some work that generalizes generation of lagrange multipliers following optimization. These multipliers provide the sensitivty of the objective function to the constraints and bounds imposed by the user, and we can use these to obtain sensitivities wrt other model inputs.

In viewing the optimization problem as its own implicit process, we're also interested in obtaining sensitivities for the resulting design variable values with respect to the bounds/constraints and inputs. The math to accomplish this requires second derivatives, and at least initially, we'll be relying upon finite differences **across compute_totals** (not across the optmization) to obtain these.

Ultimately the most robust way to do this would involve directly computing second total derivatives using the MAUD machinery in OpenMDAO. In the past this has always been hampered by the need for the user to compute their own second derivatives. Perhaps AD tools will open up a path to this.

3. AnalysisDriver to Surrogate model tool

One of the biggest use-cases for AnalysisDriver is to inform the creation of surrogate models. It mkes sense that OpenMDAO should provide some automation of this capability.

4. Partial Derivative Relevance

`compute_partials` currently has no way of knowing what partials are actually needed for the current optimization problem.  In many cases this is moot because the partial calculations are generally less expensive than conditionally checking for them.

However, there are cases where individual partial calculations are not cheap. If we could use relevance to determine which ones need to be calculated, OpenMDAO would get considerably faster in some situations.

This is also the case for using finite-difference or complex-step across big models, especially something like a file-wrapped external code. Using `declare_partials(of='*', wrt='*', method='fd')` will cause OpenMDAO to compute all of the partials, even those not needed. There should be room for some considerable performance gains here.

5. Expand use of shape_by_conn and implement units_by_conn

We have had a shape-by-connection capability for some time, but it hasn't gotten significant uptake because computing partials by hand when the shapes of inputs can be changing is too challenging. This is another scenario where AD should help.

We also frequently find ourselves in situations where the units of outputs depend upon the units of inputs. This is often the case when "pass-thru" components are used, or with something like a simple matrix-vector product. Dymos is probably the best example of this.  Given a generic ODE model, it performs some significant introspection to determine the shapes and units of variables in the ODE. Having a units-by-conn capability would make life easier here as well.
