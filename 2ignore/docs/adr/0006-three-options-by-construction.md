# ADR-0006 — The three-option requirement holds by construction, not by filtering

Status: accepted

## Context

The specification requires every child to have at least three candidate stops within `ε`. A
generator can satisfy that in two ways: sample children uniformly and discard the instance (or
the child) when the property fails, or sample children only where the property already holds.

Uniform sampling plus rejection at the *instance* level is unusable at the sizes wanted here —
with 300 children, the chance that all of them independently have three options in range is
negligible. Rejection at the *child* level biases the instance in a way that is hard to describe
in a report.

## Decision

`generate_instance` places stops first, computes the **viable anchors** (stops with at least two
others within `2ε`), and draws each child uniformly inside a random anchor's `ε`-disc, keeping the
point only if at least three stops end up in range. `validate_instance` re-checks the property for
anything loaded from disk.

The anchor test is necessary but not sufficient — three stops on an equilateral triangle of side
`2ε` have no common point — which is why the per-child check remains.

## Consequences

- Instances are describable in one sentence in the report, and the requirement is a property of
  the generator rather than a filter applied afterwards.
- Children cluster around stops rather than being uniform over the square. That is realistic for
  the problem (stops are sited where people live) but it must be stated, not implied.
- `suggest_epsilon` targets ~6 expected options per child, comfortably above the requirement, so
  rejection sampling stays cheap.
- Failure is loud: an impossible parameter combination raises `InstanceError` naming the
  parameter to relax, instead of silently returning a smaller instance.
