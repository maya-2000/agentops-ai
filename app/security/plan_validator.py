"""Plan validation: the model's investigation plan is untrusted until every step passes.

Checks:

- **Step count.** At most ``max_plan_steps`` steps, and no more than the remaining tool budget.
- **Arguments.** Every step's ``arguments_json`` parses to a JSON object.
- **Authorization.** Every step passes ``ToolAuthorizationPolicy.authorize``: allowlist, enabled,
  permitted for the intent, SQL privilege, typed and policy-checked arguments, SQL validation,
  budget and prerequisites.
- **Duplicates.** A step identical to an earlier step or an already executed call is dropped (no
  unnecessary repeats).
- **Structure.** Dependencies and cycles are ruled out: a plan is a flat, ordered list of tool
  calls executed at most once each. A tool cannot call another tool or the planner, and follow-up
  planning is bounded by ``max_planning_iterations``.

Problems are split into two kinds:

- *Model-output problems*: malformed JSON, invalid arguments, an unknown tool name, too many
  steps. The planner may be asked again with the problems as feedback, within the retry budget.
- *Security denials*: a disabled tool, a tool not permitted for the intent, revoked SQL
  privilege, unsafe SQL, the data-exposure policy, an exhausted budget. These are not retried,
  and the run fails closed.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.llm.schemas import PlanOutput
from app.security.authorization import (
    SQL_TOOL,
    AuthorizationContext,
    AuthorizationDecision,
    ToolAuthorizationPolicy,
)
from app.security.budget import BudgetUsage, RunBudget, remaining_tool_calls

# Denials that indicate a policy violation rather than a malformed model output.
NON_RETRYABLE_DENIALS = frozenset(
    {
        "tool_disabled",
        "tool_not_permitted",
        "sql_not_permitted",
        "unsafe_sql",
        "data_policy",
        "budget_exceeded",
        "prerequisite_missing",
    }
)


class ValidatedStep(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    purpose: str


class PlanValidation(BaseModel):
    steps: list[ValidatedStep] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)  # model-output problems (retry with feedback)
    denials: list[AuthorizationDecision] = Field(default_factory=list)  # every denied step
    budget_exceeded: bool = False

    @property
    def valid(self) -> bool:
        return not self.problems and not self.denials and not self.budget_exceeded

    @property
    def security_denial(self) -> AuthorizationDecision | None:
        """The first denial that must not be retried, if any."""
        return next((d for d in self.denials if d.code in NON_RETRYABLE_DENIALS), None)


class PlanValidator:
    def __init__(self, policy: ToolAuthorizationPolicy):
        self.policy = policy

    def validate(
        self,
        plan: PlanOutput,
        *,
        context: AuthorizationContext,
        budget: RunBudget,
        usage: BudgetUsage,
        executed: set[tuple[str, str]],
    ) -> PlanValidation:
        limits = self.policy.limits
        result = PlanValidation()
        remaining = remaining_tool_calls(budget, usage)
        if len(plan.steps) > limits.max_plan_steps:
            result.budget_exceeded = True  # over the step limit: never truncated, reported as a limit
            result.problems.append(
                f"The plan has {len(plan.steps)} steps; at most {limits.max_plan_steps} are allowed."
            )
        if context.iteration == 1 and not plan.steps:
            result.problems.append("The plan has no steps.")
        seen = set(executed)
        for index, step in enumerate(plan.steps[: limits.max_plan_steps], start=1):
            try:
                arguments = json.loads(step.arguments_json or "{}")
            except json.JSONDecodeError as exc:
                result.problems.append(
                    f"step {index} ({step.tool_name[:40]}): arguments are not valid JSON ({exc.msg})"
                )
                continue
            decision = self.policy.authorize(step.tool_name, arguments, context=context, budget=budget, usage=usage)
            if not decision.allowed:
                result.denials.append(decision)
                if decision.code not in NON_RETRYABLE_DENIALS:
                    result.problems.append(f"step {index} ({step.tool_name[:40]}): {decision.reason}")
                continue
            assert decision.arguments is not None
            key = (step.tool_name, json.dumps(decision.arguments, sort_keys=True))
            if key in seen:
                continue  # an identical call adds no evidence
            seen.add(key)
            result.steps.append(
                ValidatedStep(tool_name=step.tool_name, arguments=decision.arguments, purpose=step.purpose[:200])
            )
        if len(result.steps) > remaining:
            result.budget_exceeded = True
            result.problems.append(f"The plan has {len(result.steps)} tool calls but only {remaining} remain.")
        sql_steps = sum(1 for s in result.steps if s.tool_name == SQL_TOOL)
        sql_remaining = max(0, budget.max_sql_calls - usage.sql_calls)
        if sql_steps > sql_remaining:
            result.budget_exceeded = True
            result.problems.append(f"The plan has {sql_steps} SQL queries but only {sql_remaining} remain.")
        return result
