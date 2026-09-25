"""Tool authorization and plan validation: the model proposes, the application decides."""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.agent.request import validate_understanding
from app.llm.deterministic.planning import plan as deterministic_plan
from app.llm.deterministic.understanding import understand
from app.llm.schemas import Intent, PlanOutput, UnderstandingOutput
from app.security.authorization import (
    ALLOWED_TOOLS,
    INTENT_TOOL_PERMISSIONS,
    SQL_TOOL,
    AuthorizationContext,
    ToolAuthorizationPolicy,
    ToolPermissions,
)
from app.security.budget import BudgetUsage, RunBudget
from app.security.limits import SecurityLimits
from app.security.plan_validator import PlanValidator
from app.tools import TOOL_DEFINITIONS, ToolRegistry
from tests.phase4_support import AS_OF, understanding_context
from tests.phase5_support import AUGUST, JULY_COMPARISON, plan

LIMITS = SecurityLimits()
REGISTRY = ToolRegistry()
POLICY = ToolAuthorizationPolicy(REGISTRY, LIMITS)
BUDGET = RunBudget.from_limits(LIMITS)
FRESH = BudgetUsage()


def _ctx(intent: Intent | None = Intent.KPI_LOOKUP, **kwargs: Any) -> AuthorizationContext:
    return AuthorizationContext(intent=intent, **kwargs)


def _authorize(
    tool: str, args: Any, *, policy: ToolAuthorizationPolicy = POLICY, usage: BudgetUsage = FRESH, **context: Any
) -> Any:
    ctx = _ctx(**context) if context else _ctx()
    return policy.authorize(tool, args, context=ctx, budget=BUDGET, usage=usage)


# ---- allowlist and registry -------------------------------------------------------------------------


def test_allowlist_is_explicit_and_matches_the_registry() -> None:
    assert len(ALLOWED_TOOLS) == 12 and set(REGISTRY.names) == ALLOWED_TOOLS
    for definition in TOOL_DEFINITIONS:
        assert definition.handler.__module__ == "app.tools.handlers"  # plain functions, no dynamic imports
    for intent in Intent:
        assert intent in INTENT_TOOL_PERMISSIONS
        assert INTENT_TOOL_PERMISSIONS[intent] <= ALLOWED_TOOLS
    assert not INTENT_TOOL_PERMISSIONS[Intent.UNSUPPORTED]
    assert SQL_TOOL not in INTENT_TOOL_PERMISSIONS[Intent.FORECAST]


@pytest.mark.parametrize("tool", ["read_files", "exec_python", "shell", "os.system", "__import__", "", "GET_KPI"])
def test_unknown_tools_are_denied(tool: str) -> None:
    decision = _authorize(tool, {})
    assert not decision.allowed and decision.code == "unknown_tool" and "Unknown tool" in decision.reason


def test_registered_but_not_allowlisted_tool_is_denied() -> None:
    policy = ToolAuthorizationPolicy(REGISTRY, LIMITS, ToolPermissions(allowlist=ALLOWED_TOOLS - {"get_kpi"}))
    assert _authorize("get_kpi", {"kpi": "revenue"}, policy=policy).code == "unknown_tool"


def test_disabled_tools() -> None:
    policy = ToolAuthorizationPolicy(REGISTRY, LIMITS, ToolPermissions(disabled=frozenset({"get_kpi"})))
    assert _authorize("get_kpi", {"kpi": "revenue"}, policy=policy).code == "tool_disabled"
    with pytest.raises(ValueError, match="unknown tools"):
        ToolAuthorizationPolicy(REGISTRY, LIMITS, ToolPermissions(disabled=frozenset({"no_such_tool"})))


def test_intent_permissions_and_sql_privilege() -> None:
    sql = {"sql": "SELECT segment FROM customers"}
    assert _authorize(SQL_TOOL, sql, intent=Intent.FORECAST).code == "tool_not_permitted"
    assert _authorize("get_kpi", {"kpi": "revenue"}, intent=Intent.UNSUPPORTED).code == "tool_not_permitted"
    assert _authorize("get_kpi", {"kpi": "revenue"}, intent=None).code == "prerequisite_missing"
    assert _authorize(SQL_TOOL, sql, intent=Intent.KPI_LOOKUP, sql_permitted=False).code == "sql_not_permitted"
    assert _authorize(SQL_TOOL, sql, intent=Intent.KPI_LOOKUP).allowed


# ---- arguments --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "args", "code"),
    [
        ("get_kpi", ["revenue"], "invalid_arguments"),
        ("get_kpi", {"kpi": "revenue", "extra": 1}, "invalid_arguments"),
        ("get_kpi", {"kpi": "revenue", "start_date": "2026-08-31"}, "invalid_arguments"),
        ("get_kpi", {"kpi": "stock_price"}, "unsupported_kpi"),
        ("get_kpi", {"kpi": "revenue", "dimension": "star_sign"}, "unsupported_dimension"),
        ("get_kpi", {"kpi": "revenue", "dimension": "customer_id"}, "data_policy"),
        ("get_kpi", {"kpi": "revenue", "filters": {"segment": "Galactic"}}, "invalid_filter_value"),
        ("get_kpi", {"kpi": "revenue", "filters": {"planet": "Mars"}}, "unsupported_dimension"),
        ("get_kpi", {"kpi": "revenue", "start_date": "2026-08-31", "end_date": "2026-08-01"}, "invalid_date_range"),
        ("get_kpi", {"kpi": "revenue", "period": "next_decade"}, "invalid_period"),
        ("get_kpi", {"kpi": "revenue", "period": "x" * 600}, "oversized_input"),
        ("get_kpi", {"kpi": "revenue", "filters": {k: "x" for k in "abcdefghij"}}, "oversized_input"),
    ],
)
def test_invalid_arguments_are_rejected_before_execution(tool: str, args: Any, code: str) -> None:
    decision = _authorize(tool, args)
    assert not decision.allowed and decision.code == code


def test_filter_count_and_argument_size_limits() -> None:
    policy = ToolAuthorizationPolicy(REGISTRY, SecurityLimits(max_filters=1, max_argument_chars=500))
    two = {"kpi": "revenue", "filters": {"segment": "SMB", "region": "APAC"}}
    assert _authorize("get_kpi", two, policy=policy).code == "oversized_input"
    huge = {"kpi": "revenue", "filters": {"segment": "SMB"}, "period": "last_month", "padding": "x" * 600}
    assert _authorize("get_kpi", huge, policy=policy).code == "oversized_input"


@pytest.mark.parametrize(
    ("tool", "args", "intent", "code"),
    [
        ("forecast_metric", {"metric": "win_rate"}, Intent.FORECAST, "unsupported_metric"),
        ("forecast_metric", {"metric": "revenue", "horizon": 24}, Intent.FORECAST, "invalid_horizon"),
        ("forecast_metric", {"metric": "revenue", "cutoff_date": "1850-01-01"}, Intent.FORECAST, "invalid_date_range"),
        ("detect_anomalies", {"metric": "revenue", "detector": "magic"}, Intent.ANOMALY_DETECTION, "invalid_arguments"),
        ("detect_anomalies", {"metric": "cac"}, Intent.ANOMALY_DETECTION, "unsupported_metric"),
        ("get_customer_risk", {"limit": 150}, Intent.CUSTOMER_INVESTIGATION, "data_policy"),
        (
            "analyze_sales",
            {"operation": "sales_performance", "dimension": "sales_rep"},
            Intent.SALES_ANALYSIS,
            "data_policy",
        ),
        (SQL_TOOL, {"sql": "DROP TABLE customers"}, Intent.KPI_LOOKUP, "unsafe_sql"),
        (SQL_TOOL, {"sql": "SELECT company_name FROM customers"}, Intent.KPI_LOOKUP, "unsafe_sql"),
        (SQL_TOOL, {"sql": "SELECT segment FROM customers", "max_rows": 5000}, Intent.KPI_LOOKUP, "oversized_input"),
    ],
)
def test_tool_specific_policy(tool: str, args: dict[str, Any], intent: Intent, code: str) -> None:
    decision = _authorize(tool, args, intent=intent)
    assert not decision.allowed and decision.code == code


def test_budget_and_prerequisites() -> None:
    spent = BudgetUsage(tool_calls=BUDGET.max_tool_calls)
    assert _authorize("get_kpi", {"kpi": "revenue"}, usage=spent).code == "budget_exceeded"
    sql_spent = BudgetUsage(sql_calls=BUDGET.max_sql_calls)
    sql = {"sql": "SELECT segment FROM customers"}
    assert _authorize(SQL_TOOL, sql, usage=sql_spent).code == "budget_exceeded"
    rows_spent = BudgetUsage(sql_rows=BUDGET.max_sql_rows)
    assert _authorize(SQL_TOOL, sql, usage=rows_spent).code == "budget_exceeded"
    assert _authorize("get_kpi", {"kpi": "revenue"}, iteration=2).code == "prerequisite_missing"
    assert _authorize("get_kpi", {"kpi": "revenue"}, iteration=2, has_prior_evidence=True).allowed


def test_allowed_decision_records_the_checks_and_canonical_arguments() -> None:
    decision = _authorize("get_kpi", {"kpi": "revenue", **AUGUST})
    assert decision.allowed and decision.arguments == {"kpi": "revenue", **AUGUST}
    assert decision.checks_passed == [
        "allowlist",
        "enabled",
        "intent",
        "sql_privilege",
        "typed_arguments",
        "argument_policy",
        "budget",
        "prerequisites",
    ]


# ---- plan validation -------------------------------------------------------------------------------------

VALIDATOR = PlanValidator(POLICY)


def _validate(
    output: dict[str, Any],
    *,
    context: AuthorizationContext | None = None,
    usage: BudgetUsage = FRESH,
    executed: set[tuple[str, str]] | None = None,
    validator: PlanValidator = VALIDATOR,
) -> Any:
    return validator.validate(
        PlanOutput.model_validate(output),
        context=context or _ctx(Intent.REVENUE_INVESTIGATION),
        budget=BUDGET,
        usage=usage,
        executed=executed or set(),
    )


def test_valid_plan_and_duplicates() -> None:
    step = ("analyze_revenue", {"operation": "revenue_change", **AUGUST, **JULY_COMPARISON})
    result = _validate(plan(step, step))
    assert result.valid and len(result.steps) == 1
    executed = {(result.steps[0].tool_name, json.dumps(result.steps[0].arguments, sort_keys=True))}
    assert not _validate(plan(step), executed=executed).steps  # already executed: not repeated


def test_structural_problems_are_retryable() -> None:
    bad_json = {"steps": [{"tool_name": "get_kpi", "arguments_json": "{not json", "purpose": "x"}]}
    result = _validate(bad_json)
    assert result.problems and result.security_denial is None
    assert "not valid JSON" in result.problems[0]
    assert _validate({"steps": []}).problems == ["The plan has no steps."]
    unknown = _validate(plan(("launch_rocket", {})))
    assert unknown.problems and unknown.security_denial is None


def test_security_denials_are_not_retryable() -> None:
    denied = _validate(
        plan((SQL_TOOL, {"sql": "SELECT segment FROM customers"})),
        context=_ctx(Intent.REVENUE_INVESTIGATION, sql_permitted=False),
    )
    assert denied.security_denial is not None and denied.security_denial.code == "sql_not_permitted"
    unsafe = _validate(plan((SQL_TOOL, {"sql": "DELETE FROM customers"})))
    assert unsafe.security_denial is not None and unsafe.security_denial.code == "unsafe_sql"
    forbidden = _validate(plan(("forecast_metric", {"metric": "revenue"})))
    assert forbidden.security_denial is not None and forbidden.security_denial.code == "tool_not_permitted"


def test_plan_step_and_budget_limits() -> None:
    small = PlanValidator(ToolAuthorizationPolicy(REGISTRY, SecurityLimits(max_tool_calls=3)))
    steps = [("get_kpi", {"kpi": "revenue", "period": p}) for p in ("2026-05", "2026-06", "2026-07", "2026-08")]
    result = _validate(plan(*steps), validator=small)
    assert result.budget_exceeded and not result.valid
    nearly_spent = BudgetUsage(tool_calls=BUDGET.max_tool_calls - 1)
    assert _validate(plan(*steps[:2]), usage=nearly_spent).budget_exceeded
    sql = [(SQL_TOOL, {"sql": f"SELECT segment FROM customers LIMIT {n}"}) for n in (1, 2, 3, 4)]
    assert _validate(plan(*sql)).budget_exceeded  # more SQL steps than the SQL budget


QUESTIONS = [
    "What was revenue last month?",
    "How did MRR change compared with the previous month?",
    "Which segment contributed most to the revenue decline last month?",
    "Why did revenue decline last month?",
    "Forecast revenue for the next 3 months",
    "Were there any anomalies last month?",
    "How did support tickets change last month?",
    "Which segment had the highest churn last month?",
    "What caused churn?",
    "Which customers are at risk?",
    "Show cohort retention",
    "How did revenue and support tickets change last month?",
    "What was the win rate last quarter?",
    "How is marketing performing?",
    "What is adoption of Dashboards?",
]


@pytest.mark.parametrize("question", QUESTIONS)
def test_every_deterministic_playbook_is_authorized(question: str) -> None:
    """The security layer must not block legitimate plans: every playbook step passes authorization."""
    understood = UnderstandingOutput.model_validate(understand(understanding_context(question)))
    from datetime import date

    request = validate_understanding(understood, as_of=AS_OF, coverage=(date(2024, 9, 1), date(2026, 8, 31))).request
    assert request is not None
    output = PlanOutput.model_validate(deterministic_plan({"request": request.model_dump(mode="json")}))
    result = _validate(output.model_dump(), context=_ctx(request.intent))
    assert result.valid, [d.reason for d in result.denials] + result.problems
