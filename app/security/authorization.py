"""Tool authorization: the application, not the model, decides whether a tool call may run.

``ToolAuthorizationPolicy.authorize`` runs before every execution. The plan validator calls it
too, so a plan is checked when proposed and each call is checked again just before it runs.
Checks, in order (the first failure denies the call):

1. **Allowlist.** The tool is in ``ALLOWED_TOOLS`` *and* registered with its typed handler. There
   are no dynamic imports, no arbitrary callables, no shell and no filesystem tools.
2. **Enabled.** Not switched off in configuration (``AGENT_DISABLED_TOOLS``).
3. **Intent.** Permitted for the validated intent (``INTENT_TOOL_PERMISSIONS``), e.g. a forecast
   question cannot trigger ad-hoc SQL. An unknown intent permits nothing.
4. **SQL privilege.** ``run_safe_sql`` also needs the run's SQL privilege, which is revoked when
   the input screen flagged the question.
5. **Arguments.**
   - size bound;
   - typed validation against the tool's input model (unknown fields rejected);
   - the central argument policy: KPIs, metrics, dimensions, filters, periods, dates, horizons
     and text sizes;
   - data-exposure rules: no per-customer or per-rep breakdowns, a cap on customer-level rows,
     SQL row caps;
   - full SQL validation for ``run_safe_sql``.
6. **Budget.** Tool calls, SQL executions and SQL rows remain in the run budget.
7. **Prerequisites.** A validated request exists. A follow-up iteration requires evidence from
   earlier iterations.

The user's text never reaches this code: permissions depend only on the validated intent, the
screening verdict, configuration and the budget.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.analytics.errors import InvalidRequestError
from app.llm.schemas import Intent
from app.security.budget import BudgetUsage, RunBudget, check_tool_call
from app.security.data_policy import CUSTOMER_LEVEL_OPERATIONS
from app.security.events import Severity
from app.security.limits import SecurityLimits
from app.security.validators import (
    Violation,
    check_date,
    check_date_range,
    check_dimension,
    check_filters,
    check_horizon,
    check_kpi,
    check_period_spec,
    check_series_metric,
    check_text,
)
from app.tools.registry import ToolRegistry
from app.tools.sql_safety import SQLComplexityLimits, UnsafeSQLError, validate_sql

SQL_TOOL = "run_safe_sql"
ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "get_kpi",
        "analyze_revenue",
        "analyze_customers",
        "analyze_sales",
        "analyze_marketing",
        "analyze_support",
        "analyze_product",
        "get_cohort_analysis",
        "get_customer_risk",
        "forecast_metric",
        "detect_anomalies",
        SQL_TOOL,
    }
)
_ANALYTICS = frozenset(
    {
        "analyze_revenue",
        "analyze_customers",
        "analyze_sales",
        "analyze_marketing",
        "analyze_support",
        "analyze_product",
    }
)
INTENT_TOOL_PERMISSIONS: dict[Intent, frozenset[str]] = {
    Intent.KPI_LOOKUP: frozenset({"get_kpi", SQL_TOOL}),
    Intent.PERIOD_COMPARISON: frozenset({"get_kpi", SQL_TOOL}) | _ANALYTICS,
    Intent.DIMENSIONAL_COMPARISON: frozenset({"get_kpi", SQL_TOOL}) | _ANALYTICS,
    Intent.REVENUE_INVESTIGATION: frozenset(
        {"get_kpi", "analyze_revenue", "analyze_customers", "detect_anomalies", SQL_TOOL}
    ),
    Intent.CUSTOMER_INVESTIGATION: frozenset(
        {
            "get_kpi",
            "analyze_customers",
            "analyze_revenue",
            "get_cohort_analysis",
            "get_customer_risk",
            "detect_anomalies",
            SQL_TOOL,
        }
    ),
    Intent.SALES_ANALYSIS: frozenset({"get_kpi", "analyze_sales", SQL_TOOL}),
    Intent.MARKETING_ANALYSIS: frozenset({"get_kpi", "analyze_marketing", SQL_TOOL}),
    Intent.SUPPORT_ANALYSIS: frozenset({"get_kpi", "analyze_support", "detect_anomalies", SQL_TOOL}),
    Intent.PRODUCT_ANALYSIS: frozenset({"get_kpi", "analyze_product", "detect_anomalies", SQL_TOOL}),
    Intent.FORECAST: frozenset({"forecast_metric", "get_kpi"}),
    Intent.ANOMALY_DETECTION: frozenset({"detect_anomalies", "get_kpi"}),
    Intent.MIXED_INVESTIGATION: frozenset(
        {"get_kpi", "get_customer_risk", "detect_anomalies", "forecast_metric", SQL_TOOL} | _ANALYTICS
    ),
    Intent.UNSUPPORTED: frozenset(),
}
# Breakdowns that would list individual customers or people; denied by the data-exposure policy.
RESTRICTED_BREAKDOWNS = frozenset({"customer_id", "sales_rep"})
_TEXT_EXEMPT = frozenset({"sql"})
_DATE_PAIRS = (("start_date", "end_date", "period"), ("comparison_start_date", "comparison_end_date", "comparison"))


class ToolPermissions(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowlist: frozenset[str] = ALLOWED_TOOLS
    disabled: frozenset[str] = frozenset()
    intent_tools: dict[Intent, frozenset[str]] = Field(default_factory=lambda: dict(INTENT_TOOL_PERMISSIONS))

    def permitted(self, intent: Intent) -> frozenset[str]:
        return (self.intent_tools.get(intent, frozenset()) & self.allowlist) - self.disabled


class AuthorizationContext(BaseModel):
    """What authorization may depend on: validated facts about the run, never the user's text."""

    intent: Intent | None
    sql_permitted: bool = True
    iteration: int = 1
    has_prior_evidence: bool = False


class AuthorizationDecision(BaseModel):
    allowed: bool
    tool_name: str
    code: str | None = None
    reason: str = ""
    severity: Severity = Severity.INFO
    arguments: dict[str, Any] | None = None  # canonical arguments when allowed
    checks_passed: list[str] = Field(default_factory=list)


class ToolAuthorizationPolicy:
    def __init__(
        self,
        registry: ToolRegistry,
        limits: SecurityLimits,
        permissions: ToolPermissions | None = None,
    ):
        self.registry = registry
        self.limits = limits
        self.permissions = permissions or ToolPermissions()
        unknown = self.permissions.disabled - ALLOWED_TOOLS
        if unknown:
            raise ValueError(f"Cannot disable unknown tools: {', '.join(sorted(unknown))}")
        self.sql_limits = SQLComplexityLimits(
            max_length=limits.max_sql_length,
            max_joins=limits.max_sql_joins,
            max_nesting_depth=limits.max_sql_nesting_depth,
            max_ctes=limits.max_sql_ctes,
            max_parameters=limits.max_sql_parameters,
        )

    def authorize(
        self,
        tool_name: str,
        arguments: Any,
        *,
        context: AuthorizationContext,
        budget: RunBudget,
        usage: BudgetUsage,
    ) -> AuthorizationDecision:
        passed: list[str] = []

        def deny(code: str, reason: str, severity: Severity = Severity.HIGH) -> AuthorizationDecision:
            return AuthorizationDecision(
                allowed=False,
                tool_name=str(tool_name)[:80],
                code=code,
                reason=reason,
                severity=severity,
                checks_passed=passed,
            )

        if not isinstance(tool_name, str) or tool_name not in self.permissions.allowlist:
            return deny("unknown_tool", "Unknown tool: it is not on the allowlist.")
        if self.registry.get(tool_name) is None:
            return deny("unknown_tool", "Unknown tool: it is not registered.")
        passed.append("allowlist")
        if tool_name in self.permissions.disabled:
            return deny("tool_disabled", "The tool is disabled by configuration.", Severity.WARNING)
        passed.append("enabled")
        if context.intent is None:
            return deny("prerequisite_missing", "No validated request: tools cannot run.")
        if tool_name not in self.permissions.permitted(context.intent):
            return deny("tool_not_permitted", f"{tool_name} is not permitted for intent {context.intent.value}.")
        passed.append("intent")
        if tool_name == SQL_TOOL and not context.sql_permitted:
            return deny("sql_not_permitted", "Ad-hoc SQL is disabled for this run (flagged input).")
        passed.append("sql_privilege")

        if not isinstance(arguments, Mapping):
            return deny("invalid_arguments", "Arguments must be an object.", Severity.WARNING)
        size = len(json.dumps(arguments, default=str))
        if size > self.limits.max_argument_chars:
            return deny("oversized_input", f"Arguments are {size} characters (limit {self.limits.max_argument_chars}).")
        try:
            canonical = self.registry.validate_arguments(tool_name, dict(arguments))
        except InvalidRequestError as exc:
            code = "invalid_arguments" if type(exc) is InvalidRequestError else exc.code
            return deny(code, f"Invalid arguments: {str(exc)[:200]}", Severity.WARNING)
        passed.append("typed_arguments")
        violations = self.argument_violations(tool_name, canonical)
        if violations:
            first = violations[0]
            severity = Severity.HIGH if first.code in ("data_policy", "unsafe_sql") else Severity.WARNING
            return deny(first.code, "; ".join(f"{v.field}: {v.message}" for v in violations[:3]), severity)
        passed.append("argument_policy")

        exhausted = check_tool_call(budget, usage, sql=tool_name == SQL_TOOL)
        if exhausted is not None:
            return deny("budget_exceeded", f"The {exhausted.replace('_', ' ')} budget is exhausted.")
        passed.append("budget")
        if context.iteration > 1 and not context.has_prior_evidence:
            return deny("prerequisite_missing", "A follow-up step requires evidence from the first iteration.")
        passed.append("prerequisites")
        return AuthorizationDecision(
            allowed=True,
            tool_name=tool_name,
            reason="All authorization checks passed.",
            arguments=canonical,
            checks_passed=passed,
        )

    # ------------------------------------------------------------------ argument policy
    def argument_violations(self, tool_name: str, args: Mapping[str, Any]) -> list[Violation]:
        """Central value rules for canonical tool arguments (after typed validation)."""
        limits = self.limits
        problems: list[Violation] = []
        for key, value in args.items():
            if isinstance(value, str) and key not in _TEXT_EXEMPT:
                problems += check_text(value, key, limits.max_text_field_chars)
        if "filters" in args:
            problems += check_filters(args["filters"], max_filters=limits.max_filters)
        if args.get("dimension") is not None:
            problems += check_dimension(args["dimension"])
            if args["dimension"] in RESTRICTED_BREAKDOWNS:
                problems.append(
                    Violation(
                        code="data_policy",
                        field="dimension",
                        message="Breakdowns by individual customer or person are not exposed",
                    )
                )
        for field in ("period", "comparison_period"):
            if args.get(field) is not None:
                problems += check_period_spec(str(args[field]).strip().lower(), field)
        for start, end, label in _DATE_PAIRS:
            if args.get(start) is not None and args.get(end) is not None:
                problems += check_date_range(args[start], args[end], label)
        if tool_name == "get_kpi":
            problems += check_kpi(args.get("kpi"))
        if tool_name in ("forecast_metric", "detect_anomalies"):
            problems += check_series_metric(args.get("metric"))
        if tool_name == "forecast_metric":
            if "horizon" in args:
                problems += check_horizon(args["horizon"])
            if args.get("cutoff_date") is not None:
                problems += check_date(args["cutoff_date"], "cutoff_date")
        if tool_name == "detect_anomalies":
            for field in ("start_date", "end_date"):
                if args.get(field) is not None:
                    problems += check_date(args[field], field)
        if tool_name in CUSTOMER_LEVEL_OPERATIONS and int(args.get("limit", 20)) > limits.max_customer_rows:
            problems.append(
                Violation(
                    code="data_policy",
                    field="limit",
                    message=f"At most {limits.max_customer_rows} customer-level rows may be returned",
                )
            )
        if tool_name == SQL_TOOL:
            problems += self._sql_violations(args)
        return problems

    def _sql_violations(self, args: Mapping[str, Any]) -> list[Violation]:
        max_rows = args.get("max_rows")
        if max_rows is not None and int(max_rows) > self.limits.sql_row_limit:
            return [
                Violation(
                    code="oversized_input",
                    field="max_rows",
                    message=f"At most {self.limits.sql_row_limit} rows per query",
                )
            ]
        try:
            validate_sql(
                str(args.get("sql", "")),
                dict(args.get("parameters") or {}),
                self.limits.sql_row_limit,
                limits=self.sql_limits,
            )
        except UnsafeSQLError as exc:
            return [Violation(code="unsafe_sql", field="sql", message=str(exc)[:200])]
        return []
