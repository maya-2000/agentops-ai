"""Pydantic models for KPI definitions, parameters and results."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from app.analytics.dimensions import FILTER_KEYS, Filters
from app.analytics.errors import InvalidRequestError, ResultStatus
from app.analytics.models import Provenance, Scalar, safe_ratio
from app.analytics.periods import Period

RuleKind = Literal["component", "ratio", "complement_ratio", "scaled", "growth", "net_retention", "lifetime_value"]


class ValueRule(BaseModel):
    """How a KPI value is derived from the components its SQL returns.

    Kinds:
    - ``component``: value = component
    - ``ratio``: value = numerator / denominator
    - ``complement_ratio``: value = 1 - numerator / denominator
    - ``scaled``: value = component x factor
    - ``growth``: value = (numerator - denominator) / denominator
    - ``net_retention``: value = (opening_mrr - churned_mrr - contraction_mrr + expansion_mrr) / opening_mrr
    - ``lifetime_value``: value = (closing_mrr / closing_customers)
      / (churned_customers / opening_customers / period_months)
    """

    model_config = ConfigDict(frozen=True)

    kind: RuleKind
    component: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    factor: float | None = None

    def apply(self, c: dict[str, Scalar]) -> tuple[float | None, ResultStatus, str | None]:
        """Return (value, status, message) for a set of components."""

        def num(name: str | None) -> float | None:
            value = c.get(name) if name else None
            return None if value is None or isinstance(value, str) else float(value)

        if self.kind == "component":
            value = num(self.component)
            return (value, "ok", None) if value is not None else (None, "no_data", f"{self.component} is missing")
        if self.kind == "scaled":
            value = num(self.component)
            if value is None:
                return None, "no_data", f"{self.component} is missing"
            return value * float(self.factor or 1.0), "ok", None
        if self.kind in ("ratio", "complement_ratio", "growth"):
            n, d = num(self.numerator), num(self.denominator)
            if d is None or d == 0:
                return None, "insufficient_data", f"Denominator {self.denominator} is zero or missing"
            ratio = safe_ratio(n, d)
            if ratio is None:
                return None, "no_data", f"Numerator {self.numerator} is missing"
            if self.kind == "complement_ratio":
                return 1.0 - ratio, "ok", None
            if self.kind == "growth":
                return ratio - 1.0, "ok", None
            return ratio, "ok", None
        if self.kind == "net_retention":
            opening = num("opening_mrr")
            if not opening:
                return None, "insufficient_data", "Opening MRR is zero: no customers were active at the period opening"
            retained = opening - (num("churned_mrr") or 0) - (num("contraction_mrr") or 0) + (num("expansion_mrr") or 0)
            return retained / opening, "ok", None
        if self.kind == "lifetime_value":
            arpa = safe_ratio(num("closing_mrr"), num("closing_customers"))
            churn = safe_ratio(num("churned_customers"), num("opening_customers"))
            months = num("period_months")
            if arpa is None:
                return None, "insufficient_data", "No active customers at the period end"
            if churn is None or not months:
                return None, "insufficient_data", "No customers were active at the period opening"
            if churn == 0:
                return None, "insufficient_data", "No churn observed in the period: expected lifetime is unbounded"
            return arpa / (churn / months), "ok", None
        raise AssertionError(f"Unhandled rule kind {self.kind}")


class KPIDefinition(BaseModel):
    """A registered KPI: business meaning, formula, SQL and applicability."""

    model_config = ConfigDict(frozen=True)

    key: str
    name: str
    definition: str
    formula: str
    sql: str
    unit: str
    time_grain: str
    interpretation: str
    limitations: tuple[str, ...]
    dependencies: tuple[str, ...]
    template: str
    value_rule: ValueRule
    components: tuple[str, ...]
    supported_filters: tuple[str, ...]
    supported_dimensions: tuple[str, ...]
    requires_comparison: bool = False
    # True when "no rows" means "all counts are zero" (e.g. no tickets, no closed deals, empty cohort). The value
    # rule then runs on zero components: counts report a true 0 and ratios report insufficient_data (zero
    # denominator). Never a fabricated zero ratio.
    zero_when_empty: bool = False
    observation_component: str | None = None
    required_any_of: tuple[str, ...] = ()
    insufficient_grains: dict[str, str] = Field(default_factory=dict)
    insufficient_filter_values: dict[str, dict[str, str]] = Field(default_factory=dict)


class KPIParameters(BaseModel):
    """Typed parameters for ``calculate_kpi``. Filters may be given flat (``segment="SMB"``)."""

    model_config = ConfigDict(extra="forbid")

    period: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    comparison_period: str | None = None
    comparison_start_date: date | None = None
    comparison_end_date: date | None = None
    dimension: str | None = None
    dimension_value: str | None = None
    filters: Filters = Field(default_factory=Filters)

    @model_validator(mode="before")
    @classmethod
    def _collect_flat_filters(cls, data: Any) -> Any:
        if isinstance(data, dict):
            flat = {k: data[k] for k in FILTER_KEYS if k in data}
            if flat:
                rest = {k: v for k, v in data.items() if k not in flat}
                existing = rest.get("filters") or {}
                existing = existing.active() if isinstance(existing, Filters) else dict(existing)
                rest["filters"] = {**existing, **flat}
                return rest
        return data

    @model_validator(mode="after")
    def _consistent(self) -> KPIParameters:
        if (self.start_date is None) != (self.end_date is None):
            raise InvalidRequestError("start_date and end_date must be given together")
        if (self.comparison_start_date is None) != (self.comparison_end_date is None):
            raise InvalidRequestError("comparison_start_date and comparison_end_date must be given together")
        if self.start_date and self.period:
            raise InvalidRequestError("Give either period or start_date/end_date, not both")
        if self.comparison_start_date and self.comparison_period:
            raise InvalidRequestError("Give either comparison_period or comparison dates, not both")
        if self.dimension_value is not None and self.dimension is None:
            raise InvalidRequestError("dimension_value requires dimension")
        return self


class KPIBreakdownRow(BaseModel):
    dimension_value: str
    status: ResultStatus
    value: float | None
    components: dict[str, Scalar]
    message: str | None = None


class KPIResult(BaseModel):
    """Evidence-ready KPI result. ``value`` is ``None`` whenever ``status`` is not ``ok``."""

    key: str
    name: str
    status: ResultStatus
    value: float | None
    unit: str
    period: Period
    comparison_period: Period | None = None
    filters: dict[str, str] = Field(default_factory=dict)
    dimension: str | None = None
    breakdown: list[KPIBreakdownRow] = Field(default_factory=list)
    components: dict[str, Scalar] = Field(default_factory=dict)
    formula: str
    interpretation: str
    limitations: list[str] = Field(default_factory=list)
    message: str | None = None
    provenance: Provenance

    @computed_field  # type: ignore[prop-decorator]
    @property
    def calculation(self) -> str:
        return self.provenance.calculation

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sql(self) -> list[str]:
        return [q.sql for q in self.provenance.queries]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_tables(self) -> list[str]:
        return self.provenance.source_tables

    @computed_field  # type: ignore[prop-decorator]
    @property
    def query_ids(self) -> list[str]:
        return self.provenance.query_ids

    @computed_field  # type: ignore[prop-decorator]
    @property
    def operation_id(self) -> str:
        return self.provenance.operation_id

    @computed_field  # type: ignore[prop-decorator]
    @property
    def execution_timestamp(self) -> datetime:
        return self.provenance.execution_timestamp

    def row(self, dimension_value: str) -> KPIBreakdownRow:
        for row in self.breakdown:
            if row.dimension_value == dimension_value:
                return row
        raise KeyError(dimension_value)
