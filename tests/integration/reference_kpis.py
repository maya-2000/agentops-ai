"""Independent pandas reference implementations of the 20 KPIs.

These deliberately share **no code** with ``app/analytics``. Raw tables are extracted with
``SELECT *`` and every definition is re-implemented from its written business meaning in
pandas. The production SQL is never reused. Agreement between the two implementations is the
correctness evidence for the KPI layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

TABLES = (
    "customers",
    "subscriptions",
    "sales_opportunities",
    "support_tickets",
    "marketing_campaigns",
    "product_features",
    "usage_events",
)
CUSTOMER_ATTRIBUTES = ("region", "country", "segment", "industry", "acquisition_channel", "customer_id")


def load_raw_tables(db_path: Path) -> dict[str, pd.DataFrame]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        frames = {name: con.execute(f'SELECT * FROM "{name}"').df() for name in TABLES}
        frames["daily_revenue"] = con.execute(
            "SELECT date, customer_id, plan, revenue_type, revenue FROM daily_revenue"
        ).df()
    finally:
        con.close()
    for frame in frames.values():
        for col in frame.columns:
            if str(frame[col].dtype).startswith("datetime64"):
                frame[col] = frame[col].astype("datetime64[ns]")
            elif frame[col].dtype == object and col in (
                "revenue",
                "monthly_recurring_revenue",
                "previous_mrr",
                "current_mrr",
                "deal_value",
                "spend",
                "probability",
            ):
                frame[col] = frame[col].astype(float)
    for name, cols in {
        "subscriptions": ("monthly_recurring_revenue", "previous_mrr", "current_mrr"),
        "sales_opportunities": ("deal_value", "probability"),
        "marketing_campaigns": ("spend",),
        "daily_revenue": ("revenue",),
    }.items():
        for col in cols:
            frames[name][col] = frames[name][col].astype(float)
    return frames


def ts(d: date) -> pd.Timestamp:
    return pd.Timestamp(d)


@dataclass
class Reference:
    t: dict[str, pd.DataFrame]

    # ------------------------------------------------------------------ building blocks
    def customers(self, filters: dict[str, str]) -> pd.DataFrame:
        c = self.t["customers"]
        mask = np.ones(len(c), dtype=bool)
        for key, value in filters.items():
            if key in CUSTOMER_ATTRIBUTES:
                mask &= (c[key] == value).to_numpy()
        return c[mask]

    def churn_dates(self) -> pd.Series:
        s = self.t["subscriptions"]
        churned = s[s["status"] == "churned"]
        return pd.Series(churned["end_date"].to_numpy(), index=churned["customer_id"].to_numpy())

    def in_force(self, when: date) -> pd.DataFrame:
        """Records in force at the close of ``when`` (a churned record ending that day is excluded)."""
        s = self.t["subscriptions"]
        w = ts(when)
        end = s["end_date"]
        started = s["start_date"] <= w
        running = end.isna() | (end > w) | ((end == w) & (s["status"] == "superseded"))
        return s[started & running]

    # ------------------------------------------------------------------ revenue
    def revenue(self, start: date, end: date, filters: dict[str, str]) -> dict[str, float]:
        r = self.t["daily_revenue"]
        r = r[(r["date"] >= ts(start)) & (r["date"] <= ts(end))]
        if "plan" in filters:
            r = r[r["plan"] == filters["plan"]]
        if "revenue_type" in filters:
            r = r[r["revenue_type"] == filters["revenue_type"]]
        r = r[r["customer_id"].isin(self.customers(filters)["customer_id"])]
        return {"revenue": float(r["revenue"].sum()), "observations": len(r)}

    def revenue_growth(self, start: date, end: date, c_start: date, c_end: date, filters: dict[str, str]) -> float:
        current = self.revenue(start, end, filters)["revenue"]
        previous = self.revenue(c_start, c_end, filters)["revenue"]
        return current / previous - 1

    # ------------------------------------------------------------------ recurring state
    def recurring_state(self, when: date, filters: dict[str, str]) -> dict[str, float]:
        s = self.in_force(when)
        if "plan" in filters:
            s = s[s["plan"] == filters["plan"]]
        s = s[s["customer_id"].isin(self.customers(filters)["customer_id"])]
        return {"mrr": float(s["monthly_recurring_revenue"].sum()), "customers": s["customer_id"].nunique()}

    # ------------------------------------------------------------------ opening cohort
    def cohort(self, start: date, end: date, filters: dict[str, str]) -> dict[str, float]:
        opening = self.in_force(start - timedelta(days=1))
        if "plan" in filters:
            opening = opening[opening["plan"] == filters["plan"]]
        opening = opening[opening["customer_id"].isin(self.customers(filters)["customer_id"])]
        ids = set(opening["customer_id"])
        s = self.t["subscriptions"]
        in_period = (s["start_date"] >= ts(start)) & (s["start_date"] <= ts(end)) & s["customer_id"].isin(ids)
        expansion = s[in_period & (s["change_type"] == "expansion")]
        contraction = s[in_period & (s["change_type"] == "contraction")]
        churned = s[
            (s["status"] == "churned")
            & (s["end_date"] >= ts(start))
            & (s["end_date"] <= ts(end))
            & s["customer_id"].isin(ids)
        ]
        return {
            "opening_customers": len(ids),
            "churned_customers": churned["customer_id"].nunique(),
            "opening_mrr": float(opening["monthly_recurring_revenue"].sum()),
            "churned_mrr": float(churned["monthly_recurring_revenue"].sum()),
            "expansion_mrr": float((expansion["monthly_recurring_revenue"] - expansion["previous_mrr"]).sum()),
            "contraction_mrr": float((contraction["previous_mrr"] - contraction["monthly_recurring_revenue"]).sum()),
        }

    def logo_churn(self, start: date, end: date, filters: dict[str, str]) -> float:
        c = self.cohort(start, end, filters)
        return c["churned_customers"] / c["opening_customers"]

    def nrr(self, start: date, end: date, filters: dict[str, str]) -> float:
        c = self.cohort(start, end, filters)
        return (c["opening_mrr"] - c["churned_mrr"] - c["contraction_mrr"] + c["expansion_mrr"]) / c["opening_mrr"]

    def cohort_closing_mrr(self, start: date, end: date, filters: dict[str, str]) -> float:
        """MRR at the period end of the customers who were active at the opening (for the NRR identity)."""
        opening = self.in_force(start - timedelta(days=1))
        ids = set(opening[opening["customer_id"].isin(self.customers(filters)["customer_id"])]["customer_id"])
        closing = self.in_force(end)
        return float(closing[closing["customer_id"].isin(ids)]["monthly_recurring_revenue"].sum())

    def clv(self, start: date, end: date, months: float, filters: dict[str, str]) -> float:
        state = self.recurring_state(end, filters)
        arpa = state["mrr"] / state["customers"]
        c = self.cohort(start, end, filters)
        monthly_churn = c["churned_customers"] / c["opening_customers"] / months
        return arpa / monthly_churn

    # ------------------------------------------------------------------ marketing
    def marketing(self, start: date, end: date, filters: dict[str, str]) -> dict[str, float]:
        m = self.t["marketing_campaigns"]
        m = m[(m["date"] >= ts(start)) & (m["date"] <= ts(end))]
        if "acquisition_channel" in filters:
            m = m[m["channel"] == filters["acquisition_channel"]]
        if "campaign" in filters:
            m = m[m["campaign_id"] == filters["campaign"]]
        return {
            "spend": float(m["spend"].sum()),
            "leads": int(m["leads"].sum()),
            "conversions": int(m["conversions"].sum()),
        }

    # ------------------------------------------------------------------ sales
    def closed(self, start: date, end: date, filters: dict[str, str]) -> pd.DataFrame:
        o = self.t["sales_opportunities"]
        o = o[o["stage"].isin(["Won", "Lost"]) & (o["close_date"] >= ts(start)) & (o["close_date"] <= ts(end))]
        for key in ("segment", "region", "sales_rep", "opportunity_type", "customer_id"):
            if key in filters:
                o = o[o[key] == filters[key]]
        return o

    def win_rate(self, start: date, end: date, filters: dict[str, str]) -> float:
        o = self.closed(start, end, filters)
        return float((o["stage"] == "Won").mean())

    def average_order_value(self, start: date, end: date, filters: dict[str, str]) -> float:
        o = self.closed(start, end, filters)
        return float(o.loc[o["stage"] == "Won", "deal_value"].mean())

    def sales_cycle(self, start: date, end: date, filters: dict[str, str]) -> float:
        o = self.closed(start, end, filters)
        return float((o["close_date"] - o["created_date"]).dt.days.mean())

    def pipeline(self, when: date, filters: dict[str, str]) -> float:
        o = self.t["sales_opportunities"]
        mask = (o["created_date"] <= ts(when)) & (o["close_date"].isna() | (o["close_date"] > ts(when)))
        for key in ("segment", "region", "sales_rep", "opportunity_type"):
            if key in filters:
                mask &= o[key] == filters[key]
        return float(o.loc[mask, "deal_value"].sum())

    # ------------------------------------------------------------------ support
    def tickets(self, start: date, end: date, filters: dict[str, str]) -> pd.DataFrame:
        t = self.t["support_tickets"]
        t = t[(t["created_at"] >= ts(start)) & (t["created_at"] < ts(end) + pd.Timedelta(days=1))]
        for key, column in (("ticket_category", "category"), ("ticket_priority", "priority")):
            if key in filters:
                t = t[t[column] == filters[key]]
        return t[t["customer_id"].isin(self.customers(filters)["customer_id"])]

    # ------------------------------------------------------------------ product
    def adoption(self, start: date, end: date, feature: str) -> float:
        p = self.t["product_features"]
        p = p[(p["feature_name"] == feature) & (p["date"] >= ts(start)) & (p["date"] <= ts(end))]
        return float(p["adoption_rate"].mean())
