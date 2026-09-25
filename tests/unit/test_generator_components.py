"""Unit tests for generator building blocks."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from data.generator.acquisition import plan_campaigns
from data.generator.config import PLAN_PARAMS, SALES_TEAM, GeneratorConfig
from data.generator.entities import company_names, create_sales_reps, mrr_for
from data.generator.events import EVENTS, GroundTruthLog
from data.generator.timeline import Timeline, holiday_factor


class TestGeneratorConfig:
    def test_defaults_match_business_context(self) -> None:
        cfg = GeneratorConfig()
        assert cfg.seed == 42
        assert cfg.customer_count == 5000
        assert (cfg.start_date, cfg.end_date) == (date(2024, 9, 1), date(2026, 8, 31))
        assert cfg.n_months == 24

    def test_opening_and_new_customers_sum_to_total(self) -> None:
        cfg = GeneratorConfig()
        assert cfg.opening_customer_count + cfg.new_customer_count == cfg.customer_count
        assert 0 < cfg.opening_customer_count < cfg.customer_count

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            (date(2024, 9, 2), date(2026, 8, 31)),
            (date(2024, 9, 1), date(2026, 8, 30)),
            (date(2026, 1, 1), date(2025, 12, 31)),
        ],
    )
    def test_rejects_invalid_windows(self, start: date, end: date) -> None:
        with pytest.raises(ValidationError):
            GeneratorConfig(start_date=start, end_date=end)


class TestTimeline:
    def test_months(self) -> None:
        tl = Timeline(date(2024, 9, 1), date(2026, 8, 31))
        assert tl.n_months == 24
        assert tl.month_starts[0] == date(2024, 9, 1)
        assert tl.month_ends[-1] == date(2026, 8, 31)
        assert tl.month_ends[5] == date(2025, 2, 28)

    def test_weeks_are_full_mondays_inside_window(self) -> None:
        tl = Timeline(date(2024, 9, 1), date(2026, 8, 31))
        weeks = tl.week_starts
        assert all(w.weekday() == 0 for w in weeks)
        assert weeks[0] >= tl.start and weeks[-1] + timedelta(days=6) <= tl.end
        assert len(weeks) == 104

    def test_holiday_factor_regional(self) -> None:
        lunar_new_year_week = date(2026, 2, 16)
        assert holiday_factor(lunar_new_year_week, "APAC") < 1.0
        assert holiday_factor(lunar_new_year_week, "EMEA") == 1.0


class TestEntities:
    def test_mrr_is_seats_times_price_less_discount(self) -> None:
        mrr = mrr_for(np.array(["Growth", "Enterprise"]), np.array([10, 100]), np.array([0.0, 0.2]))
        assert mrr.tolist() == [
            10 * PLAN_PARAMS["Growth"].price_per_seat,
            round(100 * PLAN_PARAMS["Enterprise"].price_per_seat * 0.8, 2),
        ]

    def test_company_names_unique_and_country_specific(self) -> None:
        rng = np.random.default_rng(0)
        countries = np.array(["Singapore", "Indonesia", "Germany"] * 200, dtype=object)
        industries = np.array(["Fintech"] * 600, dtype=object)
        names = company_names(rng, countries, industries)
        assert len(set(names)) == 600
        assert names[0].endswith("Pte Ltd")
        assert names[1].startswith("PT ")
        assert names[2].endswith("GmbH")

    def test_sales_team_and_low_conversion_rep(self) -> None:
        truth = GroundTruthLog()
        reps = create_sales_reps(np.random.default_rng(1), truth)
        assert len(reps) == sum(SALES_TEAM.values()) == 20
        assert len({r.name for r in reps}) == 20
        low = truth.get("E4")["details"]["sales_rep"]
        low_rep = next(r for r in reps if r.name == low)
        assert low_rep.stage_factor == EVENTS.e4_low_conversion_rep.stage_conversion_factor
        assert low_rep.region not in EVENTS.e4_low_conversion_rep.excluded_regions
        assert all(r.stage_factor > low_rep.stage_factor for r in reps if r is not low_rep)


class TestCampaignPlan:
    def test_quarterly_campaigns_with_e3_and_launch_campaign(self) -> None:
        plans = plan_campaigns(Timeline(date(2024, 9, 1), date(2026, 8, 31)))
        ids = [p.campaign_id for p in plans]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)
        e3 = [p for p in plans if p.is_e3]
        assert len(e3) == 1 and e3[0].channel == "Paid Social"
        assert any("AI Insights Launch" in p.name for p in plans)
        assert {p.channel for p in plans} >= {"Paid Search", "Paid Social", "Email", "Organic", "Partner"}

    def test_no_e3_outside_window(self) -> None:
        plans = plan_campaigns(Timeline(date(2024, 9, 1), date(2025, 8, 31)))
        assert not any(p.is_e3 for p in plans)


class TestEventStatistics:
    def test_binomial_upper_tail_exact_values(self) -> None:
        from data.generator.event_validation import _binomial_upper_tail

        assert _binomial_upper_tail(5, 10, 0.5) == pytest.approx(0.623046875)
        assert _binomial_upper_tail(0, 10, 0.3) == pytest.approx(1.0)
        assert _binomial_upper_tail(10, 10, 0.5) == pytest.approx(0.5**10)
        # A 9-in-101 monthly churn count is extremely unlikely under a 0.2% baseline.
        assert _binomial_upper_tail(9, 101, 0.002) < 1e-10
