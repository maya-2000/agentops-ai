"""Streamlit rendering of API responses. All content comes from ``view_models``; this module only lays it out.

Every dynamic text is escaped before it reaches Streamlit markdown (``esc``), so a question, an
answer or an evidence statement is shown literally: no links, images, LaTeX or directives.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import streamlit as st

from app.ui import view_models as vm
from app.ui.client import APIFailure

esc = vm.escape_markdown


def _banner(level: str, text: str) -> None:
    {"success": st.success, "info": st.info, "warning": st.warning, "error": st.error}.get(level, st.info)(esc(text))


def render_answer(response: Mapping[str, Any]) -> vm.AnswerView:
    view = vm.answer_view(response)
    if view.outcome == "answered":
        st.subheader("Answer")
    else:
        _banner(view.banner.level, view.banner.title)
    st.markdown(esc(view.answer))
    if view.refusal_note:
        st.caption(esc(view.refusal_note))
    if view.banner.guidance:
        st.caption(esc(view.banner.guidance))
    meta = [f"Period: {esc(view.period)}"] if view.period else []
    meta.append(f"Request ID: `{view.request_id}`")
    st.caption(" · ".join(meta))
    for caveat in view.caveats:
        st.caption(f":material/info: {esc(caveat)}")
    if view.assumptions:
        st.caption("Assumptions: " + esc(" ".join(view.assumptions)))
    return view


def render_claims(response: Mapping[str, Any]) -> None:
    items = vm.claim_items(response)
    if not items:
        return
    st.markdown("#### Findings")
    for item in items:
        with st.container(border=True):
            cols = st.columns([1, 5], vertical_alignment="top")
            with cols[0]:
                st.badge(item.style.label, icon=item.style.icon, color=item.style.color)
                if item.marker:
                    st.badge(item.marker.label, icon=item.marker.icon, color=item.marker.color)
            with cols[1]:
                st.markdown(f"**{esc(item.text)}**" if item.primary else esc(item.text))
                notes = [item.claim_id, item.support, f"evidence {', '.join(item.evidence_ids) or 'none'}"]
                notes.append(item.marker.note if item.marker else item.style.note)
                st.caption(esc(" · ".join(notes)))


def render_kpis(specs: Sequence[Mapping[str, Any]]) -> None:
    for spec in specs:
        cards = vm.kpi_cards(spec)
        for start in range(0, len(cards), 4):
            row = cards[start : start + 4]
            for col, card in zip(st.columns(len(row)), row, strict=True):
                with col:
                    # delta_color "off": a change is shown with its sign, never coloured as good or bad.
                    st.metric(card.label, card.value, delta=card.delta, delta_color="off", help=card.help, border=True)
                    st.caption(esc(card.caption))


def render_charts(specs: Sequence[Mapping[str, Any]], tables: Sequence[Mapping[str, Any]], shown: set[str]) -> None:
    for spec in specs:
        chart = vm.vega_lite(spec)
        if chart is None:
            continue
        st.markdown(f"**{esc(spec.get('title', ''))}**")
        if spec.get("subtitle"):
            st.caption(esc(spec["subtitle"]))
        st.vega_lite_chart(chart, width="stretch")
        for note in spec.get("notes", []):
            if note not in shown:
                shown.add(note)
                st.caption(f":material/info: {esc(note)}")
    for spec in tables:
        with st.expander(f"Table: {esc(spec.get('title', ''))}"):
            st.dataframe(vm.table_rows(spec), hide_index=True, width="stretch")


def render_forecasts(response: Mapping[str, Any], shown: set[str]) -> None:
    for section in response.get("forecasts", []):
        panel = vm.forecast_panel(section)
        st.markdown(f"#### {esc(panel.title)}")
        if panel.notice and panel.notice not in shown:
            shown.add(panel.notice)
            st.warning(esc(panel.notice), icon=":material/query_stats:")
        st.markdown(" · ".join(f"{esc(label)}: **{esc(value)}**" for label, value in panel.facts))
        st.dataframe(panel.points, hide_index=True, width="stretch")
        if panel.backtest:
            st.caption(esc(" · ".join(f"{label}: {value}" for label, value in panel.backtest)))
        for limitation in panel.limitations[:3]:
            shown.add(limitation)
            st.caption(f"Limitation: {esc(limitation)}")


def render_anomalies(response: Mapping[str, Any], shown: set[str]) -> None:
    for section in response.get("anomalies", []):
        panel = vm.anomaly_panel(section)
        st.markdown(f"#### {esc(panel.title)}")
        if panel.notice and panel.notice not in shown:
            shown.add(panel.notice)
            st.info(esc(panel.notice), icon=":material/insights:")
        st.caption(esc(panel.summary))
        if panel.flagged:
            st.dataframe(panel.flagged, hide_index=True, width="stretch")
        else:
            st.caption("No month was flagged as statistically unusual.")


def render_evidence(response: Mapping[str, Any]) -> None:
    rows = vm.evidence_rows(response)
    with st.expander(f"Evidence & Provenance ({len(rows)} items)"):
        if not rows:
            st.caption("No evidence was produced for this request.")
            return
        st.caption(
            "Every number in the answer comes from one of these evidence items: a tool result with its period, "
            "filters, source tables, calculation and query ID."
        )
        st.dataframe(rows, hide_index=True, width="stretch")


def render_trace(response: Mapping[str, Any]) -> None:
    rows = vm.trace_rows(response)
    run = response.get("run") or {}
    with st.expander("Analysis Trace"):
        if not rows:
            st.caption("No analysis steps were run.")
        for row in rows:
            indent = "\N{EM SPACE}\N{EM SPACE}↳ " * row.level  # a visible prefix: markdown strips leading spaces
            duration = f" · {row.duration}" if row.duration else ""
            detail = f" — {esc(row.detail)}" if row.detail else ""
            st.markdown(f"{indent}{row.mark} **{esc(row.label)}**{duration}{detail}")
        st.caption(
            esc(
                f"{run.get('tool_calls', 0)} tool call(s) · agent {vm.format_ms(run.get('agent_time_ms'))} · "
                f"total {vm.format_ms(response.get('api_time_ms'))} · model provider {run.get('llm_provider', '')}"
            )
        )


def render_response(response: Mapping[str, Any]) -> None:
    view = render_answer(response)
    groups = vm.visualization_groups(response)
    shown: set[str] = set()  # notices already on the page (each is shown once)
    if view.show_analysis:
        render_kpis(groups["kpi_card"])
        render_claims(response)
        render_forecasts(response, shown)
        render_anomalies(response, shown)
        render_charts(groups["chart"], groups["table"], shown)
    if response.get("evidence") or view.outcome not in ("refused", "unsupported"):
        render_evidence(response)
    render_trace(response)


def render_error(failure: APIFailure) -> None:
    view = vm.error_view(failure)
    st.error(esc(view.title))
    st.markdown(esc(view.message))
    if view.hint:
        st.caption(esc(view.hint))
    if view.request_id:
        st.caption(f"Request ID: `{view.request_id}`")


def render_history(entries: Sequence[vm.HistoryEntry]) -> None:
    if not entries:
        return
    st.markdown("#### Earlier in this session")
    for entry in entries:
        banner = vm.OUTCOME_BANNERS.get(entry.outcome)
        status = banner.title if banner and entry.outcome != "answered" else entry.outcome.replace("_", " ")
        with st.expander(f"{entry.asked_at} · {esc(entry.question)}"):
            st.caption(esc(status))
            st.markdown(esc(entry.answer))
            if entry.request_id:
                st.caption(f"Request ID: `{entry.request_id}`")
