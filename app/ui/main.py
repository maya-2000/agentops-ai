"""The AgentOps web UI: ``streamlit run app/ui/main.py`` (the API must be running; see docs/ui.md).

The page has two modes. **Ask a question** sends one question to the AgentOps API (``UI_API_URL``)
and renders the response: the answer, typed findings, KPI cards, charts, forecast and anomaly
details, evidence and provenance, and the analysis trace. **Investigate a business issue** (Phase 10)
sends an objective to ``/investigations/stream``, shows the plan as a live checklist, then renders the
decision brief: summary, key findings, drivers, risks, recommendations, uncertainty, evidence and the
analysis trace. The page never touches the database or the agent directly. Questions, objectives and
their results of the current browser session are kept in memory (``st.session_state``, at most
``UI_HISTORY_LIMIT``) only; nothing is stored. Requests carry ``API_AUTH_TOKEN`` when it is set;
the token is never shown. Start it with ``python -m app.ui`` (``app/ui/__main__.py``).
"""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from app.config import get_settings
from app.ui import render
from app.ui import view_models as vm
from app.ui.client import AgentOpsClient, APIFailure

TITLE = "AgentOps AI"
TAGLINE = "Evidence-backed AI Business Intelligence"
DESCRIPTION = (
    "Ask business questions in natural language. AgentOps analyzes trusted business data, validates evidence, "
    "and explains the result. Or investigate a business issue and get an evidence-backed decision brief."
)
MODES = ("Ask a question", "Investigate a business issue")
EXAMPLE_OBJECTIVES = (
    "Why is revenue growth slowing?",
    "Why is customer churn increasing?",
    "Give me a management brief on the current state of the business.",
)
EXAMPLES = (
    "What was revenue in July compared with June?",
    "Which region had the largest revenue decline?",
    "Why did support tickets increase?",
    "What is our 3-month revenue forecast?",
    "Which acquisition channel has the highest CAC?",
    "Are there any unusual customer or product trends?",
)


def _client() -> AgentOpsClient:
    settings = get_settings()
    token = settings.api_auth_token.get_secret_value() if settings.api_auth_token else None
    return AgentOpsClient(settings.ui_api_url, timeout=settings.ui_request_timeout_seconds, token=token)


@st.cache_data(ttl=10, show_spinner=False)
def _readiness(base_url: str) -> dict[str, Any] | None:
    try:
        return _client().readiness()
    except APIFailure:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _capabilities(base_url: str) -> tuple[dict[str, Any] | None, str | None]:
    """(capabilities, None) or (None, the failure code), e.g. "unauthorized" when the token is wrong."""
    try:
        return _client().capabilities(), None
    except APIFailure as failure:
        return None, failure.code or failure.kind


def _init_state() -> None:
    st.session_state.setdefault("session_id", f"ui-{uuid.uuid4().hex[:12]}")
    st.session_state.setdefault("history", [])  # vm.HistoryEntry, newest first
    st.session_state.setdefault("current", None)  # {"question": str, "response": dict | None, "failure": ...}
    st.session_state.setdefault("question", "")
    st.session_state.setdefault("objective", "")


def _use_example(question: str) -> None:
    st.session_state.question = question
    st.session_state.submit = True


def _use_objective(objective: str) -> None:
    st.session_state.objective = objective
    st.session_state.run_objective = True


def _clear_history() -> None:
    st.session_state.history = []
    st.session_state.current = None


def _sidebar(client: AgentOpsClient, capabilities: dict[str, Any] | None, problem: str | None) -> None:
    with st.sidebar:
        st.markdown("### Service")
        ready = _readiness(client.base_url)
        if ready is None:
            st.error("API not reachable")
            st.caption(f"Expected at {client.base_url}. Start it with `python -m app.api`.")
        elif ready.get("status") == "ready":
            st.success("API ready")
        else:
            st.warning("API not ready")
            failed = [name.replace("_", " ") for name, ok in (ready.get("checks") or {}).items() if not ok]
            if failed:
                st.caption("Not ready: " + ", ".join(failed) + ".")
        if problem == "unauthorized":
            st.error("The UI is not authorised to call the API. Set the same API_AUTH_TOKEN for the UI and the API.")
        if capabilities:
            st.caption(
                f"Data as of {capabilities.get('as_of_date') or '—'} · dataset "
                f"{capabilities.get('dataset_version') or '—'} · model provider "
                f"{capabilities.get('llm_provider') or '—'} · API {capabilities.get('version', '')}"
            )
        st.markdown("### Session")
        st.caption(
            f"The last {get_settings().ui_history_limit} questions and answers are kept in this browser session "
            "only; nothing is stored."
        )
        st.button("Clear history", on_click=_clear_history, disabled=not st.session_state.history)


def _ask(client: AgentOpsClient, question: str) -> None:
    current: dict[str, Any] = {"question": question, "response": None, "failure": None}
    with st.status("Analyzing…", expanded=False) as status:

        def on_progress(event: dict[str, Any]) -> None:
            status.update(label=f"{event.get('label', 'Working')}…")

        try:
            current["response"] = client.ask_stream(
                question, session_id=st.session_state.session_id, on_progress=on_progress
            )
            status.update(label="Analysis complete", state="complete")
        except APIFailure as failure:
            current["failure"] = failure
            status.update(label="The request could not be completed", state="error")
    entry = vm.history_entry(question, response=current["response"], failure=current["failure"])
    st.session_state.history = vm.bounded_history(st.session_state.history, entry, get_settings().ui_history_limit)
    st.session_state.current = current


def _investigate(client: AgentOpsClient, objective: str) -> None:
    """Run an investigation with a live checklist of its plan (stage and step events only; never reasoning)."""
    current: dict[str, Any] = {"question": objective, "response": None, "failure": None, "kind": "investigation"}
    with st.status("Investigating…", expanded=True) as status:
        checklist = st.empty()
        plan: list[dict[str, Any]] = []
        statuses: dict[str, str] = {}

        def on_progress(event: dict[str, Any]) -> None:
            stage = event.get("stage")
            if stage == "plan" and isinstance(event.get("steps"), list):
                plan[:] = [s for s in event["steps"] if isinstance(s, dict)]
            elif stage == "step_started" and event.get("step_id"):
                statuses[str(event["step_id"])] = "running"
            elif stage == "step_finished" and event.get("step_id"):
                statuses[str(event["step_id"])] = str(event.get("status") or "completed")
            status.update(label=f"{event.get('label', 'Working')}…")
            if plan:
                rows = vm.plan_rows(plan, statuses)
                checklist.markdown(
                    "\n".join(f"{r.mark} {vm.escape_markdown(r.title)}" for r in rows), unsafe_allow_html=False
                )

        try:
            current["response"] = client.investigate_stream(
                objective, session_id=st.session_state.session_id, on_progress=on_progress
            )
            status.update(label="Investigation complete", state="complete", expanded=False)
        except APIFailure as failure:
            current["failure"] = failure
            status.update(label="The investigation could not be completed", state="error")
    entry = vm.investigation_history_entry(objective, response=current["response"], failure=current["failure"])
    st.session_state.history = vm.bounded_history(st.session_state.history, entry, get_settings().ui_history_limit)
    st.session_state.current = current


def main() -> None:
    st.set_page_config(page_title=TITLE, page_icon=":material/insights:", layout="wide")
    _init_state()
    client = _client()
    capabilities, problem = _capabilities(client.base_url)
    _sidebar(client, capabilities, problem)

    st.title(TITLE)
    st.markdown(f"##### {TAGLINE}")
    st.caption(DESCRIPTION)

    limits = (capabilities or {}).get("limits") or {}
    mode = st.radio("Mode", MODES, horizontal=True, key="mode", label_visibility="collapsed")
    if mode == MODES[1]:
        _investigation_form(client, capabilities, limits)
    else:
        _question_form(client, capabilities, limits)

    current = st.session_state.current
    if current:
        st.divider()
        if current.get("kind") == "investigation":
            if current["response"] is not None:
                render.render_investigation(current["response"])
            elif current["failure"] is not None:
                st.markdown(f"**Objective:** {vm.escape_markdown(current['question'])}")
                render.render_error(current["failure"])
        else:
            st.markdown(f"**Q:** {vm.escape_markdown(current['question'])}")
            if current["response"] is not None:
                render.render_response(current["response"])
            elif current["failure"] is not None:
                render.render_error(current["failure"])
    earlier = st.session_state.history[1:] if current else st.session_state.history
    if earlier:
        st.divider()
        render.render_history(earlier)


def _question_form(client: AgentOpsClient, capabilities: dict[str, Any] | None, limits: dict[str, Any]) -> None:
    with st.form("ask", border=False):
        st.text_area(
            "Your question",
            key="question",
            height=100,
            max_chars=int(limits.get("max_question_chars") or get_settings().agent_max_question_chars),
            placeholder="e.g. What was revenue in July compared with June?",
        )
        submitted = st.form_submit_button("Analyze", type="primary")

    examples = vm.examples(capabilities, EXAMPLES)
    st.caption("Try an example:")
    columns = st.columns(3)
    for index, example in enumerate(examples[:9]):
        columns[index % 3].button(example, key=f"example-{index}", on_click=_use_example, args=(example,))

    question = str(st.session_state.question or "")
    if submitted or st.session_state.pop("submit", False):
        if question.strip():
            _ask(client, question)
        else:
            st.warning("Type a question first.")


def _investigation_form(client: AgentOpsClient, capabilities: dict[str, Any] | None, limits: dict[str, Any]) -> None:
    st.caption(
        "Investigate a business issue: AgentOps plans several analysis steps, runs them through the same secured "
        "tools, validates the findings and writes an evidence-backed decision brief."
    )
    with st.form("investigate", border=False):
        st.text_area(
            "Business issue to investigate",
            key="objective",
            height=100,
            max_chars=int(limits.get("max_question_chars") or get_settings().agent_max_question_chars),
            placeholder="e.g. Why is revenue growth slowing?",
        )
        submitted = st.form_submit_button("Investigate", type="primary")
    examples = vm.example_objectives(capabilities, EXAMPLE_OBJECTIVES)
    st.caption("Try an investigation:")
    columns = st.columns(3)
    for index, example in enumerate(examples[:6]):
        columns[index % 3].button(example, key=f"objective-{index}", on_click=_use_objective, args=(example,))
    objective = str(st.session_state.objective or "")
    if submitted or st.session_state.pop("run_objective", False):
        if objective.strip():
            _investigate(client, objective)
        else:
            st.warning("Describe the business issue first.")


if __name__ == "__main__":  # `streamlit run` executes the page as __main__
    main()
