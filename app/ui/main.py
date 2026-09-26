"""The AgentOps web UI: ``streamlit run app/ui/main.py`` (the API must be running; see docs/ui.md).

The page sends each question to the AgentOps API (``UI_API_URL``) and renders the response: the
answer, typed findings, KPI cards, charts, forecast and anomaly details, evidence and provenance,
and the analysis trace. It never touches the database or the agent directly. Questions and answers
of the current browser session are kept in memory (``st.session_state``) only; nothing is stored.
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
    "and explains the result."
)
EXAMPLES = (
    "What was revenue in July compared with June?",
    "Which region had the largest revenue decline?",
    "Why did support tickets increase?",
    "What is our 3-month revenue forecast?",
    "Which acquisition channel has the highest CAC?",
    "Are there any unusual customer or product trends?",
)
MAX_HISTORY = 20


def _client() -> AgentOpsClient:
    settings = get_settings()
    return AgentOpsClient(settings.ui_api_url, timeout=settings.ui_request_timeout_seconds)


@st.cache_data(ttl=10, show_spinner=False)
def _health(base_url: str) -> dict[str, Any] | None:
    try:
        return _client().health()
    except APIFailure:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _capabilities(base_url: str) -> dict[str, Any] | None:
    try:
        return _client().capabilities()
    except APIFailure:
        return None


def _init_state() -> None:
    st.session_state.setdefault("session_id", f"ui-{uuid.uuid4().hex[:12]}")
    st.session_state.setdefault("history", [])  # vm.HistoryEntry, newest first
    st.session_state.setdefault("current", None)  # {"question": str, "response": dict | None, "failure": ...}
    st.session_state.setdefault("question", "")


def _use_example(question: str) -> None:
    st.session_state.question = question
    st.session_state.submit = True


def _clear_history() -> None:
    st.session_state.history = []
    st.session_state.current = None


def _sidebar(client: AgentOpsClient) -> None:
    with st.sidebar:
        st.markdown("### Service")
        health = _health(client.base_url)
        if health is None:
            st.error("API not reachable")
            st.caption(f"Expected at {client.base_url}. Start it with `python -m app.api`.")
        elif health.get("status") == "ok":
            st.success("API ready")
        else:
            st.warning(f"API {health.get('status', 'unavailable')}")
        if health:
            st.caption(
                f"Data as of {health.get('as_of_date') or '—'} · dataset {health.get('dataset_version') or '—'} · "
                f"model provider {health.get('llm_provider') or '—'} · API {health.get('version', '')}"
            )
        st.markdown("### Session")
        st.caption("Questions and answers are kept in this browser session only.")
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
    st.session_state.history = [entry, *st.session_state.history][:MAX_HISTORY]
    st.session_state.current = current


def main() -> None:
    st.set_page_config(page_title=TITLE, page_icon=":material/insights:", layout="wide")
    _init_state()
    client = _client()
    _sidebar(client)

    st.title(TITLE)
    st.markdown(f"##### {TAGLINE}")
    st.caption(DESCRIPTION)

    capabilities = _capabilities(client.base_url)
    limits = (capabilities or {}).get("limits") or {}
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

    current = st.session_state.current
    if current:
        st.divider()
        st.markdown(f"**Q:** {vm.escape_markdown(current['question'])}")
        if current["response"] is not None:
            render.render_response(current["response"])
        elif current["failure"] is not None:
            render.render_error(current["failure"])
    earlier = st.session_state.history[1:] if current else st.session_state.history
    if earlier:
        st.divider()
        render.render_history(earlier)


if __name__ == "__main__":  # `streamlit run` executes the page as __main__
    main()
