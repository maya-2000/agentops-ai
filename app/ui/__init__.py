"""Phase 8: the AgentOps web UI (Streamlit), a client of the HTTP API.

    Streamlit page (``main``) -> ``client`` (HTTP) -> AgentOps API -> agent -> secured tools -> data

- ``main``: the page (``streamlit run app/ui/main.py``): question input, examples, session history.
- ``client``: the HTTP client; the UI's only way to reach the agent.
- ``view_models``: pure transformations of API responses into what is shown (tested without a browser).
- ``render``: Streamlit layout of the view models.

The UI never queries the database, never imports the agent, tools, analytics or evidence layers,
never reads data files and never calculates a business number: it formats and draws what the API
returns. Documentation: ``docs/ui.md``.
"""
