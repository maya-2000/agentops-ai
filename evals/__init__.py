"""Phase 7: the AgentOps evaluation and benchmark suite, an independent measurement layer.

It measures the production system and is not part of it. Production code (``app/``) never imports
``evals``, and a static test enforces this. ``evals`` imports ``app`` to drive the real production
paths: the LangGraph agent, the MCP server, and the shared secured executor.

    datasets/   versioned scenario datasets (``eval_v1``)
    scenarios/  the typed ``EvaluationScenario`` model and the dataset loader
    reference/  expected values: the independent Phase 2/3 reference implementations and the hidden
                evaluation labels (the generator's injected-event ground truth, evaluation only)
    runners/    drive the production paths (agent, MCP, direct secured executor) and record what happened
    graders/    the deterministic grader (authoritative) and an optional, isolated LLM judge
    metrics/    aggregate metrics, latency percentiles and regression thresholds
    reports/    result models and the JSON and Markdown reports

Entry point: ``python -m evals.run``. Documentation: ``docs/evaluation.md``.
"""
