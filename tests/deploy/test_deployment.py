"""Phase 9 deployment artefacts: the Dockerfile, docker-compose.yml, .dockerignore and the CI workflows.

Static checks run everywhere. The container runtime test runs only where a Docker daemon and the
built ``agentops-api`` image are available (``docker compose build``); CI's docker job exercises
the same path with ``scripts/smoke_test.py``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
import tomllib
from typing import Any

import pytest

from app.api import API_VERSION
from app.config import PROJECT_ROOT

yaml = pytest.importorskip("yaml")

DOCKERFILE = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
COMPOSE = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
WORKFLOWS = PROJECT_ROOT / ".github" / "workflows"


def _stage(name: str) -> str:
    """The Dockerfile text of one build stage."""
    parts = re.split(r"(?m)^FROM ", DOCKERFILE)
    for part in parts:
        if part.splitlines()[0].strip().endswith(f" AS {name}"):
            return part
    raise AssertionError(f"no stage {name}")


# ---------------------------------------------------------------------------------------- images


@pytest.mark.parametrize("target", ["api", "ui"])
def test_images_run_as_an_unprivileged_user_with_a_healthcheck(target: str) -> None:
    stage = _stage(target)
    assert "USER agentops:agentops" in stage and "HEALTHCHECK" in stage
    assert re.search(r'CMD \["python", "-m", "app\.(api|ui)"\]', stage), "exec form: python receives SIGTERM"
    assert "useradd --system --uid 10001" in DOCKERFILE


def test_images_contain_no_data_secrets_or_ground_truth() -> None:
    for line in re.findall(r"(?m)^COPY .*$", DOCKERFILE):
        for forbidden in ("data/", "seeds", "database", ".env", "tests", "evals"):
            assert forbidden not in line.split("--from")[0], line
    assert "API_AUTH_TOKEN" not in DOCKERFILE and "ANTHROPIC_API_KEY" not in DOCKERFILE
    lines = [line.strip() for line in DOCKERIGNORE.splitlines() if line.strip() and not line.startswith("#")]
    assert lines[0] == "*", "the build context is an allow-list"
    allowed = {line[1:] for line in lines if line.startswith("!")}
    assert allowed == {"pyproject.toml", "README.md", "app/"}


def test_the_api_image_is_production_by_default_and_reads_mounted_data() -> None:
    api = _stage("api")
    assert "APP_ENV=production" in api and "API_HOST=0.0.0.0" in api
    assert "DATABASE_URL=duckdb:////data/database/" in api and "EXPOSE 8000" in api
    assert '".[api]"' in _stage("api-build")  # the api extra only: no dev tools, no Streamlit


def test_the_ui_image_has_no_database_or_agent_code() -> None:
    ui, build = _stage("ui"), _stage("ui-build")
    assert "COPY app/__init__.py app/config.py" in ui and "COPY app/ui" in ui
    assert "COPY app ./app" not in ui and "duckdb" not in build.lower() and "langgraph" not in build.lower()


def test_ui_image_dependency_ranges_match_pyproject() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = {*project["optional-dependencies"]["ui"], *project["dependencies"]}
    for requirement in re.findall(r'"([a-z][a-z0-9-]*[<>=][^"]+)"', _stage("ui-build")):
        name = re.split(r"[<>=]", requirement)[0]
        assert requirement in declared, (name, requirement)


# ---------------------------------------------------------------------------------------- compose


def test_compose_services_are_hardened() -> None:
    services: dict[str, Any] = COMPOSE["services"]
    assert set(services) == {"api", "ui"}
    for name, service in services.items():
        assert service["read_only"] is True and service["cap_drop"] == ["ALL"], name
        assert "no-new-privileges:true" in service["security_opt"], name
        for port in service["ports"]:
            assert port.startswith("127.0.0.1:"), (name, port)
        assert service["image"].endswith(f":{API_VERSION}"), name
        assert service["environment"]["API_AUTH_TOKEN"].startswith("${API_AUTH_TOKEN:?"), "a token is required"


def test_only_the_api_mounts_data_and_only_read_only() -> None:
    api, ui = COMPOSE["services"]["api"], COMPOSE["services"]["ui"]
    assert api["volumes"] and all(volume.endswith(":ro") for volume in api["volumes"])
    assert not any("seeds" in volume for volume in api["volumes"])
    assert "volumes" not in ui and ui["environment"]["UI_API_URL"] == "http://api:8000"
    assert ui["depends_on"]["api"]["condition"] == "service_healthy"


# ---------------------------------------------------------------------------------------- CI


def _workflow(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    return loaded


def _runs(workflow: dict[str, Any]) -> str:
    return "\n".join(step.get("run", "") for job in workflow["jobs"].values() for step in job["steps"])


def test_ci_runs_the_quality_gates_on_pull_requests_and_main() -> None:
    ci = _workflow("ci.yml")
    triggers = ci.get("on", ci.get(True))
    assert "pull_request" in triggers and triggers["push"]["branches"] == ["main"]
    runs = _runs(ci)
    for command in (
        "ruff check app data tests evals",
        "ruff format --check app data tests evals",
        "mypy app data evals",
        "pytest",
        "python -m evals.run --suite critical",
        "docker compose build",
        "scripts/smoke_test.py",
        "docker compose stop",
    ):
        assert command in runs, command
    assert ci["permissions"] == {"contents": "read"}


def test_the_evaluation_workflow_runs_the_full_benchmark_and_multi_seed() -> None:
    runs = _runs(_workflow("evaluation.yml"))
    assert "python -m evals.run --output" in runs and "--suite critical" in runs
    assert "--suite multi_seed --multi-seed 7,2027" in runs


@pytest.mark.parametrize("name", ["ci.yml", "evaluation.yml"])
def test_workflows_need_no_secrets_and_stay_deterministic(name: str) -> None:
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    assert "${{ secrets." not in text and "ANTHROPIC_API_KEY" not in text  # no repository secret is needed
    assert "LLM_PROVIDER: deterministic" in text
    assert re.search(r'(python-version|PYTHON_VERSION): "?3\.11"?', text)


# ---------------------------------------------------------------------------------------- container runtime


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "image", "inspect", f"agentops-api:{API_VERSION}"], capture_output=True, timeout=20, check=False
    )
    return probe.returncode == 0 and (PROJECT_ROOT / "database" / "northwind_cloud.duckdb").exists()


@pytest.mark.docker
@pytest.mark.skipif(not _docker_ready(), reason="needs a Docker daemon, the built API image and the generated data")
def test_the_api_container_starts_serves_and_stops_cleanly() -> None:
    import httpx

    token = "container-test-" + "t" * 40
    name = f"agentops-api-test-{int(time.time())}"
    run = [
        "docker", "run", "-d", "--name", name, "--read-only", "--tmpfs", "/tmp", "--cap-drop", "ALL",
        "-p", "127.0.0.1::8000", "-e", f"API_AUTH_TOKEN={token}",
        "-v", f"{PROJECT_ROOT / 'database'}:/data/database:ro",
        "-v", f"{PROJECT_ROOT / 'data' / 'metadata'}:/data/metadata:ro",
        f"agentops-api:{API_VERSION}",
    ]  # fmt: skip
    subprocess.run(run, check=True, capture_output=True, timeout=60)
    try:
        port = (
            subprocess.run(["docker", "port", name, "8000/tcp"], check=True, capture_output=True, text=True, timeout=20)
            .stdout.strip()
            .split(":")[-1]
        )
        base = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=base, timeout=30, trust_env=False) as client:
            for _ in range(60):
                try:
                    if client.get("/api/v1/readiness").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            assert client.get("/api/v1/health").json()["status"] == "ok"
            assert client.post("/api/v1/ask", json={"question": "What was revenue last month?"}).status_code == 401
            answer = client.post(
                "/api/v1/ask",
                json={"question": "What was revenue in July 2026 compared with June 2026?"},
                headers={"Authorization": f"Bearer {token}"},
            ).json()
            assert answer["outcome"] == "answered" and answer["evidence"]
        subprocess.run(["docker", "stop", "-t", "20", name], check=True, capture_output=True, timeout=60)
        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.ExitCode}}", name], capture_output=True, text=True, timeout=20
        )
        assert inspect.stdout.strip() == "0"  # SIGTERM: graceful shutdown
        logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True, timeout=20)
        assert '"event": "service_stopped"' in logs.stdout + logs.stderr and token not in logs.stdout + logs.stderr
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60, check=False)
