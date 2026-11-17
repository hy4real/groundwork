"""Optional loopback API contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from agentlab.api import create_app
from agentlab.reasoning import Action
from agentlab.research.corpus import Corpus
from agentlab.research.tasks import MAX_OUTSTANDING_TASKS, ResearchTaskStore, TaskStoreError
from agentlab.research.worker import ResearchWorker
from agentlab.research.workspace import PathScope


def _app(tmp_path: Path):
    (tmp_path / "fact.md").write_text("The API fixture state is healthy.", encoding="utf-8")
    return create_app(tmp_path)


def _endpoint(app, path: str):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == path)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_persistent_task_http_lifecycle_and_restart(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "fact.md").write_text("The Helix fixture status is ready.", encoding="utf-8")
    path = tmp_path / "state" / "tasks.sqlite3"
    app = create_app(root, task_db=path)
    headers = {"X-AgentLab-Token": app.state.session_token}
    payload = {"request_key": "req-1", "question": "What is the Helix fixture status?"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        response = await client.post("/v1/research/tasks", json=payload, headers=headers)
        assert response.status_code == 202
        task = response.json()
        assert task["status"] == "pending"
        assert "question" not in response.text and "req-1" not in response.text
        location = response.headers["location"]
        replay = await client.post("/v1/research/tasks", json=payload, headers=headers)
        assert replay.status_code == 200 and replay.json()["id"] == task["id"]
        conflict = await client.post("/v1/research/tasks", json=dict(payload, question="changed"), headers=headers)
        assert conflict.status_code == 409 and conflict.json() == {"error": "idempotency_conflict"}
        assert (await client.get(location, headers=headers)).json()["status"] == "pending"
        assert (await client.get("/v1/research/tasks/missing", headers=headers)).status_code == 404

    worker = ResearchWorker(ResearchTaskStore(path), Corpus.load(root), worker_id="test-worker")
    assert worker.run_once().status == "succeeded"
    restarted = create_app(root, task_db=path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restarted), base_url="http://localhost") as client:
        assert (await client.get(location, headers=headers)).status_code == 403
        new_headers = {"X-AgentLab-Token": restarted.state.session_token}
        response = await client.get(location, headers=new_headers)
        assert response.status_code == 200
        assert response.json()["result"]["answer"]["status"] == "answered"
        replay = await client.post("/v1/research/tasks", json=payload, headers=new_headers)
        assert replay.status_code == 200 and replay.json() == response.json()


@pytest.mark.anyio
async def test_task_endpoints_require_local_authorization(tmp_path: Path) -> None:
    (tmp_path / "fact.md").write_text("Synthetic corpus.", encoding="utf-8")
    app = create_app(tmp_path, task_db=tmp_path / "state" / "tasks.sqlite3")
    token = {"X-AgentLab-Token": app.state.session_token}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        for method, path, body in [
            ("POST", "/v1/research/tasks", {"request_key": "req", "question": "q"}),
            ("GET", "/v1/research/tasks/unknown", None),
            ("POST", "/v1/research/tasks/unknown/cancel", None),
            ("GET", "/v1/research/task-metrics", None),
        ]:
            for headers, expected in [
                ({}, 403), ({"X-AgentLab-Token": "wrong"}, 403),
                (dict(token, Origin="https://attacker.example"), 403),
                (dict(token, Host="attacker.example"), 400),
            ]:
                response = await client.request(method, path, json=body, headers=headers)
                assert response.status_code == expected


@pytest.mark.anyio
@pytest.mark.parametrize("payload", [
    {"request_key": "req", "question": "q", "corpus": "/private/path"},
    {"request_key": "req", "question": "   "},
    {"request_key": "bad key", "question": "q"},
    {"request_key": "req\n", "question": "q"},
    {"request_key": "req", "question": "q", "task_db": "private.sqlite3"},
])
async def test_task_schema_rejects_bad_input_without_echo(tmp_path: Path, payload: dict) -> None:
    (tmp_path / "fact.md").write_text("Synthetic corpus.", encoding="utf-8")
    app = create_app(tmp_path, task_db=tmp_path / "state" / "tasks.sqlite3")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        response = await client.post("/v1/research/tasks", json=payload, headers={"X-AgentLab-Token": app.state.session_token})
        assert response.status_code == 422 and response.json() == {"error": "invalid_request"}


@pytest.mark.anyio
async def test_task_store_failure_is_503_and_redacted(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "fact.md").write_text("Synthetic corpus.", encoding="utf-8")
    app = create_app(tmp_path, task_db=tmp_path / "state" / "tasks.sqlite3")

    def unavailable(*args, **kwargs):
        raise TaskStoreError("private database details")

    monkeypatch.setattr(ResearchTaskStore, "submit", unavailable)
    monkeypatch.setattr(ResearchTaskStore, "get", unavailable)
    monkeypatch.setattr(ResearchTaskStore, "cancel", unavailable)
    monkeypatch.setattr(ResearchTaskStore, "metrics", unavailable)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        for method, path, body in [
            ("POST", "/v1/research/tasks", {"request_key": "req", "question": "q"}),
            ("GET", "/v1/research/tasks/unknown", None),
            ("POST", "/v1/research/tasks/unknown/cancel", None),
            ("GET", "/v1/research/task-metrics", None),
        ]:
            response = await client.request(method, path, json=body, headers={"X-AgentLab-Token": app.state.session_token})
            assert response.status_code == 503 and response.json() == {"error": "task_store_unavailable"}


@pytest.mark.anyio
async def test_capacity_and_cancel_http_contract(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "fact.md").write_text("Synthetic corpus.", encoding="utf-8")
    path = tmp_path / "state" / "tasks.sqlite3"
    store = ResearchTaskStore(path)
    corpus_hash = Corpus.load(root).content_hash
    for i in range(MAX_OUTSTANDING_TASKS):
        store.submit(f"req-{i}", "q", corpus_hash)
    app = create_app(root, task_db=path)
    headers = {"X-AgentLab-Token": app.state.session_token}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        body = {"request_key": "overflow", "question": "q"}
        response = await client.post("/v1/research/tasks", json=body, headers=headers)
        assert response.status_code == 429
        assert response.json() == {"error": "task_capacity_exceeded"}
        assert response.headers["retry-after"] == "1"
        replay = await client.post("/v1/research/tasks", json=dict(body, request_key="req-0"), headers=headers)
        assert replay.status_code == 200
        task_id = replay.json()["id"]
        location = f"/v1/research/tasks/{task_id}/cancel"
        assert (await client.post(location)).status_code == 403
        cancelled = await client.post(location, headers=headers)
        assert cancelled.status_code == 200 and cancelled.json()["error_code"] == "cancelled"
        assert (await client.post(location, headers=headers)).json() == cancelled.json()
        assert (await client.post("/v1/research/tasks/missing/cancel", headers=headers)).status_code == 404
        assert (await client.post("/v1/research/tasks", json=body, headers=headers)).status_code == 202


@pytest.mark.anyio
async def test_metrics_and_events_do_not_expose_input(tmp_path: Path, capsys) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "fact.md").write_text("Private corpus marker.", encoding="utf-8")
    path = tmp_path / "state" / "tasks.sqlite3"
    app = create_app(root, task_db=path)
    headers = {"X-AgentLab-Token": app.state.session_token}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        body = {"request_key": "private-request-marker", "question": "Private question marker"}
        response = await client.post("/v1/research/tasks", headers=headers, json=body)
        task_id = response.json()["id"]
        response = await client.get("/v1/research/task-metrics", headers=headers)
        assert response.status_code == 200
        assert response.json()["counts"]["pending"] == 1
        assert response.json()["available_capacity"] == 99
        await client.post(f"/v1/research/tasks/{task_id}/cancel", headers=headers)
        response = await client.get("/v1/research/task-metrics", headers=headers)
        assert response.json()["errors"] == {"cancelled": 1}
        assert response.json()["outstanding"] == 0
        assert task_id not in response.text
    output = capsys.readouterr().err
    events = [json.loads(line) for line in output.splitlines()]
    assert [event["event"] for event in events] == ["task_submitted", "task_cancelled"]
    for private in (body["request_key"], body["question"], str(root), headers["X-AgentLab-Token"]):
        assert private not in output and private not in response.text


def test_health_and_research(tmp_path: Path) -> None:
    app = _app(tmp_path)
    assert {
        (getattr(route, "path", None), frozenset(getattr(route, "methods", ()) or ()))
        for route in app.routes
    } == {
        ("/health", frozenset({"GET"})),
        ("/v1/research", frozenset({"POST"})),
    }
    research_route = next(route for route in app.routes if getattr(route, "path", None) == "/v1/research")
    assert research_route.body_field is not None
    assert _endpoint(app, "/health")() == {"status": "ok"}
    request = app.state.research_request_type(question="What is the API fixture state?")
    assert _endpoint(app, "/v1/research")(request)["answer"]["status"] == "answered"


def test_request_cannot_submit_local_path(tmp_path: Path) -> None:
    app = _app(tmp_path)
    try:
        app.state.research_request_type.model_validate(
            {"question": "What is the state?", "corpus": "/private/secret"}
        )
    except ValidationError as exc:
        response = app.state.validation_error_handler(None, RequestValidationError(exc.errors()))
    else:
        raise AssertionError("path field should be rejected")
    assert response.status_code == 422
    assert json.loads(response.body) == {"error": "invalid_request"}
    assert b"/private/secret" not in response.body


@pytest.mark.anyio
async def test_research_http_requires_loopback_host_and_session_token(tmp_path: Path) -> None:
    app = _app(tmp_path)
    payload = {"question": "What is the API fixture state?"}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        assert (await client.post("/v1/research", json=payload)).status_code == 403
        response = await client.post(
            "/v1/research",
            json=payload,
            headers={"X-AgentLab-Token": app.state.session_token},
        )
        assert response.status_code == 200
        assert response.json()["answer"]["status"] == "answered"
        assert (await client.post(
            "/v1/research",
            json=payload,
            headers={"Host": "attacker.example", "X-AgentLab-Token": app.state.session_token},
        )).status_code == 400


@pytest.mark.anyio
async def test_research_http_rejects_non_loopback_origin(tmp_path: Path) -> None:
    app = _app(tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.post(
            "/v1/research",
            json={"question": "What is the API fixture state?"},
            headers={
                "Origin": "http://attacker.example",
                "X-AgentLab-Token": app.state.session_token,
            },
        )
    assert response.status_code == 403
    assert response.json() == {"error": "forbidden"}


@pytest.mark.anyio
async def test_api_wires_workspace_path_scope_and_enforces_task_read_only_boundary(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "fact.md").write_text("The API fixture state is healthy.", encoding="utf-8")
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    (ws_dir / "notes").mkdir()

    with pytest.raises(ValueError, match="path_scope 需要 workspace_root"):
        create_app(corpus_dir, path_scope=PathScope(mode="deny"))
    with pytest.raises(ValueError, match="persistent task API"):
        create_app(
            corpus_dir,
            task_db=tmp_path / "state" / "tasks.sqlite3",
            workspace_root=ws_dir,
            path_scope=PathScope(mode="allow", roots=("notes",)),
        )

    app_default_deny = create_app(corpus_dir, workspace_root=ws_dir)
    assert app_default_deny.state.path_scope == PathScope(mode="deny")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_default_deny), base_url="http://127.0.0.1") as client:
        response = await client.post(
            "/v1/research",
            json={"question": "What is the API fixture state?"},
            headers={"X-AgentLab-Token": app_default_deny.state.session_token},
        )
    assert response.status_code == 200
    assert response.json()["trace"]["path_scope_mode"] == "deny"

    app_allow = create_app(
        corpus_dir,
        workspace_root=ws_dir,
        path_scope=PathScope(mode="allow", roots=("notes",)),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_allow), base_url="http://127.0.0.1") as client:
        response = await client.post(
            "/v1/research",
            json={"question": "What is the API fixture state?"},
            headers={"X-AgentLab-Token": app_allow.state.session_token},
        )
    assert response.status_code == 200
    assert response.json()["trace"]["path_scope_mode"] == "allow"


@pytest.mark.anyio
async def test_api_research_endpoint_enforces_workspace_tools_through_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "fact.md").write_text("The API fixture state is healthy.", encoding="utf-8")
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    (ws_dir / "notes").mkdir()
    (ws_dir / "notes" / "ok.md").write_text("allowed", encoding="utf-8")
    (ws_dir / "notes" / "secret.key").write_text("secret-key", encoding="utf-8")
    (ws_dir / "outside.md").write_text("outside", encoding="utf-8")

    class ScriptedApiReasoner:
        def __init__(self) -> None:
            self.calls = (
                ("read_file", '{"path":"outside.md"}'),
                ("read_file", '{"path":"notes/secret.key"}'),
                ("write_file", '{"path":"notes/created.md","text":"api-ok"}'),
            )
            self.index = 0

        def decide(self, *, task, context, available_tools):
            assert "read_file" in available_tools and "write_file" in available_tools
            if self.index < len(self.calls):
                tool, arg = self.calls[self.index]
                self.index += 1
                return Action(kind="tool", tool=tool, argument=arg)
            return Action(
                kind="finish",
                answer='{"status":"insufficient_evidence","answer":"","claims":[],"reason":"checked"}',
            )

    monkeypatch.setattr("agentlab.research.agent.ReplayResearchReasoner", ScriptedApiReasoner)

    app_deny = create_app(corpus_dir, workspace_root=ws_dir)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_deny), base_url="http://127.0.0.1") as client:
        denied = await client.post(
            "/v1/research",
            json={"question": "test deny"},
            headers={"X-AgentLab-Token": app_deny.state.session_token},
        )
    assert denied.status_code == 200
    assert denied.json()["trace"]["write_calls"] == 0
    assert all(event["status"] == "error" for event in denied.json()["trace"]["tool_events"])
    assert not (ws_dir / "notes" / "created.md").exists()

    app_allow = create_app(
        corpus_dir,
        workspace_root=ws_dir,
        path_scope=PathScope(mode="allow", roots=("notes",), deny_globs=("notes/**/*.key",)),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_allow), base_url="http://127.0.0.1") as client:
        allowed = await client.post(
            "/v1/research",
            json={"question": "test allow"},
            headers={"X-AgentLab-Token": app_allow.state.session_token},
        )
    assert allowed.status_code == 200
    payload = allowed.json()
    assert payload["trace"]["write_calls"] == 1
    assert [e["status"] for e in payload["trace"]["tool_events"]] == ["error", "error", "ok"]
    assert (ws_dir / "notes" / "created.md").read_text(encoding="utf-8") == "api-ok"
