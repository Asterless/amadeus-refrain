"""Tests for usage HTTP routes."""

import pytest
from starlette.testclient import TestClient

from src.llm.usage import UsageTracker
from src.llm.usage_routes import create_usage_router


@pytest.fixture
async def tracker(tmp_path) -> UsageTracker:
    t = UsageTracker(db_path=str(tmp_path / "usage.db"))
    await t.init()
    return t


@pytest.fixture
def client(tracker: UsageTracker) -> TestClient:
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(create_usage_router(tracker))
    return TestClient(app)


def test_today_endpoint(client: TestClient) -> None:
    resp = client.get("/api/usage/today")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_calls" in data


def test_month_endpoint(client: TestClient) -> None:
    resp = client.get("/api/usage/month")
    assert resp.status_code == 200


def test_top_users_endpoint(client: TestClient) -> None:
    resp = client.get("/api/usage/top-users?days=7")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_top_groups_endpoint(client: TestClient) -> None:
    resp = client.get("/api/usage/top-groups?days=7")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
