"""Fixtures for Playwright end-to-end tests against a live Streamlit server."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PORT = 8599
HEALTH_URL = f"http://127.0.0.1:{APP_PORT}/_stcore/health"
BASE_URL = f"http://127.0.0.1:{APP_PORT}"


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "e2e" in item.keywords:
        pytest.importorskip("playwright")


def _wait_for_health(proc: subprocess.Popen[str], log_path: Path, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: str | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
            raise RuntimeError(
                f"Streamlit exited early with code {proc.returncode}:\n{out}"
            )
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2.0) as resp:
                body = resp.read().decode("utf-8", errors="ignore").strip().lower()
                if resp.status == 200 and body == "ok":
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = str(exc)
        time.sleep(0.25)
    out = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    raise RuntimeError(
        f"Streamlit did not become healthy at {HEALTH_URL} within {timeout_s}s "
        f"(last error: {last_err})\n{out}"
    )


@pytest.fixture(scope="session")
def app_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    log_dir = tmp_path_factory.mktemp("streamlit_e2e")
    log_path = log_dir / "streamlit.log"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(APP_PORT),
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--browser.gatherUsageStats",
        "false",
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_health(proc, log_path, timeout_s=60.0)
            yield BASE_URL
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


@pytest.fixture(scope="session")
def shots_dir() -> Path:
    custom = os.environ.get("E2E_SHOTS_DIR")
    path = Path(custom) if custom else (REPO_ROOT / "test-results" / "viewport")
    path.mkdir(parents=True, exist_ok=True)
    return path
