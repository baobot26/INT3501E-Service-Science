"""Smoke test for the Streamlit app: it must load and run a scenario.

Skipped in Docker CI where app.py and data/ are not present.
"""
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app.py"
pytestmark = pytest.mark.skipif(
    not APP.is_file() or not (APP.parent / "data" / "demand.csv").is_file(),
    reason="app.py or data/ not available (Docker CI)",
)


def test_app_loads_without_error():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any(b.label == "🚀 Chạy kịch bản" for b in at.button)
    # larger radii overrun the solver's time limit and return nothing
    assert at.sidebar.slider[0].max <= 800


def test_app_runs_the_default_scenario():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    # no slider touched: the defaults are what a demo runs first, so they must solve
    at.button[0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("Số trạm" in m.label for m in at.metric)
