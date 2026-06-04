from contextlib import nullcontext
from unittest.mock import patch
import unittest

import pandas as pd
import polars as pl

import app.steps.step7_replication as step7_module


class _FakeContainer:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _FakeStreamlit:
    """Minimal Streamlit stub covering the widgets used by Step 7."""

    def __init__(self, *, run_clicked, source, methods, sweeps, session_state=None):
        self._run_clicked = run_clicked
        self._source = source
        self._methods = methods
        self._sweeps = sweeps
        self.session_state = _FakeSessionState(session_state or {})
        self.info_messages = []
        self.success_messages = []
        self.writes = []
        self.dataframes = []
        self.charts = []
        self.markdowns = []

    # layout
    def markdown(self, value, unsafe_allow_html=False):
        self.markdowns.append(value)

    def caption(self, value):
        self.writes.append(value)

    def write(self, value):
        self.writes.append(value)

    def container(self, border=False):
        return _FakeContainer()

    def columns(self, count):
        return [_FakeContainer() for _ in range(count)]

    def tabs(self, labels):
        return [_FakeContainer() for _ in labels]

    def spinner(self, message):
        return nullcontext()

    # inputs
    def selectbox(self, label, options, index=0):
        return self._source

    def multiselect(self, label, options, default=None):
        return list(self._methods)

    def checkbox(self, label, value=False):
        return self._sweeps

    def button(self, label, **kwargs):
        return self._run_clicked

    # outputs
    def success(self, message):
        self.success_messages.append(message)

    def info(self, message):
        self.info_messages.append(message)

    def warning(self, message):
        self.info_messages.append(message)

    def dataframe(self, value, use_container_width=False):
        self.dataframes.append(value)

    def plotly_chart(self, fig, use_container_width=False, key=None):
        self.charts.append(key)


def _fake_report():
    return {
        "figures": {"fig01_pyramid": object(), "fig04_cumulative_distance": object(),
                    "fig09_yearly_alpha_distance": object()},
        "comparison": pl.DataFrame({"method": ["distance"], "repl_wav_return_%": [-2.1]}),
        "summaries": {"distance": pl.DataFrame({"variant": ["standard"], "mean_return": [0.1]})},
        "by_period": {"distance": pl.DataFrame()},
        "alpha_table": pd.DataFrame({"method": ["distance"], "alpha_annual_%": [-2.8]}),
        "params": {"wavelet": "sym22"},
        "n_periods": 7,
        "universe_pool": 272,
    }


class Step7ReplicationTests(unittest.TestCase):
    def test_run_calls_build_report_and_stores_result(self) -> None:
        fake_st = _FakeStreamlit(run_clicked=True, source="data",
                                 methods=["distance"], sweeps=False, session_state={})
        report = _fake_report()
        with patch.object(step7_module, "st", fake_st), \
             patch.object(step7_module, "build_report", return_value=report) as build:
            step7_module.render()

        build.assert_called_once()
        kwargs = build.call_args.kwargs
        self.assertEqual(kwargs["source"], "data")
        self.assertEqual(tuple(kwargs["methods"]), ("distance",))
        self.assertFalse(kwargs["sweeps"])
        self.assertIs(fake_st.session_state["paper_report"], report)
        # comparison + at least one summary table rendered, plus figures charted.
        self.assertGreaterEqual(len(fake_st.dataframes), 2)
        self.assertTrue(fake_st.charts)

    def test_existing_report_renders_without_running(self) -> None:
        report = _fake_report()
        fake_st = _FakeStreamlit(run_clicked=False, source="data",
                                 methods=["distance"], sweeps=False,
                                 session_state={"paper_report": report})
        with patch.object(step7_module, "st", fake_st), \
             patch.object(step7_module, "build_report") as build:
            step7_module.render()

        build.assert_not_called()
        self.assertTrue(fake_st.success_messages)
        self.assertTrue(any("alpha" in str(d).lower() or "method" in str(d).lower()
                            for d in fake_st.dataframes))

    def test_no_report_shows_info(self) -> None:
        fake_st = _FakeStreamlit(run_clicked=False, source="data",
                                 methods=["distance"], sweeps=False, session_state={})
        with patch.object(step7_module, "st", fake_st), \
             patch.object(step7_module, "build_report") as build:
            step7_module.render()
        build.assert_not_called()
        self.assertTrue(fake_st.info_messages)


if __name__ == "__main__":
    unittest.main()
