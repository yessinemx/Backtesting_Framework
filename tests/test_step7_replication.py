from contextlib import nullcontext
from unittest.mock import patch
import unittest

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
    def __init__(self, *, buttons, select_values, text_value, number_values, checkbox_value, session_state=None):
        self._buttons = list(buttons)
        self._select_values = list(select_values)
        self._text_value = text_value
        self._number_values = list(number_values)
        self._checkbox_value = checkbox_value
        self.session_state = _FakeSessionState(session_state or {})
        self.info_messages = []
        self.writes = []
        self.dataframes = []
        self.markdowns = []

    def markdown(self, value, unsafe_allow_html=False):
        del unsafe_allow_html
        self.markdowns.append(value)

    def caption(self, value):
        self.writes.append(value)

    def container(self, border=False):
        del border
        return _FakeContainer()

    def columns(self, count):
        return [_FakeContainer() for _ in range(count)]

    def selectbox(self, label, options, index=0):
        del label, options, index
        return self._select_values.pop(0)

    def text_input(self, label, value=""):
        del label, value
        return self._text_value

    def number_input(self, label, **kwargs):
        del label, kwargs
        return self._number_values.pop(0)

    def checkbox(self, label, value=False):
        del label, value
        return self._checkbox_value

    def button(self, label, use_container_width=False):
        del label, use_container_width
        return self._buttons.pop(0)

    def spinner(self, message):
        del message
        return nullcontext()

    def info(self, message):
        self.info_messages.append(message)

    def write(self, value):
        self.writes.append(value)

    def dataframe(self, value, use_container_width=False):
        del use_container_width
        self.dataframes.append(value)


class Step7ReplicationTests(unittest.TestCase):
    def test_render_runs_smoke_test_and_stores_result(self) -> None:
        fake_st = _FakeStreamlit(
            buttons=[True, False],
            select_values=["data", "SPX", "distance"],
            text_value="sym20",
            number_values=[12, 30, 63, 1.7, 2],
            checkbox_value=False,
            session_state={},
        )
        result = {
            "summary": pl.DataFrame({"variant": ["standard"], "mean_return": [0.1]}),
            "by_period": pl.DataFrame(),
            "params": {"method": "distance", "wavelet": "sym20", "index_id": "SPX"},
        }

        with patch.object(step7_module, "st", fake_st), patch.object(step7_module, "run_pipeline", return_value=result) as run_pipeline:
            step7_module.render()

        self.assertEqual(fake_st.session_state["paper_replication_result"], result)
        run_pipeline.assert_called_once()
        kwargs = run_pipeline.call_args.kwargs
        self.assertEqual(kwargs["source"], "data")
        self.assertFalse(kwargs["write_outputs"])
        self.assertFalse(kwargs["verbose"])
        self.assertEqual(kwargs["params"]["index_id"], "SPX")
        self.assertEqual(kwargs["params"]["top_n"], 12)
        self.assertEqual(kwargs["params"]["candidate_pool"], 30)
        self.assertEqual(kwargs["params"]["block_size"], 63)
        self.assertEqual(kwargs["params"]["threshold_sigma"], 1.7)
        self.assertEqual(kwargs["params"]["max_periods"], 2)

    def test_render_uses_existing_result_when_no_run_requested(self) -> None:
        result = {
            "summary": pl.DataFrame({"variant": ["standard"], "mean_return": [0.1]}),
            "by_period": pl.DataFrame({"variant": ["standard"], "period": [1]}),
            "params": {"method": "distance", "wavelet": "db4", "index_id": "NDX"},
        }
        fake_st = _FakeStreamlit(
            buttons=[False, False],
            select_values=["data", None, "distance"],
            text_value="sym20",
            number_values=[12, 30, 63, 1.7, 2],
            checkbox_value=False,
            session_state={"paper_replication_result": result},
        )

        with patch.object(step7_module, "st", fake_st), patch.object(step7_module, "run_pipeline") as run_pipeline:
            step7_module.render()

        run_pipeline.assert_not_called()
        self.assertGreaterEqual(len(fake_st.dataframes), 2)
        self.assertTrue(any("Tables:" in str(item) for item in fake_st.writes))


if __name__ == "__main__":
    unittest.main()