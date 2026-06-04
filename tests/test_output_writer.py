from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import polars as pl

from research.paper_replication import output_writer


class _FakeFigure:
    def __init__(self) -> None:
        self.html_paths: list[Path] = []
        self.image_paths: list[Path] = []

    def write_html(self, path: str, include_plotlyjs: str = "cdn") -> None:
        del include_plotlyjs
        target = Path(path)
        target.write_text("<html></html>", encoding="utf-8")
        self.html_paths.append(target)

    def write_image(
        self,
        path: str,
        format: str | None = None,
        scale: int = 2,
        width: int = 1100,
        height: int = 650,
    ) -> None:
        del format, scale, width, height
        target = Path(path)
        target.write_bytes(b"png")
        self.image_paths.append(target)


class OutputWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.old_tables_dir = output_writer.TABLES_DIR
        self.old_figures_dir = output_writer.FIGURES_DIR
        output_writer.TABLES_DIR = self.root / "tables"
        output_writer.FIGURES_DIR = self.root / "figures"

    def tearDown(self) -> None:
        output_writer.TABLES_DIR = self.old_tables_dir
        output_writer.FIGURES_DIR = self.old_figures_dir
        self.tmp_dir.cleanup()

    def test_save_table_writes_requested_formats(self) -> None:
        frame = pl.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})

        written = output_writer.save_table(frame, "sample_table")

        self.assertEqual(len(written), 1)
        self.assertTrue((self.root / "tables" / "sample_table.csv").exists())
        self.assertFalse((self.root / "tables" / "sample_table.parquet").exists())

    def test_save_figure_png_only_writes_static_image(self) -> None:
        fig = _FakeFigure()

        written = output_writer.save_figure(fig, "sample_figure")

        self.assertEqual(len(written), 1)
        self.assertTrue(fig.image_paths)
        self.assertTrue((self.root / "figures" / "sample_figure.png").exists())

    def test_clear_outputs_removes_generated_files(self) -> None:
        frame = pl.DataFrame({"x": [1]})
        fig = _FakeFigure()
        output_writer.save_table(frame, "to_clear", formats=("csv",))
        output_writer.save_figure(fig, "to_clear")

        output_writer.clear_outputs()

        self.assertEqual(list((self.root / "tables").iterdir()), [])
        self.assertEqual(list((self.root / "figures").iterdir()), [])


if __name__ == "__main__":
    unittest.main()