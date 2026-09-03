"""配布HTMLの再生成一致とオフライン参照を検査する。"""
from html.parser import HTMLParser
from pathlib import Path
import runpy


REPORT = Path("reports/プラズマ等価回路_収束性改善ガイド.html")


class ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.resources: list[str] = []
        self.svg_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag == "a" and attributes.get("href"):
            self.links.append(attributes["href"])
        if attributes.get("src"):
            self.resources.append(attributes["src"])
        self.svg_count += tag == "svg"


def test_convergence_report_is_reproducible() -> None:
    builder = runpy.run_path("scripts/build_convergence_report.py")
    assert REPORT.read_text(encoding="utf-8") == builder["build"]()


def test_convergence_report_has_valid_local_links_and_embedded_figures() -> None:
    document = REPORT.read_text(encoding="utf-8")
    parser = ReportParser()
    parser.feed(document)
    assert "@@" not in document
    assert parser.svg_count == 2
    assert not parser.resources
    for link in parser.links:
        if link.startswith("#"):
            assert link[1:] in parser.ids
        elif not link.startswith("https://"):
            assert (REPORT.parent / link).is_file(), link
