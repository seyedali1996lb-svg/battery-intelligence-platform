"""
Turn pytest's JUnit XML into GitHub check-run annotations.

Why this exists: a failing step's LOG is not readable without a GitHub session
-- the REST logs endpoint answers 403 "Must have admin rights" even on a public
repo, and the job page renders "Sign in to view logs". Check-run ANNOTATIONS,
unlike logs, are in the public API. So a suite that fails only in a place nobody
can read is a suite nothing can triage: not an agent, not a monitoring script,
not a contributor without a browser open. This re-emits each failure as an
annotation, which is where the failure list can actually be read from -- and it
shows up next to the file in the GitHub UI as a side effect.

Verified against pytest 9.1.1's output, which matters for one detail: its
<testcase> elements carry classname/name/time and NOT file/line, so a location
has to be recovered from the traceback's closing `path:lineno: ExcType` line
(that is the frame pytest prints for the test body). A parser written against
the documented JUnit schema would have produced annotations with no location.

Two of GitHub's limits shape the output, rather than our preferences:
  * a step carries at most 10 error annotations, so the first 10 failures get
    their own file/line annotation and EVERY failure is listed in one summary
    notice -- a notice is the honest level for the rest, since a long list is a
    handful of bugs, not fifty;
  * an annotation's payload is capped, so messages are trimmed rather than
    allowed to silently vanish.

Exit status: 0 when the report was read and annotated -- whether the suite
passed or failed -- and 1 when there was no usable report. This is a reporting
step, not a gate: the step's verdict is pytest's own exit code, so a bug in here
can neither turn a green build red nor a red one green, and the workflow's
`|| echo "warning: could not annotate"` keeps meaning exactly that. (An earlier
version returned 1 when tests had failed, which made that warning fire on every
genuinely red run -- the annotation would have reported success and failure at
the same time.)

Run locally:  python scripts/annotate_pytest_failures.py path/to/junit.xml
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# GitHub renders at most 10 error annotations per step; the rest are summarized.
MAX_ERROR_ANNOTATIONS = 10
# A single annotation message is capped at 64 KiB by GitHub. Trim well below it
# so the annotation stays readable rather than merely legal.
MAX_MESSAGE_CHARS = 4000
MAX_SUMMARY_CHARS = 60_000

# The last line of pytest's short-format traceback: `path:8: AssertionError`.
# The path is non-greedy so a Windows drive letter (`C:\...`) still matches.
_LOCATION = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+): (?P<exc>[A-Za-z_][\w.]*)\s*$", re.MULTILINE
)
# pytest prefixes the exception's own lines with `E` and padding.
_ERROR_LINE = re.compile(r"^E\s+(?P<text>.+?)\s*$", re.MULTILINE)

# The shape of every line this script prints, so the tests can assert that its
# output is a workflow command and not merely text that looks like one. Property
# values can never contain a literal colon (it is escaped to %3A), so the
# property block is everything up to the first `::`.
_COMMAND = re.compile(r"^::(?P<level>error|warning|notice)(?: (?P<props>[^:]*))?::(?P<message>.*)$")


@dataclass(frozen=True)
class Failure:
    """One failing testcase, reduced to what an annotation can carry."""

    name: str
    """`classname::name` -- the id pytest itself reports."""

    path: str
    """Repo-relative when it can be told, otherwise what the traceback said."""

    line: int
    """0 when the traceback carried no location (a collection error has none)."""

    message: str
    """The exception's own words, on one line."""

    kind: str
    """`failure` (a failed assertion) or `error` (fixture/collection/teardown)."""


@dataclass
class Report:
    """Counts and failures, derived from the testcase nodes themselves.

    Deliberately NOT from <testsuite tests= failures= ...>: those attributes
    double-count when a run writes nested suites, and a wrong denominator is the
    kind of number nobody re-checks.
    """

    failures: list[Failure] = field(default_factory=list)
    passed: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.skipped + len(self.failures)


def _escape_data(text: str) -> str:
    """Workflow-command escaping for the payload after `::`."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    """Property escaping: the payload rules, plus the two separators."""
    return _escape_data(text).replace(":", "%3A").replace(",", "%2C")


def _command(level: str, properties: "dict[str, str]", message: str) -> str:
    """One workflow command line, or `::level::message` when there is no location."""
    props = ",".join(f"{key}={_escape_property(value)}" for key, value in properties.items())
    head = f"::{level} {props}" if props else f"::{level}"
    return f"{head}::{_escape_data(message)}"


def _relative(path: str, root: Path) -> str:
    """A path GitHub can attach an annotation to, when we can produce one.

    pytest run from the repo root already writes repo-relative paths; an
    absolute path under the root is made relative for the same reason (GitHub
    resolves annotation paths against the repository, not the runner's disk).
    Anything else is passed through rather than mangled into a path that points
    nowhere.
    """
    if not path:
        return ""
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.relative_to(root).as_posix()
        except ValueError:
            return candidate.as_posix()
    return path.replace("\\", "/")


def _message(text: str, attribute: str) -> str:
    """The exception's own words, preferring what pytest printed with `E `.

    The `message` attribute is the fallback for the failures pytest does not
    render as short-format traceback lines (a collection error, for instance).
    """
    lines = [match.group("text") for match in _ERROR_LINE.finditer(text)]
    if lines:
        joined = " | ".join(line.strip() for line in lines)
    else:
        first = attribute.strip().splitlines()
        joined = first[0].strip() if first else ""
    return joined[:MAX_MESSAGE_CHARS]


def _failure_from(node: "ET.Element", root: Path) -> "Failure | None":
    """The Failure for a failing <testcase>, or None if it passed or skipped.

    `<error>` means the test never got to run (fixture, collection, teardown) --
    kept separate in `kind` because the fix for an error is not the fix for a
    failed assertion.
    """
    for kind in ("failure", "error"):
        element = node.find(kind)
        if element is None:
            continue
        text = element.text or ""
        classname = node.get("classname") or ""
        name = node.get("name") or ""
        where = _LOCATION.findall(text)
        path, line = "", 0
        if where:
            path, line_text = where[-1][0], where[-1][1]
            line = int(line_text)
        return Failure(
            name=f"{classname}::{name}" if classname else name,
            path=_relative(path, root),
            line=line,
            message=_message(text, element.get("message") or ""),
            kind=kind,
        )
    return None


def parse_report(xml_text: str, root: Path | None = None) -> Report:
    """Parse pytest's JUnit XML. Raises ET.ParseError on anything else."""
    root = root or Path.cwd()
    report = Report()
    for node in ET.fromstring(xml_text).iter("testcase"):
        failure = _failure_from(node, root)
        if failure is not None:
            report.failures.append(failure)
        elif node.find("skipped") is not None:
            report.skipped += 1
        else:
            report.passed += 1
    return report


def _summary_message(report: Report) -> str:
    """Every failing test by name, so the notice is the complete list.

    Bounded, because an annotation payload is: the tail says how many names were
    dropped rather than letting the message be truncated invisibly.
    """
    headline = (
        f"{len(report.failures)} failed, {report.passed} passed, "
        f"{report.skipped} skipped (of {report.total})"
    )
    if not report.failures:
        return headline
    names = [failure.name for failure in report.failures]
    listed: list[str] = []
    for index, name in enumerate(names):
        listed.append(name)
        if len(", ".join(listed)) > MAX_SUMMARY_CHARS - 200:
            dropped = len(names) - index - 1
            listed.append(f"... (+{dropped} more)")
            break
    return f"{headline} — {', '.join(listed)}"


def render(report: Report, *, max_errors: int = MAX_ERROR_ANNOTATIONS) -> "list[str]":
    """The annotation lines to print, in order, for one report."""
    lines: list[str] = []
    for failure in report.failures[:max_errors]:
        properties = {"title": failure.name}
        if failure.path:
            properties["file"] = failure.path
        if failure.line:
            properties["line"] = str(failure.line)
        lines.append(_command("error", properties, failure.message))
    lines.append(_command("notice", {"title": "pytest summary"}, _summary_message(report)))
    return lines


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit GitHub check-run annotations for pytest's JUnit XML."
    )
    parser.add_argument("xml", type=Path, help="the report pytest wrote (--junitxml=...)")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repository root that annotation paths are made relative to (default: cwd)",
    )
    args = parser.parse_args(argv)
    root = args.root or Path.cwd()

    if not args.xml.exists():
        # Not a failure of the tests: it means pytest never got far enough to
        # write a report (an import error in conftest, a killed runner). Said
        # plainly so nobody reads this as "the suite passed".
        print(
            _command(
                "error",
                {"title": "pytest"},
                f"pytest wrote no JUnit XML at {args.xml} — it did not get far enough "
                "to report. The failing step's own log has what happened.",
            )
        )
        return 1

    try:
        xml_text = args.xml.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(_command("error", {"title": "pytest"}, f"could not read {args.xml}: {exc}"))
        return 1

    try:
        report = parse_report(xml_text, root)
    except ET.ParseError as exc:
        print(
            _command(
                "error",
                {"title": "pytest"},
                f"{args.xml} is not parseable JUnit XML: {exc}",
            )
        )
        return 1

    for line in render(report):
        print(line)
    # 0 even when tests failed: the fails are reported as `error` annotations and
    # in the summary notice's headline, and pytest already failed the step. See
    # the module docstring for why the two verdicts must not be entangled.
    return 0


if __name__ == "__main__":
    sys.exit(main())
