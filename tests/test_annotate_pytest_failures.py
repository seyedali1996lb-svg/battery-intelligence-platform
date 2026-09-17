"""
Unit tests for scripts/annotate_pytest_failures.py -- the CI helper that makes a
failing test suite readable from the public API instead of only behind a
GitHub sign-in.

The XML fixtures here are TRIMMED COPIES of output captured from this project's
own pytest (9.1.1), not hand-written against the JUnit schema. That distinction
is the point: the schema does not say whether <testcase> carries a location, and
this pytest does NOT, so a fixture written from the documentation (or from
memory) would have asserted a `file=` attribute that never arrives -- and the
annotations would have shipped with no location at all, which is the one thing
they exist to provide. tests/test_annotate_pytest_failures.py's last test runs a
real pytest so the fixture cannot drift from reality unnoticed.

Run:  python -m pytest tests/test_annotate_pytest_failures.py -v
"""

import os as _os
import subprocess as _subprocess
import sys as _sys

_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401

from annotate_pytest_failures import (  # noqa: E402
    _COMMAND,
    Failure,
    Report,
    main,
    parse_report,
    render,
)

# Captured from pytest 9.1.1: two failures (one assertion, one raise), a pass and
# a skip. Note the absence of file/line attributes on <testcase>.
REAL_XML = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
    '<testsuite name="pytest" errors="0" failures="2" skipped="1" tests="4" '
    'time="0.041" timestamp="2026-09-18T01:29:00.966511+02:00" hostname="runner">'
    '<testcase classname="tests.test_sample" name="test_passes" time="0.000" />'
    '<testcase classname="tests.test_sample" name="test_assertion_fails" time="0.000">'
    '<failure message="AssertionError: assert 1 == 2&#10;  Differing items:">'
    "def test_assertion_fails():\n"
    "        value = 1\n"
    ">       assert value == 2\n"
    "E       AssertionError: assert 1 == 2\n"
    "E         Use -v to get more diff\n"
    "\n"
    "tests/test_sample.py:8: AssertionError</failure>"
    "</testcase>"
    '<testcase classname="tests.test_sample" name="test_raises" time="0.000">'
    '<failure message="RuntimeError: line one&#10;line two">'
    "def test_raises():\n"
    ">       raise RuntimeError(\"line one\\nline two\")\n"
    "E       RuntimeError: line one\n"
    "E       line two\n"
    "\n"
    "tests/test_sample.py:11: RuntimeError</failure>"
    "</testcase>"
    '<testcase classname="tests.test_sample" name="test_skipped" time="0.000">'
    '<skipped type="pytest.skip" message="demo skip">'
    "tests/test_sample.py:14: demo skip</skipped>"
    "</testcase>"
    "</testsuite></testsuites>"
)

# A collection error: pytest writes an <error> whose traceback has no
# `path:lineno: Exc` tail at all, because no test body ever ran.
COLLECTION_ERROR_XML = (
    '<testsuites><testsuite name="pytest" tests="1" errors="1">'
    '<testcase classname="tests.test_broken" name="tests.test_broken" time="0.000">'
    '<error message="collection failure">ImportError while importing test module.'
    "\n"
    "tests/test_broken.py:3: in &lt;module&gt;\n"
    "    from nothing import here\n"
    "E   ModuleNotFoundError: No module named 'nothing'</error>"
    "</testcase></testsuite></testsuites>"
)


def _report_with(count: int) -> Report:
    """A report with `count` distinct failures, for the annotation cap."""
    report = Report(passed=1, skipped=0)
    for index in range(count):
        report.failures.append(
            Failure(
                name=f"tests.test_many::test_{index}",
                path="tests/test_many.py",
                line=index + 1,
                message=f"AssertionError {index}",
                kind="failure",
            )
        )
    return report


# ---------------------------------------------------------------------------
# Parsing what pytest actually writes
# ---------------------------------------------------------------------------

def test_counts_every_testcase_by_its_outcome():
    report = parse_report(REAL_XML)
    assert report.passed == 1
    assert report.skipped == 1
    assert len(report.failures) == 2
    assert report.total == 4


def test_recovers_the_location_from_the_traceback_not_from_attributes():
    """The attribute pytest does not write, and the tail line it does."""
    report = parse_report(REAL_XML)
    assert report.failures[0].path == "tests/test_sample.py"
    assert report.failures[0].line == 8
    assert report.failures[1].line == 11


def test_a_failure_is_named_the_way_pytest_names_it():
    report = parse_report(REAL_XML)
    assert report.failures[0].name == "tests.test_sample::test_assertion_fails"


def test_message_is_the_exception_not_the_traceback():
    """The `E ` lines pytest prints, not the source it reprints above them."""
    message = parse_report(REAL_XML).failures[0].message
    assert message.startswith("AssertionError: assert 1 == 2")
    assert "Use -v to get more diff" in message, message
    assert "value = 1" not in message, "the replayed source leaked into the message"


def test_a_multiline_exception_becomes_one_line():
    """An annotation's message is one line; a raw newline would truncate it."""
    report = parse_report(REAL_XML)
    message = report.failures[1].message
    assert "\n" not in message
    assert "line one" in message and "line two" in message


def test_an_error_element_is_a_failure_of_kind_error():
    """A collection error never ran a test body; the kind says so."""
    report = parse_report(COLLECTION_ERROR_XML)
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert failure.kind == "error"
    assert failure.line == 0
    assert "ModuleNotFoundError" in failure.message


def test_xfail_counts_as_skipped_not_passed():
    xml = (
        '<testsuites><testsuite name="pytest" tests="1" skipped="1">'
        '<testcase classname="tests.test_x" name="test_x" time="0.000">'
        '<skipped type="pytest.xfail" message="expected failure">boom</skipped>'
        "</testcase></testsuite></testsuites>"
    )
    report = parse_report(xml)
    assert report.skipped == 1
    assert report.passed == 0


def test_nested_suites_do_not_double_count():
    """Counts come from the testcase nodes, not from <testsuite tests= ...>.

    A nested report makes those attributes sum to more than the test count, and
    a wrong denominator is the kind of number nobody re-checks.
    """
    xml = (
        '<testsuites><testsuite name="outer" tests="1" failures="0">'
        '<testsuite name="inner" tests="1" failures="1">'
        '<testcase classname="tests.test_x" name="test_a" time="0.000">'
        '<failure message="AssertionError">f()\nE   AssertionError\ntest_x.py:9: AssertionError'
        "</failure></testcase>"
        "</testsuite></testsuite></testsuites>"
    )
    report = parse_report(xml)
    assert report.total == 1


def test_an_absolute_path_under_the_root_is_made_relative(tmp_path):
    xml = (
        f'<testsuites><testsuite name="pytest" tests="1" failures="1">'
        f'<testcase classname="tests.test_x" name="test_a" time="0.000">'
        f'<failure message="AssertionError">f()\nE   AssertionError\n'
        f"{tmp_path}/tests/test_x.py:9: AssertionError</failure>"
        f"</testcase></testsuite></testsuites>"
    )
    report = parse_report(xml, root=tmp_path)
    assert report.failures[0].path == "tests/test_x.py"


def test_a_path_outside_the_root_is_passed_through_not_mangled(tmp_path):
    """GitHub resolves annotation paths against the repo; inventing a relative
    path for a file that is not in it would point at the wrong place."""
    xml = (
        '<testsuites><testsuite name="pytest" tests="1" failures="1">'
        '<testcase classname="x" name="test_a" time="0.000">'
        '<failure message="AssertionError">f()\nE   AssertionError\n'
        "/elsewhere/test_x.py:9: AssertionError</failure>"
        "</testcase></testsuite></testsuites>"
    )
    report = parse_report(xml, root=tmp_path)
    assert report.failures[0].path == "/elsewhere/test_x.py"


# ---------------------------------------------------------------------------
# The annotation lines themselves
# ---------------------------------------------------------------------------

def test_every_line_is_a_workflow_command():
    for line in render(_report_with(2)):
        assert _COMMAND.match(line), line


def test_the_location_travels_as_properties():
    line = render(parse_report(REAL_XML))[0]
    match = _COMMAND.match(line)
    assert match is not None and match.group("level") == "error"
    props = match.group("props") or ""
    assert "file=tests/test_sample.py" in props
    assert "line=8" in props
    # `::` is escaped here because a property value ends at `,` or `:`; the
    # unescaped id is what the summary notice carries.
    assert "title=tests.test_sample%3A%3Atest_assertion_fails" in props


def test_a_percent_sign_in_the_message_is_escaped():
    """Real messages carry percentages (`Final SOH: 72.3%`); an unescaped % is
    read as the start of an escape sequence and corrupts the annotation."""
    report = Report(passed=0, skipped=0)
    report.failures.append(
        Failure(
            name="tests.test_pct::test_pct",
            path="tests/test_pct.py",
            line=3,
            message="AssertionError: final SOH 72.3% != 71.0%",
            kind="failure",
        )
    )
    line = render(report)[0]
    assert "72.3%25" in line
    assert "72.3% !=" not in line


def test_property_separators_in_a_name_are_escaped():
    """A comma or colon inside a test id would otherwise end the property."""
    report = Report(passed=0, skipped=0)
    report.failures.append(
        Failure(
            name="tests.test_param::test_x[a,b:1]",
            path="tests/test_param.py",
            line=5,
            message="AssertionError",
            kind="failure",
        )
    )
    line = render(report)[0]
    props = _COMMAND.match(line).group("props") or ""
    assert "a%2Cb%3A1" in props
    assert props.count("::") == 0


def test_at_most_ten_error_annotations_but_every_failure_is_named():
    """GitHub renders at most 10 error annotations per step, so the rest have to
    survive somewhere: the summary notice lists all of them."""
    report = _report_with(14)
    lines = render(report)
    errors = [line for line in lines if line.startswith("::error")]
    assert len(errors) == 10
    summary = lines[-1]
    assert summary.startswith("::notice")
    for index in range(14):
        assert f"tests.test_many::test_{index}" in summary


def test_the_summary_reports_the_counts():
    summary = render(_report_with(1))[-1]
    assert "1 failed, 1 passed, 0 skipped (of 2)" in summary


def test_a_green_report_still_gets_a_summary():
    """The absence of a verdict is not evidence of one: a green run says so."""
    lines = render(Report(passed=1530, skipped=10))
    assert len(lines) == 1
    assert lines[0].startswith("::notice")
    assert "0 failed, 1530 passed, 10 skipped (of 1540)" in lines[0]


# ---------------------------------------------------------------------------
# Exit status: what the step can rely on
# ---------------------------------------------------------------------------

def test_main_exits_zero_on_a_green_report(tmp_path, capsys):
    xml = tmp_path / "green.xml"
    xml.write_text(
        '<testsuites><testsuite name="pytest" tests="1">'
        '<testcase classname="tests.test_x" name="test_a" time="0.000" />'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    assert main([str(xml)]) == 0
    out = capsys.readouterr().out
    assert "::notice" in out and "::error" not in out


def test_main_exits_one_and_says_so_when_the_report_is_missing(tmp_path, capsys):
    """No XML means pytest never got far enough to report -- never "the suite
    passed"."""
    assert main([str(tmp_path / "never-written.xml")]) == 1
    out = capsys.readouterr().out
    assert out.startswith("::error")
    assert "wrote no JUnit XML" in out


def test_main_exits_one_on_an_unparseable_report(tmp_path, capsys):
    xml = tmp_path / "garbage.xml"
    xml.write_text("this is not xml", encoding="utf-8")
    assert main([str(xml)]) == 1
    assert "not parseable" in capsys.readouterr().out


def test_main_annotates_a_failing_report(tmp_path, capsys):
    xml = tmp_path / "red.xml"
    xml.write_text(REAL_XML, encoding="utf-8")
    assert main([str(xml)]) == 0
    out = capsys.readouterr().out
    assert "::error " in out
    assert "file=tests/test_sample.py" in out
    assert "line=8" in out


def test_a_red_run_still_exits_zero_so_the_workflows_warning_keeps_its_meaning(
    tmp_path, capsys
):
    """The step's verdict belongs to pytest, not to this script.

    Pinned because the first version of this file returned 1 whenever tests had
    failed, which made ci.yml's `annotator || echo "::warning::could not annotate
    the failures"` fire on every genuinely red run: a warning claiming it had
    failed to annotate, printed directly beneath the annotations it had just
    written. 1 must mean "no usable report", and nothing else.
    """
    xml = tmp_path / "red.xml"
    xml.write_text(REAL_XML, encoding="utf-8")
    assert main([str(xml)]) == 0
    assert capsys.readouterr().out.count("::error ") == 2


# ---------------------------------------------------------------------------
# End to end: a real pytest run, so the fixture cannot drift from reality
# ---------------------------------------------------------------------------

def test_end_to_end_against_a_real_pytest_run(tmp_path, capsys):
    """Fixture-free on purpose.

    This is the test that would catch pytest changing its JUnit output (as it
    would if it ever adds or renames the location it writes), which is exactly
    what no amount of hand-written XML can catch.
    """
    test_file = tmp_path / "test_live_failure.py"
    test_file.write_text(
        "def test_this_fails():\n"
        "    value = {'a': 1}\n"
        "    assert value == {'a': 2}\n",
        encoding="utf-8",
    )
    xml = tmp_path / "live.xml"
    env = {key: value for key, value in _os.environ.items() if key != "PYTEST_CURRENT_TEST"}
    result = _subprocess.run(
        [
            _sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={xml}",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, result.stdout
    assert xml.exists(), "pytest did not write the report this script needs"

    assert main([str(xml)]) == 0
    out = capsys.readouterr().out
    assert "title=test_live_failure%3A%3Atest_this_fails" in out
    assert "file=test_live_failure.py" in out
    assert "line=" in out, f"the annotation carries no location: {out!r}"
    assert "AssertionError" in out
    # ...and the real id, unescaped, in the summary the reader scans first.
    assert "test_live_failure::test_this_fails" in out
