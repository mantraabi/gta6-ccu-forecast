"""Schema check of main.py's output for every practice case (and optionally any case).

Runs main.py the way the grader does — one input directory, one output directory, all
*.in in name order — then validates every <id>.out against the contract in BRIEF.md:

  - UTF-8 JSON, at most 65,536 bytes, an object whose only key is "hours";
  - exactly 23 rows, hours 2..24 each exactly once, in order, no other key per row;
  - each row's "q" has exactly 5 finite numbers, all > 0 and <= 1e9, non-decreasing.

Usage:
    python3 tools/selfcheck.py                       # the practice kit's input dir
    python3 tools/selfcheck.py --input DIR --out DIR # explicit directories
It exits 0 when every case is valid, 1 otherwise. It is not part of the entry main.py.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIRS = (
    REPO.parent / "gta6-twitch-ccu-forecast-01-kit" / "input",
    REPO / "input",
    Path("/input"),
)

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
HOURS = tuple(range(2, 25))
MAX_BYTES = 65536
Q_MAX = 1e9


def pick_input_dir(explicit):
    if explicit:
        return Path(explicit)
    env = os.environ.get("MAKEFUN_INPUT_DIR")
    if env:
        return Path(env)
    for directory in DEFAULT_INPUT_DIRS:
        if directory.is_dir() and any(directory.glob("*.in")):
            return directory
    raise SystemExit("selfcheck: no input directory with *.in; pass --input DIR")


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_document(raw, case_id):
    """Every contract violation of one output file, as a list of strings."""
    problems = []
    if len(raw) > MAX_BYTES:
        problems.append("file is %d bytes, over %d" % (len(raw), MAX_BYTES))
    try:
        document = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return ["not UTF-8"]
    except json.JSONDecodeError as error:
        return ["not JSON: %s" % error]

    if not isinstance(document, dict):
        return ["top level is %s, not an object" % type(document).__name__]
    if list(document.keys()) != ["hours"]:
        problems.append("top-level keys are %s, not exactly ['hours']" % list(document.keys()))
        if "hours" not in document:
            return problems

    rows = document["hours"]
    if not isinstance(rows, list):
        return problems + ["hours is not a list"]
    if len(rows) != len(HOURS):
        problems.append("%d rows, not %d" % (len(rows), len(HOURS)))

    seen = []
    for index, row in enumerate(rows):
        where = "row %d" % index
        if not isinstance(row, dict):
            problems.append("%s is not an object" % where)
            continue
        if sorted(row.keys()) != ["hour", "q"]:
            problems.append("%s keys are %s, not exactly ['hour', 'q']" % (where, sorted(row.keys())))
        hour = row.get("hour")
        if not isinstance(hour, int) or isinstance(hour, bool):
            problems.append("%s hour is not an integer" % where)
            hour = None
        else:
            seen.append(hour)
            expected = HOURS[index] if index < len(HOURS) else None
            if hour != expected:
                problems.append("%s hour is %d, expected %s" % (where, hour, expected))
        qs = row.get("q")
        if not isinstance(qs, list):
            problems.append("%s q is not a list" % where)
            continue
        if len(qs) != len(LEVELS):
            problems.append("%s has %d quantiles, not %d" % (where, len(qs), len(LEVELS)))
            continue
        for level, value in zip(LEVELS, qs):
            if not number(value):
                problems.append("%s q at %.2f is %r, not a number" % (where, level, value))
                continue
            if not math.isfinite(value):
                problems.append("%s q at %.2f is not finite" % (where, level))
            elif not (value > 0):
                problems.append("%s q at %.2f is %r, not > 0" % (where, level, value))
            elif value > Q_MAX:
                problems.append("%s q at %.2f is %r, over 1e9" % (where, level, value))
        numeric = [value for value in qs if number(value)]
        for left, right in zip(numeric, numeric[1:]):
            if right < left:
                problems.append("%s is not non-decreasing: %r" % (where, qs))
                break

    if seen != list(HOURS) and not any("hour is" in problem for problem in problems):
        problems.append("hours are %s, not 2..24 each once" % seen)
    return problems


def main_():
    input_dir = None
    output_dir = None
    argv = sys.argv[1:]
    while argv:
        flag = argv.pop(0)
        if flag == "--input" and argv:
            input_dir = argv.pop(0)
        elif flag == "--out" and argv:
            output_dir = argv.pop(0)
        elif flag in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            raise SystemExit("selfcheck: unknown argument %r" % flag)

    source = pick_input_dir(input_dir)
    cases = sorted(source.glob("*.in"))
    if not cases:
        raise SystemExit("selfcheck: no *.in in %s" % source)

    temporary = None
    if output_dir is None:
        temporary = tempfile.TemporaryDirectory()
        target = Path(temporary.name)
    else:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ, MAKEFUN_INPUT_DIR=str(source), MAKEFUN_OUTPUT_DIR=str(target))
    run = subprocess.run([sys.executable, str(REPO / "main.py")], env=environment,
                         capture_output=True, text=True)
    print("input:  %s" % source)
    print("output: %s" % target)
    print("main.py exit code: %d, stdout %d bytes, stderr %d bytes" %
          (run.returncode, len(run.stdout), len(run.stderr)))
    if run.stdout:
        print("  stdout: %r" % run.stdout[:400])
    if run.stderr:
        print("  stderr: %r" % run.stderr[:400])
    failures = 0
    if run.returncode != 0:
        failures += 1

    print()
    print("%-10s %8s %6s %8s  %s" % ("case", "bytes", "rows", "median-h24", "verdict"))
    for case in cases:
        case_id = case.stem
        path = target / (case_id + ".out")
        if not path.is_file():
            print("%-10s %8s %6s %8s  MISSING" % (case_id, "-", "-", "-"))
            failures += 1
            continue
        raw = path.read_bytes()
        problems = check_document(raw, case_id)
        row_count = 0
        median_h24 = float("nan")
        try:
            rows = json.loads(raw.decode("utf-8"))["hours"]
            row_count = len(rows)
            median_h24 = rows[-1]["q"][2]
        except Exception:
            pass
        verdict = "OK" if not problems else "INVALID"
        print("%-10s %8d %6d %8.0f  %s" % (case_id, len(raw), row_count, median_h24, verdict))
        for problem in problems:
            print("    - %s" % problem)
            failures += 1

    if temporary is not None:
        temporary.cleanup()

    print()
    if failures:
        print("FAILED: %d problem(s)" % failures)
        return 1
    print("all %d case(s) valid: 23 rows each, hours 2..24, five finite non-decreasing quantiles > 0" % len(cases))
    return 0


if __name__ == "__main__":
    sys.exit(main_())
