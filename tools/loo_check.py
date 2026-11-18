"""Leave-one-out validation of main.py's launch model on the console-only launches.

For each of the eight console-only launches in the comparables (GTA VI ships on PS5 and
Xbox Series only, so these are its right neighbours) the launch is held out of the
neighbour pool, forecast from the other launches, and scored against what it actually
did, hour by hour, hours 2..24. The score is the competition's own pinball loss on the
log scale (RULES.txt, Scoring):

    d = ln(y) - ln(q)
    loss(tau) = 2 * max(tau * d, (tau - 1) * d)

averaged over the 23 hours and 5 levels. The persistence baseline
(q_tau = p1 * exp(z_tau * 0.15 * sqrt(h - 1)), the kit's starter) is scored on the very
same items for contrast, since a prize needs a lower loss than it.

Usage (the case file supplies the comparables; nothing is fetched):
    python3 tools/loo_check.py                       # uses the practice kit's input dir
    python3 tools/loo_check.py path/to/p00.in        # or an explicit case file
    MAKEFUN_INPUT_DIR=input python3 tools/loo_check.py

It reuses main.py's own feature extraction, weighting and quantiles, so what it scores
is what main.py ships, not a reimplementation that could drift from it.
"""

import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import main  # noqa: E402  (needs the sys.path line above)

# GTA VI launches on PS5 and Xbox Series only: no PC build on day one.
CONSOLE_ONLY = (
    "Ghost of Yōtei",
    "FINAL FANTASY VII REBIRTH",
    "The Last of Us Part II",
    "Ghost of Tsushima",
    "Stellar Blade",
    "Death Stranding 2: On the Beach",
    "Saros",
    "The Legend of Zelda: Tears of the Kingdom",
)

DEFAULT_INPUT_DIRS = (
    REPO.parent / "gta6-twitch-ccu-forecast-01-kit" / "input",
    REPO / "input",
    Path("/input"),
)


def case_path(argv):
    """The case file whose comparables the check runs on."""
    if len(argv) > 1:
        given = Path(argv[1])
        return given if given.is_file() else given / "p00.in"
    explicit = os.environ.get("MAKEFUN_INPUT_DIR")
    directories = ([Path(explicit)] if explicit else []) + list(DEFAULT_INPUT_DIRS)
    for directory in directories:
        for name in ("p00.in", "p01.in", "p02.in", "p03.in"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    for directory in directories:
        if directory.is_dir():
            for candidate in sorted(directory.glob("*.in")):
                return candidate
    raise SystemExit("loo_check: no *.in case file found; pass one: python3 tools/loo_check.py <case.in>")


def pinball_item(tau, actual, forecast):
    d = math.log(max(actual, 1.0)) - math.log(max(forecast, main.Q_MIN))
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def load_launches(path):
    """Every real launch in the case's comparables, as the model's features."""
    case = json.loads(path.read_text(encoding="utf-8"))
    launches = []
    for comparable in case.get("comparables") or ():
        features = main.launch_features(comparable)
        if features is not None:
            features["name"] = comparable.get("name")
            launches.append(features)
    return launches


def forecast(launch, pool, blend=True):
    """main.py's quantiles for a held-out launch, from p1 = its own hour-1 print.

    blend=True is what main.py ships (the kernel averaged with the log-log regression);
    blend=False scores the kernel alone, for contrast.
    """
    p1 = launch["first"]
    la_case = launch["la"]
    lb_case = launch["lb"]
    rows = {}
    last = None
    for hour in main.HOURS:
        if blend:
            ratios = main.combined_ratios(pool, hour, p1, la_case, lb_case)
        else:
            ratios = main.ratio_quantiles(pool, hour, la_case, lb_case)
        if ratios is None:
            ratios = last if last is not None else [0.0] * len(main.LEVELS)
        last = ratios
        rows[hour] = main.clamp_row([p1 * math.exp(r) for r in ratios])
    return rows


def baseline(launch):
    return {hour: main.clamp_row(main.baseline_quantiles(launch["first"], hour)) for hour in main.HOURS}


def score(actual_curve, rows):
    """Mean pinball loss over every scored item (hours 2..24 x 5 levels)."""
    total = 0.0
    items = 0
    for hour in main.HOURS:
        actual = actual_curve.get(hour)
        if actual is None:
            continue
        for tau, quantile in zip(main.LEVELS, rows[hour]):
            total += pinball_item(tau, actual, quantile)
            items += 1
    return (total / items) if items else float("nan")


def main_():
    path = case_path(sys.argv)
    launches = load_launches(path)
    by_name = {}
    for launch in launches:
        by_name.setdefault(launch["name"], launch)

    print("leave-one-out, console-only launches (main.py's blend vs kernel alone vs the house baseline)")
    print("case: %s   launches in the comparables: %d" % (path, len(launches)))
    print()
    print("%-46s %10s %8s %9s %9s %9s" % ("held out launch", "hour-1", "lb", "blended", "kernel", "baseline"))

    blend_losses = []
    kernel_losses = []
    baseline_losses = []
    missing = []
    for name in CONSOLE_ONLY:
        launch = by_name.get(name)
        if launch is None:
            missing.append(name)
            continue
        pool = [other for other in launches if other["name"] != name]
        blend_loss = score(launch["curve"], forecast(launch, pool, blend=True))
        kernel_loss = score(launch["curve"], forecast(launch, pool, blend=False))
        base_loss = score(launch["curve"], baseline(launch))
        blend_losses.append(blend_loss)
        kernel_losses.append(kernel_loss)
        baseline_losses.append(base_loss)
        print("%-46s %10.0f %8.2f %9.4f %9.4f %9.4f"
              % (name[:44], launch["first"], launch["lb"], blend_loss, kernel_loss, base_loss))

    if missing:
        print()
        print("not in this case's comparables: %s" % ", ".join(missing))
    if not blend_losses:
        raise SystemExit("loo_check: none of the console-only launches is in the case")

    mean_blend = sum(blend_losses) / len(blend_losses)
    mean_kernel = sum(kernel_losses) / len(kernel_losses)
    mean_base = sum(baseline_losses) / len(baseline_losses)
    print()
    print("mean pinball loss, blended (ALPHA=%.1f): %.4f  (over %d launches)"
          % (main.ALPHA, mean_blend, len(blend_losses)))
    print("mean pinball loss, kernel alone:  %.4f" % mean_kernel)
    print("mean pinball loss, baseline:      %.4f" % mean_base)
    print("blend improvement on kernel alone:   %.1f%%" % (100.0 * (1.0 - mean_blend / mean_kernel)))
    print("blend improvement on the baseline:   %.1f%%" % (100.0 * (1.0 - mean_blend / mean_base)))
    return 0


if __name__ == "__main__":
    sys.exit(main_())
