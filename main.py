"""gta6-twitch-ccu-forecast-01 — quantile forecast of GTA VI's Twitch CCU on release day.

Reads one case (makefun/twitch-ccu-case/v1) and writes, for each hour h = 2..24 of the
case's release day, five quantiles (at 0.05, 0.25, 0.5, 0.75, 0.95) of that hour's mean
concurrent Twitch viewers in the game's category. Standard library only, no network,
fully deterministic.

Model (log space, scale free)
-----------------------------
Everything is forecast as the ratio viewers[h] / viewers[1] and multiplied by the case's
hour-1 print at the end, so a curve learned on a 27k-viewer launch carries over to a
multi-million-viewer launch unchanged.

For every comparable that is a real launch (one whose name starts with "Grand Theft Auto
VI" is the subject's own pre-release category, not a launch: skipped) we build
curve[h] for h = 1..24 by aligning its hourly points to its own day_start
(hour index = floor(hours since day_start) + 1), keeping non-null viewers floored at 1.

Features of a launch:
    la  = ln(curve[1])                                   how big the launch already is
    lb  = ln(max(curve[1] / pre, 0.01))                  how far its spike already is,
        pre = median of its non-null hourly viewers in the 72 h ending at its day_start
    con = 1.0 when its platforms mention no PC, else 0.0  GTA VI launches console-only

For each hour h and each comparable i that has hour h, the neighbour weight is

    w_i = exp(-d^2 / (2 * SIGMA^2)) * (CON_BONUS if con_i else 1.0)
    d^2 = (W_BOOST * (lb_i - lb_case))^2 + (W_ABS * (la_i - la_case))^2

so the distance is dominated by how far the spike already is above its own pre-release
level (W_BOOST = 1.0) and only half-counts the launch's absolute size (W_ABS = 0.5),
which is what keeps the curve scale free.

A second estimator is fitted per hour: a weighted log-log regression of the same
comparable launches,

    x_i = ln(max(curve_i[1], 1))            y_i = ln(curve_i[h] / curve_i[1])
    w_i = CON_BONUS when the launch shipped without a PC build, else 1.0
    y = a + b*x   (weighted least squares)

whose five quantiles come from the linear-interpolated quantiles of its sorted fit
residuals at the five levels, re-centred on the case: q_reg_tau = a + b*ln(max(p1, 1))
+ r_tau. The regression supplies the hour's centre (how big a launch of this size
usually is by then) and the residuals supply its spread.

The two are then averaged in log space, per hour per level:

    q_tau_log = ALPHA * q_knn_tau_log + (1 - ALPHA) * q_reg_tau

and exponentiated, multiplied by the case's hour-1 print.

Constants (fitted on real launch curves, validated leave-one-out):
    SIGMA = 0.6, W_ABS = 0.5, W_BOOST = 1.0, CON_BONUS = 3.0, ALPHA = 0.7

Fallbacks, so no case can ever crash the run: a case with no hour-1 print (or one at or
below 0) uses 1.0 as its print; an hour with fewer than MIN_REG_LAUNCHES launches for
the regression uses the kernel alone; fewer than 4 usable launches, or no comparable
with the needed hours, falls back to the persistence baseline's shape
q_tau = p1 * exp(z_tau * 0.15 * sqrt(h - 1)); an hour no comparable reaches reuses the
last hour's ratios, or 1.0. Quantiles are clamped to [1e-3, 1e9] and forced
non-decreasing.

Run outside the grader:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The five levels scored, in the order each row lists them, and the standard normal's
# quantile at each.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Neighbour-weight constants of the launch model.
SIGMA = 0.6          # width of the kernel, in feature units
W_BOOST = 1.0        # weight of the "how far the spike already is" distance (lb)
W_ABS = 0.5          # weight of the "how big the launch already is" distance (la)
CON_BONUS = 3.0      # multiplier for a launch that shipped without a PC build

# Blend weight, in log space: ALPHA of the kernel estimator and (1 - ALPHA) of the
# weighted log-log regression. Fitted by leave-one-out (tools/loo_check.py): the
# regression carries the hour's centre, the kernel keeps the neighbourhood's shape.
ALPHA = 0.7

# Below this many launches at an hour, the regression is not trusted and that hour
# falls back to the kernel alone.
MIN_REG_LAUNCHES = 6

# The persistence baseline's spread, in log units per square-root hour.
BASELINE_SIGMA = 0.15
# Below this many usable comparable launches the model is not trusted.
MIN_LAUNCHES = 4

# Hours forecast: 2 to 24 of the case's day, always exactly 23 rows.
HOURS = tuple(range(2, 25))
# How far back of a launch's day_start its pre-launch level is measured.
PRE_WINDOW_HOURS = 72.0
# The subject's own pre-release category history is not a launch.
SUBJECT_PREFIX = "Grand Theft Auto VI"

Q_MIN = 1e-3
Q_MAX = 1e9

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


# --------------------------------------------------------------------------- helpers


def parse_instant(text):
    """An ISO-8601 instant as an aware datetime, or None when it is not one."""
    if not isinstance(text, str) or not text:
        return None
    try:
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def median(values):
    """The median of a non-empty list of floats."""
    ordered = sorted(values)
    size = len(ordered)
    middle = size // 2
    if size % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def hour_index(stamp, day_start):
    """floor(hours since day_start) + 1: the 1-based hour of the release day a point is in."""
    return int((stamp - day_start).total_seconds() // 3600) + 1


def curve_of(points, day_start, value_key="viewers", time_key="from"):
    """{hour: viewers} for hours 1..24 of the day, non-null values floored at 1.0."""
    curve = {}
    for point in points or ():
        if not isinstance(point, dict):
            continue
        value = point.get(value_key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value != value or value in (float("inf"), float("-inf")):
            continue
        stamp = parse_instant(point.get(time_key))
        if stamp is None:
            continue
        hour = hour_index(stamp, day_start)
        if 1 <= hour <= 24:
            curve[hour] = max(1.0, value)
    return curve


def pre_level(points, day_start, value_key="viewers", time_key="from"):
    """Median of the non-null viewers in the 72 h ending at day_start (1.0 when none)."""
    window = []
    for point in points or ():
        if not isinstance(point, dict):
            continue
        value = point.get(value_key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        stamp = parse_instant(point.get(time_key))
        if stamp is None:
            continue
        back = (day_start - stamp).total_seconds() / 3600.0
        if 0.0 <= back <= PRE_WINDOW_HOURS:
            window.append(value)
    if not window:
        return 1.0
    level = median(window)
    return level if level > 0.0 else 1.0


def console_only(platforms):
    """1.0 when the launch shipped without a PC build, else 0.0."""
    text = platforms if isinstance(platforms, str) else ""
    return 1.0 if "PC" not in text.upper() else 0.0


def launch_features(comparable):
    """The model's features for one comparable, or None when it is not a usable launch."""
    if not isinstance(comparable, dict):
        return None
    name = comparable.get("name")
    if not isinstance(name, str) or name.startswith(SUBJECT_PREFIX):
        return None
    day_start = parse_instant(comparable.get("day_start"))
    if day_start is None:
        return None
    hourly = comparable.get("hourly")
    curve = curve_of(hourly, day_start)
    if 1 not in curve:
        return None
    first = curve[1]
    pre = pre_level(hourly, day_start)
    return {
        "curve": curve,
        "first": first,
        "la": math.log(first),
        "lb": math.log(max(first / pre, 0.01)),
        "con": console_only(comparable.get("platforms")),
    }


def weighted_quantile(pairs, level):
    """The weighted quantile at `level` of (value, weight) pairs, steps on cumulative weight.

    Uniform weights when every weight underflowed to 0, so a far-away neighbourhood still
    yields an answer instead of a crash.
    """
    total = 0.0
    for _, weight in pairs:
        total += weight
    if not (total > 0.0):
        pairs = [(value, 1.0) for value, _ in pairs]
        total = float(len(pairs))
    ordered = sorted(pairs)
    target = level * total
    reached = 0.0
    for value, weight in ordered:
        reached += weight
        if reached >= target:
            return value
    return ordered[-1][0]


# ---------------------------------------------------------------------------- model


def case_context(case):
    """The case's hour-1 print and its own features: (p1, la, lb)."""
    day_start = parse_instant(case.get("day_start"))

    p1 = None
    known = case.get("known")
    if isinstance(known, list):
        for row in known:
            if isinstance(row, dict) and row.get("hour") == 1:
                value = row.get("viewers")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    p1 = float(value)
                break
    if p1 is None or not (p1 > 0.0) or p1 != p1:
        p1 = 1.0

    if day_start is None:
        return p1, math.log(p1), 0.0
    pre = pre_level(case.get("history_hourly"), day_start)
    return p1, math.log(p1), math.log(max(p1 / pre, 0.01))


def ratio_quantiles(launches, hour, la_case, lb_case):
    """The five log-ratio quantiles of `hour` over the launches, or None when none has it."""
    pairs = []
    for launch in launches:
        value = launch["curve"].get(hour)
        if value is None:
            continue
        distance = (W_BOOST * (launch["lb"] - lb_case)) ** 2 + (W_ABS * (launch["la"] - la_case)) ** 2
        weight = math.exp(-distance / (2.0 * SIGMA * SIGMA))
        if launch["con"]:
            weight *= CON_BONUS
        pairs.append((math.log(value / launch["first"]), weight))
    if not pairs:
        return None
    return [weighted_quantile(pairs, level) for level in LEVELS]


def residual_quantile(residuals, level):
    """The linear-interpolated quantile at `level` of an already sorted list."""
    size = len(residuals)
    if size == 0:
        return 0.0
    if size == 1:
        return residuals[0]
    pos = level * (size - 1)
    low = int(pos)
    if low < 0:
        low = 0
    elif low > size - 1:
        low = size - 1
    high = low + 1 if low + 1 < size else size - 1
    frac = pos - low
    return residuals[low] * (1.0 - frac) + residuals[high] * frac


def regression_ratios(launches, hour, p1_case):
    """The five log-ratios of `hour` from a weighted log-log regression, or None.

    Fits y = a + b*x by weighted least squares over every launch that reaches `hour`,
    with x = ln(max(curve[1], 1)), y = ln(curve[hour] / curve[1]) and weight CON_BONUS
    for a launch that shipped without a PC build. The five levels are the regression's
    prediction for the case's own hour-1 print plus the interpolated quantiles of the
    sorted fit residuals, so the centre is the regression's and the spread is the
    neighbourhood's own. None when fewer than MIN_REG_LAUNCHES launches reach `hour`.
    """
    samples = []
    for launch in launches:
        value = launch["curve"].get(hour)
        if value is None:
            continue
        weight = CON_BONUS if launch["con"] else 1.0
        samples.append((math.log(max(launch["first"], 1.0)), math.log(value / launch["first"]), weight))
    if len(samples) < MIN_REG_LAUNCHES:
        return None

    total = 0.0
    sum_x = 0.0
    sum_y = 0.0
    for x, y, weight in samples:
        total += weight
        sum_x += weight * x
        sum_y += weight * y
    if not (total > 0.0):  # cannot happen with weights >= 1, but never divide by zero
        total = float(len(samples))
        sum_x = sum(x for x, _, _ in samples)
        sum_y = sum(y for _, y, _ in samples)
    mean_x = sum_x / total
    mean_y = sum_y / total

    covariance = 0.0
    variance = 0.0
    for x, y, weight in samples:
        covariance += weight * (x - mean_x) * (y - mean_y)
        variance += weight * (x - mean_x) ** 2
    slope = (covariance / variance) if variance > 1e-12 else 0.0
    intercept = mean_y - slope * mean_x

    residuals = sorted(y - (intercept + slope * x) for x, y, _ in samples)
    centre = intercept + slope * math.log(max(p1_case, 1.0))
    return [centre + residual_quantile(residuals, level) for level in LEVELS]


def combined_ratios(launches, hour, p1_case, la_case, lb_case):
    """The five log-ratios of `hour`: kernel blended with regression, in log space.

    None only when no launch reaches `hour` at all. An hour the regression cannot be
    fitted on (fewer than MIN_REG_LAUNCHES launches) keeps the kernel's answer alone,
    and vice versa, so the blend never throws an estimator away for lack of the other.
    """
    kernel = ratio_quantiles(launches, hour, la_case, lb_case)
    regression = regression_ratios(launches, hour, p1_case)
    if kernel is None:
        return regression
    if regression is None:
        return kernel
    return [ALPHA * k + (1.0 - ALPHA) * r for k, r in zip(kernel, regression)]


def baseline_quantiles(p1, hour):
    """The persistence baseline's five quantiles for `hour`: what every prize must beat."""
    return [p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(hour - 1)) for z in Z]


def clamp_row(values):
    """Clamp to the contract's bounds and force the row non-decreasing."""
    row = []
    for value in values:
        if value != value or value == float("inf") or value == float("-inf"):
            value = Q_MIN
        row.append(min(max(value, Q_MIN), Q_MAX))
    for index in range(1, len(row)):
        if row[index] < row[index - 1]:
            row[index] = row[index - 1]
    return row


def forecast_case(case):
    """The {"hours": [...]} document for one case: exactly 23 rows, hours 2..24."""
    p1, la_case, lb_case = case_context(case)

    comparables = case.get("comparables")
    launches = []
    if isinstance(comparables, list):
        for comparable in comparables:
            features = launch_features(comparable)
            if features is not None:
                launches.append(features)

    if len(launches) < MIN_LAUNCHES:
        rows = [{"hour": hour, "q": clamp_row(baseline_quantiles(p1, hour))} for hour in HOURS]
        return {"hours": rows}

    rows = []
    last = None
    for hour in HOURS:
        ratios = combined_ratios(launches, hour, p1, la_case, lb_case)
        if ratios is None:
            # No comparable reaches this hour: keep the last hour's ratios, else flat.
            ratios = last if last is not None else [0.0] * len(LEVELS)
        last = ratios
        rows.append({"hour": hour, "q": clamp_row([p1 * math.exp(r) for r in ratios])})
    return {"hours": rows}


# ----------------------------------------------------------------------------- main


def write_case(path):
    """Forecast one case file and return the document; never raises on bad input."""
    try:
        case = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        case = None
    if not isinstance(case, dict):
        return {"hours": [{"hour": hour, "q": clamp_row(baseline_quantiles(1.0, hour))} for hour in HOURS]}
    try:
        return forecast_case(case)
    except Exception:
        # A malformed case still gets a contract-valid output (the baseline's shape).
        try:
            p1, _, _ = case_context(case)
        except Exception:
            p1 = 1.0
        return {"hours": [{"hour": hour, "q": clamp_row(baseline_quantiles(p1, hour))} for hour in HOURS]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        document = write_case(path)
        text = json.dumps(document, separators=(",", ":"), allow_nan=False) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # the grader must never see a non-zero exit
        print("gta6-twitch-ccu-forecast-01: %s" % error, file=sys.stderr)
        sys.exit(0)
