# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/mantraabi/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/mantraabi/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's Twitch category viewership for every hour 2..24 of release day, as five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95) per hour, from nothing but the case file: the game's
own hour-1 print plus the hourly curves of past console launches. The output is pure JSON,
one `.out` per `.in`, and the whole thing is standard-library Python with no network and no
randomness — same input, same bytes, every run.

## Model

Everything is forecast as a ratio to the launch's own hour-1 print and multiplied back by the
case's hour-1 print at the end, so a curve learned on a 27k-viewer launch transfers unchanged
to a multi-million-viewer one. Two estimators are fitted per hour and averaged in log space.

**1. kNN kernel over past console launches.** Each comparable launch becomes a point in
`(ln(hour-1 viewers), ln(hour-1 over its own pre-launch level))`, with console-only launches
weighted up (`CON_BONUS`) since GTA VI ships on PS5 and Xbox Series only. For each hour, the
neighbour weight is a Gaussian in that distance (`SIGMA`, `W_ABS`, `W_BOOST`), and the hour's
five quantiles are weighted quantiles of `ln(curve_i[h] / curve_i[1])` across neighbours.

**2. Weighted log-log regression.** Per hour, `y = ln(curve_i[h] / v1_i)` is regressed on
`x = ln(max(v1_i, 1))` by weighted least squares (console-only launches get `CON_BONUS`
weight), giving a slope `b` and intercept `a`. The five levels are `a + b*ln(max(p1, 1))` plus
the linear-interpolated quantiles of the fitted residuals — the regression supplies the hour's
centre, the residuals its spread.

**Blend.** Per hour per level: `q = ALPHA * q_knn_log + (1 - ALPHA) * q_reg_log` with
`ALPHA = 0.7`, then `v1 * exp(q)`, clamped to `[1e-3, 1e9]` and forced non-decreasing. Hours
with fewer than six usable launches for the regression use the kernel alone. A case that
cannot be modelled at all falls back to the persistence baseline
`q_tau = p1 * exp(z_tau * 0.15 * sqrt(h - 1))`, so the program always emits a contract-valid
file.

## Run

```bash
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Every `*.in` in the input directory produces a matching `*.out` in the output directory.

Tools (not part of the entry point):

```bash
python3 tools/loo_check.py     # leave-one-out score: blended vs kernel alone vs baseline
python3 tools/selfcheck.py     # validates every output against the contract
```

## Leave-one-out results

Eight console-only launches held out of the neighbour pool and forecast from the rest, scored
with the competition's own pinball loss on the log scale, averaged over 23 hours x 5 levels:

| model | mean pinball loss |
| --- | --- |
| persistence baseline | 0.3226 |
| kNN kernel alone | 0.2078 |
| **kNN + log-log regression, blended 70/30** | **0.1774** |

The blend is 45% better than the baseline and 15% better than the kernel on its own.

## License

MIT — see [LICENSE](LICENSE).
