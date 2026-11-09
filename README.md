# tradelab

*[Français](README.fr.md)*

A trading-strategy test bench: realistic simulation of a margin account, then a **statistical verdict** on a period the strategy has not seen.

The project starts from a real case: a "grid + short + adaptive leverage" trading bot that ran with real money on Kraken. Its logic is ported here as a pure function, replayed over six years of Binance ETH/USDT data (not included in this repository, see [Data](#data)) and submitted to a judge that asks: *does this result differ from chance and from simply holding?*

> **Result: the strategy is not proven profitable in either of the two variants tested.** With the bot's real fees, it loses money over the validation period (−90% with the bot's leverage, −73% without leverage), whereas holding the ETH would have returned +108%. See [the results](#results). That is the tool's job: to put an idea that looks good to the test before any money goes into it.
>
> The tool's console output, error messages and code comments are in French.

## What this project demonstrates

| Skill | Where to see it |
|---|---|
| **Decision logic separated from input/output**: the strategy is a pure function, the same code serves the backtest and could sit behind a broker | `src/tradelab/strategy.py` |
| **Margin-account simulation**: leverage, locked margin, rollover on the borrowed amount, liquidation at the candle's worst price | `src/tradelab/account.py`, `costs.py` |
| **Statistical rigor**: validation window with fresh capital, bootstrap by blocks of months, circular-shift placebo, Bonferroni correction | `src/tradelab/validation.py` |
| **Data engineering**: safe reading of `.npz` (no pickle), resampling, quality report | `src/tradelab/data.py` |
| **Configurable without touching the code**: strategy parameters (`--set`, `--params`), leverage levels to compare (`--leverage`), analyses to run (`--analyses`), honest hypothesis counting (`--tests-counted`) | `src/tradelab/options.py`, `examples/` |
| **Operational robustness**: state written atomically, `.bak` backup, refusal to restart from scratch | `src/tradelab/state.py` |
| **Quality**: 285 tests, including tests written from real defects of the original bot; deliberate mutations: 10 on the original code, 42 of 43 detected on the options code (the 43rd exposed a redundant check, since removed) | `tests/` |

## Getting started

Python 3.10 or newer. No keys, no account, no network access.

```bash
pip install -e ".[dev]"
pytest                                  # 285 tests, ~40 s
tradelab backtest --sensitivity         # ~15 s, on FICTITIOUS data (see below)
tradelab carry                          # funding carry, on FICTITIOUS data
tradelab params                         # what you can change: parameters, leverage levels, analyses
```

By default the commands read `data/synthetic/`, which is **invented** data. Their figures therefore differ from those in the [Results](#results) section, which come from real data. When that is the case, the report prints a warning at the top (in French: "données FICTIVES").

## Choosing what to test

The default run reproduces the original bot (adaptive leverage, and no leverage). Without touching the code you can change the strategy's parameters, choose the leverage levels to compare, and choose the analyses to run. `tradelab params` prints everything, with default values and allowed ranges.

```bash
tradelab params                                                   # parameters, ranges, leverage levels, analyses, file format
tradelab backtest --set spread=0.02 --set levels=8                # change parameters (decimal comma accepted: spread=0,02)
tradelab backtest --params examples/params.example.json           # the same from a JSON file; --set wins over the file
tradelab backtest --leverage 1,2,3                                # one hypothesis per level (adaptive = the bot's x2/x3/x5)
tradelab backtest --analyses sensitivity --set levels=6           # only the cost-sensitivity table
tradelab backtest --tests-counted 20                              # declare 20 tests in total for the Bonferroni correction
```

- **13 adjustable parameters**: grid levels and spacing, rebalance interval, ADX and RSI thresholds, short stop and target, minimum quantities. An unknown name, a value outside its range, a wrong type or an inconsistent pair (`rsi_cover` must stay below `rsi_short`) is refused with a message, **before** any data is read. Leverage is chosen with `--leverage`, not with the parameters.
- **Every hypothesis launched counts for the Bonferroni correction**: three leverage levels means α/3. If you tried other configurations elsewhere, declare the total with `--tests-counted`; it can raise the count, never lower it below what this run launches.
- **Changing parameters changes what the verdict means.** The judge assumes the strategy was frozen before looking at VAL. Adjusting parameters until VAL looks good makes the verdict meaningless, correction or not; the report repeats this whenever parameters are modified. Tune on DEV, or fix the values first.
- **Fixed on purpose**: the decision rules (`strategy.decide`), the 24-month minimum of validation, the monthly block bootstrap and the placebo. The strategy is parametrizable, not pluggable: testing another strategy means writing another `decide()`.
- With no option the report and the JSON are identical, character for character, to those of the previous version (compared against saved output).

## Data

**This repository contains no real market data.** The files in `data/synthetic/` (1.2 MB) are fictitious: a random walk with varying volatility and fat tails (an "ETH" in one-hour candles from January 2020 to October 2026, with no gaps) and an invented daily funding series. `src/tradelab/synthetic.py` produces them with a fixed seed, `python scripts/make_synthetic.py` regenerates them identically, and a test checks that the files in the repository match the generator. They exist to run the tool and the tests without downloading anything; they say nothing about a real market.

The [results](#results) below, on the other hand, were measured by the author on real Binance data: ETH/USDT in one-hour candles and BTC daily funding. That data is not redistributed here: the [Binance Vision terms of use](https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf) place it under a CC BY-NC-SA 4.0 license (non-commercial use, attribution) and forbid hosting the datasets. To redo the measurement:

1. Download the data from Binance Vision, in compliance with those terms.
2. Convert it to UTC CSV: `time,open,high,low,close,volume` for the one-hour candles (`.csv` or `.csv.gz` file), `date,funding_sum` for the daily sum of funding.
3. Run `tradelab backtest --data my_candles.csv` and `tradelab carry --funding my_funding.csv`.

`scripts/build_sample.py` does this conversion from the author's `.npz` archives (format described in `src/tradelab/data.py`); it writes to `data/real/`, a folder ignored by git.

> Market data: Binance, via Binance Vision (CC BY-NC-SA 4.0). The figures in the Results section were computed by the author, with no affiliation to Binance, and are shown for non-commercial illustration.

## Results

*Measured on the real data described in [Data](#data); they cannot be reproduced with `data/synthetic/`. The blocks below are the tool's French output, translated: labels in English, numbers unchanged, decimal points instead of commas.*

Original bot strategy, **parameters unchanged** (nothing is tuned here), ETH/USDT hourly, starting at 50 of capital. Each window starts with a fresh account, so that a ruin occurring earlier does not decide the next window's verdict. DEV (2020-2022) gives context; **only VAL (2023-2026) judges**.

```
▌ Bot as is (adaptive leverage ×2/×3/×5)
  Window   Months   Strategy   Buy & hold   Avg. exposure   Monthly mean   Lower bound   Excess (bound)   Placebo p   Liq.
  DEV          36       -97%         849%             87%         -2.91%       -12.99%          -16.93%       0.918      2
  VAL          46       -90%         108%             83%         -3.69%        -8.95%           -8.30%       0.880      0
  VERDICT: not proven

▌ Same strategy without leverage (×1)
  DEV          36        37%         849%             36%          1.30%        -1.76%           -3.43%       0.705      0
  VAL          46       -73%         108%             51%         -2.38%        -5.74%           -5.02%       1.000      0
  VERDICT: not proven

▌ Sensitivity to costs (full period, no leverage, fresh capital)
  Scenario                       Return   Trades   Fees paid
  no costs                         184%     2108         0.0
  0.16% fees + slippage             94%     2150        64.2
  0.26% fees + slippage              6%     2140        85.9
  0.40% fees + slippage            -77%     2073       111.8
```

**How to read these rows.**

- A "proven" verdict requires three conditions at once on VAL: the net monthly mean has a lower bound > 0; the excess over holding **at the same average exposure** has a lower bound > 0; the placebo test is significant. None is met, in either variant.
- **Leverage makes everything worse.** With the bot's leverage, two liquidations (19 May 2021 and 14 June 2022) followed by wear from fees ruin the simulated account (−97% on DEV).
- **The result depends heavily on fees.** With no costs at all the strategy wins, but far less than holding. At 0.16% fees it is still positive over the full period (but already at −30% on the VAL window alone); at 0.26% (the bot's value) it makes only +6% in 6.8 years while ETH went up 19-fold; at 0.40% it loses 77%. In the VAL window without leverage, orders are tiny (about 6 of notional on average) and their average gain before fees, slippage included, is around 0.02% of notional: fees exceed it as soon as they reach a few tenths of a percent.
- **Placebo p ≥ 0.7 everywhere**: the strategy's exposure profile, randomly shifted in time, would do as well or better. Its entry and exit timing adds nothing measurable.

### Funding carry (the other lead of the original project)

Buy spot and sell a perpetual of the same size: the price cancels out, what remains is the funding paid between buyers and sellers. Measured on BTC, with 2%/year of frictions deducted up front:

```
  Year    Days   Gross/yr   Net/yr   Negative days
  2020     366      17.2%    15.2%             14%
  2021     365      30.6%    28.6%              8%
  2022     365       4.2%     2.2%             20%
  2023     365       7.9%     5.9%              8%
  2024     366      11.9%     9.9%              7%
  2025     365       5.1%     3.1%              8%
  2026     273       2.9%     0.9%             24%

  Last 12 months   gross 3.4%/yr · net 1.4%/yr
  Relative to capital tied up (spot + perpetual margin): 0.7 to 1.0%/yr
```

Funding was positive on most days (76% to 93% depending on the year), which agrees with the "steady" verdict of the original study. But most of it came from 2020-2021: over the last 12 months it yields about 3% gross, **less than 1.5% net, and less than 1% of the capital actually tied up**. Not measured: liquidation of the short leg in a sudden rally, platform risk, the perpetual/spot gap, taxes, access to the product from one's country, and the fact that Kraken's funding is not Binance's.

## Defects found in the original `bot.py`

Noted while porting the logic (the original code is not included in this repository):

1. **Displayed PnL inflated by leverage and understated in fees** (lines 550, 622, 721, 768). Fees are deducted as `× (1 − 2·FEES)`, i.e. a fraction of the *profit* rather than of the notional, then the result is multiplied by the leverage although the quantity already accounts for it. Consequence: a losing trade can display as a winner (`tests/test_costs.py`). This PnL feeds `eur_balance` (lines 726 and 774). The XRP and AVAX bots carry a comment "bug fixed: extra *LEVERAGE"; the ETH and SOL bots keep the pattern.
2. **Stale inventory within a cycle.** `eth_total` is read once, then several sell tiers can trigger in the same loop without it being decremented: the bot can sell more than it holds. Fixed in the port (`tests/test_strategy.py`, regression test).
3. **`connect()` returns 2 values on error where 4 are expected** (lines 90 and 403): a connection error crashes startup instead of stopping cleanly. The SOL, XRP and AVAX bots do return 4 values.
4. **Long and short positions assumed independent.** The original repair scripts (`close_leftover_xrp.py`, `fix_after_incident.py`) tell of a "cover" that opened a long position instead of closing one, followed by a buy-back loop. This simulator reproduces what the bot *believes*; a real margin account is more dangerous.
5. **Short stop checked only at the candle close** (`high ≥ stop`), then a market order: no stop is placed on the exchange, and the exit happens at the next price, not at the stop price.
6. **Leverage growing with capital** (×2 under 100, ×3 up to 500, ×5 above): risk increases after gains. On a position fully at ×5, a drop of about 12% in the asset is enough to bring the margin level down to 40% (the liquidation threshold assumed here, not verified).
7. **Dead code**: a fragment of `build_grid` copied after a `return` (lines 237-248).

## Acknowledged limitations

- **Unmeasured costs.** Fees 0.26% (the bot's value, close to the 0.271% observed on a real sale), slippage 0.03%, rollover 0.02% per 4 h: the latter is **an assumption** (Kraken states a variable rate set at execution, with no interval on the page consulted). The liquidation threshold at 40% margin level is also an assumption; only the 2% liquidation fee comes from the Kraken page. The sensitivity table shows the conclusion holds across the plausible range of fees, but not the exact size of the losses.
- **A different market from the real bot.** Binance ETH/USDT data, whereas the bot trades ETH/EUR on Kraken.
- **The bot's parameters are not blank.** They were chosen by their author before this project, perhaps while seeing these same years. The VAL window is therefore not out-of-sample proof in the strict sense: it removes DEV's optimism, not every selection bias.
- **Parameters are not learned.** `--set` changes them but nothing here searches for good values: a search would have to be counted (`--tests-counted`) and judged on a window it has not seen.
- **One asset, one period.** This bench does not say every grid strategy fails, only that this one is not proven on this data.
- **Indicators computed over the full history.** The real bot computes them over 100 candles: a mean gap of 0.4 RSI points but up to 5 points at worst on real ETH (`tests/test_indicators.py`).
- **Execution at the next open, at the mid price ± slippage.** No order book, no partial fills.
- **31 missing hours** in the real data (0.05%), left as is and counted in the quality report. The fictitious data has no gaps.
- **Tested on Python 3.13 only**, and CI not run: the workflow (Python 3.10 and 3.12) is written but has not run.

## Structure

| File | Role |
|---|---|
| `src/tradelab/strategy.py` | Strategy as a pure function: ADX/MA50 regime, grid, short, stops, adaptive leverage |
| `src/tradelab/account.py` | Simulation account: margin, rollover, liquidation, long and short positions |
| `src/tradelab/costs.py` | Cost model and measurement of the original calculation's errors (`legacy_reported_pnl`) |
| `src/tradelab/backtest.py` | Engine: decision at the close, execution at the next open |
| `src/tradelab/validation.py` | Judge: bootstrap by month, placebo, Bonferroni, verdict |
| `src/tradelab/carry.py` | Funding carry: yearly table, windows, return on capital |
| `src/tradelab/state.py` | Atomic state storage with backup |
| `src/tradelab/data.py` | `.npz` → candles, resampling, quality |
| `src/tradelab/synthetic.py` | Reproducible fictitious data (random walk, invented funding) |
| `src/tradelab/options.py` | What the user can choose: strategy parameters (validated), leverage levels, analyses |
| `src/tradelab/cli.py` | `tradelab backtest`, `tradelab carry` and `tradelab params` |
| `examples/params.example.json` | Example of a `--params` file |
| `data/synthetic/` | The fictitious files read by default |
| `scripts/make_synthetic.py` | Regenerates `data/synthetic/` identically |
| `scripts/build_sample.py` | Converts the author's `.npz` archives into real CSVs in `data/real/` (ignored by git) |

## Tests

```bash
pytest
```

- **Accounting invariant**: with no costs, equity is exactly the product of (1 + exposure × candle return).
- **No look-ahead**: changing future candles does not change past equity; every order executes at the open of the next candle.
- **Indicators** compared with hand-written Wilder implementations on eight series, and checked to be causal.
- **Judge** checked with a positive control (a clairvoyant strategy: proven) and a negative control (noise: not proven).
- **Fictitious data**: deterministic, no gaps, annualized volatility close to the requested one, fat tails, and repository files identical to the generator's output.
- **Options**: every parameter's range edges (accepted) and the values just outside (refused), malformed `--set` items, types, `--params` precedence, leverage lists, analyses, Bonferroni counting, and the guarantee that an invalid option stops the run before any data is read.
- **Mutation**: deliberately breaking ten behaviors (inventory not decremented, execution at the close, p-value without correction, ADX badly initialized, zero fees, etc.) makes at least one test fail every time. A test that was too weak on ADX initialization was strengthened as a result. The same exercise on the options code (43 changes) found one redundant check, removed, and no weak test left.

## Design choices

**No live trading, on purpose.** The repository contains no order-sending code, no keys, no network calls. The measurements above show the strategy is not profitable; connecting it to money would make no sense. For another use, the orders produced by `decide()` (`open_long`, `close_short`, …) translate directly into broker calls.

**Conservative verdict.** Three simultaneous conditions, Bonferroni correction for the number of variants, at least 24 months of validation. A test that cannot fail proves nothing: that is why the judge is itself tested on cases whose answer is known.

## Disclaimer

An analysis tool, not financial advice. The results describe the past on specific data and cost assumptions; they guarantee nothing for the future.

## License

© 2026 Samuel Mateo. All rights reserved.

The code is published to be read and evaluated. Without the author's written permission, it may not be copied, modified or redistributed. The real data mentioned above remains subject to the Binance Vision terms.
