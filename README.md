# CTA-lite Study

This project tests a single-market version of CTA trend following. Traditional
CTA programs diversify across many futures markets. Here the universe is limited
to Nasdaq-100 ETFs, so the model is intentionally framed as CTA-lite:

- calculate trend on `QQQ`
- trade either `QQQ` or `QLD`
- stay long or in cash
- scale exposure down when realized volatility is high
- include simple trading costs

## Why TQQQ Is Excluded

`TQQQ` uses 3x daily leverage, so path dependency and volatility drag can dominate
the experiment. `QLD` still has leverage risk, but 2x exposure is a more stable
second step after validating the signal on `QQQ`.

## Algorithm

The model uses only information known at the prior close.

1. Compute trend votes on the signal asset.

   - 21 trading day return
   - 63 trading day return
   - 126 trading day return
   - 252 trading day return
   - price above 200 day moving average

2. Convert votes to a trend score.

   ```text
   trend_score = positive_votes / 5
   ```

3. Map the score to a base position.

   The default profile is now `aggressive`:

   ```text
   score >= 0.80 -> 125%
   score >= 0.60 -> 100%
   score >= 0.40 -> 75%
   score >= 0.20 -> 35%
   otherwise     -> 0%
   ```

   The old defensive behavior is still available with `--profile balanced`:

   ```text
   score >= 0.80 -> 100%
   score >= 0.60 -> 75%
   score >= 0.40 -> 50%
   otherwise     -> 0%
   ```

4. Apply a crash filter.

   In the aggressive profile, if the signal asset has fallen 10% or more over 10
   trading days, exposure is capped at 60%. In the balanced profile, the old
   8% drop / 25% cap is used.

5. Apply volatility targeting on the traded asset.

   ```text
   final_position = base_position * min(1, target_vol / realized_vol_20d)
   ```

6. Trade the next day using the position decided at the prior close.

## Data

Place CSV files here:

```text
data/QQQ.csv
data/QLD.csv
```

Each CSV should contain `Date` and either `Adj Close` or `Close`.

You can also try automatic download if `yfinance` is installed:

```bash
python3 cta_lite.py --download --signal QQQ --trade QQQ
python3 cta_lite.py --download --signal QQQ --trade QLD --profile aggressive
```

If network access or `yfinance` is unavailable, manually export historical daily
prices from your data source and save them under `data/`.

## Suggested Runs

Validate the signal on unlevered exposure first:

```bash
python3 cta_lite.py --signal QQQ --trade QQQ --profile balanced
```

Then test the same signal on 2x exposure:

```bash
python3 cta_lite.py --signal QQQ --trade QLD --profile aggressive
```

Run a shorter performance window while still using older data for signal warmup:

```bash
python3 cta_lite.py --signal QQQ --trade QLD --profile aggressive --from-date 2026-01-01 --output results/qld_2026.csv
```

For a custom risk level, override the profile defaults:

```bash
python3 cta_lite.py --signal QQQ --trade QLD --profile aggressive --target-vol 0.40 --max-position 1.50
```

The script prints CAGR, volatility, Sharpe, max drawdown, average exposure, and
turnover. It also writes a daily equity curve to:

```text
results/equity_curve.csv
```

## Weekly DCA Mode

Use DCA mode to compare:

- buying the same dollar amount of `QLD` every Monday and holding it
- adding the same Monday contribution, then letting the trend model decide how
  much of the account should be in `QLD`

```bash
python3 cta_lite.py --signal QQQ --trade QLD --mode dca --weekly-amount 100 --profile aggressive --from-date 2026-01-01 --output results/qld_2026_dca.csv
```

For DCA mode, the default strategy exposure is capped at 100% unless you
explicitly pass `--max-position`. This keeps the comparison to cash + QLD, with
no extra margin on top of the leveraged ETF.

More aggressive DCA + auto test:

```bash
python3 cta_lite.py --signal QQQ --trade QLD --mode dca --weekly-amount 100 --profile aggressive --target-vol 0.45 --max-position 1.25 --from-date 2026-01-01 --output results/qld_2026_dca_125.csv
```

DCA mode prints total contributed, final value, profit, ROI, max drawdown,
contribution count, and average auto exposure.

## Slack Signal Automation

Use `cta_signal_report.py` to create a manual trading signal and send it to the
Slack `nasdaq` channel through the existing brefingbot token. The signal asset
is still `QQQ`, but the current manual-trading target is `TQQQ`.

The script defaults to:

- channel key: `nasdaq`
- channel id: `C0B3RR8MHCN`
- bot env file: `../Market-Briefing-Bot/config/slack_bot.env`
- signal: `QQQ`
- trade: `TQQQ`
- profile: `aggressive`

Preview without sending:

```bash
python3 cta_signal_report.py --timing pre-open --dry-run --send-slack
```

Send the pre-open signal:

```bash
python3 cta_signal_report.py --timing pre-open --refresh-data --auto-weekly-contribution --send-slack
```

Send the after-close signal:

```bash
python3 cta_signal_report.py --timing after-close --refresh-data --send-slack
```

The convenience scripts are:

```bash
scripts/run_nasdaq_signal_preopen.sh
scripts/run_nasdaq_signal_after_close.sh
```

To calculate exact order amounts, copy the sample portfolio file and update it
locally:

```bash
cp config/nasdaq_portfolio.example.json config/nasdaq_portfolio.json
```

```json
{
  "cash": 1000,
  "trade_shares": 10,
  "weekly_amount": 100
}
```

If `config/nasdaq_portfolio.json` is missing, the Slack message still sends the
target TQQQ exposure, but it will not estimate buy/sell dollars.

For Slack sending, put the brefingbot token in:

```text
../Market-Briefing-Bot/config/slack_bot.env
```

```text
SLACK_BOT_TOKEN=xoxb-...
```

## Interpretation

This is not a prediction engine. It is a risk-managed participation engine:

- hold exposure when multiple trend horizons agree
- reduce exposure when the market becomes volatile
- avoid full exposure after sharp short-term breaks
- compare against buy-and-hold to see whether drawdown reduction is worth the
  missed upside

The key questions are:

- Does the strategy reduce max drawdown versus buy-and-hold?
- Does lower drawdown compensate for lower average exposure?
- Does `QLD` improve CAGR without making drawdown unacceptable?
- Is turnover low enough after costs?
- In DCA mode, does automatic trading reduce drawdown enough to justify missed
  upside versus simple weekly accumulation?
