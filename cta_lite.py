#!/usr/bin/env python3
"""
CTA-lite trend following backtest for QQQ / QLD.

The model calculates trend signals on a signal asset, usually QQQ, and trades a
selected asset, usually QQQ or QLD. It is long/cash only because leveraged ETFs
are already aggressive instruments and the user's current universe excludes
inverse ETFs.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


TRADING_DAYS = 252


@dataclass(frozen=True)
class PriceSeries:
    ticker: str
    dates: list[date]
    closes: list[float]


@dataclass(frozen=True)
class BacktestResult:
    dates: list[date]
    trade_closes: list[float]
    strategy_returns: list[float]
    buy_hold_returns: list[float]
    positions: list[float]
    scores: list[float]
    equity: list[float]
    buy_hold_equity: list[float]


@dataclass(frozen=True)
class DcaResult:
    dates: list[date]
    trade_closes: list[float]
    contributions: list[float]
    cumulative_contributions: list[float]
    dca_only_value: list[float]
    dca_strategy_value: list[float]
    dca_strategy_position: list[float]
    scores: list[float]


@dataclass(frozen=True)
class StrategyProfile:
    name: str
    default_target_vol: float
    default_max_position: float
    score_bands: tuple[tuple[float, float], ...]
    crash_threshold: float
    crash_cap: float


PROFILES = {
    "balanced": StrategyProfile(
        name="balanced",
        default_target_vol=0.18,
        default_max_position=1.0,
        score_bands=((0.80, 1.00), (0.60, 0.75), (0.40, 0.50)),
        crash_threshold=-0.08,
        crash_cap=0.25,
    ),
    "aggressive": StrategyProfile(
        name="aggressive",
        default_target_vol=0.35,
        default_max_position=1.25,
        score_bands=((0.80, 1.25), (0.60, 1.00), (0.40, 0.75), (0.20, 0.35)),
        crash_threshold=-0.10,
        crash_cap=0.60,
    ),
}


def parse_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def read_price_csv(path: Path, ticker: str) -> PriceSeries:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV for {ticker}: {path}")

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"{path} is empty")

    fieldnames = set(rows[0].keys())
    close_field = "Adj Close" if "Adj Close" in fieldnames else "Close"
    if "Date" not in fieldnames or close_field not in fieldnames:
        raise ValueError(f"{path} must contain Date and Close or Adj Close columns")

    parsed: list[tuple[date, float]] = []
    for row in rows:
        raw_close = row.get(close_field, "")
        if not raw_close or raw_close.lower() == "null":
            continue
        parsed.append((parse_date(row["Date"]), float(raw_close)))

    parsed.sort(key=lambda item: item[0])
    dates = [item[0] for item in parsed]
    closes = [item[1] for item in parsed]
    return PriceSeries(ticker=ticker, dates=dates, closes=closes)


def maybe_download(ticker: str, data_dir: Path, start: str, force: bool = False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"{ticker}.csv"
    if output_path.exists() and not force:
        return output_path

    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Put CSV files in data/ or install yfinance."
        ) from exc

    try:
        frame = yf.download(
            ticker,
            start=start,
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )
    except TypeError:
        frame = yf.download(ticker, start=start, auto_adjust=False, progress=False)

    if frame.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")

    if getattr(frame.columns, "nlevels", 1) > 1:
        frame.columns = frame.columns.get_level_values(0)

    frame.to_csv(output_path)
    return output_path


def align_series(signal: PriceSeries, trade: PriceSeries) -> tuple[list[date], list[float], list[float]]:
    signal_map = dict(zip(signal.dates, signal.closes, strict=True))
    trade_map = dict(zip(trade.dates, trade.closes, strict=True))
    dates = sorted(set(signal_map) & set(trade_map))
    if len(dates) < 260:
        raise ValueError("Need at least about one trading year of overlapping data")
    return dates, [signal_map[d] for d in dates], [trade_map[d] for d in dates]


def pct_change(values: list[float], index: int, lookback: int) -> float | None:
    if index < lookback or values[index - lookback] <= 0:
        return None
    return values[index] / values[index - lookback] - 1.0


def moving_average(values: list[float], index: int, lookback: int) -> float | None:
    if index + 1 < lookback:
        return None
    window = values[index - lookback + 1 : index + 1]
    return sum(window) / lookback


def realized_vol(returns: list[float], index: int, lookback: int) -> float | None:
    if index < lookback:
        return None
    window = returns[index - lookback + 1 : index + 1]
    if len(window) < 2:
        return None
    return statistics.stdev(window) * math.sqrt(TRADING_DAYS)


def max_drawdown(equity: Iterable[float]) -> float:
    peak = -math.inf
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def annualized_return(equity: list[float], periods: int) -> float:
    if not equity or equity[-1] <= 0 or periods <= 0:
        return float("nan")
    return equity[-1] ** (TRADING_DAYS / periods) - 1.0


def annualized_vol(returns: list[float]) -> float:
    if len(returns) < 2:
        return float("nan")
    return statistics.stdev(returns) * math.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: list[float]) -> float:
    vol = annualized_vol(returns)
    if not vol or math.isnan(vol):
        return float("nan")
    return statistics.mean(returns) * TRADING_DAYS / vol


def trend_score(signal_closes: list[float], index: int) -> float:
    """Return a 0..1 long-only trend score known at the close of index."""
    trend_votes: list[float] = []
    for lookback in (21, 63, 126, 252):
        change = pct_change(signal_closes, index, lookback)
        if change is None:
            return 0.0
        trend_votes.append(1.0 if change > 0 else 0.0)

    sma_200 = moving_average(signal_closes, index, 200)
    if sma_200 is None:
        return 0.0
    trend_votes.append(1.0 if signal_closes[index] > sma_200 else 0.0)

    return sum(trend_votes) / len(trend_votes)


def target_position(
    score: float,
    signal_closes: list[float],
    trade_returns: list[float],
    index: int,
    target_vol: float,
    max_position: float,
    profile: StrategyProfile,
) -> float:
    """Map trend score and current volatility to next-day exposure."""
    base = 0.0
    for threshold, position in profile.score_bands:
        if score >= threshold:
            base = position
            break

    crash_10d = pct_change(signal_closes, index, 10)
    if crash_10d is not None and crash_10d <= profile.crash_threshold:
        base = min(base, profile.crash_cap)

    vol = realized_vol(trade_returns, index, 20)
    if vol is None or vol <= 0:
        return 0.0

    vol_scalar = min(1.0, target_vol / vol)
    return max(0.0, min(max_position, base * vol_scalar))


def backtest(
    signal: PriceSeries,
    trade: PriceSeries,
    target_vol: float,
    max_position: float,
    cost_bps: float,
    profile: StrategyProfile,
) -> BacktestResult:
    dates, signal_closes, trade_closes = align_series(signal, trade)
    trade_returns = [0.0]
    for i in range(1, len(trade_closes)):
        trade_returns.append(trade_closes[i] / trade_closes[i - 1] - 1.0)

    positions = [0.0] * len(dates)
    scores = [0.0] * len(dates)
    strategy_returns = [0.0] * len(dates)
    equity = [1.0] * len(dates)
    buy_hold_equity = [1.0] * len(dates)

    cost_rate = cost_bps / 10_000.0
    previous_position = 0.0

    for i in range(1, len(dates)):
        # Today's return is earned from yesterday's position. This prevents
        # look-ahead bias because the signal at i-1 can only affect day i.
        turnover = abs(positions[i - 1] - previous_position)
        strategy_returns[i] = positions[i - 1] * trade_returns[i] - turnover * cost_rate
        equity[i] = equity[i - 1] * (1.0 + strategy_returns[i])
        buy_hold_equity[i] = buy_hold_equity[i - 1] * (1.0 + trade_returns[i])

        previous_position = positions[i - 1]
        scores[i] = trend_score(signal_closes, i)
        positions[i] = target_position(
            scores[i],
            signal_closes,
            trade_returns,
            i,
            target_vol=target_vol,
            max_position=max_position,
            profile=profile,
        )

    return BacktestResult(
        dates=dates,
        trade_closes=trade_closes,
        strategy_returns=strategy_returns,
        buy_hold_returns=trade_returns,
        positions=positions,
        scores=scores,
        equity=equity,
        buy_hold_equity=buy_hold_equity,
    )


def trim_result(result: BacktestResult, from_date: date | None, to_date: date | None) -> BacktestResult:
    if from_date is None and to_date is None:
        return result

    indexes = [
        i
        for i, current_date in enumerate(result.dates)
        if (from_date is None or current_date >= from_date)
        and (to_date is None or current_date <= to_date)
    ]
    if len(indexes) < 2:
        raise ValueError("Selected backtest period has fewer than two trading days")

    dates = [result.dates[i] for i in indexes]
    trade_closes = [result.trade_closes[i] for i in indexes]
    strategy_returns = [result.strategy_returns[i] for i in indexes]
    buy_hold_returns = [result.buy_hold_returns[i] for i in indexes]
    positions = [result.positions[i] for i in indexes]
    scores = [result.scores[i] for i in indexes]

    # Rebase the selected period to 1.0 and do not count the overnight return
    # from before the selected start date.
    strategy_returns[0] = 0.0
    buy_hold_returns[0] = 0.0
    equity = [1.0] * len(dates)
    buy_hold_equity = [1.0] * len(dates)
    for i in range(1, len(dates)):
        equity[i] = equity[i - 1] * (1.0 + strategy_returns[i])
        buy_hold_equity[i] = buy_hold_equity[i - 1] * (1.0 + buy_hold_returns[i])

    return BacktestResult(
        dates=dates,
        trade_closes=trade_closes,
        strategy_returns=strategy_returns,
        buy_hold_returns=buy_hold_returns,
        positions=positions,
        scores=scores,
        equity=equity,
        buy_hold_equity=buy_hold_equity,
    )


def simulate_weekly_dca(result: BacktestResult, weekly_amount: float, cost_bps: float) -> DcaResult:
    if weekly_amount <= 0:
        raise ValueError("--weekly-amount must be positive")

    cost_rate = cost_bps / 10_000.0
    dca_shares = 0.0
    strategy_shares = 0.0
    strategy_cash = 0.0
    cumulative_contribution = 0.0

    contributions: list[float] = []
    cumulative_contributions: list[float] = []
    dca_only_value: list[float] = []
    dca_strategy_value: list[float] = []

    for i, current_date in enumerate(result.dates):
        price = result.trade_closes[i]
        contribution = weekly_amount if current_date.weekday() == 0 else 0.0

        if contribution:
            cumulative_contribution += contribution
            dca_shares += contribution / (1.0 + cost_rate) / price
            strategy_cash += contribution

        strategy_value = strategy_cash + strategy_shares * price
        target_position_value = strategy_value * result.positions[i]
        current_position_value = strategy_shares * price
        delta_value = target_position_value - current_position_value

        if abs(delta_value) > 1e-9:
            trade_cost = abs(delta_value) * cost_rate
            if delta_value > 0 and result.positions[i] <= 1.0:
                max_buy_value = max(0.0, strategy_cash / (1.0 + cost_rate))
                if delta_value > max_buy_value:
                    delta_value = max_buy_value
                    trade_cost = delta_value * cost_rate

            strategy_shares += delta_value / price
            if delta_value >= 0:
                strategy_cash -= delta_value + trade_cost
            else:
                strategy_cash += -delta_value - trade_cost

        contributions.append(contribution)
        cumulative_contributions.append(cumulative_contribution)
        dca_only_value.append(dca_shares * price)
        dca_strategy_value.append(strategy_cash + strategy_shares * price)

    return DcaResult(
        dates=result.dates,
        trade_closes=result.trade_closes,
        contributions=contributions,
        cumulative_contributions=cumulative_contributions,
        dca_only_value=dca_only_value,
        dca_strategy_value=dca_strategy_value,
        dca_strategy_position=result.positions,
        scores=result.scores,
    )


def summarize_dca(result: DcaResult) -> dict[str, float]:
    total_contributed = result.cumulative_contributions[-1]
    if total_contributed <= 0:
        raise ValueError("No Monday contributions found in selected period")

    dca_final = result.dca_only_value[-1]
    strategy_final = result.dca_strategy_value[-1]
    return {
        "total_contributed": total_contributed,
        "dca_final": dca_final,
        "strategy_final": strategy_final,
        "dca_profit": dca_final - total_contributed,
        "strategy_profit": strategy_final - total_contributed,
        "dca_roi": dca_final / total_contributed - 1.0,
        "strategy_roi": strategy_final / total_contributed - 1.0,
        "dca_mdd": max_drawdown(result.dca_only_value),
        "strategy_mdd": max_drawdown(result.dca_strategy_value),
        "avg_strategy_exposure": sum(result.dca_strategy_position) / len(result.dca_strategy_position),
        "contribution_count": sum(1 for value in result.contributions if value > 0),
    }


def summarize(result: BacktestResult) -> dict[str, float]:
    active_returns = result.strategy_returns[1:]
    buy_hold_returns = result.buy_hold_returns[1:]
    periods = len(active_returns)
    exposure = sum(result.positions) / len(result.positions)
    turnover = sum(abs(result.positions[i] - result.positions[i - 1]) for i in range(1, len(result.positions)))

    return {
        "strategy_final": result.equity[-1],
        "buy_hold_final": result.buy_hold_equity[-1],
        "strategy_cagr": annualized_return(result.equity, periods),
        "buy_hold_cagr": annualized_return(result.buy_hold_equity, periods),
        "strategy_vol": annualized_vol(active_returns),
        "buy_hold_vol": annualized_vol(buy_hold_returns),
        "strategy_sharpe": sharpe_ratio(active_returns),
        "buy_hold_sharpe": sharpe_ratio(buy_hold_returns),
        "strategy_mdd": max_drawdown(result.equity),
        "buy_hold_mdd": max_drawdown(result.buy_hold_equity),
        "avg_exposure": exposure,
        "annual_turnover": turnover / periods * TRADING_DAYS,
    }


def write_equity_curve(path: Path, result: BacktestResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Date",
                "Position",
                "TrendScore",
                "StrategyReturn",
                "BuyHoldReturn",
                "StrategyEquity",
                "BuyHoldEquity",
            ]
        )
        for i, current_date in enumerate(result.dates):
            writer.writerow(
                [
                    current_date.isoformat(),
                    f"{result.positions[i]:.6f}",
                    f"{result.scores[i]:.6f}",
                    f"{result.strategy_returns[i]:.8f}",
                    f"{result.buy_hold_returns[i]:.8f}",
                    f"{result.equity[i]:.6f}",
                    f"{result.buy_hold_equity[i]:.6f}",
                ]
            )


def write_dca_curve(path: Path, result: DcaResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Date",
                "Close",
                "Contribution",
                "CumulativeContribution",
                "DcaOnlyValue",
                "DcaStrategyValue",
                "StrategyPosition",
                "TrendScore",
            ]
        )
        for i, current_date in enumerate(result.dates):
            writer.writerow(
                [
                    current_date.isoformat(),
                    f"{result.trade_closes[i]:.6f}",
                    f"{result.contributions[i]:.2f}",
                    f"{result.cumulative_contributions[i]:.2f}",
                    f"{result.dca_only_value[i]:.2f}",
                    f"{result.dca_strategy_value[i]:.2f}",
                    f"{result.dca_strategy_position[i]:.6f}",
                    f"{result.scores[i]:.6f}",
                ]
            )


def print_summary(
    summary: dict[str, float],
    result: BacktestResult,
    signal_ticker: str,
    trade_ticker: str,
    profile_name: str,
) -> None:
    print(f"Signal asset: {signal_ticker}")
    print(f"Trade asset : {trade_ticker}")
    print(f"Profile     : {profile_name}")
    print(f"Period      : {result.dates[0]} to {result.dates[-1]}")
    print()
    print("Metric                 Strategy     Buy & Hold")
    print("-" * 47)
    print(f"Final equity           {summary['strategy_final']:>8.2f}x     {summary['buy_hold_final']:>8.2f}x")
    print(f"CAGR                   {summary['strategy_cagr']:>8.2%}     {summary['buy_hold_cagr']:>8.2%}")
    print(f"Annual vol             {summary['strategy_vol']:>8.2%}     {summary['buy_hold_vol']:>8.2%}")
    print(f"Sharpe                 {summary['strategy_sharpe']:>8.2f}     {summary['buy_hold_sharpe']:>8.2f}")
    print(f"Max drawdown           {summary['strategy_mdd']:>8.2%}     {summary['buy_hold_mdd']:>8.2%}")
    print()
    print(f"Average exposure       {summary['avg_exposure']:.2%}")
    print(f"Annual turnover        {summary['annual_turnover']:.2f}x")


def print_dca_summary(
    summary: dict[str, float],
    result: DcaResult,
    signal_ticker: str,
    trade_ticker: str,
    profile_name: str,
    weekly_amount: float,
) -> None:
    print(f"Signal asset : {signal_ticker}")
    print(f"Trade asset  : {trade_ticker}")
    print(f"Profile      : {profile_name}")
    print(f"Mode         : weekly DCA every Monday")
    print(f"Weekly amount: {weekly_amount:,.2f}")
    print(f"Period       : {result.dates[0]} to {result.dates[-1]}")
    print()
    print("Metric                    DCA Only     DCA + Auto")
    print("-" * 54)
    print(f"Total contributed       {summary['total_contributed']:>10,.2f}   {summary['total_contributed']:>10,.2f}")
    print(f"Final value             {summary['dca_final']:>10,.2f}   {summary['strategy_final']:>10,.2f}")
    print(f"Profit                  {summary['dca_profit']:>10,.2f}   {summary['strategy_profit']:>10,.2f}")
    print(f"ROI                     {summary['dca_roi']:>10.2%}   {summary['strategy_roi']:>10.2%}")
    print(f"Max drawdown            {summary['dca_mdd']:>10.2%}   {summary['strategy_mdd']:>10.2%}")
    print()
    print(f"Contribution count      {summary['contribution_count']:.0f}")
    print(f"Avg auto exposure       {summary['avg_strategy_exposure']:.2%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CTA-lite trend following backtest for QQQ / QLD")
    parser.add_argument("--signal", default="QQQ", help="Ticker used to calculate trend signals")
    parser.add_argument("--trade", default="QQQ", help="Ticker to trade: QQQ or QLD")
    parser.add_argument("--data-dir", default="data", help="Directory containing ticker CSV files")
    parser.add_argument("--start", default="2010-01-01", help="Download start date when --download is used")
    parser.add_argument("--from-date", help="First date included in performance stats, YYYY-MM-DD")
    parser.add_argument("--to-date", help="Last date included in performance stats, YYYY-MM-DD")
    parser.add_argument("--download", action="store_true", help="Download missing CSVs with yfinance")
    parser.add_argument("--refresh-data", action="store_true", help="Force-refresh CSVs with yfinance before running")
    parser.add_argument(
        "--mode",
        choices=("lump-sum", "dca"),
        default="lump-sum",
        help="lump-sum compares one initial investment; dca compares weekly Monday contributions.",
    )
    parser.add_argument("--weekly-amount", type=float, default=100.0, help="Weekly Monday contribution for --mode dca")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="aggressive",
        help="Risk profile. aggressive is the new default; balanced keeps the original defensive behavior.",
    )
    parser.add_argument("--target-vol", type=float, help="Annualized volatility target")
    parser.add_argument("--max-position", type=float, help="Maximum portfolio weight in trade asset")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="One-way trading cost in basis points")
    parser.add_argument("--output", default="results/equity_curve.csv", help="Output CSV path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)

    try:
        profile = PROFILES[args.profile]
        target_vol = args.target_vol if args.target_vol is not None else profile.default_target_vol
        if args.max_position is not None:
            max_position = args.max_position
        elif args.mode == "dca":
            max_position = min(profile.default_max_position, 1.0)
        else:
            max_position = profile.default_max_position

        if args.download or args.refresh_data:
            signal_path = maybe_download(args.signal, data_dir, args.start, force=args.refresh_data)
            trade_path = maybe_download(args.trade, data_dir, args.start, force=args.refresh_data)
        else:
            signal_path = data_dir / f"{args.signal}.csv"
            trade_path = data_dir / f"{args.trade}.csv"

        signal = read_price_csv(signal_path, args.signal)
        trade = read_price_csv(trade_path, args.trade)
        result = backtest(
            signal,
            trade,
            target_vol=target_vol,
            max_position=max_position,
            cost_bps=args.cost_bps,
            profile=profile,
        )
        result = trim_result(
            result,
            parse_date(args.from_date) if args.from_date else None,
            parse_date(args.to_date) if args.to_date else None,
        )
        if args.mode == "dca":
            dca_result = simulate_weekly_dca(result, args.weekly_amount, args.cost_bps)
            dca_summary = summarize_dca(dca_result)
            write_dca_curve(Path(args.output), dca_result)
            print_dca_summary(dca_summary, dca_result, args.signal, args.trade, profile.name, args.weekly_amount)
        else:
            summary = summarize(result)
            write_equity_curve(Path(args.output), result)
            print_summary(summary, result, args.signal, args.trade, profile.name)
        print()
        print(f"Saved equity curve to {args.output}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(file=sys.stderr)
        print("Expected CSV paths:", file=sys.stderr)
        print(f"  {data_dir / (args.signal + '.csv')}", file=sys.stderr)
        print(f"  {data_dir / (args.trade + '.csv')}", file=sys.stderr)
        print("Or try: python3 cta_lite.py --download --signal QQQ --trade QQQ", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
