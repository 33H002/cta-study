#!/usr/bin/env python3
"""
Create a QQQ/TQQQ trading signal report and optionally post it to Slack.

The Slack path intentionally mirrors ../Market-Briefing-Bot: it reads a local
SLACK_BOT_TOKEN from that bot's config/slack_bot.env and posts with the bot token,
so the visible sender is the configured Slack bot, not a user connector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from cta_lite import (
    PROFILES,
    PriceSeries,
    align_series,
    backtest,
    maybe_download,
    moving_average,
    parse_date,
    pct_change,
    read_price_csv,
    realized_vol,
)


DEFAULT_CHANNEL_KEY = "nasdaq"
DEFAULT_NASDAQ_CHANNEL_ID = "C0B3RR8MHCN"
DEFAULT_BOT_ENV = Path("../Market-Briefing-Bot/config/slack_bot.env")
TRADING_DAYS = 252


@dataclass(frozen=True)
class Portfolio:
    cash: float | None
    trade_shares: float | None
    weekly_amount: float


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_portfolio(path: Path | None, weekly_amount: float) -> Portfolio:
    if path is None or not path.exists():
        return Portfolio(cash=None, trade_shares=None, weekly_amount=weekly_amount)

    payload = json.loads(path.read_text(encoding="utf-8"))
    return Portfolio(
        cash=float(payload["cash"]) if "cash" in payload else None,
        trade_shares=float(payload["trade_shares"]) if "trade_shares" in payload else None,
        weekly_amount=float(payload.get("weekly_amount", weekly_amount)),
    )


def format_money(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"${value:,.2f}"


def format_pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.1%}"


def signal_breakdown(signal: PriceSeries, trade: PriceSeries) -> dict[str, float | bool | date]:
    dates, signal_closes, trade_closes = align_series(signal, trade)
    trade_returns = [0.0]
    for i in range(1, len(trade_closes)):
        trade_returns.append(trade_closes[i] / trade_closes[i - 1] - 1.0)

    i = len(dates) - 1
    sma_200 = moving_average(signal_closes, i, 200)
    vol_20 = realized_vol(trade_returns, i, 20)
    return {
        "date": dates[i],
        "signal_close": signal_closes[i],
        "trade_close": trade_closes[i],
        "ret_21": pct_change(signal_closes, i, 21) or 0.0,
        "ret_63": pct_change(signal_closes, i, 63) or 0.0,
        "ret_126": pct_change(signal_closes, i, 126) or 0.0,
        "ret_252": pct_change(signal_closes, i, 252) or 0.0,
        "sma_200": sma_200 or float("nan"),
        "above_sma_200": bool(sma_200 is not None and signal_closes[i] > sma_200),
        "crash_10d": pct_change(signal_closes, i, 10) or 0.0,
        "trade_vol_20": vol_20 or float("nan"),
    }


def trade_instruction(
    target_position: float,
    trade_close: float,
    portfolio: Portfolio,
    include_contribution: bool,
    min_trade_amount: float,
) -> tuple[str, str]:
    if portfolio.cash is None or portfolio.trade_shares is None:
        return (
            "CHECK",
            "현재 현금/보유수량 정보가 없어서 주문 금액은 계산하지 않았습니다. "
            f"계좌 기준 목표 비중을 {format_pct(target_position)}로 맞춰 주세요.",
        )

    cash = portfolio.cash + (portfolio.weekly_amount if include_contribution else 0.0)
    position_value = portfolio.trade_shares * trade_close
    account_value = cash + position_value
    target_value = account_value * target_position
    delta = target_value - position_value

    if abs(delta) < min_trade_amount:
        return ("HOLD", f"목표 비중과 현재 비중 차이가 {format_money(min_trade_amount)} 미만입니다.")

    shares = abs(delta) / trade_close
    current_position = position_value / account_value if account_value > 0 else 0.0
    side = "BUY" if delta > 0 else "SELL"
    message = (
        f"{side} {format_money(abs(delta))} of the trade asset, 약 {shares:.3f}주. "
        f"현재 비중 {format_pct(current_position)} -> 목표 {format_pct(target_position)}."
    )
    if include_contribution:
        message += f" 이번 주 적립금 {format_money(portfolio.weekly_amount)}를 현금에 더해 계산했습니다."
    return side, message


def send_slack_message(token: str, channel_id: str, message: str) -> dict:
    body = json.dumps(
        {
            "channel": channel_id,
            "text": message,
            "unfurl_links": False,
            "unfurl_media": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(f"Slack API error: {payload.get('error', 'unknown_error')}")
    return payload


def write_report_csv(path: Path, row: dict[str, str | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_message(
    signal_ticker: str,
    trade_ticker: str,
    timing: str,
    profile_name: str,
    score: float,
    target_position: float,
    breakdown: dict[str, float | bool | date],
    action: str,
    instruction: str,
    include_contribution: bool,
) -> str:
    contribution_line = "포함" if include_contribution else "미포함"
    return "\n".join(
        [
            f"*Nasdaq {trade_ticker} 수동매매 신호* ({timing})",
            f"- 데이터 기준: {breakdown['date']} 종가",
            f"- 신호/매매: {signal_ticker} -> {trade_ticker}",
            f"- 프로파일: {profile_name}",
            f"- 결론: *{action}*",
            f"- 목표 {trade_ticker} 비중: *{format_pct(target_position)}*",
            f"- 추세 점수: {format_pct(score)}",
            f"- 주문 가이드: {instruction}",
            "",
            "*신호 구성*",
            f"- 1M/3M/6M/12M: {format_pct(float(breakdown['ret_21']))} / "
            f"{format_pct(float(breakdown['ret_63']))} / {format_pct(float(breakdown['ret_126']))} / "
            f"{format_pct(float(breakdown['ret_252']))}",
            f"- 200일선 위: {'YES' if breakdown['above_sma_200'] else 'NO'}",
            f"- 10거래일 QQQ 변화: {format_pct(float(breakdown['crash_10d']))}",
            f"- {trade_ticker} 20일 연율화 변동성: {format_pct(float(breakdown['trade_vol_20']))}",
            f"- 월요일 적립금 계산: {contribution_line}",
            "",
            "_전일 종가 기반 수동 주문 참고용입니다. 최종 주문 전 계좌/가격/세금/체결조건을 확인하세요._",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and optionally send a Nasdaq CTA signal report.")
    parser.add_argument("--signal", default="QQQ")
    parser.add_argument("--trade", default="TQQQ")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--download", action="store_true", help="Download missing CSVs before creating the report")
    parser.add_argument("--refresh-data", action="store_true", help="Force-refresh CSVs before creating the report")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="aggressive")
    parser.add_argument("--target-vol", type=float)
    parser.add_argument("--max-position", type=float)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--weekly-amount", type=float, default=100.0)
    parser.add_argument("--portfolio-file", default="config/nasdaq_portfolio.json")
    parser.add_argument("--include-weekly-contribution", action="store_true")
    parser.add_argument("--auto-weekly-contribution", action="store_true")
    parser.add_argument("--timezone", default="Asia/Seoul")
    parser.add_argument("--timing", choices=("pre-open", "after-close"), default="pre-open")
    parser.add_argument("--min-trade-amount", type=float, default=10.0)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--history-csv", default="results/nasdaq_signal_history.csv")
    parser.add_argument("--send-slack", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bot-env", default=str(DEFAULT_BOT_ENV))
    parser.add_argument("--channel-key", default=DEFAULT_CHANNEL_KEY)
    parser.add_argument("--channel-id", default=DEFAULT_NASDAQ_CHANNEL_ID)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    profile = PROFILES[args.profile]
    default_target_vol = 0.30 if args.trade.upper() == "TQQQ" else profile.default_target_vol
    target_vol = args.target_vol if args.target_vol is not None else default_target_vol
    max_position = args.max_position if args.max_position is not None else min(profile.default_max_position, 1.0)

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
    breakdown = signal_breakdown(signal, trade)
    latest_score = result.scores[-1]
    target_position = result.positions[-1]

    portfolio_path = Path(args.portfolio_file)
    portfolio = load_portfolio(portfolio_path if portfolio_path.exists() else None, args.weekly_amount)
    local_today = datetime.now(ZoneInfo(args.timezone)).date()
    include_contribution = args.include_weekly_contribution or (
        args.auto_weekly_contribution and local_today.weekday() == 0
    )
    action, instruction = trade_instruction(
        target_position=target_position,
        trade_close=float(breakdown["trade_close"]),
        portfolio=portfolio,
        include_contribution=include_contribution,
        min_trade_amount=args.min_trade_amount,
    )
    message = build_message(
        signal_ticker=args.signal,
        trade_ticker=args.trade,
        timing=args.timing,
        profile_name=profile.name,
        score=latest_score,
        target_position=target_position,
        breakdown=breakdown,
        action=action,
        instruction=instruction,
        include_contribution=include_contribution,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"nasdaq_signal_{breakdown['date']}_{args.timing}.md"
    output_path.write_text(message + "\n", encoding="utf-8")

    write_report_csv(
        Path(args.history_csv),
        {
            "run_date": date.today().isoformat(),
            "market_date": str(breakdown["date"]),
            "timing": args.timing,
            "profile": profile.name,
            "score": round(latest_score, 6),
            "target_position": round(target_position, 6),
            "action": action,
            "trade_close": round(float(breakdown["trade_close"]), 6),
            "output_path": str(output_path),
        },
    )

    print(message)
    print()
    print(f"Saved report to {output_path}")

    if args.send_slack:
        if args.dry_run:
            print(f"Dry run: would send to Slack channel {args.channel_key} ({args.channel_id})")
        else:
            load_env_file(Path(args.bot_env))
            token = os.environ.get("SLACK_BOT_TOKEN")
            if not token:
                raise RuntimeError(f"SLACK_BOT_TOKEN is not set. Fill {args.bot_env} for brefingbot.")
            payload = send_slack_message(token, args.channel_id, message)
            print(json.dumps({"ok": True, "channel": payload.get("channel"), "ts": payload.get("ts")}))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
