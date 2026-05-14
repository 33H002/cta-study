#!/usr/bin/env bash
set -euo pipefail

cd /Users/soohee/Documents/Codex/study/cta

python3 cta_signal_report.py \
  --signal QQQ \
  --trade TQQQ \
  --profile aggressive \
  --timing pre-open \
  --refresh-data \
  --auto-weekly-contribution \
  --send-slack
