#!/usr/bin/env bash
# Default container behaviour: rank the FULL candidate pool.
#   - RANK_SAMPLE=1        -> rank the bundled 50-candidate sample (instant, no download)
#   - CANDIDATES=/path     -> rank that file (e.g. a mounted candidates.jsonl)
#   - otherwise            -> download candidates.jsonl from DRIVE_FILE_URL and rank all
set -euo pipefail

OUT="${OUT:-/data/submission.csv}"
mkdir -p "$(dirname "$OUT")"

if [ "${RANK_SAMPLE:-0}" = "1" ]; then
  exec python rank.py --job jobs/senior_ai_engineer --sample --out "$OUT" --topk "${TOPK:-100}"
fi

CAND="${CANDIDATES:-/data/candidates.jsonl}"
if [ ! -f "$CAND" ]; then
  echo "Downloading full candidate pool from: $DRIVE_FILE_URL"
  python -m gdown --fuzzy -O "$CAND" "$DRIVE_FILE_URL"
fi

exec python rank.py --job jobs/senior_ai_engineer --candidates "$CAND" --out "$OUT" --topk "${TOPK:-100}"
