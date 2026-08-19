#!/usr/bin/env bash
# Wrap a resumable run_arms.py invocation with periodic Kaggle Dataset checkpointing, so
# progress survives a session getting killed by Kaggle's idle/time limit.
#
# WHY: /kaggle/working only persists into a COMMITTED version's Output. A plain interactive
# session that gets killed writes nothing durable, so run_arms.py's own resumability (it skips
# rows already present in --out, via done_ids()) only helps within a session that is still
# alive. This script makes the --out file itself durable: it pushes it to a private Kaggle
# Dataset every --interval seconds, and pulls the latest checkpoint back down before starting,
# so re-running the SAME cell after a kill resumes for real, from a fresh session.
#
# ONE-TIME SETUP (per Kaggle account, not per notebook):
#   1. kaggle.com/settings/account -> API -> "Create New Token" -> downloads kaggle.json
#   2. In the notebook: Add-ons -> Secrets -> add KAGGLE_USERNAME and KAGGLE_KEY from that file
#   3. The account must be phone-verified -- the Datasets API (create/version) 403s otherwise
#   4. Run the auth-setup cell (writes ~/.kaggle/kaggle.json from the secrets) before this script
#
# USAGE (inside a %%bash cell):
#   bash "$REPO/scripts/kaggle_checkpoint_wrap.sh" \
#       --out /kaggle/working/arms_1-adp-ourops_1of5.jsonl \
#       --slug acorec-arm1-1of5-ckpt \
#       --interval 600 \
#       -- \
#       python "$REPO/scripts/run_arms.py" --arm 1-adp-ourops --shard 1/5 ...
#
# If ~/.kaggle/kaggle.json is missing, this degrades to running the command directly with NO
# checkpointing (loud warning, not a silent no-op) rather than blocking the run on auth setup.
set -uo pipefail

OUT=""
SLUG=""
INTERVAL=600
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --slug) SLUG="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --) shift; ARGS=("$@"); break ;;
    *) echo "[ckpt] unknown arg: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$OUT" ] || [ -z "$SLUG" ] || [ ${#ARGS[@]} -eq 0 ]; then
  echo "usage: $0 --out FILE --slug SLUG [--interval SECS] -- CMD..." >&2
  exit 2
fi

USERNAME=$(python -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])" 2>/dev/null)
if [ -z "$USERNAME" ]; then
  echo "[ckpt] WARNING: no ~/.kaggle/kaggle.json -- run the auth-setup cell first." >&2
  echo "[ckpt] WARNING: continuing WITHOUT checkpointing. Progress will NOT survive a session kill." >&2
  exec "${ARGS[@]}"
fi

FULL_SLUG="$USERNAME/$SLUG"
CKPT_DIR=$(mktemp -d)
mkdir -p "$(dirname "$OUT")"
touch "$OUT"

echo "[ckpt] pulling prior checkpoint from $FULL_SLUG (if any) ..."
kaggle datasets download -d "$FULL_SLUG" -p "$CKPT_DIR" --unzip >/dev/null 2>&1
if [ -f "$CKPT_DIR/arms.jsonl" ]; then
  cp "$CKPT_DIR/arms.jsonl" "$OUT"
  echo "[ckpt] resumed: $(wc -l < "$OUT") row(s) carried over from the last checkpoint"
else
  echo "[ckpt] no prior checkpoint found for $FULL_SLUG; starting fresh"
fi

push() {
  cp "$OUT" "$CKPT_DIR/arms.jsonl"
  cat > "$CKPT_DIR/dataset-metadata.json" <<JSON
{"title": "$SLUG", "id": "$FULL_SLUG", "licenses": [{"name": "CC0-1.0"}]}
JSON
  if kaggle datasets version -p "$CKPT_DIR" -m "checkpoint $(date -u +%FT%TZ), $(wc -l < "$OUT") rows" -q -r zip \
      > "$CKPT_DIR/.push.log" 2>&1; then
    echo "[ckpt] pushed $(wc -l < "$OUT") row(s) to $FULL_SLUG"
    return 0
  fi
  # First checkpoint ever for this slug: `version` fails because the dataset doesn't exist yet.
  if kaggle datasets create -p "$CKPT_DIR" -q -r zip >> "$CKPT_DIR/.push.log" 2>&1; then
    echo "[ckpt] created $FULL_SLUG with $(wc -l < "$OUT") row(s)"
    return 0
  fi
  echo "[ckpt] WARNING: checkpoint push failed (this run's progress is not durable right now):" >&2
  tail -5 "$CKPT_DIR/.push.log" >&2
  return 1
}

"${ARGS[@]}" &
RUN_PID=$!
echo "[ckpt] watching pid=$RUN_PID, checkpointing every ${INTERVAL}s to $FULL_SLUG"

while kill -0 "$RUN_PID" 2>/dev/null; do
  sleep "$INTERVAL"
  kill -0 "$RUN_PID" 2>/dev/null && push
done
wait "$RUN_PID"
STATUS=$?
echo "[ckpt] run finished (exit $STATUS); final checkpoint ..."
push
exit "$STATUS"
