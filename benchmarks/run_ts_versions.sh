#!/bin/sh
# Benchmark of the twelve further JEPA versions of TSJEPAOD (REPORT.md section 7.8):
# 5 seeds, trained and zero-epoch control, two threads per run, JOBS runs at a time.
#   sh benchmarks/run_ts_versions.sh [JOBS]
# Finished runs (a log ending in "ALL DONE") are skipped, so the script can be restarted.
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-.venv/bin/python}
OUT=results/ts_variants
mkdir -p "$OUT"
if [ "$1" = --one ]; then   # one run: version, kind
    v=$2; kind=$3
    if tail -n 1 "$OUT/${v}_${kind}.log" 2>/dev/null | grep -q 'ALL DONE'; then exit 0; fi
    modes=block,forecast
    case $v in lenepa|lewm|cfjepa) modes=block;; esac
    epochs=
    if [ "$kind" = untrained ]; then epochs="--epochs 0"; fi
    exec $PY benchmarks/bench_ts_jepa_final.py --seeds 5 --threads 2 --variant "$v" --modes $modes $epochs \
        --out "$OUT/${v}_${kind}.csv" > "$OUT/${v}_${kind}.log" 2>&1
fi
JOBS=${1:-6}
# longest first
for v in cfjepa cjepa lpjepa lenepa vjepa21 kerjepa lambdajepa vicreg sjepa salt vjepa lewm; do
    echo "$v trained"
done > "$OUT/.queue"
for v in cfjepa cjepa lpjepa lenepa vjepa21 kerjepa lambdajepa vicreg sjepa salt vjepa lewm; do
    echo "$v untrained"
done >> "$OUT/.queue"
xargs -P "$JOBS" -n 2 sh "$0" --one < "$OUT/.queue"
rm -f "$OUT/.queue"
echo "all runs finished"
