#!/bin/sh
# Benchmark of the twelve further JEPA versions of TSJEPAOD (REPORT.md section 7.8):
# 5 seeds, trained and zero-epoch control, two threads per run, JOBS runs at a time.
#   sh benchmarks/run_ts_versions.sh [JOBS]
#   SET=ten sh benchmarks/run_ts_versions.sh [JOBS]     the ten versions of 0.5.0 instead (section 7.11)
# Finished runs (a log ending in "ALL DONE") are skipped, so the script can be restarted.
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-.venv/bin/python}
OUT=results/ts_variants
mkdir -p "$OUT"
if [ "$1" = --one ]; then   # one run: version, kind
    v=$2; kind=$3
    if tail -n 1 "$OUT/${v}_${kind}.log" 2>/dev/null | grep -q 'ALL DONE'; then exit 0; fi
    modes=block,forecast
    case $v in lenepa|lewm|cfjepa|hepa|subjepa|straight|motionjepa) modes=block;; esac
    epochs=
    if [ "$kind" = untrained ]; then epochs="--epochs 0"; fi
    exec $PY benchmarks/bench_ts_jepa_final.py --seeds 5 --threads 2 --variant "$v" --modes $modes $epochs \
        --out "$OUT/${v}_${kind}.csv" > "$OUT/${v}_${kind}.log" 2>&1
fi
JOBS=${1:-6}
# longest first
TWELVE="cfjepa cjepa lpjepa lenepa vjepa21 kerjepa lambdajepa vicreg sjepa salt vjepa lewm"
# the ten versions of 0.5.0 (REPORT.md section 7.11); subjepa, straight and motionjepa have the network of lewm,
# so their zero-epoch control is lewm's and is not run again
TEN="hepa njepa phijepa variational gmmjepa ststjepa harjepa motionjepa subjepa straight"
if [ "$SET" = ten ]; then
    for v in $TEN; do echo "$v trained"; done > "$OUT/.queue"
    for v in hepa njepa phijepa variational gmmjepa ststjepa harjepa; do echo "$v untrained"; done >> "$OUT/.queue"
else
    for v in $TWELVE; do echo "$v trained"; done > "$OUT/.queue"
    for v in $TWELVE; do echo "$v untrained"; done >> "$OUT/.queue"
fi
xargs -P "$JOBS" -n 2 sh "$0" --one < "$OUT/.queue"
rm -f "$OUT/.queue"
echo "all runs finished"
