#!/bin/sh
# TSB-AD benchmark of the sixteen JEPA versions of TSJEPA (REPORT.md section 7.9): benchmarks/bench_tsb.py for
# every version, trained and untrained, on TSB-AD-M-Eva and TSB-AD-U-Eva, JOBS runs at a time over the GPUs in GPUS.
#   ROOT=<work folder> GPUS=2,5,6,7 JOBS=16 sh benchmarks/run_tsb.sh
# Every run takes a GPU of GPUS on which nobody else computes and gives it up when somebody comes (bench_tsb.py
# --gpus); it is then started again here and continues on another GPU.
# ROOT holds .venv/, pyod/ (the PyOD checkout), ref/TSB-AD/ (a clone, for the file lists), data/tsb/sets/ (TSB-AD-M/
# and TSB-AD-U/) and receives results/tsb/. Finished runs (a log ending in "ALL DONE") are skipped, so the script
# can be restarted. The queue is in order of interest: the default version first, the untrained controls last.
# It also holds the runs of the default version with other seeds, fitted on the whole series, and with constant
# training channels scaled by 1 (--unit-const).
#   QUEUE=confirm ROOT=<work folder> GPUS=2,5,6,7 JOBS=12 sh benchmarks/run_tsb.sh
# runs the confirmation of results/tsb_confirm_plan.md instead (REPORT.md section 7.10): every version, trained and
# untrained, and the control without a network, on the lists M-Tuning, U-Tuning and U-Rest.
#   QUEUE=scjepa ROOT=<work folder> GPUS=2,5,6,7 JOBS=10 sh benchmarks/run_tsb.sh
# runs scjepa of 0.5.1 the same way (ten runs, REPORT.md section 7.13).
#   QUEUE=more ROOT=<work folder> GPUS=2,5,6,7 JOBS=12 sh benchmarks/run_tsb.sh
# runs the ten versions of 0.5.0 on all five lists (REPORT.md section 7.11). They are run with jepa_od.TSJEPAOD of
# ROOT/jepa-od (a copy of this repository), since the PyOD port does not have them. The zero-epoch controls of
# subjepa, straight and motionjepa are not run: they have the network of lewm and the same scores.
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$HOME/jepa-od-gpu}
MORE="ststjepa hepa subjepa straight motionjepa njepa gmmjepa variational phijepa harjepa"
PROTOTYPE="$MORE scjepa"   # the versions that only jepa_od.TSJEPAOD has
GPUS=${GPUS:-0}
OUT=$ROOT/results/tsb
mkdir -p "$OUT/logs"
if [ "$1" = --one ]; then   # one run: list (M or U: the evaluation list; or M-Tuning, U-Tuning, U-Rest), version, kind
    suite=${2%%-*}; split=Eva; v=$3; kind=$4; gpu="--gpus $GPUS"
    case $2 in *-*) split=${2#*-};; esac
    case $kind in
        trained)   opts="";               tag=${v}_trained_s42_semi;;
        untrained) opts="--epochs 0";     tag=${v}_e0_s42_semi;;
        seed*)     opts="--seed ${kind#seed}"; tag=${v}_trained_s${kind#seed}_semi;;
        unsup)     opts="--mode unsup";   tag=${v}_trained_s42_unsup;;
        unitconst)  opts="--unit-const";            tag=${v}_trained_s42_semi_unitconst;;
        unitconst0) opts="--epochs 0 --unit-const"; tag=${v}_e0_s42_semi_unitconst;;
        rawpatch)  opts="--rawpatch --device cpu"; tag=rawpatch_s42_semi; gpu="";;   # no network, on the CPU
        *) echo "unknown kind $kind"; exit 1;;
    esac
    pkg=""   # the ten versions of 0.5.0 are in the prototype only
    case " $PROTOTYPE " in *" $v "*) pkg="--jepa-od $ROOT/jepa-od";; esac
    log=$OUT/logs/$2_${tag}.log
    if tail -n 1 "$log" 2>/dev/null | grep -q 'ALL DONE'; then exit 0; fi
    while :; do
        "$ROOT/.venv/bin/python" -I "$HERE/bench_tsb.py" --pyod "$ROOT/pyod" --data "$ROOT/data/tsb/sets" \
            --lists "$ROOT/ref/TSB-AD/Datasets/File_List" --suite "$suite" --split "$split" --variant "$v" \
            $pkg $gpu $opts --out "$OUT" >> "$log" 2>&1
        [ $? -eq 75 ] || exit 0
        sleep 10
    done
fi
JOBS=${JOBS:-3}
FIRST="lejepa kerjepa vicreg lambdajepa ijepa tsjepa tjepa"
REST="vjepa vjepa21 cjepa sjepa salt lpjepa lenepa lewm cfjepa"
if [ "$QUEUE" = scjepa ]; then   # version 0.5.1 (REPORT.md section 7.13)
    for l in M U U-Rest U-Tuning M-Tuning; do echo "$l scjepa trained"; echo "$l scjepa untrained"; done > "$OUT/.queue"
elif [ "$QUEUE" = more ]; then
    {
        for l in M U U-Rest U-Tuning M-Tuning; do for v in $MORE; do echo "$l $v trained"; done; done
        for l in M U U-Rest U-Tuning M-Tuning; do
            for v in ststjepa hepa njepa gmmjepa variational phijepa harjepa; do echo "$l $v untrained"; done
        done
    } > "$OUT/.queue"
elif [ "$QUEUE" = confirm ]; then   # the runs the tests of the plan need first, the longest list first
    LISTS="U-Rest U-Tuning M-Tuning"
    OTHER="kerjepa ijepa tsjepa tjepa vjepa vjepa21 cjepa sjepa salt lpjepa lenepa"
    {
        for l in $LISTS; do
            for v in cfjepa lejepa vicreg lambdajepa lewm; do echo "$l $v trained"; done
            echo "$l lejepa untrained"; echo "$l lejepa rawpatch"
        done
        for l in $LISTS; do for v in $OTHER; do echo "$l $v trained"; done; done
        for l in $LISTS; do for v in cfjepa vicreg lambdajepa lewm $OTHER; do echo "$l $v untrained"; done; done
    } > "$OUT/.queue"
else
{
    for s in M U; do echo "$s lejepa trained"; done
    for s in M U; do echo "$s lejepa untrained"; done
    for v in kerjepa vicreg lambdajepa ijepa tsjepa tjepa $REST; do for s in M U; do echo "$s $v trained"; done; done
    echo "M lejepa unitconst"; echo "M lejepa unitconst0"; echo "U lejepa unitconst0"
    for s in M U; do echo "$s lejepa seed0"; echo "$s lejepa seed1"; echo "$s lejepa unsup"; done
    for v in kerjepa vicreg lambdajepa ijepa tsjepa tjepa $REST; do for s in M U; do echo "$s $v untrained"; done; done
} > "$OUT/.queue"
fi
xargs -P "$JOBS" -n 3 sh "$0" --one < "$OUT/.queue"
rm -f "$OUT/.queue"
echo "all runs finished"
