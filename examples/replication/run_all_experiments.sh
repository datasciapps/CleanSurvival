#!/bin/bash
# run_all_experiments.sh

set -e

cd "$(dirname "$0")/../../"

PYTHON_ENV="../env/bin/python"
if [ -x "$PYTHON_ENV" ]; then
    PYTHON="$PYTHON_ENV"
else
    PYTHON="python3"
fi

echo "Using Python interpreter: $PYTHON"

N_EPISODES="${N_EPISODES:-20}"
echo "Using Q-learning n_episodes: $N_EPISODES"

mkdir -p results/qlearning

TOTAL=620
CURRENT=1

echo "=== Baseline: Optimize for C-Index ==="

for dataset in "rotterdam" "gbsg"; do
    if [ "$dataset" == "rotterdam" ]; then
        tc="dtime"
        ec="death"
    else
        tc="rfstime"
        ec="status"
    fi

    for missing in "MCAR" "MAR" "MNAR"; do
        for pct in 10 20 30 40 50; do
            for iter in {1..20}; do
                DONE_FILE="results/qlearning/.run_all_cindex_${dataset}_${pct}_${missing}_${iter}"
                if [ -f "$DONE_FILE" ]; then
                    echo "[$CURRENT/$TOTAL] Skipping $dataset $pct% $missing iteration $iter (already done)"
                    CURRENT=$((CURRENT+1))
                    continue
                fi
                echo "[$CURRENT/$TOTAL] -> Running $dataset $pct% $missing for C-Index iteration $iter"
                $PYTHON run.py \
                    -d cleansurvival/datasets/${dataset}_missing_${missing}/${dataset}_missing_${pct}_${missing}.csv \
                    -r config.json \
                    -md COX \
                    -lm D \
                    -lf disable.txt \
                    -a CleanSurvival \
                    -ne "$N_EPISODES" \
                    -tc $tc \
                    -ec $ec \
                    -dc pid \
                    -mt c-index > /dev/null 2>&1 || true
                touch "$DONE_FILE"
                CURRENT=$((CURRENT+1))
            done
        done
    done
done

echo "=== Running FLCHAIN ==="
for iter in {1..20}; do
    DONE_FILE_FLC="results/qlearning/.run_all_cindex_flchain_${iter}"
    if [ -f "$DONE_FILE_FLC" ]; then
        CURRENT=$((CURRENT+1))
    else
        echo "[$CURRENT/$TOTAL] -> Running flchain for C-Index iteration $iter"
        $PYTHON run.py \
            -d cleansurvival/datasets/flchain.csv \
            -r config.json \
            -md COX \
            -lm D \
            -lf disable.txt \
            -a CleanSurvival \
            -ne "$N_EPISODES" \
            -tc futime \
            -ec death \
            -dc rownames \
            -mt c-index > /dev/null 2>&1 || true
        touch "$DONE_FILE_FLC"
        CURRENT=$((CURRENT+1))
    fi
done

echo "Experiments execution fully completed."
