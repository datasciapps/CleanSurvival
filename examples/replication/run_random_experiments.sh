#!/bin/bash
# run_random_experiments.sh
# Reproduces the statistical baseline using Random search method

set -e

cd "$(dirname "$0")/../../"

PYTHON_ENV="../env/bin/python"
if [ -x "$PYTHON_ENV" ]; then
    PYTHON="$PYTHON_ENV"
else
    PYTHON="python3"
fi

echo "=== Running Random Baselines ==="

mkdir -p results/random

TOTAL=620
CURRENT=1

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
            echo "Running Random loops for stats generation for $dataset $pct% $missing (20 iterations)..."
            for i in {1..20}; do
                DONE_FILE="results/random/.random_done_${dataset}_${pct}_${missing}_${i}"
                if [ -f "$DONE_FILE" ]; then
                    echo "[$CURRENT/$TOTAL] Skipping Random iteration $i/20 for $dataset $pct% $missing (already done)"
                    CURRENT=$((CURRENT+1))
                    continue
                fi
                echo "[$CURRENT/$TOTAL] Random iteration $i/20 for $dataset $pct% $missing"
                $PYTHON run.py \
                    -d cleansurvival/datasets/${dataset}_missing_${missing}/${dataset}_missing_${pct}_${missing}.csv \
                    -r config.json \
                    -md COX \
                    -lm D \
                    -lf disable.txt \
                    -a Random \
                    -ao 15 \
                    -tc $tc \
                    -ec $ec \
                    -dc pid > /dev/null
                touch "$DONE_FILE"
                CURRENT=$((CURRENT+1))
            done
        done
    done
done

echo "Running Random loops for FLCHAIN (20 iterations)..."
for i in {1..20}; do
    DONE_FILE="results/random/.random_done_flchain_${i}"
    if [ -f "$DONE_FILE" ]; then
        echo "[$CURRENT/$TOTAL] Skipping Random iteration $i/20 for flchain (already done)"
        CURRENT=$((CURRENT+1))
        continue
    fi
    echo "[$CURRENT/$TOTAL] Random iteration $i/20 for flchain"
    $PYTHON run.py \
        -d cleansurvival/datasets/flchain.csv \
        -r config.json \
        -md COX \
        -lm D \
        -lf disable.txt \
        -a Random \
        -ao 15 \
        -tc futime \
        -ec death \
        -dc rownames > /dev/null
    touch "$DONE_FILE"
    CURRENT=$((CURRENT+1))
done

echo "Optuna iterations completed."