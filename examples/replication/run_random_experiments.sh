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

ITERATIONS="${ITERATIONS:-20}"
RANDOM_TRIALS="${RANDOM_TRIALS:-15}"
DATASETS=("rotterdam" "gbsg")
MECHANISMS=("MNAR" "MAR" "MCAR")
PCTS=(50 40 30 20 10)

TOTAL=$(( (${#DATASETS[@]} * ${#MECHANISMS[@]} * ${#PCTS[@]} * ITERATIONS) + ITERATIONS ))
CURRENT=1

for dataset in "${DATASETS[@]}"; do
    if [ "$dataset" == "rotterdam" ]; then
        tc="dtime"
        ec="death"
    else
        tc="rfstime"
        ec="status"
    fi

    for missing in "${MECHANISMS[@]}"; do
        for pct in "${PCTS[@]}"; do
            echo "Running Random loops for stats generation for $dataset $pct% $missing ($ITERATIONS iterations)..."
            for ((i=1; i<=ITERATIONS; i++)); do
                DONE_FILE="results/random/.random_done_${dataset}_${pct}_${missing}_${i}"
                if [ -f "$DONE_FILE" ]; then
                    echo "[$CURRENT/$TOTAL] Skipping Random iteration $i/$ITERATIONS for $dataset $pct% $missing (already done)"
                    CURRENT=$((CURRENT+1))
                    continue
                fi
                echo "[$CURRENT/$TOTAL] Random iteration $i/$ITERATIONS for $dataset $pct% $missing"
                $PYTHON run.py \
                    -d cleansurvival/datasets/${dataset}_missing_${missing}/${dataset}_missing_${pct}_${missing}.csv \
                    -r config.json \
                    -md COX \
                    -lm D \
                    -lf disable.txt \
                    -a Random \
                    -ao "$RANDOM_TRIALS" \
                    -tc $tc \
                    -ec $ec \
                    -dc pid > /dev/null 2>&1 || true
                touch "$DONE_FILE"
                CURRENT=$((CURRENT+1))
            done
        done
    done
done

echo "Running Random loops for FLCHAIN ($ITERATIONS iterations)..."
for ((i=1; i<=ITERATIONS; i++)); do
    DONE_FILE="results/random/.random_done_flchain_${i}"
    if [ -f "$DONE_FILE" ]; then
        echo "[$CURRENT/$TOTAL] Skipping Random iteration $i/$ITERATIONS for flchain (already done)"
        CURRENT=$((CURRENT+1))
        continue
    fi
    echo "[$CURRENT/$TOTAL] Random iteration $i/$ITERATIONS for flchain"
    $PYTHON run.py \
        -d cleansurvival/datasets/flchain.csv \
        -r config.json \
        -md COX \
        -lm D \
        -lf disable.txt \
        -a Random \
        -ao "$RANDOM_TRIALS" \
        -tc futime \
        -ec death \
        -dc rownames > /dev/null 2>&1 || true
    touch "$DONE_FILE"
    CURRENT=$((CURRENT+1))
done

echo "Random iterations completed."