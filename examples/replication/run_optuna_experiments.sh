#!/bin/bash
# run_optuna_experiments.sh
# Reproduces the statistical baseline using Optuna search method

set -e

cd "$(dirname "$0")/../../"

PYTHON_ENV="../env/bin/python"
if [ -x "$PYTHON_ENV" ]; then
    PYTHON="$PYTHON_ENV"
else
    PYTHON="python3"
fi

echo "=== Running Optuna Baselines ==="

mkdir -p results/optuna

ITERATIONS="${ITERATIONS:-20}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-15}"
METRICS=("c-index" "ibs")
DATASETS=("rotterdam" "gbsg")
MECHANISMS=("MNAR" "MAR" "MCAR")
PCTS=(50 40 30 20 10)

TOTAL=$(( (${#DATASETS[@]} * ${#MECHANISMS[@]} * ${#PCTS[@]} * ITERATIONS * ${#METRICS[@]}) + (ITERATIONS * ${#METRICS[@]}) ))
CURRENT=1

for metric in "${METRICS[@]}"; do
    echo "=== Metric: $metric ==="
    metric_tag="${metric//-/_}"

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
            echo "Running Optuna loops for stats generation for $dataset $pct% $missing metric=$metric ($ITERATIONS iterations)..."
            for ((i=1; i<=ITERATIONS; i++)); do
                DONE_FILE="results/optuna/.optuna_done_${metric_tag}_${dataset}_${pct}_${missing}_${i}"
                if [ -f "$DONE_FILE" ]; then
                    echo "[$CURRENT/$TOTAL] Skipping Optuna iteration $i/$ITERATIONS for $dataset $pct% $missing metric=$metric (already done)"
                    CURRENT=$((CURRENT+1))
                    continue
                fi
                echo "[$CURRENT/$TOTAL] Optuna iteration $i/$ITERATIONS for $dataset $pct% $missing metric=$metric"
                $PYTHON run.py \
                    -d cleansurvival/datasets/${dataset}_missing_${missing}/${dataset}_missing_${pct}_${missing}.csv \
                    -r config.json \
                    -md COX \
                    -lm D \
                    -lf disable.txt \
                    -a O \
                    -ao "$OPTUNA_TRIALS" \
                    -tc $tc \
                    -ec $ec \
                    -dc pid \
                    -mt "$metric" > /dev/null 2>&1 || true
                touch "$DONE_FILE"
                CURRENT=$((CURRENT+1))
            done
        done
    done
done

echo "Running Optuna loops for FLCHAIN metric=$metric ($ITERATIONS iterations)..."
for ((i=1; i<=ITERATIONS; i++)); do
    DONE_FILE="results/optuna/.optuna_done_${metric_tag}_flchain_${i}"
    if [ -f "$DONE_FILE" ]; then
        echo "[$CURRENT/$TOTAL] Skipping Optuna iteration $i/$ITERATIONS for flchain metric=$metric (already done)"
        CURRENT=$((CURRENT+1))
        continue
    fi
    echo "[$CURRENT/$TOTAL] Optuna iteration $i/$ITERATIONS for flchain metric=$metric"
    $PYTHON run.py \
        -d cleansurvival/datasets/flchain.csv \
        -r config.json \
        -md COX \
        -lm D \
        -lf disable.txt \
        -a O \
        -ao "$OPTUNA_TRIALS" \
        -tc futime \
        -ec death \
        -dc rownames \
        -mt "$metric" > /dev/null 2>&1 || true
    touch "$DONE_FILE"
    CURRENT=$((CURRENT+1))
done
done

echo "Optuna iterations completed."