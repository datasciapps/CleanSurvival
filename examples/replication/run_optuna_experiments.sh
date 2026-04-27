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
            echo "Running Optuna loops for stats generation for $dataset $pct% $missing (20 iterations)..."
            for i in {1..20}; do
                $PYTHON run.py \
                    -d cleansurvival/datasets/${dataset}_missing_${missing}/${dataset}_missing_${pct}_${missing}.csv \
                    -r config.json \
                    -md COX \
                    -lm D \
                    -lf disable.txt \
                    -a O \
                    -ao 15 \
                    -tc $tc \
                    -ec $ec \
                    -dc pid > /dev/null
            done
        done
    done
done

echo "Running Optuna loops for FLCHAIN (20 iterations)..."
for i in {1..20}; do
    $PYTHON run.py \
        -d cleansurvival/datasets/flchain.csv \
        -r config.json \
        -md COX \
        -lm D \
        -lf disable.txt \
        -a O \
        -ao 15 \
        -tc futime \
        -ec death \
        -dc rownames > /dev/null
done

echo "Optuna iterations completed."