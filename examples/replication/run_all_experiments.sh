#!/bin/bash
# run_all_experiments.sh
# This script strictly replicates the paper's experiments for Rotterdam missingness models.
# It invokes CleanSurvival aiming for 1) Discriminative properties (c-index)
# and 2) Calibrated properties (ibs).

set -e

# Change directory to the root of the package so paths align naturally
cd "$(dirname "$0")/../../"

# Ensure the python environment is available or use the system python
PYTHON_ENV="../env/bin/python"
if [ -x "$PYTHON_ENV" ]; then
    PYTHON="$PYTHON_ENV"
else
    PYTHON="python3"
fi

echo "Using Python interpreter: $PYTHON"

# 1. Base Learn2Clean Optimization for C-index (Discriminative)
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
            echo "-> Running $dataset $pct% $missing for C-Index"
            $PYTHON run.py \
                -d cleansurvival/datasets/${dataset}_missing_${missing}/${dataset}_missing_${pct}_${missing}.csv \
                -r config.json \
                -md COX \
                -lm D \
                -lf disable.txt \
                -a L \
                -tc $tc \
                -ec $ec \
                -dc pid \
                -mt c-index
        done
    done
done

# 2. Alternative Calibration Optimization for IBS
echo "=== Alternative: Optimize for IBS ==="

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
            echo "-> Running $dataset $pct% $missing for IBS"
            $PYTHON run.py \
                -d cleansurvival/datasets/${dataset}_missing_${missing}/${dataset}_missing_${pct}_${missing}.csv \
                -r config.json \
                -md COX \
                -lm D \
                -lf disable.txt \
                -a L \
                -tc $tc \
                -ec $ec \
                -dc pid \
                -mt ibs
        done
    done
done

echo "Experiments execution fully completed."
# 3. Optimize for C-Index and IBS on FLCHAIN dataset
echo "=== Running FLCHAIN ==="
$PYTHON run.py \
    -d cleansurvival/datasets/flchain.csv \
    -r config.json \
    -md COX \
    -lm D \
    -lf disable.txt \
    -a L \
    -tc futime \
    -ec death \
    -dc rownames \
    -mt c-index

$PYTHON run.py \
    -d cleansurvival/datasets/flchain.csv \
    -r config.json \
    -md COX \
    -lm D \
    -lf disable.txt \
    -a L \
    -tc futime \
    -ec death \
    -dc rownames \
    -mt ibs

