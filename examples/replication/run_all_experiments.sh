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
ITERATIONS="${ITERATIONS:-20}"
echo "Using Q-learning n_episodes: $N_EPISODES"
echo "Using repeated runs per combo: $ITERATIONS"

mkdir -p results/qlearning

METRICS=("c-index" "ibs")
DATASETS=("rotterdam" "gbsg")
MECHANISMS=("MNAR" "MAR" "MCAR")
PCTS=(50 40 30 20 10)

TOTAL=$(( (${#DATASETS[@]} * ${#MECHANISMS[@]} * ${#PCTS[@]} * ITERATIONS * ${#METRICS[@]}) + (ITERATIONS * ${#METRICS[@]}) ))
CURRENT=1

echo "=== Running CleanSurvival (priority ordered) ==="

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
            for ((iter=1; iter<=ITERATIONS; iter++)); do
                DONE_FILE="results/qlearning/.run_all_${metric_tag}_${dataset}_${pct}_${missing}_${iter}"
                if [ -f "$DONE_FILE" ]; then
                    echo "[$CURRENT/$TOTAL] Skipping $dataset $pct% $missing metric=$metric iteration $iter (already done)"
                    CURRENT=$((CURRENT+1))
                    continue
                fi
                echo "[$CURRENT/$TOTAL] -> Running $dataset $pct% $missing metric=$metric iteration $iter"
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
                    -mt "$metric" > /dev/null 2>&1 || true
                touch "$DONE_FILE"
                CURRENT=$((CURRENT+1))
            done
        done
    done
done

    echo "=== Running FLCHAIN metric=$metric ==="
    for ((iter=1; iter<=ITERATIONS; iter++)); do
        DONE_FILE_FLC="results/qlearning/.run_all_${metric_tag}_flchain_${iter}"
        if [ -f "$DONE_FILE_FLC" ]; then
            echo "[$CURRENT/$TOTAL] Skipping flchain metric=$metric iteration $iter (already done)"
            CURRENT=$((CURRENT+1))
            continue
        fi

        echo "[$CURRENT/$TOTAL] -> Running flchain metric=$metric iteration $iter"
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
            -mt "$metric" > /dev/null 2>&1 || true
        touch "$DONE_FILE_FLC"
        CURRENT=$((CURRENT+1))
    done
done

echo "Experiments execution fully completed."
