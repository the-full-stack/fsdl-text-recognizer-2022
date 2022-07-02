#!/bin/bash
set -uo pipefail
set +e

# tests whether we can achieve a criterion loss
#  on a single batch within a certain number of epochs

FAILURE=false

# constants and CLI args set by aiming for <5 min test on commodity GPU
MAX_EPOCHS="${1:-100}"
CRITERION="${2:-2.0}"

python ./training/run_experiment.py \
  --data_class=IAMOriginalAndSyntheticParagraphs --model_class=ResnetTransformer --loss=transformer \
  --limit_test_batches 0.0 --overfit_batches 1 --num_sanity_val_steps 0 \
  --augment_data false --tf_dropout 0.0 \
  --gpus 1 --precision 16 --batch_size 16 --lr 0.0001 \
  --log_every_n_steps 25 --max_epochs "$MAX_EPOCHS"  --wandb || FAILURE=true

python -c "import json; loss = json.load(open('training/logs/wandb/latest-run/files/wandb-summary.json'))['train/loss']; assert loss < $CRITERION" || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Overfitting test failed"
  exit 1
fi
echo "Overfitting test passed"
exit 0
