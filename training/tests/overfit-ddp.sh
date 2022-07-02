#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

# constants and CLI args set by aiming for <10 min test on 8xV100
MAX_EPOCHS="${1:-600}"
CRITERION="${2:-0.1}"

echo "running with configuration tuned on 8xV100"
echo "- note that num_workers > 1 speeds up training but results in multiprocessing errors in terminal"
python ./training/run_experiment.py \
  --data_class=IAMOriginalAndSyntheticParagraphs --model_class=ResnetTransformer --loss=transformer \
  --limit_test_batches 0.0 --overfit_batches 1 --num_sanity_val_steps 0 \
  --augment_data false --tf_dropout 0.0 \
  --gpus 8 --precision 16 --strategy=ddp_find_unused_parameters_false --num_workers 1 --batch_size 16 --lr 0.0001 \
  --log_every_n_steps 50 --max_epochs "$MAX_EPOCHS"  --wandb || FAILURE=true

python -c "import json; loss = json.load(open('training/logs/wandb/latest-run/files/wandb-summary.json'))['train/loss']; assert loss < $CRITERION" || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Overfitting test failed"
  exit 1
fi
echo "Overfitting test passed"
exit 0
