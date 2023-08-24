#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

echo "training Pythia-70m on subset of real data"
python fit.py --checkpoint_dir checkpoints/EleutherAI/pythia-70m --data_dir data/smol-chess --out_dir out/full/smol-chess  || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Single batch test failed"
  exit 1
fi

echo "Single batch test passed"
exit 0
