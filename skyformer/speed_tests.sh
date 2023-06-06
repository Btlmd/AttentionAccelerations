export CUDA_VISIBLE_DEVICES=7

set -e

tasks=(lra-text lra-image lra-listops lra-retrieval lra-pathfinder)
models=(softmax lara skyformer cosformer)

for TASK in ${tasks[@]}; do
    for MODLE in ${models[@]}; do
        echo "===== Task: $TASK, Model: $MODLE ====="
        python speed_test.py --mode train --attn $MODLE --task $TASK
    done
done
