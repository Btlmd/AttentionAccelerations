# Set up training envs. Same for all tasks.
seed=1

source dataset.source

echo "Chunk size: $1"
echo "GPU: $2"

DATA=${DATA_ROOT}/listops
CHUNK=$1
SAVE=save_time/listops_${CHUNK}
rm -r ${SAVE}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

model=mega_lra_listop
CUDA_VISIBLE_DEVICES=$2 python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task lra-text --input-type text \
    --encoder-layers 6 --n-dim 16 --chunk-size $CHUNK \
    --activation-fn 'silu' --attention-activation-fn 'softmax' \
    --norm-type 'layernorm' --sen-rep-type 'mp' \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.001 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.1 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
    --batch-size 64 --batch-size-valid 32 --sentence-avg --update-freq 1 --max-update 90000 --max-sentences-valid 256 \
    --lr-scheduler linear_decay --total-num-update 903000 --end-learning-rate 0.0 \
    --warmup-updates 3000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --required-batch-size-multiple 1 \
    --valid-subset test \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 2>&1 | tee ${SAVE}/log.txt
    