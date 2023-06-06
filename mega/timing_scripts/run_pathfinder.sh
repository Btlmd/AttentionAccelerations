# Set up training envs. Same for all tasks.
seed=99

source dataset.source

echo "Chunk size: $1"
echo "GPU: $2"

DATA=${DATA_ROOT}/pathfinder
CHUNK=$1
SAVE=save_time/pathfinder_${CHUNK}_pow2
rm -r ${SAVE}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

model=mega_lra_pf32
CUDA_VISIBLE_DEVICES=$2 python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task lra-image --input-type image --pixel-normalization 0.5 0.5 \
    --encoder-layers 6 --n-dim 16 --chunk-size ${CHUNK} \
    --activation-fn 'silu' --attention-activation-fn 'laplace' \
    --norm-type 'batchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.0 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
    --batch-size 128 --batch-size-valid 128 --sentence-avg --update-freq 1 --max-update 250000 \
    --lr-scheduler linear_decay --total-num-update 250000 --end-learning-rate 0.0 \
    --warmup-updates 50000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --max-sentences-valid 512 \
    --valid-subset test \
    --warmup-power 2 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 2>&1 | tee ${SAVE}/log.txt
