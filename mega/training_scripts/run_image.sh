# Set up training envs. Same for all tasks.
seed=1

source dataset.source

DATA=${DATA_ROOT}/cifar10
CHUNK=512
SAVE=save/image_${CHUNK}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

model=mega_lra_cifar10
CUDA_VISIBLE_DEVICES=4 python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task lra-image --input-type image --pixel-normalization 0.48 0.24 \
    --encoder-layers 8 --n-dim 16 --chunk-size ${CHUNK} \
    --activation-fn 'silu' --attention-activation-fn 'laplace' \
    --norm-type 'batchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.0 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.02 \
    --batch-size 50 --sentence-avg --update-freq 1 --max-update 180000 \
    --lr-scheduler linear_decay --total-num-update 180000 --end-learning-rate 0.0 \
    --warmup-updates 9000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --required-batch-size-multiple 1 \
    --valid-subset valid,test \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 00 2>&1 | tee -a ${SAVE}/log.txt