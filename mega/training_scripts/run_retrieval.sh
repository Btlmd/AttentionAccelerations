# Set up training envs. Same for all tasks.
seed=1

source dataset.source

DATA=${DATA_ROOT}/aan
CHUNK=128
SAVE=save/retrieval_${CHUNK}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

model=mega_lra_aan
CUDA_VISIBLE_DEVICES="2,3,6,7" python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task lra-text --input-type text \
    --encoder-layers 6 --n-dim 16 --chunk-size $CHUNK \
    --activation-fn 'silu' --attention-activation-fn 'softmax' \
    --norm-type 'scalenorm' --sen-rep-type 'mp' \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.003 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.1 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.04 \
    --batch-size 16 --sentence-avg --update-freq 8 --max-update 91960 \
    --lr-scheduler linear_decay --total-num-update 91960 --end-learning-rate 0.0 \
    --warmup-updates 10000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --max-sentences-valid 64 \
    --valid-subset valid,test \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0
