# Set up training envs. Same for all tasks.
seed=1

DATA=/workspace/mingdao/data/lra_mega/imdb-4000
SAVE=save/text
CHUNK=128
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

model=mega_lra_imdb
CUDA_VISIBLE_DEVICES=1 python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task lra-text --input-type text \
    --encoder-layers 4 --n-dim 16 --chunk-size ${CHUNK} \
    --activation-fn 'silu' --attention-activation-fn 'softmax' \
    --norm-type 'scalenorm' --sen-rep-type 'mp' \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.004 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.1 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
    --batch-size 50 --sentence-avg --update-freq 2 --max-update 25000 --required-batch-size-multiple 1 \
    --lr-scheduler linear_decay --total-num-update 25000 --end-learning-rate 0.0 \
    --warmup-updates 10000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --max-sentences-valid 100 \
    --valid-subset valid,test \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0
