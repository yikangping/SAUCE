#!/bin/bash

model=naru
# model=transformer

# 循环执行命令，将query_seed分别设置为1到10
echo "Model evaluation"
for query_seed in {31..50}
do
    # python Naru/eval_model.py --dataset bjaq --model $model --query_seed $query_seed
    # python Naru/eval_model.py --dataset forest --model $model --query_seed $query_seed
    python Naru/eval_model.py --dataset census --model $model --query_seed $query_seed
done