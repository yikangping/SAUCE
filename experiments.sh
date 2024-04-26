#!/bin/bash

# 执行第一个命令
echo "Data update and "
python end2end/multi_experiments.py

model=naru
# model=transformer

# 执行第二个命令
echo "Incremental learning"

# Naru/Transformer incremental train
# python Naru/incremental_train.py --dataset bjaq --epochs 40 --model $model --update_size 80000 --bs 4000
# python Naru/incremental_train.py --dataset forest --epochs 40 --model $model --update_size 100000 --bs 4000
# python Naru/incremental_train.py --dataset census --epochs 40 --model $model --update_size 8000

# FACE incremental train
# python FACE/incremental_train.py --dataset bjaq --epochs 40 --update_size 80000
python FACE/incremental_train.py --dataset power --epochs 40 --update_size 400000

# echo "Retrain"
# python Naru/train_model.py --dataset bjaq --epochs 300 --model $model --training_type retrain --bs 4000
# python Naru/train_model.py --dataset census --epochs 300 --model $model --training_type retrain
# python Naru/train_model.py --dataset forest --epochs 300 --model $model --training_type retrain --bs 4000

# echo "Model evaluation"
# python Naru/eval_model.py --dataset bjaq --model transformer --update_size 80000 --query_seed 8

echo "Model evaluation"

model_type=(origin update adaptST adaptTA finetn)
for query_seed in {0..10}
do
    # Naru/Transformer
    # python Naru/eval_model.py --dataset bjaq --model $model --query_seed $query_seed
    # python Naru/eval_model.py --dataset forest --model $model --query_seed $query_seed
    # python Naru/eval_model.py --dataset census --model $model --query_seed $query_seed

    #FACE
    for type in ${model_type[@]}
    do
        python FACE/evaluate/Evaluate-BJAQ-end2end.py --dataset bjaq --model_type $type --query_seed $query_seed
    done
done