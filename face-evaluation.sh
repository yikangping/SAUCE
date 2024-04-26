#!/bin/bash

model_type=(origin update adaptST adaptTA finetn)
seeds=(11 12 13 14 15 16 17 18 21 24 26 30)
for query_seed in {11..30}
# for query_seed in ${seeds[@]}
do
    for type in ${model_type[@]}
    do
        # python FACE/evaluate/Evaluate-BJAQ-end2end.py --dataset bjaq --model_type  $type --query_seed $query_seed
        python FACE/evaluate/Evaluate-BJAQ-end2end.py --dataset power --model_type  $type --query_seed $query_seed
    done
done

# for type in ${model_type[@]}
# do
#     python FACE/evaluate/Evaluate-BJAQ-end2end.py --dataset bjaq --model_type  $type --query_seed 1
# done