#!/bin/bash

# python table_sample.py --run init

echo "\n*************************Random sample mode**************************\n"
for ((i=1;i<=10;i++))
do
    python table_sample.py --run update --update sample
done

echo "\n*************************Single sample mode**************************\n"
for ((i=1;i<=10;i++))
do
    python table_sample.py --run update --update single
done

# for ((i=1;i<=5;i++))
# do
#     python table_sample.py --run update --update permute
# done