#!/bin/bash 

sleep 2

path_dataset="../../data/text_success_with_summary_sample.csv"

nohup python3 main.py $path_dataset > main.log 2>&1 &