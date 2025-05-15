#!/bin/bash 

sleep 2

path_dataset="../../data/text_success_with_summary_sample.csv"

class="relevant_sample"

nohup python3 main.py $path_dataset $class > main.log 2>&1 &