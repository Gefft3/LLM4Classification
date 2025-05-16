#!/bin/bash 

sleep 2

path_dataset="../../data/text_success_with_summary.csv"

class="relevant"

nohup python3 classifier.py $path_dataset $class > classifier.log 2>&1 &