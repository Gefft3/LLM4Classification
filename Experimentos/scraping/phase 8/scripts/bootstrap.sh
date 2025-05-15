#!/bin/bash

sleep 2
# Segue as colunas que devem existir no dataset: expanded_url, short_url, html_content, summary. 
path_dataset_text="../../data/text_success_with_summary_sample.csv"

path_classes="../../phase 7/logs/relevant_sample/classes.txt"

path_output="../data/text_success_with_summary_sample_filtered.csv"

nohup python3 text_filter.py "$path_dataset_text" "$path_classes" "$path_output" > main.log 2>&1 &


