#!/bin/bash

csv_path='../../../data/retrieved_html_success_abstract_sample.csv'

python3 main.py $csv_path > main.log 2>&1 &