#!/bin/bash

csv_path='../../../data/retrieved_html_success_sample.csv'

python3 filter.py $csv_path > filter.log 2>&1 &