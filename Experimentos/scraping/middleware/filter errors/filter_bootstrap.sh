#!/bin/bash

csv_path='../../data/samples_to_filter.csv'

python3 filter.py $csv_path > filter.log 2>&1 &