#!/usr/bin/env bash

python3 train_definition_model.py --summary_dir ./logs/default_params/ --save_dir ./saves/default_params/ --num_epochs 100

python3 train_definition_model.py --optimizer rmsprop --summary_dir ./logs/rmsprop/ --save_dir ./saves/rmsprop/ --num_epochs 100

python3 train_definition_model.py --optimizer gradientdescent --summary_dir ./logs/gradientdescent/ --save_dir ./saves/gradientdescent/ --num_epochs 100
