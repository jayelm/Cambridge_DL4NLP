#!/usr/bin/env bash

# ADAM (default)
# python3 train_definition_model.py --summary_dir ./logs/default_params/ --save_dir ./saves/default_params/ --num_epochs 100

# Faster learning rates
python3 train_definition_model.py --summary_dir ./logs/adam_001/ --save_dir ./saves/adam_001/ --num_epochs 100 --learning_rate 0.001
python3 train_definition_model.py --summary_dir ./logs/adam_01/ --save_dir ./saves/adam_01/ --num_epochs 100 --learning_rate 0.01

# RMSPROP
# python3 train_definition_model.py --optimizer rmsprop --summary_dir ./logs/rmsprop/ --save_dir ./saves/rmsprop/ --num_epochs 100

# GRAD DESCENT
# python3 train_definition_model.py --optimizer gradientdescent --summary_dir ./logs/gradientdescent/ --save_dir ./saves/gradientdescent/ --num_epochs 100

# Faster learning rate for gradient descent
python3 train_definition_model.py --optimizer gradientdescent --summary_dir ./logs/gradientdescent_001/ --save_dir ./saves/gradientdescent_001/ --num_epochs 100 --learning_rate 0.001
python3 train_definition_model.py --optimizer gradientdescent --summary_dir ./logs/gradientdescent_01/ --save_dir ./saves/gradientdescent_01/ --num_epochs 100 --learning_rate 0.01
