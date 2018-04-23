#!/usr/bin/env bash

set -e

# ADAM (default)
# python3 train_definition_model.py --summary_dir ./logs/default_params/ --save_dir ./saves/default_params/ --num_epochs 100

# BOW encoder
# python3 train_definition_model.py --summary_dir ./logs/bow/ --save_dir ./saves/bow/ --num_epochs 100 --encoder_type BOW --model_name BOW

# BOW, rmsprop
# python3 train_definition_model.py --summary_dir ./logs/bow_rmsprop/ --save_dir ./saves/bow_rmsprop/ --num_epochs 100 --encoder_type BOW --model_name BOW --optimizer rmsprop

# BOW, rmsprop, GloVe vectors
# python3 train_definition_model.py --summary_dir ./logs/bow_rmsprop_glove/ --save_dir ./saves/bow_rmsprop_glove/ --num_epochs 100 --encoder_type BOW --model_name BOW --optimizer rmsprop --embeddings_path ./embeddings/glove.6B.300d.pkl

# RNN, rmsprop, GloVe vectors
# python3 train_definition_model.py --summary_dir ./logs/rnn_rmsprop_glove/ --save_dir ./saves/rnn_rmsprop_glove/ --num_epochs 100 --optimizer rmsprop --embeddings_path ./embeddings/glove.6B.300d.pkl

# BOW, rmsprop, glove, pretrained input
# python3 train_definition_model.py --summary_dir ./logs/bow_rmsprop_glove_pretrained/ --save_dir ./saves/bow_rmsprop_glove_pretrained/ --num_epochs 100 --encoder_type BOW --model_name BOW --optimizer rmsprop --pretrained_input --embeddings_path ./embeddings/glove.6B.300d.pkl

# BOW, rmsprop, glove, pretrained input, but further refine
# python3 train_definition_model.py --summary_dir ./logs/bow_rmsprop_glove_pretrained_refine/ --save_dir ./saves/bow_rmsprop_glove_pretrained_refine/ --num_epochs 100 --encoder_type BOW --model_name BOW --optimizer rmsprop --pretrained_input --embeddings_path ./embeddings/glove.6B.300d.pkl --input_trainable --oov_init normal

# EMBEDDINGS VS COVERAGE EXPTS
# BOW RMSPROP GLOVE, default coverage only
python3 train_definition_model.py --summary_dir ./logs/bow_rmsprop_glove_lowc/ --save_dir ./saves/bow_rmsprop_glove_lowc/ --num_epochs 100 --encoder_type BOW --model_name BOW --optimizer rmsprop --embeddings_path ./embeddings/glove.6B.300d.pkl --cover_only ./embeddings/GoogleWord2Vec.clean.normed.pkl

# WORD2VEC (higher coverage!)
python3 train_definition_model.py --summary_dir ./logs/bow_rmsprop_lowc/ --save_dir ./saves/bow_rmsprop_lowc/ --num_epochs 100 --encoder_type BOW --model_name BOW --optimizer rmsprop --embeddings_path ./embeddings/GoogleNews-vectors-negative300.pkl --cover_only ./embeddings/glove.6B.300d.pkl

# Faster learning rates
# python3 train_definition_model.py --summary_dir ./logs/adam_001/ --save_dir ./saves/adam_001/ --num_epochs 100 --learning_rate 0.001
# python3 train_definition_model.py --summary_dir ./logs/adam_01/ --save_dir ./saves/adam_01/ --num_epochs 100 --learning_rate 0.01

# RMSPROP
# python3 train_definition_model.py --optimizer rmsprop --summary_dir ./logs/rmsprop/ --save_dir ./saves/rmsprop/ --num_epochs 100

# GRAD DESCENT
# python3 train_definition_model.py --optimizer gradientdescent --summary_dir ./logs/gradientdescent/ --save_dir ./saves/gradientdescent/ --num_epochs 100

# Faster learning rate for gradient descent
# python3 train_definition_model.py --optimizer gradientdescent --summary_dir ./logs/gradientdescent_001/ --save_dir ./saves/gradientdescent_001/ --num_epochs 100 --learning_rate 0.001
# python3 train_definition_model.py --optimizer gradientdescent --summary_dir ./logs/gradientdescent_01/ --save_dir ./saves/gradientdescent_01/ --num_epochs 100 --learning_rate 0.01
