#!/usr/bin/env bash

set -e

# Train Mimick LSTM-based model on GloVe embeddings.

# Load OOV vocab
python get_oovs.py

# mimick subfolder needs utils
cp Mimick/utils.py Mimick/mimick/

cd Mimick/mimick/

# Make dataset
python2 make_dataset.py --vectors ../../embeddings/glove.6B.300d.txt --w2v-format --output ../../embeddings/glove.6B.300d.mimick.pkl

# Train model
python2 model.py --dataset ../../embeddings/glove.6B.300d.mimick.pkl \
    --vocab ../../embeddings/glove.6B.300d.oov.txt  \
    --output ../../embeddings/glove.6B.300d.oov.mimick.txt  \
    --model-out ../../embeddings/glove.6B.300d.mimick.model.bin

# Combine oovs and original embeddings
cd ../../

tail -n +2 embeddings/glove.6B.300d.oov.mimick.txt > embeddings/temp
# Trim whitespace
cat embeddings/temp embeddings/glove.6B.300d.txt | awk '{$1=$1};1' > embeddings/glove.6B.300d.with_oov.txt
rm embeddings/temp

cd embeddings

python parse_glove_embeddings.py glove.6B.300d.with_oov.txt

cd ..
