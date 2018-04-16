# Embeddings

Default (Felix's) embeddings obtainable via

    wget https://www.cl.cam.ac.uk/~sc609/downloads/data_practical.tgz
    tar -xzf data_practical.tgz

Which also generates the `../data/` folder.

Download GloVe embeddings:

    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip

Parse them into Felix's format:

    # Generates ./glove.6B.300d.pkl
    python parse_glove_embeddings.py ./glove.6B.300d.txt

To generate mimick embeddings, use `train_mimick.sh` in the parent diretory after downloading GloVe embeddings and parsing them with `parse_glove_embeddings.py`
