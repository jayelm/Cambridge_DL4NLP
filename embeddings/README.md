# Embeddings

Default (Felix's) embeddings obtainable via

    wget https://www.cl.cam.ac.uk/~sc609/downloads/data_practical.tgz
    tar -xzf data_practical.tgz

Which also generates the `../data/` folder.

Download GloVe embeddings:

    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip

Parse them into Felix's format:

    python parse_glove_embeddings.py
