import numpy as np
import pandas as pd
import csv
import pickle


embeddings = pd.read_csv('glove.6B.300d.txt',
                         header=None,
                         index_col=False,
                         sep=' ',
                         quoting=csv.QUOTE_NONE)

words = np.array(embeddings[0])
vecs = embeddings.iloc[:, 1:].as_matrix().astype(np.float32)
del embeddings
wv_map = dict(zip(words, vecs))

with open('glove.6B.300d.pkl', 'wb') as fout:
    pickle.dump(wv_map, fout)
