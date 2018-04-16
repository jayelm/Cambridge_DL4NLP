import numpy as np
import pandas as pd
import csv
import pickle
import sys


embeddings = pd.read_csv(sys.argv[1],
                         header=None,
                         index_col=False,
                         sep=' ',
                         quoting=csv.QUOTE_NONE)

words = np.array(embeddings[0])
vecs = embeddings.iloc[:, 1:].as_matrix().astype(np.float32)
del embeddings
wv_map = dict(zip(words, vecs))

with open(sys.argv[1].replace('.txt', '.pkl'), 'wb') as fout:
    pickle.dump(wv_map, fout)
