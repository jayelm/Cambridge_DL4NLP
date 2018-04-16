"""
Write OOVs to file.
"""

from train_definition_model import load_pretrained_embeddings
from data_utils import initialize_vocabulary


VOCAB_FILE = './data/definitions/definitions_100000.vocab'

if __name__ == '__main__':
    embs_dict, pre_emb_dim = load_pretrained_embeddings('./embeddings/glove.6B.300d.pkl')
    vocab, _ = initialize_vocabulary(VOCAB_FILE)
    oovs = [v for v in vocab if v not in embs_dict]
    with open('./embeddings/glove.6B.300d.oov.txt', 'w') as fout:
        fout.write('\n'.join(oovs))
        fout.write('\n')
