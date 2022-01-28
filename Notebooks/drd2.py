import numpy as np
import pandas as pd
import scipy.sparse
from pathlib import Path
# import sys; sys.path.append('../lib-python/')
from desmiles import *
from desmiles.config import DATA_DIR

def load_training_data(raise_prob=True):
    smiles, encoded, fps, pairs, probs = load_drd2_data()
    smiles, encoded, fps, pairs, probs, smile_to_enc, smile_to_fp, smile_to_prob = clean_up_data(smiles, encoded, fps, pairs, probs)
    original_smile, train_fp, train_enc = clean_data_to_training_data(smiles, encoded, fps, pairs, probs, smile_to_enc, smile_to_fp, smile_to_prob, raise_prob=raise_prob)
    return original_smile, train_fp, train_enc

def load_drd2_data():
    datadir = Path(os.path.join(DATA_DIR, 'notebooks', 'DRD2'))
    smiles = np.asarray([s.strip() for s in open(datadir / "drd2.smi")])
    encoded = np.load(datadir / "drd2.enc8000.npy")
    fps = scipy.sparse.load_npz(datadir / "fps_drd2.npz")
    pairs = pd.read_csv(datadir / 'train_pairs.txt', header=None, sep=" ").values
    probs = np.load(datadir / 'drd2_probs.npy')
    return smiles, encoded, fps, pairs, probs

def clean_data_to_training_data(smiles, encoded, fps, pairs, probs, smile_to_enc, smile_to_fp, smile_to_prob, raise_prob=True):
    train_enc = []
    train_fp = []
    original_smile = []
    for s1, s2 in pairs:
        # raise prob
        if raise_prob:
            if smile_to_prob[s1] < smile_to_prob[s2]:
                train_enc.append(smile_to_enc[s2])
                train_fp.append(smile_to_fp[s1]) 
                original_smile.append(s1)
            else:
                train_enc.append(smile_to_enc[s1])
                train_fp.append(smile_to_fp[s2])
                original_smile.append(s2)
        else:
            if smile_to_prob[s1] < smile_to_prob[s2]:
                train_enc.append(smile_to_enc[s1])
                train_fp.append(smile_to_fp[s2])
                original_smile.append(s2)
            else:
                train_enc.append(smile_to_enc[s2])
                train_fp.append(smile_to_fp[s1])
                original_smile.append(s1)
    train_enc = np.asarray(train_enc)
    original_smile = np.asarray(original_smile)
    train_fp = scipy.sparse.csr_matrix(np.asarray(train_fp))
    return original_smile, train_fp, train_enc

def clean_up_data(smiles, encoded, fps, pairs, probs):
    doesnt_have_unk = ~(encoded == 3).sum(axis=1).astype(np.bool)
    is_less_than_25 = (encoded > 0).sum(axis=1) < 25
    to_keep = doesnt_have_unk & is_less_than_25
    smiles = smiles[to_keep]
    encoded = encoded[to_keep]
    fps = np.asarray(fps[to_keep].todense())
    probs = probs[to_keep]
    smiles_set = set(smiles)
    pairs = np.asarray([p for p in pairs if p[0] in smiles_set and p[1] in smiles_set])
    smile_to_enc = {s:e for s,e in zip(smiles, encoded)}
    smile_to_fp = {s:fp for s,fp in zip(smiles, fps)}
    smile_to_prob = {s:prob for s,prob in zip(smiles, probs)}
    return smiles, encoded, fps, pairs, probs, smile_to_enc, smile_to_fp, smile_to_prob

def create_databunch(train_fp, train_enc, itos_fn, bs):
    itos = [s.strip() for i,s in enumerate(open(itos_fn, encoding='utf-8'))]
    vocab = Vocab(itos)
    num_tokens = len(itos)

    inds = np.arange(train_enc.shape[0])
    inds = np.random.permutation(inds)
    val_inds = inds[int(0.8*train_enc.shape[0]):]
    # validate on random %20 of training data
    # true validation will be measured with inversion
    trn_ds = FpSmilesList(train_enc, train_fp, vocab=vocab)
    val_ds = FpSmilesList(train_enc[val_inds], train_fp[val_inds], vocab=vocab)

    trn_dl = DesmilesLoader(trn_ds, bs=bs, vocab=vocab)
    val_dl = DesmilesLoader(val_ds, bs=bs, vocab=vocab)
    db = DataBunch(trn_dl, val_dl)
    return db

