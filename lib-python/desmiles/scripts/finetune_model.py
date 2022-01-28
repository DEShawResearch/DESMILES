#!/usr/bin/env python3


import sys
import os
import argparse
import tempfile
import multiprocessing
import functools
import subprocess
from pathlib import Path


import numpy as np
import pandas as pd
import scipy


from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

import desmiles
from desmiles.data import Vocab, FpSmilesList, DesmilesLoader, DataBunch
from desmiles.config import DATA_DIR, MODEL_train_val1
from desmiles.learner import desmiles_model_learner
from desmiles.models import Desmiles, RecurrentDESMILES
from desmiles.models import get_fp_to_embedding_model, get_embedded_fp_to_smiles_model
from desmiles.utils import load_old_pretrained_desmiles, load_pretrained_desmiles
from desmiles.utils import accuracy4
from desmiles.learner import OriginalFastaiOneCycleScheduler, Learner


def load_pairs(csv_fname, col1="SMILES_1", col2="SMILES_2"):
    "Load pairs of SMILES from columns SMILES_1, SMILES_2"
    df = pd.read_csv(csv_fname)
    return df.loc[:, df.columns.isin((col1, col2))].copy()


def canon_smiles(x):
    return Chem.CanonSmiles(x, useChiral=True)


def smiles_list_to_canon(slist):
    "convert a list of smiles to a list of rdkit canonical chiral smiles"
    with multiprocessing.Pool() as p:
        result = p.map(canon_smiles, slist)
    return result


def pairs_to_canon_dict(pairs):
    "Take pairs of smiles, get set of all smiles and return dict smile: canon"
    s1 = pairs['SMILES_1'].values.tolist()
    s2 = pairs['SMILES_2'].values.tolist()
    smiles = list(set(s1) | set(s2))
    canon = smiles_list_to_canon(smiles)
    return {x: y for x, y in zip(smiles, canon)}


def canon2bpe(canon):
    "Convert a list of smiles (typically canonical) to their BPE, return dict canon: BPE"
    #  Work in a temporary directory and cleanup in case of errors
    with tempfile.TemporaryDirectory() as tmpdirname:
        #  Write smiles into temporary smiles.smi file
        sm_fname = os.path.join(tmpdirname, 'smiles.smi')
        with open(sm_fname, 'w') as f:
            for s in canon:
                f.write(s + '\n')
        #  Write corresponding BPE into smiles.bpe file
        bpe_fname = os.path.join(tmpdirname, 'smiles.bpe')
        cmd = f"spm_encode --model {DATA_DIR}/pretrained/bpe_v8000.model --output {bpe_fname} {sm_fname} --output_format=id --extra_options=bos:eos"
        run = subprocess.run(cmd.split())
        assert run.returncode == 0
        #  Create pbe list (of lists) before the temporary directory is deleted
        with open(bpe_fname) as bpe_file:
            bpe = [[int(x) for x in y.strip().split()] for y in bpe_file]

    # Return a dictionary from canonical smiles to byte pair encoding
    return {x: y for x, y in zip(canon, bpe)}


def c2b2rectbpe(canon2bpe):
    "Convert a map of canon: bpe to a map that has BPE as numpy int16 arrays of equal length"
    smiles, bpe = map(list, zip(*canon2bpe.items()))
    maxlen = max(list(map(len, bpe)))
    desmiles_enc = np.zeros((len(smiles), maxlen), dtype=np.int16)
    for i, x in enumerate(bpe):
        desmiles_enc[i, :len(x)] = x
    return {x: y for x, y in zip(smiles, desmiles_enc)}


def smiles_to_fingerprint(smiles_str, sparse=False, as_tensor=False):
    "Return the desmiles fp"
    rdmol = Chem.MolFromSmiles(smiles_str)
    fp = np.concatenate([
        np.asarray(GetMorganFingerprintAsBitVect(rdmol, 2, useChirality=True), dtype=np.int8),
        np.asarray(GetMorganFingerprintAsBitVect(rdmol, 3, useChirality=True), dtype=np.int8)])
    if sparse:
        return scipy.sparse.csr_matrix(fp)
    if as_tensor:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.tensor(fp.astype(np.float32)).to(device)
    return fp


def canon2fp(canon):
    "Convert a list of (canonical) smiles to desmiles fp"
    with multiprocessing.Pool() as p:
        f = functools.partial(smiles_to_fingerprint, sparse=True)
        result = p.map(f, canon)
    return {x: y for x, y in zip(canon, result)}


#######


def load_top_pretrained_learner(db):
    model_fn = Path(MODEL_train_val1)
    learner = load_old_pretrained_desmiles(model_fn, return_learner=True)
    learner.metrics = [accuracy4]
    learner.data = db
    return learner

    
############


def create_my_db(
        fp,
        enc,
        bs,
        true_validation = False,
        start_validation = 0.8,
        itos_fn=os.path.join(DATA_DIR, 'pretrained', 'id.dec8000')):
    "Create a databunch for transfer learning"
    
    itos = [s.strip() for i, s in enumerate(open(itos_fn, encoding='utf-8'))]
    vocab = Vocab(itos)

    n = len(enc)
    enc = np.array(enc)
    inds = np.arange(n)
    inds = np.random.permutation(inds)
    val_inds = inds[int(start_validation*n):]
    if true_validation:
        trn_inds = inds[:int(start_validation*n)]
        trn_ds = FpSmilesList(enc[trn_inds], fp[trn_inds], vocab=vocab)
    else:
        trn_ds = FpSmilesList(enc, fp, vocab=vocab)        
    val_ds = FpSmilesList(enc[val_inds], fp[val_inds], vocab=vocab)

    trn_dl = DesmilesLoader(trn_ds, bs=bs, vocab=vocab)
    val_dl = DesmilesLoader(val_ds, bs=bs, vocab=vocab)
    db = DataBunch(trn_dl, val_dl)
    return db


def main():
    """
    Finetune the best DESMILES model.
    The conceptual process is the following:
    We start with a list of pairs of smiles (A->B with A, B neighbors, and B better than A)
    and we want to convert it to a list of pairs of fingerprints (of A) and BPE (of B).
    Schematically, this process works as follows:

    pairs(smiles) -> set(canon_smiles) -> canon: BPE -> canon: rectBPE
                              |
                               -> canon: fp

    pairs(smiles) -> pairs(canon_smiles) -> pairs(fp: rectBPE)
    """

    args = get_parser().parse_args()
    # Load the pairs of SMILES for fine tuning.
    p = load_pairs(args.training_pairs)
    print(f"FINETUNE: Loaded pairs dataset that looks like:\n{p[:2]}")
    print(f'FINETUNE: Total length of dataset is: {len(p)}')
    # Dictionary that transforms a SMILES from those pairs to a canonical SMILES
    s2c = pairs_to_canon_dict(p)
    # The lists of unique SMILES and corresponding canonical SMILES
    smiles, canon = map(list, zip(*s2c.items()))
    print(f'FINETUNE: Total number of SMILES: {len(smiles)}')
    print(f'FINETUNE: Total number of canonical SMILES: {len(set(canon))}')
    # How many SMILES were already canonicalized in input (often all or none)
    print('FINETUNE: Number of canonical smiles in input:', sum([x == y for x, y in zip(smiles, canon)]))
    # Get the byte-pair encoding for each unique smiles
    c2b = canon2bpe(canon)
    print(f'FINETUNE: got the Byte Pair Encoding of {len(c2b)} canonical SMILES')
    for c, b in list(c2b.items())[:2]:
        print(c, b)
    # Get the rectangular form of the dictionary
    c2r = c2b2rectbpe(c2b)
    # Get the desmiles fingerprints for each unique smiles
    c2fp = canon2fp(canon)
    print(f'FINETUNE: got the fingeprints for {len(c2fp)} canonical SMILES')
    for c, fp in list(c2fp.items())[:2]:
        print(c, fp.shape)
    # Get a set of pairs of fp -> rectBPE for each of the original pairs
    fps = []
    enc = []
    for s1, s2 in p.values.tolist():
        c1 = s2c[s1]
        c2 = s2c[s2]
        fps.append(c2fp[c1])
        enc.append(c2r[c2])
    fps = scipy.sparse.vstack(fps)
    print(f'FINETUNE: Got the lists of fingerprints: {fps.shape}')
    print(f'FINETUNE: Got the lists of encodings: {len(enc)}')
    print("FINETUNE: DONE with setup")
    print("#################################")

    db = create_my_db(fps, enc,
                      bs=args.batch_size,
                      true_validation=args.true_validation,
                      start_validation=args.start_validation)

    learner = load_top_pretrained_learner(db)

    num_epochs = args.num_epochs   # 1  
    max_lr = args.max_lr   #  0.001
    div_factor = args.div_factor   #   7

    one_cycle_linear_cb = OriginalFastaiOneCycleScheduler(learner, max_lr, div_factor=div_factor)
    if num_epochs > 0:
        learner.fit(num_epochs, callbacks=[one_cycle_linear_cb])

    learner.save(args.learner)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # finetuned model name
    parser.add_argument('-t', '--training_pairs',
                        help="filename of the training pairs (csv with SMILES_1, SMILES_2)",
                        type=os.path.abspath, required=True)
    parser.add_argument('-l', '--learner',
                        help="filename of the finetuned model",
                        type=os.path.abspath, required=True)
    parser.add_argument('-e', '--num_epochs',
                        help="number of epochs (0 for none)",
                        type=int, default=1)
    parser.add_argument('-m', '--max_lr',
                        help="maximal learning rate",
                        type=float, default=0.001)
    parser.add_argument('-d', '--div_factor',
                        help="div factor in one-cycle training",
                        type=int, default=7)
    parser.add_argument('-b', '--batch_size',
                        help="batch size",
                        type=int, default=200)
    parser.add_argument('-v', '--true_validation',
                        help="dont train on validation subset",
                        type=bool, default=False)
    parser.add_argument('-s', '--start_validation',
                        help="fraction of data at start of validation",
                        type=float, default=0.8)
    return parser


if __name__ == "__main__":
    maindoc = """
    finetune a DESMILES model using pairs of A->B, with B improved.
    """
    main()
