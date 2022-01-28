#!/usr/bin/env python3

import sys
import os
import argparse
import multiprocessing
from collections import Counter


import numpy as np
import pandas as pd
import scipy
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect


import desmiles
from desmiles.data import Vocab, FpSmilesList, DesmilesLoader, DataBunch
from desmiles.learner import desmiles_model_learner
from desmiles.models import Desmiles, RecurrentDESMILES
from desmiles.models import get_fp_to_embedding_model, get_embedded_fp_to_smiles_model
from desmiles.utils import load_old_pretrained_desmiles, load_pretrained_desmiles
from desmiles.utils import accuracy4
from desmiles.utils import smiles_idx_to_string
from desmiles.learner import OriginalFastaiOneCycleScheduler, Learner
from desmiles.decoding.astar import AstarTreeParallelHybrid as AstarTree


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


## check: this might be in desmiles.utils
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


#######


def simple_smiles_fail(sm):
    # faster and safer processing of really bad SMILES
    return ((sm.count("(") != sm.count(")")) |
            (sm.count("[") != sm.count("]")) |
            (len(sm.strip()) == 0))


# Return num_return molecules, if possible within num_max_try iterations of the algorithm,
# otherwise return as many as you got.
def sample_astar(model, smiles, fp=None, num_return=20, cutoff=0, num_expand=2000, num_max_try=1000):
    "sample using parallel hybrid astar"
    if fp is None:
        fp = smiles_to_fingerprint(smiles, as_tensor=True)
    astar = AstarTree(fp, model, num_expand=num_expand)
    results = set()
    for i in range(num_max_try):
        nlp, generated_smiles_idx = next(astar)
        generated_smiles = smiles_idx_to_string(generated_smiles_idx)
        if simple_smiles_fail(generated_smiles):
            continue
        print(i, generated_smiles)
        try:
            mol = Chem.MolFromSmiles(generated_smiles)
            print(i, mol)
            if mol is not None:
                results.add(canon_smiles(generated_smiles))  # keep set of canonical smiles
        except:
            pass
        if len(results) >= num_return:
            return results
    print("NOTE: sample_astar didn't return enough molecules")
    return results


#######


def get_training_smiles(fname, col1="SMILES_1", col2="SMILES_2"):
    "return all canonical smiles in the training set"
    tmp = load_pairs(fname, col1, col2)
    training_smiles = smiles_list_to_canon(list(set(tmp.SMILES_1) | set(tmp.SMILES_2)))
    return training_smiles


def read_enamine_real_smiles(fname):
    return [x.strip().split()[0] for x in open(fname)]


########


def main():
    args = get_parser().parse_args()
    # First setup the workdir and change into it
    try:
        os.mkdir(args.workdir, 0o755)
    except OSError:
        print(f'failed to make directory {args.workdir}')
        sys.exit(1)
    os.chdir(args.workdir)

    # Read the input (random) molecules
    smiles = read_enamine_real_smiles(args.input_smiles)

    # Read the set of training molecules in canonical smiles form
    training_smiles = get_training_smiles(args.training_pairs)

    # Read the pre-trained learner
    learner = load_pretrained_desmiles(args.learner, return_learner=True)

    # Create the recurrent DESMILES model from fingerprint input
    model = learner.model
    model.eval()
    model = RecurrentDESMILES(model)

    # How many molecules per molecule
    num_return = args.num_return
    num_expand = args.num_expand
    num_max_try = args.num_max_try
    
    total = Counter()  # Keep track of the times we generated each molecule
    with open("samples.csv", "w") as out:
        out.write("SMILES_from,SMILES_to\n")

        for s in tqdm(smiles):
            results = sample_astar(model, s, num_return=num_return, num_expand=num_expand, num_max_try=num_max_try)
            total.update(results)
            for x in results:
                out.write(f'{s},{x}\n')

    # The rest is optional, since we've saved the new molecules already.

    with open("uniques.csv", 'w') as out:
        out.write("SMILES,count\n")
        for k, v in total.most_common():
            out.write(f"{k},{v}\n")

    unique_training = set(training_smiles)
    novel_results = set(total.keys()).difference(unique_training)
    with open("novel.csv", 'w') as out:
        out.write("SMILES\n")
        for x in novel_results:
            out.write(f'{x}\n')


def get_parser():
    parser = argparse.ArgumentParser()
    # Directory where all output goes.
    # Will create 3 output files: samples.csv, uniques.csv, novel.csv
    parser.add_argument('-w', '--workdir',
                        help="directory with output",
                        type=os.path.abspath, required=True)
    parser.add_argument('-l', '--learner',
                        help="name of saved model",
                        type=os.path.abspath, required=True)
    parser.add_argument('-n', '--num_return',
                        help="molecules to output for each input molecule",
                        type=int, default=30)
    parser.add_argument('-m', '--num_max_try',
                        help="maximal number of astar iterations",
                        type=int, default=1000)
    parser.add_argument('-x', '--num_expand',
                        help="batch expansions to try on GPU astar",
                        type=int, default=1000)
    # The list of input molecules, one smiles per line, with the smiles as first element.
    parser.add_argument('-i', '--input_smiles',
                        help="list of input smiles; no header",
                        type=os.path.abspath, required=True)
    # In principle the next argument is optional.
    # We use the training molecules to eliminate them from the output file novel.csv
    parser.add_argument('-t', '--training_pairs',
                        help="list of training molecules",
                        type=os.path.abspath, required=True)
    return parser


if __name__ == "__main__":
    maindoc = """
    Resurrect a finetuned model and apply it to new molecules.
    """
    main()
