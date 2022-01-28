#!/usr/bin/env python3

"""
Sample a number of variants of each molecule in a CSV file 
using a pretrained DESMILES model.  Example usage for getting
100 molecules for each input molecule:

time ./sample_variants_of_input.py --max_try 1000 --num_expand 500 -n 100 -i ml_results.batch4.csv --verbose  --dont-dump-model -w variants_sample_1080ti_10K_20K --min-row=10000 --max-row=20000 2>&1 >& 10K-20K_1080ti.log


The output is placed in the file variants_sample_1080ti_10K_20K/samples.csv
"""


import sys
import socket
import argparse
import time
import os
from pathlib import Path
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import pickle
import fastai
from desmiles.data import Vocab, FpSmilesList, DesmilesLoader, DataBunch
from desmiles.config import DATA_DIR, MODEL_train_val1
from desmiles.utils import load_old_pretrained_desmiles, load_pretrained_desmiles, decoder
from desmiles.decoding.astar import AstarTreeParallelHybrid as AstarTree
from desmiles.models import Desmiles, RecurrentDESMILES


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


def canon_smiles(x):
    "Shortcut to return canonical smiles with chiral info"
    return Chem.CanonSmiles(x, useChiral=True)


def load_good_model(model_fn, itos_fn):
    "Load an old desmiles model with default hyperparameters"
    model = load_old_pretrained_desmiles(model_fn, return_learner=True, itos_fn=itos_fn)
    return model
    

def simple_smiles_fail(sm):
    # faster and safer processing of really bad SMILES
    return ((sm.count("(") != sm.count(")")) |
            (sm.count("[") != sm.count("]")) |
            (len(sm.strip()) == 0))


def get_my_decoder(itos_fn):
    from functools import partial
    itos = [s.strip() for i,s in enumerate(open(itos_fn, encoding='utf-8'))]
    return partial(decoder, itos=itos)


def get_smiles_idx_to_string(itos_fn):
    my_decoder = get_my_decoder(itos_fn)
    def smiles_idx_to_string(smiles_idx):
        return my_decoder(smiles_idx[smiles_idx > 0].tolist())
    return smiles_idx_to_string


# Return num_return molecules, if possible within num_max_try iterations of the algorithm,
# otherwise return as many as you got.
def sample_astar(model, smiles, smiles_idx_to_string, fp=None, num_return=20, cutoff=0, num_expand=2000, num_max_try=1000):
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


def main():
    """
    Get num_return molecules for each of the input smiles
    """
    t_start = time.time()
    args = parse_args()
    df = pd.read_csv(args.input_smiles)
    df = df.iloc[args.min_row:args.max_row]
    df['canon_SMILES'] = df['SMILES'].map(canon_smiles)
    smiles = [canon_smiles(x) for x in df['canon_SMILES']]

    if args.verbose:
        print(f'Canonicalized the input smiles. Time so far {time.time() - t_start}')
    os.makedirs(args.workdir, mode=0o755, exist_ok=True)
    os.chdir(args.workdir)
    if Path(args.fpname).exists():
        fps = pickle.load(open(args.fpname, 'rb'))
    else:
        fps = [smiles_to_fingerprint(x) for x in smiles]
        pickle.dump(fps, open(args.fpname, 'wb'))
    if args.verbose:
        print(f'Got fingerprints. Time so far {time.time() - t_start}')
    model_dump_fname = os.path.join("models", args.modeldump + ".pth")    
    if Path(model_dump_fname).exists():
        if args.verbose:
            print(f'Loading dumped model {model_dump_fname}')
        model = load_pretrained_desmiles(args.modeldump, return_learner=True, itos_fn=args.itos_fn)
    else:
        if args.verbose:
            print(f'loading default model {args.model}')
        model = load_good_model(args.model, itos_fn=args.itos_fn)
    if args.verbose:
        print(f'Loaded model.  Time so far {time.time() - t_start}')
    if not Path(model_dump_fname).exists() and not args.dont_dump_model:
        model.save(args.modeldump)
        if args.verbose:
            print(f'Dumped local model. Time so far {time.time() - t_start}')
    rnn_desmiles = RecurrentDESMILES(model.model).eval()
    if args.verbose:
        print(f'Got recurrent model. Time so far {time.time() - t_start}')
    smiles_idx_to_string = get_smiles_idx_to_string(args.itos_fn)
    with open('samples.csv', 'w') as out:
        out.write('SMILES, SMILES_out\n')
        total = set()
        for s in smiles:
            results = sample_astar(rnn_desmiles, s, smiles_idx_to_string, num_return=args.num_return,
                                   num_expand=args.num_expand, num_max_try=args.max_try)
            total.update(results)
            for x in results:
                out.write(f'{s},{x}\n')
    t_end = time.time()

    if args.verbose:
        print(f'total walltime was: {t_end - t_start}')
        print(f'total different smiles generated: {len(total)}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workdir',
                        help="directory with output",
                        type=os.path.abspath, required=True)
    parser.add_argument('-l', '--model',
                        help="name of saved model",
                        type=os.path.abspath,
                        default=MODEL_train_val1)
    parser.add_argument('--itos_fn',
                        help="filename of int to string mapping for BPE",
                        type=os.path.abspath,
                        default=os.path.join(DATA_DIR, 'pretrained', 'id.dec8000'))
    parser.add_argument('--dont-dump-model',
                        help='do not keep a backup of the loaded model in workdir',
                        action='store_true')
    parser.add_argument('-n', '--num_return',
                        help="molecules to output for each input molecule",
                        type=int, default=100)
    parser.add_argument('-m', '--max_try',
                        help="maximal number of astar iterations",
                        type=int, default=2000)
    parser.add_argument('-x', '--num_expand',
                        help="batch expansions to try on GPU astar",
                        type=int, default=1000)
    parser.add_argument('--fpname',
                        help="pickle of the fingerprint file",
                        type=str, default="temp_fingerprints.pkl")
    parser.add_argument('--modeldump',
                        help="pickle of the particular model",
                        type=str, default="model_dump")
    parser.add_argument('--verbose',
                        help="print a few timining log messages",
                        action='store_true')
    # The list of input molecules, one smiles per line, with the smiles as first element.
    parser.add_argument('-i', '--input_smiles',
                        help="csv file with input smiles in column SMILES",
                        type=os.path.abspath, required=True)
    parser.add_argument('--min-row',
                        help='minimum row to keep in CSV input file',
                        default=0, type=int)
    parser.add_argument('--max-row',
                        help='maximal row (exclusive, or -1) to keep in CSV input file',
                        default=-1, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    maindoc = """
    Resurrect a finetuned model and apply it to new molecules.
    """
    main()
