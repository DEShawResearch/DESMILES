#!/usr/bin/env python3


import argparse
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.sparse
import multiprocessing


def smiles_to_fingerprint(smiles_str, sparse=False):
    rdmol = Chem.MolFromSmiles(smiles_str)
    fp = np.concatenate([np.asarray(GetMorganFingerprintAsBitVect(rdmol, 2, useChirality=True), dtype=np.int8), np.asarray(
        GetMorganFingerprintAsBitVect(rdmol, 3, useChirality=True), dtype=np.int8)])
    if sparse:
        return scipy.sparse.csr_matrix(fp)
    return fp

def smiles_to_fingerprints(many_smiles_strs):
    return scipy.sparse.vstack([smiles_to_fingerprint(s, sparse=True) for s in many_smiles_strs])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, help="input file")
    parser.add_argument('output', type=Path, help="output file")
    args = parser.parse_args()
    return args

def chunks(l, n):
    return (l[i:i + n] for i in range(0, len(l), n))


def main():
    args = parse_args()
    smiles = [s.strip() for s in open(args.input)]
    num_cpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cpu)
    chunked = list(chunks(smiles, 10000))
    fps = pool.map(smiles_to_fingerprints, chunked)
    fps = scipy.sparse.vstack(fps)
    scipy.sparse.save_npz(args.output, fps)
    return


if __name__ == "__main__":
    main()
    
