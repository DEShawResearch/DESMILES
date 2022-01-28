#!/usr/bin/env python3

import argparse
from rdkit import Chem
import numpy as np
import pickle
from rdkit.Chem import AllChem


def load_model():
    with open('./clf_py36.pkl', 'rb') as infile:
        model = pickle.load(infile)
    return model


def get_score(smile, model):
    mol = Chem.MolFromSmiles(smile)
    if mol:
        fp = fingerprints_from_mol(mol)
        score = model.predict_proba(fp)[:, 1]
        return float(score)
    return 0.0


def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = idx % size
        nfp[0, nidx] += int(v)
    return nfp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = load_model()
    smiles = [s.strip() for s in open(args.input_file)]
    drd2_scores = np.asarray([get_score(s, model) for s in smiles])
    np.save(args.output_file, drd2_scores)
    return


if __name__ == "__main__":
    main()
    
