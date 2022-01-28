#!/usr/bin/env python3


import argparse
from rdkit import Chem
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, help="input file")
    parser.add_argument('output', type=Path, help="output file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    smiles = [s.strip() for s in open(args.input)]
    canon_smiles = [Chem.CanonSmiles(s) for s in smiles]
    with open(args.output, 'w') as outfile:
        outfile.writelines((f'{s}\n' for s in canon_smiles))
    return


if __name__ == "__main__":
    main()
    
